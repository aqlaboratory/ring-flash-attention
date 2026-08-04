"""Bind the fa3 adapter's kwargs against the real flash-attn-3 signatures, without a GPU.

flash-attn-3 cannot be imported on a machine with no `flash_attn_3._C`, and its kernels
need sm90-class hardware. But the highest-risk errors in the adapter are pure
argument-binding mistakes, and those are checkable against the signature alone:

  * backward renames `causal` to `is_causal` -- a straight copy of the FA2 call raises
  * backward puts `cu_seqlens_*`/`max_seqlen_*` BEFORE `dq`/`dk`/`dv`, where FA2 puts
    `dq`/`dk`/`dv` immediately after `softmax_lse`, so a positional call misbinds
  * `dq`/`dk`/`dv` are Optional and the C++ silently allocates throwaways when omitted,
    computing the gradient and discarding it with no error
  * forward takes no `dropout_p`/`alibi_slopes`/`return_softmax`

The signatures are loaded from a flash-attention checkout's hopper/flash_attn_interface.py
with `flash_attn_3._C` stubbed, since `_init_fn` is the undecorated python function and its
signature survives without any kernel.

Set FLASH_ATTENTION_HOPPER to point at a checkout's hopper/ directory; the test skips if
it is not found.
"""

import inspect
import os
import sys
import types
import unittest
from pathlib import Path

_DEFAULT_HOPPER = Path.home() / "projects/repos/flash-attention/hopper"


def _hopper_dir():
    return Path(os.environ.get("FLASH_ATTENTION_HOPPER", _DEFAULT_HOPPER))


def _load_fa3_interface():
    """Import hopper/flash_attn_interface.py with the CUDA extension stubbed out."""
    import importlib.util

    path = _hopper_dir() / "flash_attn_interface.py"
    if not path.is_file():
        raise unittest.SkipTest(f"no flash-attn-3 interface at {path}")

    import torch

    # The module does `import flash_attn_3._C` at module scope purely to register ops.
    stub_pkg = types.ModuleType("flash_attn_3")
    stub_pkg.__path__ = []
    stub_ext = types.ModuleType("flash_attn_3._C")
    saved = {k: sys.modules.get(k) for k in ("flash_attn_3", "flash_attn_3._C")}
    sys.modules["flash_attn_3"] = stub_pkg
    sys.modules["flash_attn_3._C"] = stub_ext

    # torch.ops.flash_attn_3 resolves lazily, so binding it needs no real library.
    try:
        spec = importlib.util.spec_from_file_location("_fa3_iface_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:  # pragma: no cover - environment dependent
        raise unittest.SkipTest(f"could not load flash-attn-3 interface: {e!r}")
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return module


def _signature_of(op):
    """Real signature of a torch.library custom op (or a plain function)."""
    # inspect.signature on a CustomOpDef only reports (*args, **kwargs).
    return inspect.signature(getattr(op, "_init_fn", op))


class TestFA3ArgBinding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iface = _load_fa3_interface()
        cls.fwd_sig = _signature_of(cls.iface._flash_attn_forward)
        cls.bwd_sig = _signature_of(cls.iface._flash_attn_backward)

    # -- forward ---------------------------------------------------------------

    def test_forward_rejects_fa2_only_kwargs(self):
        """dropout_p / alibi_slopes / return_softmax must not be passed to FA3."""
        for absent in ("dropout_p", "alibi_slopes", "return_softmax"):
            self.assertNotIn(
                absent,
                self.fwd_sig.parameters,
                f"{absent} unexpectedly present; the adapter assumes FA3 has no such arg",
            )

    def test_forward_window_is_two_ints(self):
        self.assertIn("window_size_left", self.fwd_sig.parameters)
        self.assertIn("window_size_right", self.fwd_sig.parameters)
        self.assertNotIn("window_size", self.fwd_sig.parameters)
        # -1 is the native "unbounded" sentinel; FA4's None mapping must not leak here.
        self.assertEqual(self.fwd_sig.parameters["window_size_left"].default, -1)
        self.assertEqual(self.fwd_sig.parameters["window_size_right"].default, -1)

    def test_forward_binds_adapter_kwargs(self):
        params = {
            "q": object(),
            "k": object(),
            "v": object(),
            "softmax_scale": 0.125,
            "causal": True,
            "softcap": 0.0,
            "window_size_left": -1,
            "window_size_right": -1,
            "cu_seqlens_q": object(),
            "cu_seqlens_k": object(),
            "max_seqlen_q": 16,
            "max_seqlen_k": 32,
        }
        bound = self.fwd_sig.bind(**params)
        self.assertEqual(bound.arguments["causal"], True)
        self.assertEqual(bound.arguments["max_seqlen_q"], 16)

    # -- backward --------------------------------------------------------------

    def test_backward_uses_is_causal_not_causal(self):
        self.assertIn("is_causal", self.bwd_sig.parameters)
        self.assertNotIn("causal", self.bwd_sig.parameters)

    def test_backward_dqdkdv_bind_to_grad_params_not_cu_seqlens(self):
        """The reordering trap: cu_seqlens_*/max_seqlen_* sit before dq/dk/dv in FA3."""
        dq, dk, dv = object(), object(), object()
        params = {
            "dout": object(),
            "q": object(),
            "k": object(),
            "v": object(),
            "out": object(),
            "softmax_lse": object(),
            "dq": dq,
            "dk": dk,
            "dv": dv,
            "softmax_scale": 0.125,
            "is_causal": True,
            "softcap": 0.0,
            "deterministic": False,
            "window_size_left": -1,
            "window_size_right": -1,
        }
        bound = self.bwd_sig.bind(**params)
        # Identity, not just presence -- a positional misbind can still bind cleanly.
        self.assertIs(bound.arguments["dq"], dq)
        self.assertIs(bound.arguments["dk"], dk)
        self.assertIs(bound.arguments["dv"], dv)
        for grad_name, grad in (("dq", dq), ("dk", dk), ("dv", dv)):
            for other in ("cu_seqlens_q", "cu_seqlens_k", "sequed_q", "sequed_k"):
                self.assertIsNot(
                    bound.arguments.get(other),
                    grad,
                    f"{grad_name} misbound to {other}",
                )

    def test_backward_grad_params_are_optional_so_must_be_passed(self):
        """Documents why the adapter always passes dq/dk/dv: omitting them is silent."""
        for name in ("dq", "dk", "dv"):
            self.assertIs(
                self.bwd_sig.parameters[name].default,
                None,
                f"{name} is Optional; the C++ allocates a throwaway and discards the "
                "gradient without error when it is omitted",
            )

    def test_backward_seqused_typo_is_still_present(self):
        """Upstream misspells seqused_* as sequed_* in backward only.

        The adapter never passes seqused, but if this typo is ever fixed upstream a
        future change that does pass it would silently no-op. Fail loudly instead.
        """
        self.assertIn("sequed_q", self.bwd_sig.parameters)
        self.assertIn("sequed_k", self.bwd_sig.parameters)
        self.assertIn("seqused_q", self.fwd_sig.parameters)


class TestFA3AdapterParams(unittest.TestCase):
    """Bind the params the adapter actually builds against the real FA3 signatures.

    The tests above check the signature; these check *our* call against it, which is the
    thing that can actually be wrong. The seam is imported with RING_FLASH_ATTN_BACKEND=fa3
    while flash_attn_3 is stubbed to serve the checkout's real ops, so the adapter
    configures itself exactly as it would on Hopper -- including the introspection that
    picks `is_causal` over `causal`.
    """

    @classmethod
    def setUpClass(cls):
        import importlib

        import importlib.machinery

        iface = _load_fa3_interface()

        # The seam's probe uses importlib.util.find_spec, which consults sys.modules and
        # reads __spec__ -- a bare ModuleType has __spec__ = None and would look absent.
        pkg = types.ModuleType("flash_attn_3")
        pkg.__path__ = []
        pkg.__spec__ = importlib.machinery.ModuleSpec(
            "flash_attn_3", loader=None, is_package=True
        )
        ext = types.ModuleType("flash_attn_3._C")
        ext.__spec__ = importlib.machinery.ModuleSpec("flash_attn_3._C", loader=None)
        cls._saved_modules = {
            k: sys.modules.get(k)
            for k in (
                "flash_attn_3",
                "flash_attn_3._C",
                "flash_attn_3.flash_attn_interface",
                "ring_flash_attn.flash_attn_backend",
            )
        }
        sys.modules["flash_attn_3"] = pkg
        sys.modules["flash_attn_3._C"] = ext
        sys.modules["flash_attn_3.flash_attn_interface"] = iface
        pkg.flash_attn_interface = iface

        cls._saved_env = os.environ.get("RING_FLASH_ATTN_BACKEND")
        os.environ["RING_FLASH_ATTN_BACKEND"] = "fa3"
        sys.modules.pop("ring_flash_attn.flash_attn_backend", None)
        try:
            cls.seam = importlib.import_module("ring_flash_attn.flash_attn_backend")
        except Exception as e:  # pragma: no cover - environment dependent
            cls._restore()
            raise unittest.SkipTest(f"could not load seam under fa3: {e!r}")
        cls.fwd_sig = _signature_of(iface._flash_attn_forward)
        cls.bwd_sig = _signature_of(iface._flash_attn_backward)

    @classmethod
    def _restore(cls):
        if cls._saved_env is None:
            os.environ.pop("RING_FLASH_ATTN_BACKEND", None)
        else:
            os.environ["RING_FLASH_ATTN_BACKEND"] = cls._saved_env
        for k, v in cls._saved_modules.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    @classmethod
    def tearDownClass(cls):
        cls._restore()

    def test_backend_selected_and_capabilities(self):
        self.assertEqual(self.seam.BACKEND, "fa3")
        self.assertFalse(self.seam.CAPABILITIES.supports_dropout)
        self.assertFalse(self.seam.CAPABILITIES.supports_alibi)

    def test_adapter_picked_is_causal(self):
        self.assertEqual(self.seam._FA3_BWD_CAUSAL_KEY, "is_causal")

    def test_adapter_forward_params_bind(self):
        params = self.seam._fa3_forward_params(
            q="Q", k="K", v="V",
            softmax_scale=0.125, causal=True, window_size=(-1, -1),
            softcap=0.0, dropout_p=0.0, alibi_slopes=None,
        )
        bound = self.fwd_sig.bind(**params)  # must not raise
        self.assertEqual(bound.arguments["causal"], True)
        self.assertEqual(bound.arguments["window_size_left"], -1)
        self.assertNotIn("dropout_p", params)
        self.assertNotIn("alibi_slopes", params)

    def test_adapter_backward_params_bind_and_place_grads(self):
        dq, dk, dv = object(), object(), object()
        params = self.seam._fa3_backward_params(
            dout="DO", q="Q", k="K", v="V", out="O", softmax_lse="LSE",
            dq=dq, dk=dk, dv=dv,
            softmax_scale=0.125, causal=True, window_size=(-1, -1),
            softcap=0.0, deterministic=False, dropout_p=0.0, alibi_slopes=None,
        )
        bound = self.bwd_sig.bind(**params)  # must not raise
        self.assertIs(bound.arguments["dq"], dq)
        self.assertIs(bound.arguments["dk"], dk)
        self.assertIs(bound.arguments["dv"], dv)
        self.assertEqual(bound.arguments["is_causal"], True)
        self.assertNotIn("causal", params)

    def test_adapter_varlen_backward_params_bind(self):
        dq, dk, dv = object(), object(), object()
        params = self.seam._fa3_backward_params(
            dout="DO", q="Q", k="K", v="V", out="O", softmax_lse="LSE",
            dq=dq, dk=dk, dv=dv,
            softmax_scale=0.125, causal=True, window_size=(-1, -1),
            softcap=0.0, deterministic=False, dropout_p=0.0, alibi_slopes=None,
        )
        params.update({
            "cu_seqlens_q": "CQ", "cu_seqlens_k": "CK",
            "max_seqlen_q": 16, "max_seqlen_k": 32,
        })
        bound = self.bwd_sig.bind(**params)
        self.assertIs(bound.arguments["dq"], dq)
        self.assertEqual(bound.arguments["cu_seqlens_q"], "CQ")
        self.assertEqual(bound.arguments["max_seqlen_k"], 32)

    def test_adapter_rejects_dropout_and_alibi(self):
        with self.assertRaises(NotImplementedError):
            self.seam._fa3_forward_params(
                q="Q", k="K", v="V", softmax_scale=0.125, causal=True,
                window_size=(-1, -1), softcap=0.0, dropout_p=0.1, alibi_slopes=None,
            )
        with self.assertRaises(NotImplementedError):
            self.seam._fa3_forward_params(
                q="Q", k="K", v="V", softmax_scale=0.125, causal=True,
                window_size=(-1, -1), softcap=0.0, dropout_p=0.0, alibi_slopes="SLOPES",
            )

    def _assert_extras_are_declared_defaults(self, params, explicit, sig):
        """Every arg we pass but do not set must equal its own declared default.

        get_default_args makes the adapter pass all ~34 forward args explicitly, of
        which we set only a handful. That is a semantic no-op *only* if the rest carry
        the function's own defaults -- the same property that settled the equivalent
        question for FA2. _get_default_args force-sets `softcap` and pads non-defaulted
        params with None, so this is worth asserting rather than assuming.
        """
        for name, value in params.items():
            if name in explicit:
                continue
            declared = sig.parameters[name].default
            if declared is inspect.Parameter.empty:
                # Required params get None-padded by _get_default_args; we must be
                # setting all of those explicitly, or the call is malformed.
                self.fail(f"{name} is required but not set explicitly (got {value!r})")
            self.assertEqual(
                value, declared, f"{name} passed as {value!r}, declared default {declared!r}"
            )

    def test_forward_extras_equal_declared_defaults(self):
        params = self.seam._fa3_forward_params(
            q="Q", k="K", v="V",
            softmax_scale=0.125, causal=True, window_size=(-1, -1),
            softcap=0.0, dropout_p=0.0, alibi_slopes=None,
        )
        explicit = {
            "q", "k", "v", "softmax_scale", "causal", "softcap",
            "window_size_left", "window_size_right",
        }
        self._assert_extras_are_declared_defaults(params, explicit, self.fwd_sig)

    def test_backward_extras_equal_declared_defaults(self):
        params = self.seam._fa3_backward_params(
            dout="DO", q="Q", k="K", v="V", out="O", softmax_lse="LSE",
            dq="DQ", dk="DK", dv="DV",
            softmax_scale=0.125, causal=True, window_size=(-1, -1),
            softcap=0.0, deterministic=False, dropout_p=0.0, alibi_slopes=None,
        )
        explicit = {
            "dout", "q", "k", "v", "out", "softmax_lse", "dq", "dk", "dv",
            "softmax_scale", "is_causal", "softcap", "deterministic",
            "window_size_left", "window_size_right",
        }
        self._assert_extras_are_declared_defaults(params, explicit, self.bwd_sig)

    def test_out_of_scope_variant_refused_under_fa3(self):
        self.seam.check_variant_supported("llama3_flash_attn_varlen")  # in scope
        self.seam.check_variant_supported("ring_flash_attn_backward")  # in scope
        with self.assertRaises(NotImplementedError):
            self.seam.check_variant_supported("zigzag_ring_flash_attn")


class TestFA3BuildFlagGuard(unittest.TestCase):
    """The guard must fire on a build that compiled out something ring attention needs.

    Regression test for a real bug: the guard originally used the env-var spelling
    FLASH_ATTENTION_DISABLE_* while the generated config keys are FLASHATTENTION_DISABLE_*
    (no underscore after FLASH), so every lookup missed and the check was silently dead.
    """

    def _load_seam_with_config(self, build_flags):
        import importlib
        import importlib.machinery

        iface = _load_fa3_interface()

        pkg = types.ModuleType("flash_attn_3")
        pkg.__path__ = []
        pkg.__spec__ = importlib.machinery.ModuleSpec(
            "flash_attn_3", loader=None, is_package=True
        )
        ext = types.ModuleType("flash_attn_3._C")
        ext.__spec__ = importlib.machinery.ModuleSpec("flash_attn_3._C", loader=None)
        cfg = types.ModuleType("flash_attn_3.flash_attn_config")
        cfg.CONFIG = {"build_flags": build_flags}

        names = (
            "flash_attn_3",
            "flash_attn_3._C",
            "flash_attn_3.flash_attn_interface",
            "flash_attn_3.flash_attn_config",
            "ring_flash_attn.flash_attn_backend",
        )
        saved = {k: sys.modules.get(k) for k in names}
        saved_env = os.environ.get("RING_FLASH_ATTN_BACKEND")
        sys.modules["flash_attn_3"] = pkg
        sys.modules["flash_attn_3._C"] = ext
        sys.modules["flash_attn_3.flash_attn_interface"] = iface
        sys.modules["flash_attn_3.flash_attn_config"] = cfg
        os.environ["RING_FLASH_ATTN_BACKEND"] = "fa3"
        sys.modules.pop("ring_flash_attn.flash_attn_backend", None)
        try:
            return importlib.import_module("ring_flash_attn.flash_attn_backend")
        finally:
            if saved_env is None:
                os.environ.pop("RING_FLASH_ATTN_BACKEND", None)
            else:
                os.environ["RING_FLASH_ATTN_BACKEND"] = saved_env
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    def test_guard_fires_on_disabled_varlen(self):
        with self.assertRaises(ImportError) as cm:
            self._load_seam_with_config({"FLASHATTENTION_DISABLE_VARLEN": True})
        self.assertIn("FLASHATTENTION_DISABLE_VARLEN", str(cm.exception))

    def test_guard_fires_on_disabled_backward(self):
        with self.assertRaises(ImportError) as cm:
            self._load_seam_with_config({"FLASHATTENTION_DISABLE_BACKWARD": True})
        self.assertIn("training", str(cm.exception))

    def test_guard_silent_on_a_healthy_build(self):
        seam = self._load_seam_with_config(
            {
                "FLASHATTENTION_DISABLE_VARLEN": False,
                "FLASHATTENTION_DISABLE_LOCAL": False,
                "FLASHATTENTION_DISABLE_BACKWARD": False,
            }
        )
        self.assertEqual(seam.BACKEND, "fa3")

    def test_guard_tolerates_unexpected_config_shape(self):
        """An unrecognised layout means 'no opinion', never a failed import."""
        for weird in ([], "TRUE", None, {"SOMETHING_ELSE": True}):
            seam = self._load_seam_with_config(weird)
            self.assertEqual(seam.BACKEND, "fa3")


if __name__ == "__main__":
    unittest.main()
