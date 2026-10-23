"""Bind the fa4 adapter's calls against the real flash-attn-4 signatures, without a GPU.

flash-attn-4 needs cutlass-dsl and quack to import, and sm80+ hardware to run. But the
adapter's riskiest mistakes are argument-binding ones, which the signature alone decides:

  * ``_flash_attn_bwd`` orders ``(q, k, v, out, dout, lse)``, unlike FA2's ``dout`` first
  * windows are ``window_size_left``/``window_size_right`` and take ``None``, not ``-1``
  * ``lse`` only comes back with ``return_lse=True``
  * ``max_seqlen_q/k`` are rejected by older builds, so they must be dropped there

The signatures are parsed with ``ast`` from a flash-attention checkout's
``flash_attn/cute/interface.py`` -- no import, so none of FA4's dependencies are needed.
Set FLASH_ATTENTION_CUTE to that ``cute/`` directory; the test skips if it is not found.
"""

import ast
import inspect
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fa4_fakes import fake_fa4_seam

_DEFAULT_CUTE = Path.home() / "projects/repos/flash-attention/flash_attn/cute"
_NO_DEFAULT = object()


def _cute_dir():
    return Path(os.environ.get("FLASH_ATTENTION_CUTE", _DEFAULT_CUTE))


def signature_from_source(path, func_name):
    """Build an ``inspect.Signature`` for ``func_name`` from source, without importing it."""
    tree = ast.parse(Path(path).read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            a = node.args
            params = []
            positional = a.posonlyargs + a.args
            defaults = [_NO_DEFAULT] * (len(positional) - len(a.defaults)) + [
                object() for _ in a.defaults
            ]
            for arg, default in zip(positional, defaults):
                kind = (
                    inspect.Parameter.POSITIONAL_ONLY
                    if arg in a.posonlyargs
                    else inspect.Parameter.POSITIONAL_OR_KEYWORD
                )
                params.append(
                    inspect.Parameter(
                        arg.arg, kind,
                        default=inspect.Parameter.empty if default is _NO_DEFAULT else default,
                    )
                )
            if a.vararg:
                params.append(inspect.Parameter(a.vararg.arg, inspect.Parameter.VAR_POSITIONAL))
            for arg, default in zip(a.kwonlyargs, a.kw_defaults):
                params.append(
                    inspect.Parameter(
                        arg.arg, inspect.Parameter.KEYWORD_ONLY,
                        default=inspect.Parameter.empty if default is None else object(),
                    )
                )
            if a.kwarg:
                params.append(inspect.Parameter(a.kwarg.arg, inspect.Parameter.VAR_KEYWORD))
            return inspect.Signature(params)
    raise LookupError(f"{func_name} not found in {path}")


class Buf:
    """Stands in for a gradient tensor: identity is its address."""

    def __init__(self):
        self.copied_from = None

    def data_ptr(self):
        return id(self)

    def copy_(self, src):
        self.copied_from = src


def recorder(signature, returns):
    """A stand-in kernel entry point that reports like the real one under inspect."""
    calls = []

    def fn(*args, **kwargs):
        calls.append((args, kwargs))
        return returns

    fn.__signature__ = signature
    fn.calls = calls
    return fn


def strip_max_seqlen(signature):
    """A signature as older FA4 builds had it."""
    return signature.replace(
        parameters=[
            p for p in signature.parameters.values()
            if p.name not in ("max_seqlen_q", "max_seqlen_k")
        ]
    )


def strip_params(signature, names):
    return signature.replace(
        parameters=[p for p in signature.parameters.values() if p.name not in names]
    )


class _FA4Case(unittest.TestCase):
    max_seqlen_supported = True
    # bwd params absent from the build under test, and whether its bwd then allocates
    # fresh gradients instead of writing into ours
    bwd_missing = ()
    bwd_reallocates = False

    @classmethod
    def setUpClass(cls):
        path = _cute_dir() / "interface.py"
        if not path.is_file():
            raise unittest.SkipTest(f"no flash-attn-4 interface at {path}")
        fwd_sig = signature_from_source(path, "_flash_attn_fwd")
        bwd_sig = signature_from_source(path, "_flash_attn_bwd")
        if not cls.max_seqlen_supported:
            fwd_sig, bwd_sig = strip_max_seqlen(fwd_sig), strip_max_seqlen(bwd_sig)
        bwd_sig = strip_params(bwd_sig, cls.bwd_missing)
        cls.fwd_sig, cls.bwd_sig = fwd_sig, bwd_sig
        cls.dq, cls.dk, cls.dv = Buf(), Buf(), Buf()
        cls.kernel_grads = (
            (Buf(), Buf(), Buf()) if cls.bwd_reallocates else (cls.dq, cls.dk, cls.dv)
        )
        cls.fwd = recorder(fwd_sig, ("OUT", "LSE"))
        cls.bwd = recorder(bwd_sig, cls.kernel_grads)
        cm = fake_fa4_seam(fwd=cls.fwd, bwd=cls.bwd)
        cls.seam = cm.__enter__()
        cls.addClassCleanup(cm.__exit__, None, None, None)
        cap = mock.patch("torch.cuda.get_device_capability", return_value=(9, 0))
        cap.start()
        cls.addClassCleanup(cap.stop)
        cls.q = SimpleNamespace(device="cuda:0")

    def setUp(self):
        self.fwd.calls.clear()
        self.bwd.calls.clear()

    def last(self, fn):
        args, kwargs = fn.calls[-1]
        return args, kwargs


class TestFA4ArgBinding(_FA4Case):
    def test_dense_forward_binds_and_asks_for_lse(self):
        out, lse = self.seam.fa_forward(
            "Q", "K", "V", softmax_scale=0.125, causal=True, window_size=(-1, -1)
        )
        self.assertEqual((out, lse), ("OUT", "LSE"))
        args, kwargs = self.last(self.fwd)
        bound = self.fwd_sig.bind(*args, **kwargs)  # must not raise
        self.assertEqual([bound.arguments[n] for n in ("q", "k", "v")], ["Q", "K", "V"])
        self.assertIs(bound.arguments["return_lse"], True)
        self.assertEqual(bound.arguments["causal"], True)
        # FA2's -1 sentinel is a literal one-column window to FA4; it must become None
        self.assertIsNone(bound.arguments["window_size_left"])
        self.assertIsNone(bound.arguments["window_size_right"])
        self.assertNotIn("cu_seqlens_q", bound.arguments)

    def test_window_bounds_pass_through(self):
        self.seam.fa_forward("Q", "K", "V", window_size=(64, 0))
        _, kwargs = self.last(self.fwd)
        self.assertEqual((kwargs["window_size_left"], kwargs["window_size_right"]), (64, 0))

    def test_dense_backward_positional_order_and_grad_buffers(self):
        self.seam.fa_backward(
            "DO", self.q, "K", "V", "O", "LSE", self.dq, self.dk, self.dv,
            softmax_scale=0.125, causal=True, deterministic=True,
        )
        args, kwargs = self.last(self.bwd)
        bound = self.bwd_sig.bind(*args, **kwargs)
        # FA4 order is (q, k, v, out, dout, lse): not FA2's dout-first
        self.assertEqual(bound.arguments["out"], "O")
        self.assertEqual(bound.arguments["dout"], "DO")
        self.assertEqual(bound.arguments["lse"], "LSE")
        self.assertIs(bound.arguments["dq"], self.dq)
        self.assertIs(bound.arguments["dk"], self.dk)
        self.assertIs(bound.arguments["dv"], self.dv)
        self.assertIs(bound.arguments["deterministic"], True)

    def test_varlen_forward_binds(self):
        self.seam.fa_varlen_forward("Q", "K", "V", "CQ", "CK", 16, 32, causal=True)
        args, kwargs = self.last(self.fwd)
        bound = self.fwd_sig.bind(*args, **kwargs)
        self.assertEqual(bound.arguments["cu_seqlens_q"], "CQ")
        self.assertEqual(bound.arguments["cu_seqlens_k"], "CK")
        self.assertEqual(bound.arguments["max_seqlen_q"], 16)
        self.assertEqual(bound.arguments["max_seqlen_k"], 32)

    def test_varlen_backward_binds(self):
        self.seam.fa_varlen_backward(
            "DO", self.q, "K", "V", "O", "LSE", self.dq, self.dk, self.dv,
            "CQ", "CK", 16, 32, causal=True,
        )
        args, kwargs = self.last(self.bwd)
        bound = self.bwd_sig.bind(*args, **kwargs)
        self.assertEqual(bound.arguments["cu_seqlens_k"], "CK")
        self.assertEqual(bound.arguments["max_seqlen_q"], 16)
        self.assertIs(bound.arguments["dq"], self.dq)

    def test_copy_back_only_when_kernel_reallocated(self):
        self.seam.fa_backward(
            "DO", self.q, "K", "V", "O", "LSE", self.dq, self.dk, self.dv
        )
        self.assertIsNone(self.dq.copied_from)  # written in place: no copy
        new_dq = Buf()
        self.seam._copy_back(self.dq, self.dk, self.dv, new_dq, self.dk, self.dv)
        self.assertIs(self.dq.copied_from, new_dq)
        self.dq.copied_from = None

    def test_rejects_dropout_and_alibi(self):
        with self.assertRaises(NotImplementedError):
            self.seam.fa_forward("Q", "K", "V", dropout_p=0.1)
        with self.assertRaises(NotImplementedError):
            self.seam.fa_forward("Q", "K", "V", alibi_slopes=object())

    def test_backward_requires_hopper(self):
        with mock.patch("torch.cuda.get_device_capability", return_value=(8, 0)):
            with self.assertRaises(RuntimeError):
                self.seam.fa_backward(
                    "DO", self.q, "K", "V", "O", "LSE", self.dq, self.dk, self.dv
                )

    def test_deterministic_refused_on_sm12x_before_the_kernel(self):
        # FA4's own refusal is a bare assert; ours must fire first, for both entry points
        for cap in ((12, 0), (12, 1)):
            with self.subTest(cap=cap), mock.patch(
                "torch.cuda.get_device_capability", return_value=cap
            ):
                with self.assertRaisesRegex(RuntimeError, "deterministic=False"):
                    self.seam.fa_backward(
                        "DO", self.q, "K", "V", "O", "LSE", self.dq, self.dk, self.dv,
                        deterministic=True,
                    )
                with self.assertRaisesRegex(RuntimeError, "deterministic=False"):
                    self.seam.fa_varlen_backward(
                        "DO", self.q, "K", "V", "O", "LSE", self.dq, self.dk, self.dv,
                        "CQ", "CK", 16, 32, deterministic=True,
                    )
                self.assertEqual(self.bwd.calls, [])
                # non-deterministic backward still reaches the kernel there
                self.seam.fa_backward(
                    "DO", self.q, "K", "V", "O", "LSE", self.dq, self.dk, self.dv
                )
                self.assertEqual(len(self.bwd.calls), 1)
                self.bwd.calls.clear()

    def test_deterministic_allowed_on_datacenter_parts(self):
        for cap in ((9, 0), (10, 0)):
            with self.subTest(cap=cap), mock.patch(
                "torch.cuda.get_device_capability", return_value=cap
            ):
                self.seam.fa_backward(
                    "DO", self.q, "K", "V", "O", "LSE", self.dq, self.dk, self.dv,
                    deterministic=True,
                )
                _, kwargs = self.last(self.bwd)
                self.assertIs(kwargs["deterministic"], True)


class TestFA4OlderBuildWithoutMaxSeqlen(_FA4Case):
    """Older FA4 builds infer max seqlens from cu_seqlens and reject the kwargs."""

    max_seqlen_supported = False

    def test_signature_really_lacks_max_seqlen(self):
        self.assertNotIn("max_seqlen_q", self.fwd_sig.parameters)
        self.assertNotIn("max_seqlen_q", self.bwd_sig.parameters)

    def test_varlen_forward_drops_max_seqlen(self):
        self.seam.fa_varlen_forward("Q", "K", "V", "CQ", "CK", 16, 32)
        args, kwargs = self.last(self.fwd)
        self.fwd_sig.bind(*args, **kwargs)  # would raise TypeError if passed through
        self.assertNotIn("max_seqlen_q", kwargs)

    def test_varlen_backward_drops_max_seqlen(self):
        self.seam.fa_varlen_backward(
            "DO", self.q, "K", "V", "O", "LSE", self.dq, self.dk, self.dv,
            "CQ", "CK", 16, 32,
        )
        args, kwargs = self.last(self.bwd)
        self.bwd_sig.bind(*args, **kwargs)
        self.assertNotIn("max_seqlen_k", kwargs)


class TestFA4OlderBuildWithoutGradBuffers(_FA4Case):
    """FA4 builds before preallocated grads (and ``deterministic``) reject those kwargs."""

    bwd_missing = ("deterministic", "dq", "dk", "dv")
    bwd_reallocates = True

    def setUp(self):
        super().setUp()
        for buf in (self.dq, self.dk, self.dv):
            buf.copied_from = None

    def test_signature_really_lacks_them(self):
        for name in self.bwd_missing:
            self.assertNotIn(name, self.bwd_sig.parameters)

    def test_backward_drops_them_and_copies_back(self):
        for call in (
            lambda: self.seam.fa_backward(
                "DO", self.q, "K", "V", "O", "LSE", self.dq, self.dk, self.dv
            ),
            lambda: self.seam.fa_varlen_backward(
                "DO", self.q, "K", "V", "O", "LSE", self.dq, self.dk, self.dv,
                "CQ", "CK", 16, 32,
            ),
        ):
            call()
            args, kwargs = self.last(self.bwd)
            self.bwd_sig.bind(*args, **kwargs)  # would raise TypeError if passed through
            for name in self.bwd_missing:
                self.assertNotIn(name, kwargs)
            # the kernel's own gradients must land in the caller's buffers
            for dst, src in zip((self.dq, self.dk, self.dv), self.kernel_grads):
                self.assertIs(dst.copied_from, src)

    def test_deterministic_true_is_refused_not_dropped(self):
        with self.assertRaises(NotImplementedError):
            self.seam.fa_backward(
                "DO", self.q, "K", "V", "O", "LSE", self.dq, self.dk, self.dv,
                deterministic=True,
            )
        self.assertEqual(self.bwd.calls, [])


class TestMaxSeqlenCheckedPerFunction(unittest.TestCase):
    """fwd and bwd are checked independently: a build may differ between them."""

    def test_independent(self):
        sig_with = inspect.Signature(
            [inspect.Parameter("max_seqlen_q", inspect.Parameter.KEYWORD_ONLY, default=None)]
        )
        sig_without = inspect.Signature([])
        fwd = recorder(sig_with, ("O", "L"))
        bwd = recorder(sig_without, (None, None, None))
        with fake_fa4_seam(fwd=fwd, bwd=bwd) as seam:
            self.assertEqual(
                seam._max_seqlen_kwargs(seam._flash_attn_fwd, 1, 2),
                {"max_seqlen_q": 1, "max_seqlen_k": 2},
            )
            self.assertEqual(seam._max_seqlen_kwargs(seam._flash_attn_bwd, 1, 2), {})

    def test_var_keyword_counts_as_accepting(self):
        fn = recorder(
            inspect.Signature([inspect.Parameter("kw", inspect.Parameter.VAR_KEYWORD)]),
            None,
        )
        with fake_fa4_seam() as seam:
            self.assertEqual(
                seam._max_seqlen_kwargs(fn, 1, 2), {"max_seqlen_q": 1, "max_seqlen_k": 2}
            )


if __name__ == "__main__":
    unittest.main()
