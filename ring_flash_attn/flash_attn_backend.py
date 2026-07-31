"""Backend seam between the ring attention variants and the flash-attn kernels.

The ring variants only need four primitives: a block forward that returns
``(out, lse)`` and a block backward that writes into caller-supplied ``dq``/``dk``/``dv``,
each in a dense and a varlen flavour. This module normalises those four onto the
FA2 keyword set so the variant modules contain no version- or backend-specific
branching.

Three backends are available:

``fa2``
    ``flash_attn.flash_attn_interface`` (FlashAttention 2.x). The default. Routes
    through ``torch.ops.flash_attn.*`` where the installed versions register it, so
    the dense path stays traceable under ``torch.compile``.

``fa3``
    ``flash_attn_3`` (FlashAttention 3). Opt in with ``RING_FLASH_ATTN_BACKEND=fa3``.
    Keeps FA2's LSE layout, in-place ``dq``/``dk``/``dv`` contract, and ``-1`` window
    sentinel, and registers real ``torch.library`` custom ops, so it traces under
    ``torch.compile``. No dropout and no ALiBi. Dense and varlen share one op per
    direction. Its backward renames ``causal`` to ``is_causal`` and orders
    ``cu_seqlens_*`` before ``dq``/``dk``/``dv``, so calls here are keyword-only.

``fa4``
    ``flash_attn.cute.interface`` (FlashAttention 4, distributed as ``flash-attn-4``).
    Opt in with ``RING_FLASH_ATTN_BACKEND=fa4``. No dropout, no ALiBi, and no
    ``torch.library`` registration upstream, so it is not traceable under
    ``torch.compile``. Its backward additionally requires compute capability 9.0+.
"""

import importlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
from typing import Optional, Tuple

import torch

from .utils import get_default_args

__all__ = [
    "BACKEND",
    "CAPABILITIES",
    "Capabilities",
    "available_backends",
    "check_variant_supported",
    "fa_backward",
    "fa_forward",
    "fa_varlen_backward",
    "fa_varlen_forward",
    "is_fa3_available",
    "is_fa4_available",
]

_ENV_VAR = "RING_FLASH_ATTN_BACKEND"
_VALID_BACKENDS = ("fa2", "fa3", "fa4")
_FA4_INSTALL_HINT = (
    "install it with `pip install -e <flash-attention-checkout>/flash_attn/cute`"
)
_FA3_INSTALL_HINT = (
    "build it with `pip install -e <flash-attention-checkout>/hopper` (needs CUDA_HOME "
    "and compiles for sm90a by default)"
)

# FA3's interface module has moved between releases. Tried in this order; the one that
# resolves is recorded in FA3_MODULE so errors can name it.
#
# `hopper.flash_attn_interface` is only reachable if someone puts a source checkout on
# the path *and* has a built flash_attn_3._C from elsewhere -- the vendored hopper/ trees
# that appear in site-packages have no extension, so is_fa3_available() rejects them
# before this list is consulted. Kept as a last resort rather than a supported layout.
_FA3_MODULE_CANDIDATES = (
    "flash_attn_3.flash_attn_interface",
    "flash_attn_interface",
    "hopper.flash_attn_interface",
)


def _flash_attn_search_locations() -> Tuple[str, ...]:
    """Locate the ``flash_attn`` package directories without importing it.

    Deliberately avoids ``find_spec("flash_attn.cute.interface")``: resolving a
    *submodule* spec imports the parent package, which is precisely what fails when
    the FA2 extension module is broken. ``find_spec("flash_attn")`` only locates the
    package and does not execute its ``__init__.py``.
    """
    try:
        spec = importlib.util.find_spec("flash_attn")
    except Exception:
        # A broken parent package can make even locating it fail.
        return ()
    if spec is None or not spec.submodule_search_locations:
        return ()
    return tuple(spec.submodule_search_locations)


def _fa4_distribution_installed() -> bool:
    """Whether a flash-attn-4 distribution is registered with the installer.

    FA4's ``pyproject.toml`` names the distribution ``flash-attn-4`` while its
    ``cute/__init__.py`` looks up ``version("fa4")``, so both spellings are checked.
    Name lookup is normalised, so ``flash_attn_4`` matches ``flash-attn-4``.
    """
    for name in ("flash-attn-4", "fa4"):
        try:
            importlib.metadata.distribution(name)
            return True
        except Exception:
            continue
    return False


def is_fa4_available() -> bool:
    """Return whether flash-attn-4 looks importable, without importing it.

    A real import is unsuitable here -- FA4 pulls in ``nvidia-cutlass-dsl`` and
    ``quack-kernels`` and is slow to load -- so this works off packaging metadata and
    the filesystem.

    Note the presence of ``flash_attn/cute/`` alone is *not* sufficient: flash-attn
    2.8.x bundles an older, unrelated ``cute`` module (its own ``__version__`` is
    "0.1.0") whose ``_flash_attn_fwd`` has a different signature. Selecting it as FA4
    would fail at call time with a confusing TypeError. FA4 proper is distinguished by
    a registered distribution, or failing that by ``flash_fwd_combine.py``, which the
    bundled 2.8.x copy does not ship.
    """
    # Metadata first: an editable install (`pip install -e .../flash_attn/cute`) routes
    # through a path hook and never materialises a `cute/` directory next to flash_attn,
    # so a filesystem-only probe misses it.
    if _fa4_distribution_installed():
        return True
    # Fall back to the filesystem for a vendored or PYTHONPATH checkout with no metadata.
    return any(
        (Path(location) / "cute" / "interface.py").is_file()
        and (Path(location) / "cute" / "flash_fwd_combine.py").is_file()
        for location in _flash_attn_search_locations()
    )


def is_fa3_available() -> bool:
    """Return whether flash-attn-3 looks importable, without importing its interface.

    Probes the compiled extension ``flash_attn_3._C`` rather than the interface module,
    because the interface alone is not enough: several environments here carry a raw
    ``hopper/`` directory copied into site-packages with no extension built, so an
    interface-only probe reports available and then dies at the first kernel call.

    ``find_spec("flash_attn_3._C")`` is safe in a way the FA4 equivalent was not --
    ``flash_attn_3/__init__.py`` does not import the interface, so resolving this
    submodule does not pull in the kernels. We deliberately do *not* probe
    ``flash_attn_3.flash_attn_interface``, which would execute that module's
    ``import flash_attn_3._C`` at module scope.
    """
    try:
        return importlib.util.find_spec("flash_attn_3._C") is not None
    except Exception:
        # A broken or partially-installed parent package can make even locating it fail.
        return False


def available_backends() -> Tuple[str, ...]:
    """Backends usable in this environment, for tests to skip on.

    ``fa2`` is reported only if it genuinely imports -- an installed-but-broken
    flash-attn (e.g. built against a different torch) is not "available". That import
    is free when fa2 is already the active backend.

    ``fa4`` reflects installation, not a trial import, because importing it pulls in
    ``nvidia-cutlass-dsl`` and ``quack-kernels``. Note FA4 shares the ``flash_attn``
    package namespace, so a broken flash-attn 2.x makes FA4 unimportable too even
    when it is installed.
    """
    found = []
    try:
        importlib.import_module("flash_attn.flash_attn_interface")
        found.append("fa2")
    except Exception:
        pass
    if is_fa3_available():
        found.append("fa3")
    if is_fa4_available():
        found.append("fa4")
    return tuple(found)


class Capabilities:
    """What the active backend actually supports.

    Every flag gates a real call-time check in this module. Properties that no ring
    code can act on (such as ``torch.compile`` traceability) are documented rather
    than expressed here, so nothing in this record can be mistaken for a runtime
    guard that does not exist.
    """

    def __init__(
        self,
        name: str,
        supports_dropout: bool,
        supports_alibi: bool,
        supports_softcap: bool,
        supports_window: bool,
        supports_backward: bool,
    ):
        self.name = name
        self.supports_dropout = supports_dropout
        self.supports_alibi = supports_alibi
        self.supports_softcap = supports_softcap
        self.supports_window = supports_window
        self.supports_backward = supports_backward

    def __repr__(self) -> str:
        return f"Capabilities(name={self.name!r})"


def _select_backend() -> str:
    backend = os.environ.get(_ENV_VAR, "fa2").strip().lower()
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"{_ENV_VAR}={backend!r} is not a known backend. "
            f"Expected one of {', '.join(_VALID_BACKENDS)}."
        )
    return backend


BACKEND = _select_backend()


def _set_window(params: dict, window_size) -> None:
    """Apply the window in whichever arity the resolved signature uses.

    Both FA2 and FA3 have shipped releases taking either a ``window_size`` tuple or the
    two ints ``window_size_left``/``window_size_right``. ``params`` comes from signature
    introspection, so the key-presence check is meaningful rather than a guess.

    Shared by the fa2 and fa3 adapters. Not used by fa4, whose schema needs ``None``
    rather than the ``-1`` sentinel -- see ``_split_window`` in that branch.
    """
    if "window_size" in params:
        params["window_size"] = window_size
    else:
        params["window_size_left"] = window_size[0]
        params["window_size_right"] = window_size[1]


# ---------------------------------------------------------------------------
# fa2
# ---------------------------------------------------------------------------

if BACKEND == "fa2":
    try:
        import flash_attn
        from flash_attn.flash_attn_interface import (
            _flash_attn_backward,
            _flash_attn_forward,
            _flash_attn_varlen_backward,
            _flash_attn_varlen_forward,
        )
    except Exception as e:
        # Catch broadly: a flash_attn built against a different torch fails at
        # extension-load time as ImportError (undefined symbol), OSError, or
        # RuntimeError depending on the mismatch -- not ModuleNotFoundError.
        _alternates = []
        if is_fa3_available():
            _alternates.append(
                f"flash-attn-3 is installed -- set {_ENV_VAR}=fa3 to use it. It is a "
                "separate `flash_attn_3` package, so it is unaffected by whatever is "
                "wrong with flash-attn 2.x here."
            )
        if is_fa4_available():
            _alternates.append(
                f"flash-attn-4 is installed -- set {_ENV_VAR}=fa4 to use it. Note it "
                "installs into the same `flash_attn` package namespace, so if the "
                "failure above came from `flash_attn/__init__.py` itself, the broken "
                "flash-attn 2.x must be repaired or uninstalled first."
            )
        if _alternates:
            extra = " ".join(_alternates)
        else:
            extra = (
                f"No other backend was found either -- for flash-attn-3 {_FA3_INSTALL_HINT}, "
                f"or for flash-attn-4 {_FA4_INSTALL_HINT}."
            )
        raise ImportError(
            f"ring_flash_attn could not load the flash_attn (FA2) backend: {e}. {extra}"
        ) from e

    # FA >= 2.7 registers the python entry points as torch library custom ops, which
    # keeps them traceable under torch.compile. Preserved verbatim from the per-module
    # gates this seam replaces, including the asymmetry that the backward gate does not
    # consult flash_attn.__version__.
    if torch.__version__ >= "2.4.0" and flash_attn.__version__ >= "2.7.0":
        _wrapped_flash_attn_forward = torch.ops.flash_attn._flash_attn_forward
    else:
        _wrapped_flash_attn_forward = _flash_attn_forward

    if torch.__version__ >= "2.4.0":
        _wrapped_flash_attn_backward = torch.ops.flash_attn._flash_attn_backward
    else:
        _wrapped_flash_attn_backward = _flash_attn_backward

    CAPABILITIES = Capabilities(
        name="fa2",
        supports_dropout=True,
        supports_alibi=True,
        supports_softcap=True,
        supports_window=True,
        supports_backward=True,
    )

    def _unpack_forward(outputs):
        """FA <= 2.6 returns 8 values, FA >= 2.7 returns 4. We want out and lse."""
        if len(outputs) == 8:
            return outputs[0], outputs[5]
        assert len(outputs) == 4
        return outputs[0], outputs[1]

    def fa_forward(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        use_custom_op=True,
    ):
        fn = _wrapped_flash_attn_forward if use_custom_op else _flash_attn_forward
        params = get_default_args(_flash_attn_forward).copy()
        params.update(
            {
                "q": q,
                "k": k,
                "v": v,
                "dropout_p": dropout_p,
                "softmax_scale": softmax_scale,
                "causal": causal,
                "softcap": softcap,
                "alibi_slopes": alibi_slopes,
                "return_softmax": True and dropout_p > 0,
            }
        )
        _set_window(params, window_size)
        return _unpack_forward(fn(**params))

    def fa_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        use_custom_op=True,
    ):
        fn = _wrapped_flash_attn_backward if use_custom_op else _flash_attn_backward
        params = get_default_args(_flash_attn_backward).copy()
        params.update(
            {
                "dout": dout,
                "q": q,
                "k": k,
                "v": v,
                "out": out,
                "softmax_lse": softmax_lse,
                "dq": dq,
                "dk": dk,
                "dv": dv,
                "dropout_p": dropout_p,
                "softmax_scale": softmax_scale,
                "causal": causal,
                "softcap": softcap,
                "alibi_slopes": alibi_slopes,
                "deterministic": deterministic,
            }
        )
        _set_window(params, window_size)
        fn(**params)

    def fa_varlen_forward(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
    ):
        params = get_default_args(_flash_attn_varlen_forward).copy()
        params.update(
            {
                "q": q,
                "k": k,
                "v": v,
                "cu_seqlens_q": cu_seqlens_q,
                "cu_seqlens_k": cu_seqlens_k,
                "max_seqlen_q": max_seqlen_q,
                "max_seqlen_k": max_seqlen_k,
                "dropout_p": dropout_p,
                "softmax_scale": softmax_scale,
                "causal": causal,
                "softcap": softcap,
                "alibi_slopes": alibi_slopes,
                "return_softmax": True and dropout_p > 0,
            }
        )
        _set_window(params, window_size)
        return _unpack_forward(_flash_attn_varlen_forward(**params))

    def fa_varlen_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
    ):
        params = get_default_args(_flash_attn_varlen_backward).copy()
        params.update(
            {
                "dout": dout,
                "q": q,
                "k": k,
                "v": v,
                "out": out,
                "softmax_lse": softmax_lse,
                "dq": dq,
                "dk": dk,
                "dv": dv,
                "cu_seqlens_q": cu_seqlens_q,
                "cu_seqlens_k": cu_seqlens_k,
                "max_seqlen_q": max_seqlen_q,
                "max_seqlen_k": max_seqlen_k,
                "dropout_p": dropout_p,
                "softmax_scale": softmax_scale,
                "causal": causal,
                "softcap": softcap,
                "alibi_slopes": alibi_slopes,
                "deterministic": deterministic,
            }
        )
        _set_window(params, window_size)
        _flash_attn_varlen_backward(**params)


# ---------------------------------------------------------------------------
# fa3
# ---------------------------------------------------------------------------

elif BACKEND == "fa3":
    if not is_fa3_available():
        raise ImportError(
            f"{_ENV_VAR}=fa3 was requested but the flash_attn_3._C extension was not "
            f"found. A bare `hopper/` source directory is not enough -- the CUDA "
            f"extension must be built; {_FA3_INSTALL_HINT}."
        )

    FA3_MODULE = None
    _fa3_errors = []
    for _candidate in _FA3_MODULE_CANDIDATES:
        try:
            _fa3 = importlib.import_module(_candidate)
            FA3_MODULE = _candidate
            break
        except Exception as e:  # noqa: PERF203 - we want the reason for each candidate
            _fa3_errors.append(f"{_candidate}: {type(e).__name__}: {e}")
    if FA3_MODULE is None:
        raise ImportError(
            "ring_flash_attn found flash_attn_3._C but could not import the flash-attn-3 "
            "interface module. Tried:\n  " + "\n  ".join(_fa3_errors) + f"\n{_FA3_INSTALL_HINT}."
        )

    _flash_attn_forward = _fa3._flash_attn_forward
    _flash_attn_backward = _fa3._flash_attn_backward

    # FA3 can compile out features ring attention depends on; those surface only as
    # TORCH_CHECK failures deep in C++, so report them here instead.
    #
    # The config module is generated at build time (hopper/setup.py:89) and sits beside
    # the interface, so look for it next to whichever module resolved above before
    # falling back to the top-level name that setup.py's py_modules installs.
    #
    # Note the keys are FLASHATTENTION_DISABLE_* -- no underscore after FLASH -- while
    # the env vars that set them are FLASH_ATTENTION_DISABLE_*. Using the env-var
    # spelling here silently disables this whole check.
    _fa3_disable_flags = {
        "FLASHATTENTION_DISABLE_VARLEN": "the varlen ring paths",
        "FLASHATTENTION_DISABLE_LOCAL": "sliding-window attention",
        "FLASHATTENTION_DISABLE_BACKWARD": "training (backward)",
    }
    _fa3_config_candidates = []
    if "." in FA3_MODULE:
        _fa3_config_candidates.append(FA3_MODULE.rsplit(".", 1)[0] + ".flash_attn_config")
    _fa3_config_candidates.append("flash_attn_config")

    try:
        _flags = None
        for _cand in _fa3_config_candidates:
            try:
                _flags = importlib.import_module(_cand).CONFIG["build_flags"]
                break
            except Exception:
                continue
        # Values are real booleans (hopper/setup.py:51 computes them as `== "TRUE"`),
        # so plain truthiness is correct here.
        if isinstance(_flags, dict):
            for _flag, _why in _fa3_disable_flags.items():
                if _flags.get(_flag):
                    raise ImportError(
                        f"this flash-attn-3 build was compiled with {_flag}=TRUE, which "
                        f"disables {_why}. Rebuild without it, or use {_ENV_VAR}=fa2."
                    )
    except ImportError:
        raise
    except Exception:
        # The config module is optional and its shape is a build-time detail; an
        # unrecognised layout means "no opinion", never a failed import.
        pass

    CAPABILITIES = Capabilities(
        name="fa3",
        supports_dropout=False,
        supports_alibi=False,
        supports_softcap=True,
        supports_window=True,
        supports_backward=True,
    )

    # FA3 unifies dense and varlen into ONE op per direction -- there is no
    # _flash_attn_varlen_forward to import. The varlen wrappers below call the same two
    # ops, just supplying cu_seqlens_*/max_seqlen_*.
    #
    # Two shapes of drift are absorbed by introspecting the real signature: releases
    # differ on window arity (`window_size` tuple vs `window_size_left`/`_right` ints)
    # and on the backward causal kwarg (`causal` vs `is_causal`). get_default_args
    # handles the CustomOpDef via its `_init_fn` fallback and is cached per function,
    # so this costs nothing per ring step.
    _FA3_BWD_CAUSAL_KEY = (
        "is_causal" if "is_causal" in get_default_args(_flash_attn_backward) else "causal"
    )

    def _fa3_forward_params(
        q, k, v, softmax_scale, causal, window_size, softcap, dropout_p, alibi_slopes
    ):
        _check_unsupported(dropout_p, alibi_slopes)
        params = get_default_args(_flash_attn_forward).copy()
        params.update(
            {
                "q": q,
                "k": k,
                "v": v,
                "softmax_scale": softmax_scale,
                "causal": causal,
                "softcap": softcap,
            }
        )
        _set_window(params, window_size)
        return params

    def _fa3_backward_params(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        softmax_scale,
        causal,
        window_size,
        softcap,
        deterministic,
        dropout_p,
        alibi_slopes,
    ):
        _check_unsupported(dropout_p, alibi_slopes)
        params = get_default_args(_flash_attn_backward).copy()
        params.update(
            {
                "dout": dout,
                "q": q,
                "k": k,
                "v": v,
                "out": out,
                "softmax_lse": softmax_lse,
                # Must always be supplied: they are Optional, and the C++ silently
                # allocates throwaway buffers when omitted, computing the gradient and
                # discarding it with no error.
                "dq": dq,
                "dk": dk,
                "dv": dv,
                "softmax_scale": softmax_scale,
                _FA3_BWD_CAUSAL_KEY: causal,
                "softcap": softcap,
                "deterministic": deterministic,
            }
        )
        _set_window(params, window_size)
        return params

    def fa_forward(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        use_custom_op=True,  # accepted and ignored: the FA3 symbols *are* the custom ops
    ):
        params = _fa3_forward_params(
            q, k, v, softmax_scale, causal, window_size, softcap, dropout_p, alibi_slopes
        )
        outputs = _flash_attn_forward(**params)
        # (out, softmax_lse, out_accum, softmax_lse_accum); the accum pair is empty
        # unless num_splits > 1, which we never set.
        #
        # `out_` is deliberately left None: FA3's fake impl raises on a preallocated
        # output under tracing, so writing straight into a ring buffer here would cost
        # torch.compile support.
        return outputs[0], outputs[1]

    def fa_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        use_custom_op=True,  # accepted and ignored: the FA3 symbols *are* the custom ops
    ):
        params = _fa3_backward_params(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            softmax_scale,
            causal,
            window_size,
            softcap,
            deterministic,
            dropout_p,
            alibi_slopes,
        )
        # Writes into dq/dk/dv in place (mutates_args); returns softmax_d, unused here.
        _flash_attn_backward(**params)

    def fa_varlen_forward(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
    ):
        params = _fa3_forward_params(
            q, k, v, softmax_scale, causal, window_size, softcap, dropout_p, alibi_slopes
        )
        params.update(
            {
                "cu_seqlens_q": cu_seqlens_q,
                "cu_seqlens_k": cu_seqlens_k,
                "max_seqlen_q": max_seqlen_q,
                "max_seqlen_k": max_seqlen_k,
            }
        )
        outputs = _flash_attn_forward(**params)
        return outputs[0], outputs[1]

    def fa_varlen_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
    ):
        params = _fa3_backward_params(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            softmax_scale,
            causal,
            window_size,
            softcap,
            deterministic,
            dropout_p,
            alibi_slopes,
        )
        params.update(
            {
                "cu_seqlens_q": cu_seqlens_q,
                "cu_seqlens_k": cu_seqlens_k,
                "max_seqlen_q": max_seqlen_q,
                "max_seqlen_k": max_seqlen_k,
            }
        )
        _flash_attn_backward(**params)


# ---------------------------------------------------------------------------
# fa4
# ---------------------------------------------------------------------------

elif BACKEND == "fa4":
    if not is_fa4_available():
        raise ImportError(
            f"{_ENV_VAR}=fa4 was requested but flash_attn.cute was not found; "
            f"{_FA4_INSTALL_HINT}."
        )
    try:
        from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd
    except Exception as e:
        raise ImportError(
            f"ring_flash_attn could not load the flash-attn-4 backend: {e}. "
            "flash-attn-4 lives under the `flash_attn` package namespace, so a broken "
            "flash-attn 2.x install in the same environment breaks it too -- if the "
            "error above names flash_attn_2_cuda or flash_attn/__init__.py, repair or "
            "uninstall flash-attn 2.x rather than reinstalling FA4. FA4 also requires "
            f"nvidia-cutlass-dsl and quack-kernels; {_FA4_INSTALL_HINT}."
        ) from e

    CAPABILITIES = Capabilities(
        name="fa4",
        supports_dropout=False,
        supports_alibi=False,
        supports_softcap=True,
        supports_window=True,
        supports_backward=True,  # gated per-device, see _check_backward_arch
    )

    # Forward accepts sm_80 and up; backward is sm_90 and up.
    _MIN_BACKWARD_MAJOR = 9

    def _check_backward_arch(device) -> None:
        major, minor = torch.cuda.get_device_capability(device)
        if major < _MIN_BACKWARD_MAJOR:
            raise RuntimeError(
                f"flash-attn-4 backward requires compute capability "
                f"{_MIN_BACKWARD_MAJOR}.0+, but this device is sm_{major}{minor}. "
                "FA4 forward works on sm_80+; only the backward kernel is gated. "
                "Use RING_FLASH_ATTN_BACKEND=fa2 to train on this device."
            )

    def _split_window(window_size) -> Tuple[Optional[int], Optional[int]]:
        """Map FA2's -1 sentinel onto FA4's ``None``.

        FA4 reads these as literal bounds, so passing -1 through would be taken as a
        one-column window rather than "unbounded". Its own ``(-1, -1)`` special case
        only fires when *both* sides are negative, which is not general enough.
        """
        left, right = window_size
        return (
            None if left is None or left < 0 else left,
            None if right is None or right < 0 else right,
        )

    def fa_forward(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        use_custom_op=True,  # accepted and ignored: FA4 registers no torch ops
    ):
        _check_unsupported(dropout_p, alibi_slopes)
        window_left, window_right = _split_window(window_size)
        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            softcap=softcap,
            window_size_left=window_left,
            window_size_right=window_right,
            # Required: FA4 returns lse=None unless asked, and autograd.Function.forward
            # runs with grad disabled so the requires_grad shortcut never fires.
            return_lse=True,
        )
        return out, lse

    def fa_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        use_custom_op=True,  # accepted and ignored: FA4 registers no torch ops
    ):
        _check_unsupported(dropout_p, alibi_slopes)
        _check_backward_arch(q.device)
        window_left, window_right = _split_window(window_size)
        out_dq, out_dk, out_dv = _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            softmax_lse,
            softmax_scale=softmax_scale,
            causal=causal,
            softcap=softcap,
            window_size_left=window_left,
            window_size_right=window_right,
            deterministic=deterministic,
            dq=dq,
            dk=dk,
            dv=dv,
        )
        _copy_back(dq, dk, dv, out_dq, out_dk, out_dv)

    def fa_varlen_forward(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
    ):
        _check_unsupported(dropout_p, alibi_slopes)
        window_left, window_right = _split_window(window_size)
        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            softcap=softcap,
            window_size_left=window_left,
            window_size_right=window_right,
            return_lse=True,
        )
        return out, lse

    def fa_varlen_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
    ):
        _check_unsupported(dropout_p, alibi_slopes)
        _check_backward_arch(q.device)
        window_left, window_right = _split_window(window_size)
        out_dq, out_dk, out_dv = _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            softmax_lse,
            softmax_scale=softmax_scale,
            causal=causal,
            softcap=softcap,
            window_size_left=window_left,
            window_size_right=window_right,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            deterministic=deterministic,
            dq=dq,
            dk=dk,
            dv=dv,
        )
        _copy_back(dq, dk, dv, out_dq, out_dk, out_dv)

    def _copy_back(dq, dk, dv, out_dq, out_dk, out_dv) -> None:
        """Honour FA2's write-into-the-caller's-buffer contract.

        FA4 writes into the tensors it is handed, so these copies are normally no-ops.
        They matter only if a future FA4 version reallocates internally -- callers here
        hold views into ring buffers (see llama3's ``dkv_buffer[0][local_k_slice]``) and
        would silently lose the gradient.
        """
        for dst, src in ((dq, out_dq), (dk, out_dk), (dv, out_dv)):
            if dst is not None and src is not None and dst.data_ptr() != src.data_ptr():
                dst.copy_(src)


# ---------------------------------------------------------------------------
# Backend-agnostic guards (shared by all adapters, driven by CAPABILITIES)
# ---------------------------------------------------------------------------


def _check_unsupported(dropout_p, alibi_slopes) -> None:
    """Fail loudly rather than silently dropping an argument the backend cannot honour.

    Reads CAPABILITIES so the flags are load-bearing rather than descriptive. Under fa2
    every flag is True and this is a no-op.
    """
    if dropout_p and not CAPABILITIES.supports_dropout:
        raise NotImplementedError(
            f"the {CAPABILITIES.name} backend has no dropout support; got "
            f"dropout_p={dropout_p}. Use {_ENV_VAR}=fa2 if you need it."
        )
    if alibi_slopes is not None and not CAPABILITIES.supports_alibi:
        raise NotImplementedError(
            f"the {CAPABILITIES.name} backend has no ALiBi support; got a non-None "
            f"alibi_slopes. Use {_ENV_VAR}=fa2 if you need it."
        )


# Ring variants that are in scope for each non-default backend. Everything else is
# routed through this seam but was never exercised against those kernels, so it is
# refused rather than silently run with unverified numerics.
#
# This marks *scope*, not proven correctness -- adding a name asserts only that the
# variant is intended to work there. fa2 is absent because it is the reference backend
# and every variant is in scope.
_VALIDATED_VARIANTS = {
    "fa3": frozenset(
        {
            "llama3_flash_attn_varlen",
            "llama_fwd_ring_bwd_flash_attn",
            "ring_flash_attn_backward",
        }
    ),
    "fa4": frozenset(
        {
            "llama3_flash_attn_varlen",
            "llama_fwd_ring_bwd_flash_attn",
            "ring_flash_attn_backward",
        }
    ),
}

_BACKEND_DISPLAY_NAME = {"fa2": "flash-attn 2.x", "fa3": "flash-attn-3", "fa4": "flash-attn-4"}


def check_variant_supported(variant: str) -> None:
    """Refuse ring variants that are out of scope for the active backend.

    Called by the variant modules deliberately left out of the fa3/fa4 scope. Under
    fa2 (the default) this never fires, since fa2 has no entry in _VALIDATED_VARIANTS.
    """
    in_scope = _VALIDATED_VARIANTS.get(BACKEND)
    if in_scope is not None and variant not in in_scope:
        raise NotImplementedError(
            f"'{variant}' is not in scope for the "
            f"{_BACKEND_DISPLAY_NAME.get(BACKEND, BACKEND)} backend, so it is refused "
            f"rather than run with unverified numerics. In scope under {BACKEND}: "
            f"{', '.join(sorted(in_scope))}. "
            f"Unset {_ENV_VAR} (or set it to fa2) to use this variant."
        )
