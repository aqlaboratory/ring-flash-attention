"""Fake flash-attn installs for the CPU tests of backend selection.

Nothing here needs a GPU or a real flash-attn. Fakes are injected through ``sys.modules``
rather than ``sys.path``: a regular ``flash_attn`` package found anywhere on the path
beats a namespace one, so a fake placed on the path would silently lose to a real FA2 in
the environment and the "FA4-only" tests would test nothing.

``flash-attn-4`` registers a distribution, which ``is_fa4_available()`` consults, so the
fake FA4 also writes a real ``.dist-info`` directory and puts it on ``sys.path``.
"""

import contextlib
import importlib
import importlib.machinery
import os
import sys
import importlib.metadata
import tempfile
import types
from pathlib import Path
from unittest import mock

ENV_VAR = "RING_FLASH_ATTN_BACKEND"
_PURGED_PREFIXES = ("ring_flash_attn", "flash_attn", "flash_attn_3")


def write_dist_info(root, name="flash_attn_4", version="0.0.0"):
    d = Path(root) / f"{name}-{version}.dist-info"
    d.mkdir(parents=True, exist_ok=True)
    (d / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name.replace('_', '-')}\nVersion: {version}\n"
    )
    return d


def pkg(name, **attrs):
    """A namespace-package-like module: has a __path__ but no submodules of its own."""
    m = types.ModuleType(name)
    m.__path__ = []
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def module(name, **attrs):
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def fa4_modules(fwd=None, bwd=None):
    """An FA4-only ``flash_attn``: a namespace package holding just ``cute.interface``."""
    fwd = fwd or (lambda *a, **k: None)
    bwd = bwd or (lambda *a, **k: None)
    interface = module(
        "flash_attn.cute.interface", _flash_attn_fwd=fwd, _flash_attn_bwd=bwd
    )
    cute = pkg("flash_attn.cute", interface=interface)
    top = pkg("flash_attn", cute=cute)
    return {
        "flash_attn": top,
        "flash_attn.cute": cute,
        "flash_attn.cute.interface": interface,
    }


def fa3_modules():
    """Enough of ``flash_attn_3`` for ``is_fa3_available()``; no usable interface."""
    ext = module("flash_attn_3._C")
    return {"flash_attn_3": pkg("flash_attn_3"), "flash_attn_3._C": ext}


_HIDDEN_DISTRIBUTIONS = {"flash-attn", "flash_attn"}


def _hiding_metadata(real):
    """Wrap importlib.metadata.version/distribution so flash-attn (FA2) looks uninstalled.

    transformers decides whether FA2 is usable from package metadata, then imports
    ``flash_attn.bert_padding``. An environment that really has FA2 installed would leak
    that into an "FA4-only" scenario, where the namespace package has no such module.
    """

    def wrapper(name, *args, **kwargs):
        if name.lower().replace("_", "-") in {n.replace("_", "-") for n in _HIDDEN_DISTRIBUTIONS}:
            raise importlib.metadata.PackageNotFoundError(name)
        return real(name, *args, **kwargs)

    return wrapper


class _Absent:
    """Meta path finder making ``name`` and its submodules unimportable."""

    def __init__(self, name, exc=ModuleNotFoundError):
        self.name, self.exc = name, exc

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.name or fullname.startswith(self.name + "."):
            raise self.exc(f"fake: {fullname} is unavailable")
        return None


@contextlib.contextmanager
def fake_environment(
    modules=None, env=None, dist_infos=(), absent=(), absent_exc=ModuleNotFoundError
):
    """Run with a clean slate for ring/flash-attn modules, then restore everything.

    ``absent``: top-level names made unimportable (raising ``absent_exc``), i.e. a
    package that is not installed, or whose import blows up. ``env`` values of ``None``
    unset the variable.
    """
    saved_modules = {
        k: v
        for k, v in sys.modules.items()
        if k.split(".")[0] in _PURGED_PREFIXES
    }
    for k in saved_modules:
        del sys.modules[k]
    saved_env = {k: os.environ.get(k) for k in (env or {})}
    saved_path = list(sys.path)
    saved_meta = list(sys.meta_path)
    tmp = tempfile.TemporaryDirectory()
    patches = [
        mock.patch.object(
            importlib.metadata, fn, _hiding_metadata(getattr(importlib.metadata, fn))
        )
        for fn in ("version", "distribution")
    ]
    for patch in patches:
        patch.start()
    try:
        for k, v in (env or {}).items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for d in dist_infos:
            write_dist_info(tmp.name, d)
        if dist_infos:
            sys.path.insert(0, tmp.name)
        for name in absent:
            sys.meta_path.insert(0, _Absent(name, absent_exc))
        sys.modules.update(modules or {})
        importlib.invalidate_caches()
        yield
    finally:
        for patch in patches:
            patch.stop()
        for k in [k for k in sys.modules if k.split(".")[0] in _PURGED_PREFIXES]:
            del sys.modules[k]
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path
        sys.meta_path[:] = saved_meta
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.invalidate_caches()
        tmp.cleanup()


@contextlib.contextmanager
def fake_fa4_seam(fwd=None, bwd=None):
    """Import ``ring_flash_attn.flash_attn_backend`` as it would load on an FA4-only box."""
    with fake_environment(
        modules=fa4_modules(fwd, bwd),
        env={ENV_VAR: "fa4"},
        dist_infos=("flash_attn_4",),
    ):
        yield importlib.import_module("ring_flash_attn.flash_attn_backend")


# Script run in a fresh interpreter for the cold-import tests. It reuses fake_environment
# so the scenarios are built the same way as in-process, but starts from a process in
# which nothing under ring_flash_attn has ever been imported.
COLD_IMPORT_SCRIPT = """
import importlib, json, sys
import fa4_fakes as F

scenario = json.loads(sys.argv[1])
modules = {}
if "fa4" in scenario.get("install", ()):
    modules.update(F.fa4_modules())
if "fa3" in scenario.get("install", ()):
    modules.update(F.fa3_modules())
exc = {"OSError": OSError, "ModuleNotFoundError": ModuleNotFoundError}[
    scenario.get("absent_exc", "ModuleNotFoundError")
]
with F.fake_environment(
    modules=modules,
    env={F.ENV_VAR: scenario.get("env")},
    dist_infos=("flash_attn_4",) if "fa4" in scenario.get("install", ()) else (),
    absent=scenario.get("absent", ()),
    absent_exc=exc,
):
    pkg = importlib.import_module("ring_flash_attn")
    seam = importlib.import_module("ring_flash_attn.flash_attn_backend")
    print("RESULT", seam.BACKEND, hasattr(pkg, "llama3_flash_attn_varlen_func"))
"""
