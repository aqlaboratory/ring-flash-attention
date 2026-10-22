"""Import ring_flash_attn cold, in a fresh interpreter, against fake flash-attn installs.

Backend resolution happens once, at import, so each scenario needs its own process. The
fakes (see fa4_fakes.py) never execute a kernel and need no GPU. This doubles as an
import-cycle check for the backend seam.

The case that motivated the work: flash-attn-4 installs only ``flash_attn/cute`` as a
namespace package (no ``__init__``, no ``flash_attn_interface``), so ``import flash_attn``
and ``find_spec("flash_attn")`` both succeed on a box where FA2 does not exist.
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from fa4_fakes import COLD_IMPORT_SCRIPT

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = Path(__file__).resolve().parent


def run_scenario(**scenario):
    env = {k: v for k, v in os.environ.items() if k != "RING_FLASH_ATTN_BACKEND"}
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), str(TEST_DIR)])
    return subprocess.run(
        [sys.executable, "-c", COLD_IMPORT_SCRIPT, json.dumps(scenario)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True,
    )


class ColdImportTest(unittest.TestCase):
    def result(self, proc):
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT")][-1]
        backend, has_variant = line.split()[1:]
        self.assertEqual(has_variant, "True")  # the package, not just the seam, imported
        return backend

    def test_fa4_only_install_auto_selects_fa4(self):
        self.assertEqual(self.result(run_scenario(install=["fa4"])), "fa4")

    def test_empty_env_var_means_auto(self):
        self.assertEqual(self.result(run_scenario(install=["fa4"], env="")), "fa4")

    def test_fa4_only_install_explicit_fa4(self):
        self.assertEqual(self.result(run_scenario(install=["fa4"], env="fa4")), "fa4")

    def test_auto_falls_back_to_fa4_when_fa2_extension_fails_to_load(self):
        # flash_attn_2_cuda .so failing to load surfaces as OSError, not ImportError
        proc = run_scenario(
            install=["fa4"],
            absent=["flash_attn.flash_attn_interface"],
            absent_exc="OSError",
        )
        self.assertEqual(self.result(proc), "fa4")

    def test_explicit_fa2_on_fa4_only_install_raises_without_fallback(self):
        proc = run_scenario(install=["fa4"], env="fa2")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("ImportError", proc.stderr)
        self.assertIn("could not load the flash_attn (FA2) backend", proc.stderr)

    def test_auto_with_nothing_loadable_raises_naming_both_failures(self):
        proc = run_scenario(absent=["flash_attn"])
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("ImportError", proc.stderr)
        self.assertIn("fa2:", proc.stderr)
        self.assertIn("fa4:", proc.stderr)

    def test_fa3_is_never_auto_selected(self):
        proc = run_scenario(install=["fa3"], absent=["flash_attn"])
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("never auto-selected", proc.stderr)

    def test_invalid_env_var_raises(self):
        proc = run_scenario(install=["fa4"], env="fa5")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("not a known backend", proc.stderr)


if __name__ == "__main__":
    unittest.main()
