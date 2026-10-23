"""Unit tests for ``select_backend`` with fake probes (no GPU, no real flash-attn).

The seam resolves a backend at import, so it is loaded once under a fake FA4-only install
purely to get at ``select_backend`` and ``_requested_backend``.
"""

import os
import unittest
from unittest import mock

from fa4_fakes import ENV_VAR, fake_fa4_seam


def ok():
    return None


def failing(exc):
    def probe():
        raise exc

    return probe


class SelectBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cm = fake_fa4_seam()
        cls.seam = cm.__enter__()
        cls.addClassCleanup(cm.__exit__, None, None, None)

    def select(self, requested, **probes):
        return self.seam.select_backend(requested, probes)

    def test_auto_prefers_fa2(self):
        self.assertEqual(self.select("auto", fa2=ok, fa4=ok), "fa2")

    def test_auto_falls_back_to_fa4_on_any_fa2_failure(self):
        # ImportError, OSError (.so fails to load), RuntimeError (torch ABI mismatch)
        for exc in (ImportError("x"), OSError("x"), RuntimeError("x")):
            with self.subTest(exc=type(exc).__name__):
                self.assertEqual(self.select("auto", fa2=failing(exc), fa4=ok), "fa4")

    def test_auto_reports_both_failures(self):
        with self.assertRaises(ImportError) as cm:
            self.select(
                "auto",
                fa2=failing(OSError("flash_attn_2_cuda.so")),
                fa4=failing(ImportError("no cute")),
            )
        msg = str(cm.exception)
        self.assertIn("fa2: OSError: flash_attn_2_cuda.so", msg)
        self.assertIn("fa4: ImportError: no cute", msg)

    def test_auto_never_selects_fa3(self):
        # even with a perfectly good fa3 probe on offer
        with self.assertRaises(ImportError) as cm:
            self.select(
                "auto", fa2=failing(ImportError("a")), fa4=failing(ImportError("b")),
                fa3=ok,
            )
        self.assertIn("never auto-selected", str(cm.exception))

    def test_result_is_never_auto(self):
        # BACKEND keys _VALIDATED_VARIANTS; "auto" there would silently disable the gate
        for requested in ("auto", "fa2", "fa3", "fa4"):
            self.assertNotEqual(self.select(requested, fa2=ok, fa4=ok), "auto")

    def test_explicit_backend_is_returned_without_probing(self):
        boom = failing(AssertionError("must not probe"))
        for name in ("fa2", "fa3", "fa4"):
            self.assertEqual(self.select(name, fa2=boom, fa4=boom), name)

    def test_invalid_value_raises(self):
        with self.assertRaises(ValueError):
            self.select("fa5", fa2=ok, fa4=ok)

    def test_variant_gate_is_active_for_auto_resolved_fa4(self):
        # The loaded seam resolved fa4 explicitly; the gate must key off that name.
        self.assertEqual(self.seam.BACKEND, "fa4")
        with self.assertRaises(NotImplementedError):
            self.seam.check_variant_supported("zigzag_ring_flash_attn")
        self.seam.check_variant_supported("llama3_flash_attn_varlen")
        self.seam.check_variant_supported("ring_flash_attn_forward")

    def test_auto_records_why_it_skipped_fa2(self):
        failures = []
        self.seam.select_backend(
            "auto", {"fa2": failing(OSError("bad .so")), "fa4": ok}, failures
        )
        self.assertEqual(failures, ["fa2: OSError: bad .so"])

    def test_refusal_names_env_var_only_when_it_was_set(self):
        with self.assertRaises(NotImplementedError) as cm:
            self.seam.check_variant_supported("zigzag_ring_flash_attn")
        self.assertIn(f"Unset {ENV_VAR}", str(cm.exception))

    def test_refusal_after_auto_fallthrough_shows_the_fa2_failure(self):
        # the variable is already unset here, so telling the user to unset it is useless
        with mock.patch.object(self.seam, "_AUTO_SKIPPED", ["fa2: OSError: bad .so"]):
            with self.assertRaises(NotImplementedError) as cm:
                self.seam.check_variant_supported("zigzag_ring_flash_attn")
        msg = str(cm.exception)
        self.assertIn("fa2: OSError: bad .so", msg)
        self.assertIn("Repair flash-attn 2.x", msg)
        self.assertNotIn(f"Unset {ENV_VAR}", msg)


class RequestedBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cm = fake_fa4_seam()
        cls.seam = cm.__enter__()
        cls.addClassCleanup(cm.__exit__, None, None, None)

    def requested(self, value):
        saved = os.environ.get(ENV_VAR)
        try:
            if value is None:
                os.environ.pop(ENV_VAR, None)
            else:
                os.environ[ENV_VAR] = value
            return self.seam._requested_backend()
        finally:
            if saved is None:
                os.environ.pop(ENV_VAR, None)
            else:
                os.environ[ENV_VAR] = saved

    def test_unset_empty_and_blank_mean_auto(self):
        for value in (None, "", "   "):
            with self.subTest(value=value):
                self.assertEqual(self.requested(value), "auto")

    def test_value_is_normalised(self):
        self.assertEqual(self.requested(" FA4 "), "fa4")


if __name__ == "__main__":
    unittest.main()
