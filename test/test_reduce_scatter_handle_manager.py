import unittest
from ring_flash_attn.utils import ReduceScatterHandleManager
from unittest.mock import patch


class TestReduceScatterHandleManager(unittest.TestCase):
    def test_init_with_explicit_group_name(self):
        with patch("ring_flash_attn.utils.dist.is_initialized", return_value=True), \
            patch("ring_flash_attn.utils.dist.get_world_size", return_value=4):
            mgr = ReduceScatterHandleManager(group=object(), group_name="mygrp")
            self.assertEqual(mgr._group_name, "mygrp")
            self.assertEqual(mgr._world_size, 4)

    def test_init_autodetect_group_name_from_group_name_attr(self):
        class G:
            def group_name(self):
                return "grp"

        g = G()
        with patch("ring_flash_attn.utils.dist.is_initialized", return_value=True), \
            patch("ring_flash_attn.utils.dist.get_world_size", return_value=2):
            mgr = ReduceScatterHandleManager(group=g)
            self.assertEqual(mgr._group_name, "grp")
            self.assertEqual(mgr._world_size, 2)

    def test_init_raises_if_not_initialized_and_no_group_or_name(self):
        with patch("ring_flash_attn.utils.dist.is_initialized", return_value=False):
            with self.assertRaises(RuntimeError):
                ReduceScatterHandleManager(group=None, group_name=None)


if __name__ == "__main__":
    unittest.main()
