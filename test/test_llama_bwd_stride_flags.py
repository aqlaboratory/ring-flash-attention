import unittest
import torch
from unittest.mock import patch

from ring_flash_attn import llama_fwd_ring_bwd_flash_attn as mod


class DummyComm:
    def __init__(self, pg):
        self.pg = pg

    def all_gather(self, dst, src):
        # place src at index 0 (simulate rank 0 contribution)
        try:
            dst.zero_()
            dst[0].copy_(src)
        except Exception:
            # best-effort: if shapes mismatch, try a simple copy
            dst.copy_(src)

    def wait(self):
        return


class DummyRSManager:
    def __init__(self, *args, **kwargs):
        pass

    def issue(self, *args, **kwargs):
        return

    def drain_to(self, dk, dv):
        # no-op: assume dk/dv already set by backward stub
        return


def fake_backward(**params):
    # Fill any floating-point tensor argument with a recognizable value so tests
    # can detect that the backward stub ran. Use different values for dq/dk/dv
    # when present.
    for k, v in params.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            if k == "dq":
                v.fill_(3.0)
            elif k == "dk":
                v.fill_(4.0)
            elif k == "dv":
                v.fill_(5.0)
            else:
                v.fill_(7.0)


class TestBwdStrideFlags(unittest.TestCase):
    def setUp(self):
        # small tensors
        self.batch = 1
        self.seq = 4
        self.nheads = 4
        self.nheads_k = 4
        self.head_dim = 8

    @patch("ring_flash_attn.llama_fwd_ring_bwd_flash_attn.Comm", DummyComm)
    @patch("ring_flash_attn.llama_fwd_ring_bwd_flash_attn.ReduceScatterHandleManager", DummyRSManager)
    @patch("ring_flash_attn.llama_fwd_ring_bwd_flash_attn._wrapped_flash_attn_backward", fake_backward)
    @patch("ring_flash_attn.llama_fwd_ring_bwd_flash_attn.dist.get_world_size", return_value=1)
    @patch("ring_flash_attn.llama_fwd_ring_bwd_flash_attn.dist.get_rank", return_value=0)
    def test_valid_first_and_last_stride(self, mock_rank, mock_wsize):
        q = torch.zeros((self.batch, self.seq, self.nheads, self.head_dim))
        k = torch.zeros((self.batch, self.seq, self.nheads_k, self.head_dim))
        v = torch.zeros_like(k)
        out = torch.zeros_like(q)
        softmax_lse = torch.zeros((self.nheads, self.seq))
        dout = torch.zeros_like(q)

        dq, dk, dv = mod.llama_flash_attn_backward(
            process_group=None,
            dout=dout,
            q=q,
            k=k,
            v=v,
            out=out,
            softmax_lse=softmax_lse,
            heads_k_stride=2,
            softmax_scale=1.0,
            dropout_p=0,
            causal=False,
            bwd_head_first_stride=1,
            bwd_head_last_stride=1,
        )

        # shapes preserved
        self.assertEqual(dq.shape, q.shape)
        self.assertEqual(dk.shape, k.shape)
        self.assertEqual(dv.shape, v.shape)

        # our fake backward wrote to dq; ensure shapes for dk/dv preserved
        self.assertTrue(torch.any(dq == 3.0))
        self.assertEqual(dk.shape, k.shape)
        self.assertEqual(dv.shape, v.shape)

    @patch("ring_flash_attn.llama_fwd_ring_bwd_flash_attn.dist.get_world_size", return_value=1)
    @patch("ring_flash_attn.llama_fwd_ring_bwd_flash_attn.dist.get_rank", return_value=0)
    def test_invalid_splits_raise(self, mock_rank, mock_wsize):
        q = torch.zeros((1, 2, 2, 4))
        k = torch.zeros((1, 2, 2, 4))
        v = torch.zeros_like(k)
        out = torch.zeros_like(q)
        softmax_lse = torch.zeros((2, 2))
        dout = torch.zeros_like(q)

        # bwd_head_first_stride equals heads_k_stride → should assert
        with self.assertRaises(AssertionError):
            mod.llama_flash_attn_backward(
                process_group=None,
                dout=dout,
                q=q,
                k=k,
                v=v,
                out=out,
                softmax_lse=softmax_lse,
                heads_k_stride=2,
                softmax_scale=1.0,
                bwd_head_first_stride=2,
            )


if __name__ == "__main__":
    unittest.main()
