import unittest
import torch
import torch.nn.functional as F
from ring_flash_attn.llama_standard_attn import chunked_query_self_attn, chunked_query_self_attn_backward


class TestAttention(unittest.TestCase):
    def test_attention_equivalence(self):
        torch.manual_seed(0)
        batch_size, seq_len, nheads, head_dim = 1, 3, 2, 8
        softmax_scale = 1.0 / head_dim # Use float division for clarity

        # --- PyTorch F.scaled_dot_product_attention (Reference) ---
        torch.manual_seed(0) # Ensure Q,K,V are identical for both implementations
        torch_q = torch.randn(batch_size, nheads, seq_len, head_dim, requires_grad=True)
        torch_k = torch.randn(batch_size, nheads, seq_len, head_dim, requires_grad=True)
        torch_v = torch.randn(batch_size, nheads, seq_len, head_dim, requires_grad=True)
        
        pytorch_output_bshd = F.scaled_dot_product_attention(
            torch_q,
            torch_k,
            torch_v,
            attn_mask=None, dropout_p=0.0, is_causal=False, scale=softmax_scale
        )
        
        # Deterministic gradient for backward pass
        torch.manual_seed(42) 
        torch_d_output = torch.randn_like(pytorch_output_bshd)

        pytorch_output_bshd.backward(torch_d_output)
        
        expected_dq = torch_q.grad.clone()
        expected_dk = torch_k.grad.clone()
        expected_dv = torch_v.grad.clone()

        # --- Custom chunked_query_self_attn (Implementation under test) ---
        torch.manual_seed(0) # Ensure Q,K,V are identical by re-seeding
        q = torch.randn(batch_size, nheads, seq_len, head_dim, requires_grad=True)
        k = torch.randn(batch_size, nheads, seq_len, head_dim, requires_grad=True)
        v = torch.randn(batch_size, nheads, seq_len, head_dim, requires_grad=True)

        # Forward pass with custom function
        output, probs = chunked_query_self_attn(
            q, 
            k,
            v,
            softmax_scale,
            attn_q_chunk_size=2, # This parameter is specific to your custom function
        )
        
        # Prepare gradient tensors for custom backward pass
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        
        # Backward pass with custom function
        chunked_query_self_attn_backward(
            dout=torch_d_output, # Use the same d_output as for PyTorch's backward
            q=q,
            k=k,
            v=v,
            dv=dv,
            probs=probs,
            dq=dq,
            dk=dk,
            attn_q_chunk_size=2, # This parameter is specific to your custom function
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            key_padding_mask=None,
            causal=False,
        )

        # --- Verification ---
        # Forward output comparison
        self.assertTrue(
            torch.allclose(pytorch_output_bshd, output), 
            "Forward output mismatch between PyTorch and custom implementation."
        )
        
        # Gradient comparison for dQ
        self.assertTrue(
            torch.allclose(expected_dq, dq), 
            "dQ gradient mismatch."
        )
        # Gradient comparison for dK
        self.assertTrue(
            torch.allclose(expected_dk, dk),
            "dK gradient mismatch."
        )
        # Gradient comparison for dV
        self.assertTrue(
            torch.allclose(expected_dv, dv),
            "dV gradient mismatch."
        )

if __name__ == '__main__':
    unittest.main()