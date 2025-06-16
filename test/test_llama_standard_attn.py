import unittest
import torch
import torch.nn.functional as F
from ring_flash_attn.llama_standard_attn import chunked_query_self_attn, chunked_query_self_attn_backward
from einops import rearrange

def torch_attention(q, k, v, scale=None, key_padding_mask=None):
    """
    A simple implementation of attention using PyTorch's built-in functions.
    This is for reference and comparison with the custom chunked attention.
    """
    # 1. Calculate unscaled attention scores
    unscaled_scores = q @ k.transpose(-2, -1)

    # 2. Scale the scores if scale is provided
    if scale is not None:
        scores = unscaled_scores * scale
    else:
        scores = unscaled_scores

    # 3. Apply key padding mask if provided
    if key_padding_mask is not None:
        scores = scores.masked_fill(
            ~key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
        )

    # 4. Apply softmax to get attention weights
    attention_weights = torch.nn.functional.softmax(scores, dim=-1)


    # 5. Compute the output by weighting the values
    output = attention_weights @ v

    return output, attention_weights


param_mask = [False, True]  # Test both with and without masks
class TestAttention(unittest.TestCase):
    def test_attention_against_sdpa(self):
        torch.manual_seed(0)
        batch_size, seq_len, nheads, head_dim = 3, 10, 2, 8
        softmax_scale = 1.0 / head_dim # Use float division for clarity

        # --- PyTorch F.scaled_dot_product_attention (Reference) ---
        torch.manual_seed(0) # Ensure Q,K,V are identical for both implementations
        torch_q = torch.randn(batch_size, nheads, seq_len, head_dim, requires_grad=True,device='cuda')
        torch_k = torch.randn(batch_size, nheads, seq_len, head_dim, requires_grad=True,device='cuda')
        torch_v = torch.randn(batch_size, nheads, seq_len, head_dim, requires_grad=True,device='cuda')
        # Create masks if use_mask is True
        key_padding_mask = None
        for use_mask in param_mask:
            if use_mask:
                # True for valid tokens, False for padding. Shape (batch_size, seq_len_kv)
                key_padding_mask = torch.ones(batch_size, seq_len, device='cuda', requires_grad=True).bool()  # Example mask
                key_padding_mask[0, -1] = False  # Example: mask the last token in the sequence
                key_padding_mask[0, -5] = False  # Example: mask the last token in the sequence
            else:
                key_padding_mask = None
            # pytorch_output_bhsd = F.scaled_dot_product_attention(
            #     torch_q,
            #     torch_k,
            #     torch_v,
            #     attn_mask=key_padding_mask.view(batch_size, 1, 1, seq_len) if key_padding_mask is not None else None,
            #     dropout_p=0.0, is_causal=False, scale=softmax_scale
            # )

            pytorch_output_bhsd, _ = torch_attention(
                torch_q,
                torch_k,
                torch_v,
                scale=softmax_scale,
                key_padding_mask=key_padding_mask,)
            # Deterministic gradient for backward pass
            torch.manual_seed(42) 
            torch_d_output = torch.randn_like(pytorch_output_bhsd)
            # torch_d_output *= key_padding_mask.view(batch_size, 1, seq_len, 1) if key_padding_mask is not None else 1.0  # Zero out gradients for masked positions

            pytorch_output_bhsd.backward(torch_d_output)
            
            expected_dq = torch_q.grad.clone()
            expected_dk = torch_k.grad.clone()
            expected_dv = torch_v.grad.clone()

            # --- Custom chunked_query_self_attn (Implementation under test) ---
            torch.manual_seed(0) # Ensure Q,K,V are identical by re-seeding
            q = torch.randn(batch_size, nheads, seq_len, head_dim, requires_grad=True,device='cuda')
            k = torch.randn(batch_size, nheads, seq_len, head_dim, requires_grad=True,device='cuda')
            v = torch.randn(batch_size, nheads, seq_len, head_dim, requires_grad=True,device='cuda')
            # q, k, v, dout = rearrange([q, k, v, torch_d_output], 'qkvo b h s d -> qkvo b s h d')

            # Forward pass with custom function
            output, probs = chunked_query_self_attn(
                q, 
                k,
                v,
                softmax_scale=softmax_scale,
                key_padding_mask=key_padding_mask,
                attn_q_chunk_size=None, # This parameter is specific to your custom function
            )
            
            # Prepare gradient tensors for custom backward pass
            dq = torch.zeros_like(q)
            dk = torch.zeros_like(k)
            dv = torch.zeros_like(v)
            
            # Backward pass with custom function
            chunked_query_self_attn_backward(
                dout=torch_d_output.clone(), # Use the same d_output as for PyTorch's backward
                q=q,
                k=k,
                v=v,
                dv=dv,
                probs=probs,
                dq=dq,
                dk=dk,
                attn_q_chunk_size=None, # This parameter is specific to your custom function
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                key_padding_mask=key_padding_mask,
                causal=False,
            )
            # dq, dk, dv, output = rearrange([q, k, v, output], 'qkvo b s h d -> qkvo b h s d')

            # --- Verification ---
            # Forward output comparison

            self.assertTrue(
                torch.allclose(pytorch_output_bhsd, output, atol=1e-6, rtol=1e-5), 
                "Forward output mismatch between PyTorch and custom implementation."+
                f"pytorch_output_bhsd: {pytorch_output_bhsd[0,:2,3,:4]}, " +
                f"output: {output[0,:2,3,:4]}, " +
                f"use_mask: {use_mask}"
            )
            
            # Gradient comparison for dQ
            self.assertTrue(
                torch.allclose(expected_dq, dq, atol=1e-6, rtol=1e-5),
                "dQ gradient mismatch."  +
                f"expected_dq: {expected_dq[0,-2:,3,:4]}, " +
                f"dq: {dq[0,-2:,3,:4]}, " +
                f"use_mask: {use_mask}"
            )
            # Gradient comparison for dK
            self.assertTrue(
                torch.allclose(expected_dk, dk, atol=1e-6, rtol=1e-5), 
                "dK gradient mismatch." +
                f"use_mask: {use_mask}"
            )
            # Gradient comparison for dV
            self.assertTrue(
                torch.allclose(expected_dv, dv, atol=1e-6, rtol=1e-5), 
                "dV gradient mismatch." +
                f"use_mask: {use_mask}"
            )

if __name__ == '__main__':
    unittest.main()