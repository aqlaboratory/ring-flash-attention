import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Tuple, List
from .utils import AllGatherComm as Comm
from einops import rearrange


class LlamaStandardAttn(torch.nn.Module):
    """
    Standard attention with Llama-style context parallelism.
    K and V tensors are gathered across the sequence dimension if a process_group is provided.
    Don't support Grouped-Query (GQA), and Multi-Query (MQA) Attention.
    """

    def __init__(
        self,
        dropout_p: float = 0.0,
        heads_k_stride: int=2,
        attn_query_chunks: int=100,
        # bwd_event_sync: bool=False, # Not relevant for standard autograd
    ):
        super().__init__()
        self.dropout_p = dropout_p
        self.sp_mesh = None
        self.heads_k_stride = heads_k_stride
        self.attn_query_chunks = attn_query_chunks
        # self.bwd_event_sync = bwd_event_sync

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        #causal: bool = False,  # TODO: To implement
        return_attn_probs: bool = False,
        time_event=None,  # Sync GPU,CPU to lower vRAM allocation; no sync by default
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the Llama standard attention module.
        Intput entsor shape matches flash-attention's
        [batch_size, local_seq_len_q, num_q_heads, head_dim]
        i.e., [B S H D]
        Internally, the function uses [B H S D]

        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.
            key_padding_mask (Optional[torch.Tensor]): Boolean mask for padding in keys.
                Shape (batch_size, local_seq_len_kv). True for valid tokens, False for padding.
            softmax_scale (Optional[float]): Scale for softmax. Defaults to 1/sqrt(head_dim).
            causal (bool): Whether to apply causal masking.
            return_attn_probs (bool): Whether to return attention probabilities.

        Returns:
            torch.Tensor: Output of the attention mechanism, shape (batch_size, local_seq_len_q, num_q_heads, head_dim).
            Optional[torch.Tensor]: Attention probabilities if return_attn_probs is True, else None.
                                     Shape (batch_size, num_kv_heads, num_q_groups, local_seq_len_q, global_seq_len_kv).
        """
        batch_size, local_seq_len_q, num_q_heads, head_dim = q.shape
        _, local_seq_len_kv, num_kv_heads, _ = k.shape
        assert num_q_heads == num_kv_heads
        assert num_q_heads % self.heads_k_stride == 0

        gathered_key_padding_mask = key_padding_mask
        
        world_size = 1
        rank = 0
        q, k, v = rearrange([q, k, v], 'qkv b s h d -> qkv b h s d')

        if softmax_scale is None:
            softmax_scale = 1.0 / (head_dim**0.5)
        q = q * softmax_scale

        output_list: List[torch.Tensor] = []
        probs_list: List[torch.Tensor] = []
        if self.sp_mesh is not None:
            process_group = self.sp_mesh.get_group()
            comm = Comm(process_group)
            world_size = dist.get_world_size(process_group)
            rank = dist.get_rank(process_group)

            if key_padding_mask is not None:
                mask_list = [torch.empty_like(key_padding_mask) for _ in range(world_size)]
                dist.all_gather(mask_list, key_padding_mask, group=self.process_group)
                gathered_key_padding_mask = torch.cat(mask_list, dim=1)

            kv_buffer = torch.empty(
                (2, world_size, batch_size, self.heads_k_stride, local_seq_len_q, head_dim),
                dtype=k.dtype,
                device=k.device,
            )
            kv_buffer_copy = torch.empty_like(kv_buffer)

            k_0 = k[:, :self.heads_k_stride, :].contiguous()
            v_0 = v[:, :self.heads_k_stride, :].contiguous()

            comm = Comm(process_group)
            # Pass the main tensor slices to all_gather
            comm.all_gather(kv_buffer_copy[0], k_0)
            comm.all_gather(kv_buffer_copy[1], v_0)

            for i in range(0, num_kv_heads, self.heads_k_stride):
                # Optimization: No sync on last head stride
                if (i == num_kv_heads - self.heads_k_stride) and (time_event is not None):
                    time_event.record()
                comm.wait()
                # Swap the main storage tensors
                kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer

                if i < num_kv_heads - self.heads_k_stride:
                    # all_gather the next kv slice
                    kv_slice_left = i + self.heads_k_stride
                    kv_slice_right = kv_slice_left + self.heads_k_stride
                    send_k = k[:, kv_slice_left:kv_slice_right].contiguous()
                    send_v = v[:, kv_slice_left:kv_slice_right].contiguous()
                    # Pass the main tensor slices for the next round
                    comm.all_gather(kv_buffer_copy[0], send_k)
                    comm.all_gather(kv_buffer_copy[1], send_v)

                q_i = q[:, i : (i + self.heads_k_stride)]
                # kv_buffer[0] has shape [world_size b h s d ]
                # We want k_i to be [b h world_size*s d ]
                k_i = rearrange(kv_buffer[0], 'w b h s d -> b h (w s) d')
                v_i = rearrange(kv_buffer[1], 'w b h s d -> b h (w s) d')

                output, probs = self.self_attn(
                    q_i,
                    k_i,
                    v_i,
                    key_padding_mask=gathered_key_padding_mask,
                )
                output_list.append(output)
                probs_list.append(probs)
        # output concat heads [B H S D]; S is the local sequence length
        output = rearrange(output_list,'w b h s d -> b (w s) h d')

        if return_attn_probs:
            # probs concat heads [B H Sq Sk]; Sq is local sequence length of q
            probs = rearrange(probs_list,'w b h s d -> b (w s) h d')
            return output, probs
        else:
            return output, None

    def self_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Self-attention.

        Intput tensor shape [B H S D]
        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.

        Returns:
            torch.Tensor: Output of the self-attention mechanism.
            Optional[torch.Tensor]: Attention probabilities if required.
        """
        batch_size, num_q_heads, local_seq_len_q, head_dim = q.shape
        _, _, global_seq_len_kv, _ = k.shape
        attn_query_chunks = self.attn_query_chunks
        if self.attn_query_chunks is None:
            attn_query_chunks = local_seq_len_q
        query_chunk_size = local_seq_len_q // self.attn_query_chunks
        if local_seq_len_q % self.attn_query_chunks:
            query_chunk_size += 1

        all_attn_output = torch.empty(
            (batch_size, num_q_heads, local_seq_len_q, head_dim),
            dtype=q.dtype,
            device=q.device,
        )
        all_attn_probs = torch.empty(
            (batch_size, num_q_heads, local_seq_len_q, global_seq_len_kv),
            dtype=q.dtype,
            device=q.device,
        )

        for q_start in range(0, local_seq_len_q, query_chunk_size):
            q_chunk_end = min(q_start + query_chunk_size, local_seq_len_q)
            attn_score = q[:, :, q_start:q_chunk_end] @ k.transpose(-2, -1)
            if key_padding_mask is not None:
                mask_to_apply = key_padding_mask.view(batch_size, 1, 1, global_seq_len_kv)
                attn_score = attn_score.masked_fill(mask_to_apply == False, float('-inf'))
            attn_probs = torch.softmax(attn_score, dim=-1)
            if self.dropout_p > 0.0:
                # TODO: Check if this is correct dropout application
                attn_probs = nn.functional.dropout(attn_probs, p=self.dropout_p, training=self.training)                
            all_attn_probs[:, :, q_start:q_chunk_end] = attn_probs
            all_attn_output[:, :, q_start:q_chunk_end] = attn_probs @ v

        return all_attn_output, all_attn_probs
