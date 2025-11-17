import sys
import torch
import torch.distributed as dist
from flash_attn import flash_attn_qkvpacked_func
from ring_flash_attn.llama_fwd_ring_bwd_flash_attn import llama_flash_attn_func
from utils import log, set_seed

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    set_seed(rank)
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    batch_size = 1
    seqlen = 3816
    nheads = 5
    d = 128
    dropout_p = 0
    causal = False
    deterministic = False

    assert seqlen % world_size == 0
    assert d % 8 == 0

    # --- Setup Inputs ---
    qkv = torch.randn(
        batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    dist.broadcast(qkv, src=0)

    dout = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    dist.broadcast(dout, src=0)

    # Create local shards (detached from global graph for fresh start)
    local_qkv = qkv.chunk(world_size, dim=1)[rank].detach().clone()
    local_qkv.requires_grad = True
    local_dout = dout.chunk(world_size, dim=1)[rank].detach().clone()

    dist.barrier()

    # --- Baseline: Standard Flash Attention ---
    if rank == 0:
        print("#" * 30)
        print("# forward:")
        print("#" * 30)

    out, lse, _ = flash_attn_qkvpacked_func(
        qkv,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    local_out = out.chunk(world_size, dim=1)[rank]
    local_lse = lse.chunk(world_size, dim=-1)[rank]

    # --- Test Subject: Ring Attention ---
    # FIX 1: Correct variable name for compilation hook
    fn = llama_flash_attn_func 
    
    # FIX 2: Removed .squeeze(2). Indexing [:,:,0] results in (B, S, H, D)
    ring_out, ring_lse, _ = fn(
        local_qkv[:,:,0], 
        local_qkv[:,:,1], 
        local_qkv[:,:,2], 
        heads_k_stride=1,
        bwd_event_sync=False,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    log("out", out, rank0_only=True)
    log("lse", lse, rank0_only=True)
    log("out diff", local_out - ring_out)
    # Note: LSE might require adjustment depending on implementation (scaling factors)
    log("lse diff", local_lse - ring_lse) 

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# backward:")
        print("#" * 30)

    # --- Backward Baseline ---
    out.backward(dout)
    dqkv = qkv.grad
    local_dqkv = dqkv.chunk(world_size, dim=1)[rank]

    # --- Backward Ring ---
    ring_out.backward(local_dout)
    ring_dqkv = local_qkv.grad

    log("local_dqkv", local_dqkv)
    
    # FIX 3: Correct indexing to compare Q, K, and V specifically
    # Shape is (Batch, Seq, 3, Heads, Dim). We slice dim 2.
    log("dq diff", local_dqkv[:, :, 0] - ring_dqkv[:, :, 0])
    log("dk diff", local_dqkv[:, :, 1] - ring_dqkv[:, :, 1])
    log("dv diff", local_dqkv[:, :, 2] - ring_dqkv[:, :, 2])

    dist.destroy_process_group()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compile":
        torch._dynamo.config.capture_scalar_outputs = True
        # FIX 1 (continued): Compile the actual function used
        llama_flash_attn_func = torch.compile(llama_flash_attn_func)
    main()