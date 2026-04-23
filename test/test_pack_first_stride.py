import sys
import torch
import torch.distributed as dist
from ring_flash_attn.llama_fwd_ring_bwd_flash_attn import llama_fwd_ring_bwd_flash_attn_func
from utils import log, set_seed

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    set_seed(rank)
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    batch_size = 8
    seqlen = 3816
    nheads = 6
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
    # We create two identical local sets to test the two branches separately without mixing grads
    local_qkv_true = qkv.chunk(world_size, dim=1)[rank].detach().clone()
    local_qkv_true.requires_grad = True

    local_qkv_false = qkv.chunk(world_size, dim=1)[rank].detach().clone()
    local_qkv_false.requires_grad = True

    local_dout = dout.chunk(world_size, dim=1)[rank].detach().clone()

    dist.barrier()

    if rank == 0:
        print("#" * 30)
        print("# forward:")
        print("#" * 30)

    # --- Test Subject: Pack First Stride = True ---
    fn = llama_fwd_ring_bwd_flash_attn_func
    
    out_true, lse_true, _ = fn(
        local_qkv_true[:,:,0], 
        local_qkv_true[:,:,1], 
        local_qkv_true[:,:,2], 
        heads_k_stride=3,
        head_first_stride=1,
        pack_first_stride=True,
        bwd_event_sync=False,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    # --- Test Subject: Pack First Stride = False ---
    out_false, lse_false, _ = fn(
        local_qkv_false[:,:,0], 
        local_qkv_false[:,:,1], 
        local_qkv_false[:,:,2], 
        heads_k_stride=3,
        head_first_stride=1,
        pack_first_stride=False,
        bwd_event_sync=False,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    log("out true", out_true, rank0_only=True)
    log("out diff (True vs False)", out_true - out_false)
    log("lse diff (True vs False)", lse_true - lse_false) 

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# backward:")
        print("#" * 30)

    out_true.backward(local_dout)
    dqkv_true = local_qkv_true.grad

    out_false.backward(local_dout)
    dqkv_false = local_qkv_false.grad

    log("dq diff (True vs False)", dqkv_true[:, :, 0] - dqkv_false[:, :, 0])
    log("dk diff (True vs False)", dqkv_true[:, :, 1] - dqkv_false[:, :, 1])
    log("dv diff (True vs False)", dqkv_true[:, :, 2] - dqkv_false[:, :, 2])

    dist.destroy_process_group()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compile":
        torch._dynamo.config.capture_scalar_outputs = True
        llama_fwd_ring_bwd_flash_attn_func = torch.compile(llama_fwd_ring_bwd_flash_attn_func)
    main()