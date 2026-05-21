import sys
import torch
import torch.distributed as dist
from flash_attn import flash_attn_qkvpacked_func
from ring_flash_attn.llama_fwd_ring_bwd_flash_attn import llama_flash_attn_func
from utils import log, set_seed


CONFIGS = [
    {"name": "A_none",      "fwd": None, "bwd_f": None, "bwd_l": None},
    {"name": "B_fwd",       "fwd": 1,    "bwd_f": None, "bwd_l": None},
    {"name": "C_bwd_first", "fwd": None, "bwd_f": 1,    "bwd_l": None},
    {"name": "D_bwd_last",  "fwd": None, "bwd_f": None, "bwd_l": 1},
    {"name": "E_all",       "fwd": 1,    "bwd_f": 1,    "bwd_l": 1},
]


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    set_seed(rank)
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    batch_size = 2
    seqlen = 512
    nheads = 8
    d = 128
    heads_k_stride = 2
    dropout_p = 0
    causal = False
    deterministic = False

    assert seqlen % world_size == 0
    assert nheads % heads_k_stride == 0
    assert d % 8 == 0

    # --- Setup Inputs ---
    qkv = torch.randn(
        batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    dist.broadcast(qkv, src=0)

    dout = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    dist.broadcast(dout, src=0)

    dist.barrier()

    # --- Vanilla Baseline ---
    if rank == 0:
        print("#" * 30)
        print("# vanilla flash_attn forward/backward:")
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
    out.backward(dout)
    dqkv = qkv.grad

    local_out = out.chunk(world_size, dim=1)[rank]
    local_lse = lse.chunk(world_size, dim=-1)[rank]
    local_dqkv = dqkv.chunk(world_size, dim=1)[rank]
    local_dout = dout.chunk(world_size, dim=1)[rank].detach().clone()

    # --- Run each stride config ---
    results = {}
    for cfg in CONFIGS:
        local_qkv = qkv.chunk(world_size, dim=1)[rank].detach().clone()
        local_qkv.requires_grad = True

        ring_out, ring_lse, _ = llama_flash_attn_func(
            local_qkv[:, :, 0],
            local_qkv[:, :, 1],
            local_qkv[:, :, 2],
            heads_k_stride=heads_k_stride,
            head_first_stride=cfg["fwd"],
            bwd_head_first_stride=cfg["bwd_f"],
            bwd_head_last_stride=cfg["bwd_l"],
            bwd_event_sync=False,
            dropout_p=dropout_p,
            causal=causal,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=deterministic,
            return_attn_probs=True,
        )
        ring_out.backward(local_dout)
        ring_dqkv = local_qkv.grad

        results[cfg["name"]] = {
            "out": ring_out.detach(),
            "lse": ring_lse.detach(),
            "dq": ring_dqkv[:, :, 0].detach().clone(),
            "dk": ring_dqkv[:, :, 1].detach().clone(),
            "dv": ring_dqkv[:, :, 2].detach().clone(),
        }

    # --- Report: vs vanilla flash_attn ---
    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# diff vs vanilla flash_attn:")
        print("#" * 30)

    for cfg in CONFIGS:
        name = cfg["name"]
        r = results[name]
        if rank == 0:
            print(f"--- {name} (fwd={cfg['fwd']}, bwd_f={cfg['bwd_f']}, bwd_l={cfg['bwd_l']}) ---", flush=True)
        dist.barrier()
        log(f"{name} out diff", local_out - r["out"])
        log(f"{name} lse diff", local_lse - r["lse"])
        log(f"{name} dq diff", local_dqkv[:, :, 0] - r["dq"])
        log(f"{name} dk diff", local_dqkv[:, :, 1] - r["dk"])
        log(f"{name} dv diff", local_dqkv[:, :, 2] - r["dv"])

    # --- Report: vs no-stride baseline (A) ---
    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# diff vs no-stride baseline (A_none):")
        print("#" * 30)

    a = results["A_none"]
    for cfg in CONFIGS:
        name = cfg["name"]
        if name == "A_none":
            continue
        r = results[name]
        if rank == 0:
            print(f"--- {name} vs A_none ---", flush=True)
        dist.barrier()
        log(f"{name} out diff", a["out"] - r["out"])
        log(f"{name} lse diff", a["lse"] - r["lse"])
        log(f"{name} dq diff", a["dq"] - r["dq"])
        log(f"{name} dk diff", a["dk"] - r["dk"])
        log(f"{name} dv diff", a["dv"] - r["dv"])

    dist.destroy_process_group()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compile":
        torch._dynamo.config.capture_scalar_outputs = True
        llama_flash_attn_func = torch.compile(llama_flash_attn_func)
    main()
