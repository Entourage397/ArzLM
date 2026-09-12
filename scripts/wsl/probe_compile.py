"""Minimal torch.compile probes for WSL. Prints each stage so a crash is locatable."""

from __future__ import annotations

import sys
import time

import torch
import torch.nn as nn


def main() -> int:
    print("python", sys.version.split()[0], flush=True)
    print("torch", torch.__version__, "cuda", torch.version.cuda, flush=True)
    try:
        import triton

        print("triton", triton.__version__, flush=True)
    except Exception as exc:
        print("triton import failed", exc, flush=True)
        return 1
    print("gpu", torch.cuda.get_device_name(0), flush=True)

    m = nn.Linear(768, 768).cuda().to(torch.bfloat16)
    x = torch.randn(2, 1024, 768, device="cuda", dtype=torch.bfloat16)
    t0 = time.perf_counter()
    compiled = torch.compile(m, backend="inductor")
    y = compiled(x)
    torch.cuda.synchronize()
    print(f"linear compile+forward ok in {time.perf_counter() - t0:.2f}s shape={tuple(y.shape)}", flush=True)

    from arzlm.model.config import build_gpt, build_model_config
    from litgpt.pretrain import initialize_weights
    import lightning as L

    fabric = L.Fabric(devices=1, precision="bf16-true")
    fabric.launch()
    fabric.seed_everything(42)
    config = build_model_config()
    with fabric.init_module(empty_init=False):
        model = build_gpt(config, tie_embeddings=True)
    initialize_weights(fabric, model, n_layer=config.n_layer, n_embd=config.n_embd)
    model.max_seq_length = 1024
    ids = torch.randint(0, 32000, (1, 1024), device=fabric.device)
    print("eager forward...", flush=True)
    with torch.no_grad():
        eager = model(ids)
    torch.cuda.synchronize()
    print("eager logits", tuple(eager.shape), float(eager.float().mean()), flush=True)
    print("compiling GPT...", flush=True)
    t1 = time.perf_counter()
    model = torch.compile(model, backend="inductor")
    with torch.no_grad():
        compiled_out = model(ids)
    torch.cuda.synchronize()
    print(f"compiled GPT forward ok in {time.perf_counter() - t1:.2f}s", flush=True)
    delta = (eager.float() - compiled_out.float()).abs().max().item()
    print("max_abs_logit_diff", delta, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
