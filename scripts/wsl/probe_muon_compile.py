"""Isolate Muon + torch.compile crash: forward, backward, then optimizer.step."""

from __future__ import annotations

import os
import time
import traceback

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
from litgpt.utils import chunked_cross_entropy

from arzlm.model.config import build_gpt, build_model_config
from arzlm.training.optim.factory import build_optimizer

SPEC = {
    "name": "muon_hybrid",
    "muon": {"lr": 0.05, "momentum": 0.95, "weight_decay": 0.1},
    "adamw": {"lr": 6e-4, "weight_decay": 0.1, "betas": [0.9, 0.95], "eps": 1e-8},
}


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    disable_cg = os.environ.get("ARZLM_DISABLE_CUDA_GRAPHS", "1") == "1"
    if disable_cg:
        import torch._inductor.config as inductor_config

        inductor_config.triton.cudagraphs = False
        log("cudagraphs disabled")
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)
    device = torch.device("cuda")
    config = build_model_config()
    model = build_gpt(config, tie_embeddings=True).to(device=device, dtype=torch.bfloat16)
    model.max_seq_length = 1024
    opt = build_optimizer(SPEC, model, fused_cuda=True)
    x = torch.randint(0, 32000, (2, 1024), device=device)
    y = torch.randint(0, 32000, (2, 1024), device=device)
    log("eager step")
    logits = model(x)
    loss = chunked_cross_entropy(logits, y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    log(f"eager ok loss={float(loss.detach())}")
    log("compiling model")
    compiled = torch.compile(model, backend="inductor")
    log("compiled forward")
    t0 = time.perf_counter()
    logits = compiled(x)
    torch.cuda.synchronize()
    log(f"forward ok in {time.perf_counter()-t0:.2f}s")
    log("compiled backward")
    loss = chunked_cross_entropy(logits, y)
    loss.backward()
    torch.cuda.synchronize()
    log("backward ok")
    log("muon step after compiled backward")
    opt.step()
    torch.cuda.synchronize()
    log("muon step ok")
    log("second compiled train step")
    opt.zero_grad()
    logits = compiled(x)
    loss = chunked_cross_entropy(logits, y)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    log(f"second step ok loss={float(loss.detach())}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
