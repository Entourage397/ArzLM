"""Local inference performance: load time, VRAM, prefill and decode tok/s."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch


def _vram() -> dict[str, int | None]:
    if not torch.cuda.is_available():
        return {"allocated": None, "reserved": None}
    torch.cuda.synchronize()
    return {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
        "max_allocated": int(torch.cuda.max_memory_allocated()),
    }


def _make_ids(n: int, vocab: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(1234)
    return torch.randint(4, max(5, vocab - 1), (1, n), generator=g, dtype=torch.long).to(device)


def measure_hf(repo: str, dtype: str, prompt_tokens: int, gen_tokens: int, warmup: int, repeats: int) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16}[dtype]
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch_dtype, device_map="cuda")
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    after_load = _vram()
    vocab = int(getattr(model.config, "vocab_size", 32000))
    device = next(model.parameters()).device
    ids = _make_ids(prompt_tokens, vocab, device)
    model.eval()
    times_prefill = []
    times_decode = []
    with torch.no_grad():
        for i in range(warmup + repeats):
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            out = model(ids)
            torch.cuda.synchronize()
            prefill = time.perf_counter() - t1
            del out
            x = ids
            t2 = time.perf_counter()
            for _ in range(gen_tokens):
                logits = model(x).logits
                nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
                x = torch.cat([x, nxt], dim=1)
            torch.cuda.synchronize()
            decode = time.perf_counter() - t2
            if i >= warmup:
                times_prefill.append(prefill)
                times_decode.append(decode)
    return {
        "kind": "hf",
        "repo": repo,
        "load_s": load_s,
        "vram_after_load": after_load,
        "vram_peak": _vram(),
        "prefill_s": {"mean": statistics.mean(times_prefill), "median": statistics.median(times_prefill)},
        "decode_s": {"mean": statistics.mean(times_decode), "median": statistics.median(times_decode)},
        "prefill_tok_s": prompt_tokens / statistics.mean(times_prefill),
        "decode_tok_s": gen_tokens / statistics.mean(times_decode),
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "dtype": dtype,
        "device": str(device),
    }


def measure_arzlm(checkpoint: str, dtype: str, prompt_tokens: int, gen_tokens: int, warmup: int, repeats: int) -> dict[str, Any]:
    from arzlm.infer import load_inference_checkpoint

    t0 = time.perf_counter()
    loaded = load_inference_checkpoint(checkpoint, dtype=dtype)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    after_load = _vram()
    vocab = int(loaded.config.padded_vocab_size or loaded.config.vocab_size)
    ids = _make_ids(prompt_tokens, vocab, loaded.device)
    times_prefill = []
    times_decode = []
    with torch.no_grad():
        for i in range(warmup + repeats):
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            _ = loaded.model(ids)
            torch.cuda.synchronize()
            prefill = time.perf_counter() - t1
            x = ids
            t2 = time.perf_counter()
            for _ in range(gen_tokens):
                logits = loaded.model(x)
                nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
                x = torch.cat([x, nxt], dim=1)
            torch.cuda.synchronize()
            decode = time.perf_counter() - t2
            if i >= warmup:
                times_prefill.append(prefill)
                times_decode.append(decode)
    return {
        "kind": "arzlm",
        "checkpoint": checkpoint,
        "parameters": loaded.parameters,
        "load_s": load_s,
        "vram_after_load": after_load,
        "vram_peak": _vram(),
        "prefill_s": {"mean": statistics.mean(times_prefill), "median": statistics.median(times_prefill)},
        "decode_s": {"mean": statistics.mean(times_decode), "median": statistics.median(times_decode)},
        "prefill_tok_s": prompt_tokens / statistics.mean(times_prefill),
        "decode_tok_s": gen_tokens / statistics.mean(times_decode),
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "dtype": str(loaded.dtype),
        "device": str(loaded.device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["arzlm", "hf"], required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--repo", default=None)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--gen-tokens", type=int, default=256)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if args.kind == "arzlm":
        rec = measure_arzlm(args.checkpoint, args.dtype, args.prompt_tokens, args.gen_tokens, args.warmup, args.repeats)
    else:
        rec = measure_hf(args.repo, args.dtype, args.prompt_tokens, args.gen_tokens, args.warmup, args.repeats)
    text = json.dumps(rec, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
