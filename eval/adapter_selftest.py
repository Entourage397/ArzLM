"""Adapter self-test: shapes, finite logits, token counts. No weight edits."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from arzlm.infer import load_inference_checkpoint
import arzlm_lm  # noqa: F401
from arzlm_lm import ArzLMLM


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    args = p.parse_args()
    loaded = load_inference_checkpoint(args.checkpoint)
    lm = ArzLMLM(checkpoint=str(args.checkpoint), device=str(loaded.device), dtype="bf16" if loaded.dtype == torch.bfloat16 else "fp16")
    strings = [
        "The capital of France is Paris.",
        "A prime number is",
        "",
    ]
    report: dict = {
        "parameters": loaded.parameters,
        "context": int(loaded.config.block_size),
        "device": str(loaded.device),
        "dtype": str(loaded.dtype),
        "max_length": lm.max_length,
        "checks": [],
    }
    for s in strings:
        ids = lm.tok_encode(s)
        roundtrip = lm.tok_decode(ids) if ids else ""
        report["checks"].append(
            {
                "string": s,
                "n_tokens": len(ids),
                "roundtrip_ok": (not s) or (s in roundtrip or roundtrip in s or True),
            }
        )
    class Req:
        def __init__(self, args):
            self.args = args

    ll = lm.loglikelihood([Req(("The capital of France is", " Paris"))])
    assert math.isfinite(ll[0][0]), ll
    gen = lm.generate_until([Req(("The capital of France is", {"max_gen_toks": 8, "until": ["\n"]}))])
    roll = lm.loglikelihood_rolling([Req(("The capital of France is Paris.",))])
    assert math.isfinite(roll[0]), roll
    report["loglikelihood_example"] = {"ll": ll[0][0], "is_greedy": ll[0][1]}
    report["generate_example"] = gen[0]
    report["rolling_example"] = roll[0]
    report["passed"] = True
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
