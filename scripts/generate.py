#!/usr/bin/env python3
"""Greedy generation helper. Same as `python -m arzlm generate`."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arzlm.cli import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    args = parser.parse_args()
    argv = ["generate", "--checkpoint", args.checkpoint, "--max-new-tokens", str(args.max_new_tokens)]
    if args.device:
        argv += ["--device", args.device]
    if args.dtype:
        argv += ["--dtype", args.dtype]
    for p in args.prompt:
        argv += ["--prompt", p]
    raise SystemExit(main(argv))
