"""Write Muon weight-decay 2M configs around a chosen LR (aux AdamW wd unchanged)."""

from __future__ import annotations

import argparse
from pathlib import Path

from arzlm.paths import CONFIGS_DIR

TEMPLATE = """name: tune-muon-wd{wd_tag}-lr{lr_tag}-2m
model_config: ../model-arzlm-100m.yaml
out_dir: runs/tune/muon-wd{wd_tag}-lr{lr_tag}-2m
data_dir: data/prepared/10m
tokenizer_dir: tokenizer/trained
precision: bf16-true
compile: false
seed: 42
devices: 1
logger_name: csv
resume: false
num_workers: 0

train:
  save_interval: 100000
  log_interval: 10
  global_batch_size: 8
  micro_batch_size: 2
  lr_warmup_steps: 50
  max_tokens: 2000000
  max_seq_length: 1024
  tie_embeddings: true
  max_norm: 1.0
  min_lr: 6.0e-5

eval:
  interval: 50
  max_iters: 8
  initial_validation: false
  final_validation: true

optimizer:
  name: muon_hybrid
  class_path: arzlm.training.optim.muon.SingleDeviceMuonWithAuxAdam
  muon:
    lr: {lr}
    momentum: 0.95
    weight_decay: {wd}
    nesterov: true
    ns_steps: 5
    scale: jordan
  adamw:
    lr: 6.0e-4
    weight_decay: 0.1
    betas: [0.9, 0.95]
    eps: 1.0e-8
"""


def _tag(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--wds", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    args = parser.parse_args()
    out_dir = CONFIGS_DIR / "tune"
    paths = []
    for wd in args.wds:
        path = out_dir / f"muon-wd{_tag(wd)}-lr{_tag(args.lr)}-2m.yaml"
        path.write_text(
            TEMPLATE.format(lr=args.lr, wd=wd, lr_tag=_tag(args.lr), wd_tag=_tag(wd)),
            encoding="utf-8",
        )
        paths.append(path)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
