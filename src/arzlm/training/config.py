"""Experiment YAML loading. Training fields map onto litgpt.args.TrainArgs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from litgpt.args import EvalArgs, TrainArgs
from litgpt.config import Config

from arzlm.model.config import build_model_config, load_model_config
from arzlm.paths import REPO_ROOT


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def resolve_path(value: str | Path | None, *, base: Path = REPO_ROOT) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


@dataclass
class ExperimentConfig:
    model: Config
    train: TrainArgs
    eval: EvalArgs
    optimizer: dict[str, Any]
    out_dir: Path
    data_dir: Path
    tokenizer_dir: Path
    precision: str
    compile: str
    seed: int
    devices: int
    logger_name: str
    resume: bool | str
    num_workers: int
    raw: dict[str, Any]
    mixture: dict[str, float] | None = None


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw = _read_yaml(path)
    model_ref = raw.get("model_config")
    if isinstance(model_ref, str):
        model = load_model_config(resolve_path(model_ref, base=path.parent) or Path(model_ref))
    elif isinstance(model_ref, dict):
        model = build_model_config(model_ref)
    else:
        model = build_model_config()

    train_raw = dict(raw.get("train") or {})
    eval_raw = dict(raw.get("eval") or {})
    # TrainArgs does not accept unknown keys.
    train = TrainArgs(**train_raw)
    eval_args = EvalArgs(**eval_raw)

    optimizer = raw.get("optimizer") or {
        "class_path": "torch.optim.AdamW",
        "init_args": {"lr": 6e-4, "weight_decay": 0.1, "betas": [0.9, 0.95]},
    }

    out_dir = resolve_path(raw.get("out_dir", "runs/default"))
    data_dir = resolve_path(raw.get("data_dir", "data/prepared/tiny"))
    tokenizer_dir = resolve_path(raw.get("tokenizer_dir", "tokenizer/trained"))
    assert out_dir and data_dir and tokenizer_dir

    compile_flag = str(raw.get("compile", "auto")).lower()
    if compile_flag not in {"auto", "true", "false", "1", "0"}:
        raise ValueError("compile must be auto|true|false")
    if compile_flag in {"1"}:
        compile_flag = "true"
    if compile_flag in {"0"}:
        compile_flag = "false"

    resume = raw.get("resume", False)
    mixture = raw.get("mixture")
    if mixture is None and isinstance(raw.get("data"), dict):
        mixture = raw["data"].get("mixture")
    return ExperimentConfig(
        model=model,
        train=train,
        eval=eval_args,
        optimizer=optimizer,
        out_dir=out_dir,
        data_dir=data_dir,
        tokenizer_dir=tokenizer_dir,
        precision=raw.get("precision") or "bf16-true",
        compile=compile_flag,
        seed=int(raw.get("seed", 42)),
        devices=int(raw.get("devices", 1)),
        logger_name=raw.get("logger_name", "tensorboard"),
        resume=resume,
        num_workers=int(raw.get("num_workers", 0)),
        raw=raw,
        mixture=mixture,
    )
