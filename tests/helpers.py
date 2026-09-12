from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from litgpt.args import EvalArgs, TrainArgs

from arzlm.data.packed import FORMAT_NAME, write_packed_split
from arzlm.tokenizer.train import save_litgpt_tokenizer_files, train_bpe_tokenizer
from arzlm.training.config import ExperimentConfig


def write_random_packed(tmp_path: Path, *, n_train: int, n_val: int, seq_length: int, vocab: int = 1000, seed: int = 0) -> Path:
    block = seq_length + 1
    rng = np.random.default_rng(seed)
    train = rng.integers(0, vocab, size=(n_train // block) * block, dtype=np.uint16)
    val = rng.integers(0, vocab, size=(n_val // block) * block, dtype=np.uint16)
    data_dir = tmp_path / "packed"
    data_dir.mkdir(parents=True, exist_ok=True)
    write_packed_split(data_dir / "train.bin", train)
    write_packed_split(data_dir / "val.bin", val)
    meta = {
        "format": FORMAT_NAME,
        "dtype": "uint16",
        "splits": {
            "train": {"files": ["train.bin"], "n_tokens": int(train.size)},
            "val": {"files": ["val.bin"], "n_tokens": int(val.size)},
        },
        "seq_length": seq_length,
        "eos_id": 2,
    }
    (data_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return data_dir


def write_mini_tokenizer(tmp_path: Path) -> Path:
    texts = [
        "photosynthesis converts light energy into chemical energy.",
        "force equals mass times acceleration in newtonian mechanics.",
        "the water cycle includes evaporation condensation and precipitation.",
    ] * 200
    tok_dir = tmp_path / "tokenizer"
    trained = train_bpe_tokenizer(texts, vocab_size=512, min_frequency=1)
    save_litgpt_tokenizer_files(trained, tok_dir)
    return tok_dir


def cpu_experiment(tmp_path: Path, *, seq_length: int = 32, max_steps: int = 2, seed: int = 123) -> ExperimentConfig:
    from arzlm.model.config import build_model_config

    data_dir = write_random_packed(
        tmp_path,
        n_train=seq_length * 16,
        n_val=seq_length * 8,
        seq_length=seq_length,
        vocab=1000,
        seed=seed,
    )
    tok_dir = write_mini_tokenizer(tmp_path)
    out_dir = tmp_path / "run"
    model = build_model_config()
    train = TrainArgs(
        save_interval=1000,
        log_interval=1,
        global_batch_size=1,
        micro_batch_size=1,
        lr_warmup_steps=1,
        max_tokens=10_000_000,
        max_steps=max_steps,
        max_seq_length=seq_length,
        tie_embeddings=True,
        max_norm=1.0,
        min_lr=6e-5,
    )
    eval_args = EvalArgs(interval=10_000, max_iters=1, initial_validation=False, final_validation=False)
    return ExperimentConfig(
        model=model,
        train=train,
        eval=eval_args,
        optimizer={
            "class_path": "torch.optim.AdamW",
            "init_args": {"lr": 6e-4, "weight_decay": 0.1, "betas": [0.9, 0.95]},
        },
        out_dir=out_dir,
        data_dir=data_dir,
        tokenizer_dir=tok_dir,
        precision="32-true",
        compile="false",
        seed=seed,
        devices=1,
        logger_name="csv",
        resume=False,
        num_workers=0,
        raw={},
        mixture=None,
    )


def gpu_available() -> bool:
    return torch.cuda.is_available()
