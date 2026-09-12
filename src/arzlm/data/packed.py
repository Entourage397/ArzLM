"""Local packed token shards for high-throughput pretraining.

Format: little-endian uint16 raw `.bin` plus `meta.json`. Documents are
concatenated with an explicit EOS token after every document. Training
samples are contiguous non-overlapping windows of `seq_length + 1` tokens
(input + next-token target), matching litgpt.pretrain.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

DTYPE = np.uint16
FORMAT_NAME = "arzlm-packed-v1"


def pack_documents(documents: list[list[int]], eos_id: int) -> np.ndarray:
    """Concatenate tokenized documents, appending eos_id after each one."""
    if eos_id < 0 or eos_id > np.iinfo(DTYPE).max:
        raise ValueError(f"eos_id {eos_id} does not fit in {DTYPE}")
    pieces: list[np.ndarray] = []
    for doc in documents:
        if not doc:
            continue
        arr = np.asarray(doc, dtype=np.int64)
        if (arr < 0).any() or (arr > np.iinfo(DTYPE).max).any():
            raise ValueError("token id outside uint16 range")
        pieces.append(arr.astype(DTYPE, copy=False))
        pieces.append(np.array([eos_id], dtype=DTYPE))
    if not pieces:
        return np.zeros((0,), dtype=DTYPE)
    return np.concatenate(pieces)


def write_packed_split(path: Path, tokens: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens = np.asarray(tokens, dtype=DTYPE)
    tokens.tofile(path)
    return path


STEM_FORMAT_NAME = "arzlm-stem-v1"


class PackedTokenDataset(Dataset):
    """Memory-mapped packed tokens; each item is a `seq_length + 1` int64 tensor."""

    def __init__(self, bin_path: str | Path, seq_length: int, dtype=DTYPE) -> None:
        self.bin_path = Path(bin_path)
        self.seq_length = int(seq_length)
        self.block = self.seq_length + 1
        self.dtype = np.dtype(dtype)
        self.data = np.memmap(self.bin_path, dtype=self.dtype, mode="r")
        if len(self.data) < self.block:
            raise ValueError(
                f"{self.bin_path} has {len(self.data)} tokens; need at least {self.block} "
                f"(seq_length + 1)"
            )
        self.n_sequences = len(self.data) // self.block

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0:
            idx += self.n_sequences
        if idx < 0 or idx >= self.n_sequences:
            raise IndexError(idx)
        start = idx * self.block
        chunk = np.asarray(self.data[start : start + self.block], dtype=np.int64)
        return torch.from_numpy(chunk)

    @property
    def n_tokens(self) -> int:
        return int(self.n_sequences * self.block)


class ConcatPackedDataset(Dataset):
    """Several packed shards as one window sequence. Each shard is block-aligned."""

    def __init__(self, paths: list[Path] | list[str], seq_length: int, dtype: str = "uint16") -> None:
        self.parts = [PackedTokenDataset(p, seq_length=seq_length, dtype=dtype) for p in paths]
        if not self.parts:
            raise ValueError("no shards")
        self.seq_length = int(seq_length)
        self._cum = np.cumsum([len(p) for p in self.parts], dtype=np.int64)
        self._n = int(self._cum[-1])

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0:
            idx += self._n
        if idx < 0 or idx >= self._n:
            raise IndexError(idx)
        part_i = int(np.searchsorted(self._cum, idx, side="right"))
        prev = 0 if part_i == 0 else int(self._cum[part_i - 1])
        return self.parts[part_i][idx - prev]

    @property
    def n_tokens(self) -> int:
        return int(sum(p.n_tokens for p in self.parts))


def read_meta(data_dir: str | Path) -> dict[str, Any]:
    meta_path = Path(data_dir) / "meta.json"
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    fmt = meta.get("format")
    if fmt not in {FORMAT_NAME, STEM_FORMAT_NAME}:
        raise ValueError(f"Unsupported packed format in {meta_path}: {fmt}")
    return meta


def load_split_dataset(data_dir: str | Path, split: str, seq_length: int):
    meta = read_meta(data_dir)
    if meta.get("format") == STEM_FORMAT_NAME:
        raise TypeError(
            f"{data_dir} is {STEM_FORMAT_NAME}; load train via arzlm.data.mixture.load_stem_train_dataset"
        )
    files = meta["splits"][split]["files"]
    dtype = meta.get("dtype", "uint16")
    paths = [Path(data_dir) / f for f in files]
    if len(paths) == 1:
        return PackedTokenDataset(paths[0], seq_length=seq_length, dtype=dtype)
    return ConcatPackedDataset(paths, seq_length=seq_length, dtype=dtype)
