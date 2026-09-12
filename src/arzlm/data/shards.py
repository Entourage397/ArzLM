"""Atomic uint16 shard writes with SHA-256 sidecars."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from arzlm.data.packed import DTYPE, FORMAT_NAME

# 2^23 tokens = 16 MiB of uint16. Chosen after a local sequential-read check:
# 4/16/64 MiB shards have similar memmap throughput; 16 MiB keeps file count
# around 125 for 1B tokens without tiny-file overhead.
DEFAULT_SHARD_TOKENS = 8_388_608


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fsync_file(fh) -> None:
    try:
        os.fsync(fh.fileno())
    except OSError:
        pass


def write_shard_atomic(
    path: Path,
    tokens: np.ndarray,
    *,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Write tokens to path via a sibling .tmp, then fsync+replace. Sidecar JSON next to it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens = np.asarray(tokens, dtype=DTYPE)
    if tokens.size == 0:
        raise ValueError(f"refusing empty shard {path}")
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with open(tmp, "wb") as fh:
        tokens.tofile(fh)
        fh.flush()
        _fsync_file(fh)
    digest = sha256_file(tmp)
    os.replace(tmp, path)
    record = {
        "file": path.name,
        "n_tokens": int(tokens.size),
        "sha256": digest,
        "dtype": "uint16",
        "endian": "little",
        "format": FORMAT_NAME,
        **meta,
    }
    sidecar = Path(str(path) + ".json")
    tmp_json = Path(str(sidecar) + ".tmp")
    payload = json.dumps(record, indent=2) + "\n"
    tmp_json.write_text(payload, encoding="utf-8")
    os.replace(tmp_json, sidecar)
    return record


def drop_block_remainder(tokens: np.ndarray, seq_length: int) -> np.ndarray:
    block = int(seq_length) + 1
    n = (len(tokens) // block) * block
    return tokens[:n]


def load_shard_records(shard_dir: Path) -> list[dict[str, Any]]:
    """Recover shard sidecars after a crash that flushed files but not state."""
    shard_dir = Path(shard_dir)
    if not shard_dir.is_dir():
        return []
    recs: list[dict[str, Any]] = []
    for path in shard_dir.glob("shard-*.bin.json"):
        try:
            recs.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
    recs.sort(key=lambda r: int(r.get("creation_order", 0)))
    return recs


def shard_token_sum(records: list[dict[str, Any]]) -> int:
    return int(sum(int(r.get("n_tokens", 0)) for r in records))


def partial_bin_path(state_dir: Path, domain: str, split: str) -> Path:
    return Path(state_dir) / f"{domain}.{split}.partial.bin"


def write_partial(path: Path, tokens: np.ndarray, *, n_documents: int) -> None:
    path = Path(path)
    sidecar = Path(str(path) + ".json")
    if tokens.size == 0:
        if path.exists():
            path.unlink()
        if sidecar.exists():
            sidecar.unlink()
        return
    write_shard_atomic(
        path,
        np.asarray(tokens, dtype=DTYPE),
        meta={"partial": True, "n_documents": int(n_documents), "creation_order": -1},
    )


def read_partial(path: Path) -> tuple[np.ndarray, int]:
    path = Path(path)
    if not path.is_file():
        return np.zeros((0,), dtype=DTYPE), 0
    tokens = np.fromfile(path, dtype=DTYPE)
    n_docs = 0
    sidecar = Path(str(path) + ".json")
    if sidecar.is_file():
        try:
            n_docs = int(json.loads(sidecar.read_text(encoding="utf-8")).get("n_documents", 0))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            n_docs = 0
    return np.asarray(tokens, dtype=DTYPE), n_docs


def unlink_partial(path: Path) -> None:
    path = Path(path)
    sidecar = Path(str(path) + ".json")
    for p in (path, sidecar, Path(str(path) + ".tmp"), Path(str(sidecar) + ".tmp")):
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass


class ShardAccumulator:
    """Buffer tokens and flush complete shards. Remainder stays until flush(final=True)."""

    def __init__(
        self,
        shard_dir: Path,
        *,
        domain: str,
        split: str,
        shard_tokens: int = DEFAULT_SHARD_TOKENS,
        seq_length: int = 1024,
        prefix: str = "shard",
        start_index: int = 0,
    ) -> None:
        self.shard_dir = Path(shard_dir)
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.domain = domain
        self.split = split
        self.shard_tokens = int(shard_tokens)
        self.seq_length = int(seq_length)
        self.prefix = prefix
        self.next_index = int(start_index)
        self.buffer: list[np.ndarray] = []
        self.buffered = 0
        self.records: list[dict[str, Any]] = []
        self.n_documents = 0
        self.n_eos = 0
        self._docs_since_flush = 0

    def append(self, packed: np.ndarray, *, n_docs: int = 1) -> None:
        packed = np.asarray(packed, dtype=DTYPE)
        if packed.size == 0:
            return
        self.buffer.append(packed)
        self.buffered += int(packed.size)
        self.n_documents += n_docs
        self.n_eos += n_docs
        self._docs_since_flush += n_docs
        while self.buffered >= self.shard_tokens:
            self._flush_one(final=False)

    def export_partial(self) -> tuple[np.ndarray, int]:
        if not self.buffer:
            return np.zeros((0,), dtype=DTYPE), 0
        cat = np.concatenate(self.buffer) if len(self.buffer) > 1 else self.buffer[0]
        return np.asarray(cat, dtype=DTYPE), int(self._docs_since_flush)

    def import_partial(self, tokens: np.ndarray, n_documents: int = 0) -> None:
        tokens = np.asarray(tokens, dtype=DTYPE)
        if tokens.size == 0:
            return
        self.buffer = [tokens]
        self.buffered = int(tokens.size)
        self.n_documents += int(n_documents)
        self.n_eos += n_documents
        self._docs_since_flush += int(n_documents)

    def flush(self, *, final: bool) -> list[dict[str, Any]]:
        if self.buffered == 0:
            return list(self.records)
        if final:
            self._flush_one(final=True)
        else:
            while self.buffered >= self.shard_tokens:
                self._flush_one(final=False)
        return list(self.records)

    def _next_free_name(self) -> tuple[str, int]:
        while True:
            name = f"{self.prefix}-{self.next_index:05d}.bin"
            path = self.shard_dir / name
            if not path.exists():
                return name, self.next_index
            sidecar = Path(str(path) + ".json")
            if sidecar.is_file() and not any(r.get("file") == name for r in self.records):
                try:
                    self.records.append(json.loads(sidecar.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    pass
            self.next_index += 1

    def _flush_one(self, *, final: bool) -> None:
        if not self.buffer:
            return
        cat = np.concatenate(self.buffer) if len(self.buffer) > 1 else self.buffer[0]
        block = self.seq_length + 1
        docs_this = int(self._docs_since_flush)
        if final:
            n = (cat.size // block) * block
            chunk = cat[:n]
            self.buffer = []
            self.buffered = 0
            self._docs_since_flush = 0
        else:
            target = min(self.shard_tokens, cat.size)
            n = (target // block) * block
            if n == 0:
                return
            chunk = cat[:n]
            tail = cat[n:]
            self.buffer = [tail] if tail.size else []
            self.buffered = int(tail.size)
            self._docs_since_flush = 0
        if chunk.size == 0:
            return
        name, order = self._next_free_name()
        self.next_index = order + 1
        rec = write_shard_atomic(
            self.shard_dir / name,
            chunk,
            meta={
                "domain": self.domain,
                "split": self.split,
                "creation_order": order,
                "n_documents": docs_this,
            },
        )
        self.records.append(rec)
