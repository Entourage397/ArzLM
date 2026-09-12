"""Count remote bytes actually transferred during corpus preparation."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class ByteCounter:
    name: str
    bytes_read: int = 0
    requests: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, n: int, *, requests: int = 0) -> None:
        if n <= 0 and requests <= 0:
            return
        with self._lock:
            self.bytes_read += int(n)
            self.requests += int(requests)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {"bytes": self.bytes_read, "requests": self.requests}


class CountingReader:
    """Wrap a binary stream and count read() bytes."""

    def __init__(self, inner, counter: ByteCounter) -> None:
        self.inner = inner
        self.counter = counter
        self.counter.add(0, requests=1)

    def read(self, n: int = -1) -> bytes:
        data = self.inner.read(n)
        if data:
            self.counter.add(len(data))
        return data

    def close(self) -> None:
        closer = getattr(self.inner, "close", None)
        if closer is not None:
            closer()

    def readinto(self, b) -> int:
        if hasattr(self.inner, "readinto"):
            n = self.inner.readinto(b)
            if n:
                self.counter.add(n)
            return n
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)

    def __getattr__(self, name: str):
        return getattr(self.inner, name)


def dir_nbytes(path) -> int:
    from pathlib import Path

    root = Path(path)
    if not root.exists():
        return 0
    total = 0
    for p in root.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def prune_project_hf_dataset_cache(hf_home, dataset_ids: list[str]) -> dict[str, int]:
    """Delete Hub dataset blobs that live under this project's HF_HOME only.

    Never touches ~/.cache/huggingface or other user caches. Only deletes
    dataset ids the caller lists. Ordinary FineWeb remains forbidden.
    """
    from pathlib import Path

    from arzlm.data.catalog import FORBIDDEN_DATASETS

    root = Path(hf_home)
    hub = root / "hub"
    removed = 0
    removed_files = 0
    for dataset_id in dataset_ids:
        if dataset_id in FORBIDDEN_DATASETS:
            continue
        lower = dataset_id.lower()
        if lower.startswith("huggingfacefw/fineweb") and "fineweb-edu" not in lower:
            continue
        slug = "datasets--" + dataset_id.replace("/", "--")
        target = hub / slug
        if not target.is_dir():
            continue
        for p in target.rglob("*"):
            if p.is_file():
                try:
                    removed += p.stat().st_size
                    p.unlink()
                    removed_files += 1
                except OSError:
                    pass
        # Remove empty dirs deepest-first.
        for p in sorted(target.rglob("*"), key=lambda x: len(x.parts), reverse=True):
            if p.is_dir():
                try:
                    p.rmdir()
                except OSError:
                    pass
        try:
            target.rmdir()
        except OSError:
            pass
    return {"bytes_removed": removed, "files_removed": removed_files}
