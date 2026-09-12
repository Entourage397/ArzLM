"""Record which upstream physical files were actually read during packing."""

from __future__ import annotations

import json
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arzlm.data.security.policy import RemoteFileMeta
from arzlm.training.report import utc_now

_LEDGER: ContextVar["UpstreamLedger | None"] = ContextVar("arzlm_upstream_ledger", default=None)


@dataclass
class UpstreamRecord:
    dataset_id: str
    config: str | None
    revision: str
    path: str
    sha256: str | None
    size: int | None
    security_status: str | None
    av_status: str | None
    checked_at: str
    local_sha256: str | None = None
    local_bytes: int | None = None


class UpstreamLedger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.records: list[dict[str, Any]] = []

    def record(
        self,
        meta: RemoteFileMeta,
        *,
        config: str | None = None,
        local_sha256: str | None = None,
        local_bytes: int | None = None,
    ) -> None:
        payload = {
            "dataset_id": meta.dataset_id,
            "config": config,
            "revision": meta.revision,
            "path": meta.path,
            "upstream_hash": meta.sha256,
            "size": meta.size,
            "huggingface_security_status": meta.security_status,
            "av_status": meta.av_status,
            "timestamp_checked": utc_now(),
            "local_sha256": local_sha256,
            "local_bytes": local_bytes,
        }
        line = json.dumps(payload, sort_keys=True)
        with self._lock:
            self.records.append(payload)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def current_ledger() -> UpstreamLedger | None:
    return _LEDGER.get()


def set_ledger(ledger: UpstreamLedger | None):
    return _LEDGER.set(ledger)


def reset_ledger(token) -> None:
    _LEDGER.reset(token)
