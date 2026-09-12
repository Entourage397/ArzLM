"""Fail-closed classification of remote dataset artifacts.

File security (can we fetch this blob?) is separate from content quality
(does the text mention exploits). This module only answers the former.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

VerdictKind = Literal["approved", "denied", "unresolved", "mismatch"]

HARD_DENY_STATUSES = frozenset({"unsafe", "infected", "malware", "unwanted"})
PENDING_STATUSES = frozenset({"unscanned", "queued", "scanning", "error", "unknown", ""})

ALLOWED_SUFFIXES = (
    ".parquet",
    ".arrow",
    ".json",
    ".jsonl",
    ".json.gz",
    ".jsonl.gz",
    ".txt",
    ".txt.gz",
    ".csv",
    ".tsv",
    ".csv.gz",
)

# Never ingest these as training-source artifacts.
DENIED_SUFFIXES = (
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".bin",
    ".exe",
    ".so",
    ".dll",
    ".dylib",
    ".sh",
    ".bat",
    ".ps1",
    ".cmd",
)

PICKLE_SUFFIXES = (".pkl", ".pickle", ".pt", ".pth")


@dataclass(frozen=True)
class RemoteFileMeta:
    dataset_id: str
    revision: str
    path: str
    size: int | None = None
    sha256: str | None = None
    xet_hash: str | None = None
    blob_id: str | None = None
    security_status: str | None = None
    av_status: str | None = None
    av_message: str | None = None
    pickle_status: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DenylistEntry:
    dataset_id: str
    path: str
    sha256: str | None = None
    reason: str = ""
    revision: str | None = None
    xet_hash: str | None = None


@dataclass(frozen=True)
class LockFileEntry:
    path: str
    sha256: str | None = None
    size: int | None = None
    xet_hash: str | None = None
    security_status: str | None = None
    av_status: str | None = None
    reviewed_allow_unscanned: bool = False


@dataclass(frozen=True)
class Verdict:
    kind: VerdictKind
    reason: str
    meta: RemoteFileMeta
    denylist: bool = False


def normalize_status(value: object | None) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def path_suffixes(path: str) -> str:
    name = path.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if name.endswith(".json.gz"):
        return ".json.gz"
    if name.endswith(".jsonl.gz"):
        return ".jsonl.gz"
    if name.endswith(".txt.gz"):
        return ".txt.gz"
    if name.endswith(".csv.gz"):
        return ".csv.gz"
    dot = name.rfind(".")
    return name[dot:] if dot >= 0 else ""


def is_allowed_source_format(path: str) -> bool:
    return path_suffixes(path) in ALLOWED_SUFFIXES


def is_denied_source_format(path: str) -> bool:
    suf = path_suffixes(path)
    if suf in DENIED_SUFFIXES:
        return True
    name = path.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if name.endswith(".py") or name.endswith(".ipynb"):
        return True
    return False


def is_pickle_like(path: str) -> bool:
    return path_suffixes(path) in PICKLE_SUFFIXES


def is_fineweb_full_dump_path(dataset_id: str, path: str) -> bool:
    ds = (dataset_id or "").lower()
    rel = path.replace("\\", "/")
    return "fineweb-edu" in ds and rel.startswith("data/CC-MAIN")


def load_denylist(path: Path) -> list[DenylistEntry]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = []
    for raw in payload.get("entries") or []:
        entries.append(
            DenylistEntry(
                dataset_id=str(raw["dataset_id"]),
                path=str(raw["path"]).replace("\\", "/"),
                sha256=(str(raw["sha256"]).lower() if raw.get("sha256") else None),
                reason=str(raw.get("reason") or ""),
                revision=str(raw["revision"]) if raw.get("revision") else None,
                xet_hash=str(raw["xet_hash"]) if raw.get("xet_hash") else None,
            )
        )
    return entries


def denylist_match(
    meta: RemoteFileMeta,
    denylist: list[DenylistEntry],
) -> DenylistEntry | None:
    path = meta.path.replace("\\", "/")
    digest = (meta.sha256 or "").lower() or None
    for entry in denylist:
        if entry.dataset_id != meta.dataset_id:
            if digest and entry.sha256 and digest == entry.sha256:
                return entry
            continue
        if path == entry.path:
            return entry
        if digest and entry.sha256 and digest == entry.sha256:
            return entry
    return None


def classify_remote_file(
    meta: RemoteFileMeta,
    *,
    denylist: list[DenylistEntry],
    lock_entry: LockFileEntry | None = None,
    allow_unscanned_review: bool = False,
) -> Verdict:
    """Return approved / denied / unresolved / mismatch. Never warn-only."""
    hit = denylist_match(meta, denylist)
    if hit is not None:
        return Verdict("denied", f"denylist: {hit.reason or hit.path}", meta, denylist=True)

    if is_fineweb_full_dump_path(meta.dataset_id, meta.path):
        return Verdict("denied", "FineWeb-Edu full CC-MAIN dump is not an allowed fetch path", meta)

    if is_denied_source_format(meta.path) or not is_allowed_source_format(meta.path):
        return Verdict("denied", f"disallowed source format {path_suffixes(meta.path)!r}", meta)

    if lock_entry is not None and lock_entry.sha256 and meta.sha256:
        if lock_entry.sha256.lower() != meta.sha256.lower():
            return Verdict(
                "mismatch",
                f"sha256 mismatch lock={lock_entry.sha256} live={meta.sha256}",
                meta,
            )

    status = normalize_status(meta.security_status)
    av = normalize_status(meta.av_status)
    pickle_st = normalize_status(meta.pickle_status)

    if status in HARD_DENY_STATUSES or av in HARD_DENY_STATUSES:
        detail = meta.av_message or status or av
        return Verdict("denied", f"Hugging Face security failed ({detail})", meta)

    if is_pickle_like(meta.path):
        if pickle_st in HARD_DENY_STATUSES:
            return Verdict("denied", "pickle scan failed", meta)
        if pickle_st != "safe":
            return Verdict("unresolved", "pickle artifact lacks a safe pickle scan", meta)

    reviewed = allow_unscanned_review or (lock_entry.reviewed_allow_unscanned if lock_entry else False)

    if status == "safe" or av == "safe":
        return Verdict("approved", "Hugging Face AV status is safe", meta)

    # A pin recorded Hub-safe. Live list_repo_tree sometimes omits security
    # fields; do not treat omission as a new unknown if hashes still match.
    if lock_entry is not None:
        lock_st = normalize_status(lock_entry.security_status)
        lock_av = normalize_status(lock_entry.av_status)
        if (lock_st == "safe" or lock_av == "safe") and status not in HARD_DENY_STATUSES and av not in HARD_DENY_STATUSES:
            if not status and not av:
                return Verdict("approved", "lock-pinned Hub-safe; live security metadata omitted", meta)

    if not status and not av:
        if reviewed:
            return Verdict("approved", "missing security metadata explicitly reviewed", meta)
        return Verdict("unresolved", "security metadata missing; fail closed", meta)

    if status in PENDING_STATUSES or av in PENDING_STATUSES:
        if reviewed:
            return Verdict("approved", f"pending scan {status or av!r} explicitly reviewed", meta)
        return Verdict("unresolved", f"security scan not resolved (status={status or 'missing'} av={av or 'missing'})", meta)

    if reviewed:
        return Verdict("approved", f"unexpected status {status!r} explicitly reviewed", meta)
    return Verdict("unresolved", f"unexpected security status {status!r}", meta)
