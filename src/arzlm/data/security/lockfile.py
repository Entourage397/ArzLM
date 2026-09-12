"""Pinned upstream source lockfile (training-data equivalent of a package lock)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arzlm.data.security.policy import LockFileEntry, RemoteFileMeta
from arzlm.paths import SECURITY_DIR
from arzlm.training.report import utc_now

LOCK_SCHEMA_VERSION = 1
DEFAULT_LOCK_PATH = SECURITY_DIR / "source-lock.json"


def default_lock_path() -> Path:
    import os

    override = os.environ.get("ARZLM_SOURCE_LOCK")
    if override:
        return Path(override)
    return DEFAULT_LOCK_PATH


@dataclass
class LockedSource:
    domain: str
    dataset_id: str
    config: str | None
    revision: str
    prefix: str
    license: str | None = None
    files: dict[str, LockFileEntry] = field(default_factory=dict)


@dataclass
class SourceLock:
    schema_version: int
    corpus: str
    created_at: str
    policy: dict[str, Any]
    sources: dict[str, LockedSource]
    supplements: list[LockedSource] = field(default_factory=list)

    def source(self, domain: str) -> LockedSource:
        if domain not in self.sources:
            raise KeyError(f"source-lock has no domain {domain!r}")
        return self.sources[domain]

    def all_locked_sources(self) -> list[LockedSource]:
        return list(self.sources.values()) + list(self.supplements)

    def sources_for_domain(self, domain: str) -> list[LockedSource]:
        out = []
        primary = self.sources.get(domain)
        if primary is not None:
            out.append(primary)
        out.extend(src for src in self.supplements if src.domain == domain)
        return out

    def entry(self, dataset_id: str, path: str) -> LockFileEntry | None:
        path = path.replace("\\", "/")
        for src in self.all_locked_sources():
            if src.dataset_id != dataset_id:
                continue
            hit = src.files.get(path)
            if hit is not None:
                return hit
        return None

    def expected_revision(self, dataset_id: str) -> str | None:
        for src in self.all_locked_sources():
            if src.dataset_id == dataset_id:
                return src.revision
        return None


def _entry_from_mapping(path: str, raw: dict[str, Any]) -> LockFileEntry:
    return LockFileEntry(
        path=path.replace("\\", "/"),
        sha256=str(raw["sha256"]) if raw.get("sha256") else None,
        size=int(raw["size"]) if raw.get("size") is not None else None,
        xet_hash=str(raw["xet_hash"]) if raw.get("xet_hash") else None,
        security_status=str(raw["security_status"]) if raw.get("security_status") else None,
        av_status=str(raw["av_status"]) if raw.get("av_status") else None,
        reviewed_allow_unscanned=bool(raw.get("reviewed_allow_unscanned", False)),
    )


def _locked_source_from_mapping(raw: dict[str, Any], *, domain: str | None = None) -> LockedSource:
    files = {}
    for rec in raw.get("files") or []:
        p = str(rec["path"]).replace("\\", "/")
        files[p] = _entry_from_mapping(p, rec)
    return LockedSource(
        domain=str(raw.get("domain") or domain or ""),
        dataset_id=str(raw["dataset_id"]),
        config=raw.get("config"),
        revision=str(raw["revision"]),
        prefix=str(raw.get("prefix") or ""),
        license=raw.get("license"),
        files=files,
    )


def load_source_lock(path: Path | None = None) -> SourceLock:
    lock_path = Path(path) if path is not None else default_lock_path()
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    sources: dict[str, LockedSource] = {}
    for domain, raw in (payload.get("sources") or {}).items():
        sources[domain] = _locked_source_from_mapping(raw, domain=domain)
    supplements = [_locked_source_from_mapping(raw) for raw in (payload.get("supplements") or [])]
    return SourceLock(
        schema_version=int(payload.get("schema_version") or 0),
        corpus=str(payload.get("corpus") or ""),
        created_at=str(payload.get("created_at") or ""),
        policy=dict(payload.get("policy") or {}),
        sources=sources,
        supplements=supplements,
    )


def _dump_locked_source(src: LockedSource) -> dict[str, Any]:
    files = []
    for path in sorted(src.files):
        e = src.files[path]
        files.append(
            {
                "path": e.path,
                "size": e.size,
                "sha256": e.sha256,
                "xet_hash": e.xet_hash,
                "security_status": e.security_status,
                "av_status": e.av_status,
                "reviewed_allow_unscanned": e.reviewed_allow_unscanned,
            }
        )
    return {
        "domain": src.domain,
        "dataset_id": src.dataset_id,
        "config": src.config,
        "revision": src.revision,
        "prefix": src.prefix,
        "license": src.license,
        "files": files,
    }


def dump_source_lock(lock: SourceLock) -> dict[str, Any]:
    sources = {domain: _dump_locked_source(src) for domain, src in lock.sources.items()}
    return {
        "schema_version": lock.schema_version,
        "corpus": lock.corpus,
        "created_at": lock.created_at,
        "policy": lock.policy,
        "sources": sources,
        "supplements": [_dump_locked_source(src) for src in lock.supplements],
    }


def write_source_lock(lock: SourceLock, path: Path | None = None) -> Path:
    lock_path = Path(path) if path is not None else default_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dump_source_lock(lock), indent=2) + "\n"
    lock_path.write_text(payload, encoding="utf-8")
    return lock_path


def lock_entry_from_meta(meta: RemoteFileMeta, *, reviewed_allow_unscanned: bool = False) -> LockFileEntry:
    return LockFileEntry(
        path=meta.path,
        sha256=meta.sha256,
        size=meta.size,
        xet_hash=meta.xet_hash,
        security_status=meta.security_status,
        av_status=meta.av_status,
        reviewed_allow_unscanned=reviewed_allow_unscanned,
    )


def default_lock_policy() -> dict[str, Any]:
    return {
        "fail_closed": True,
        "never_build_from_unpinned_main": True,
        "allow_missing_security": False,
        "fineweb_edu_prefix": "sample/10BT/",
        "forbid_fineweb_full_dump": True,
        "trust_remote_code": False,
        "documents_are_inert_text": True,
        "checked_at_note": utc_now(),
        "local_av": {
            "enabled": False,
            "skip_if_unavailable": True,
            "engine": "clamav",
        },
    }
