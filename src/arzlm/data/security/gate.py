"""Pre-fetch validation. Fail closed. Never downloads content to decide."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_UNSET = object()

from arzlm.data.catalog import FINEWEB_EDU, assert_dataset_allowed
from arzlm.data.security.hub import HubClient, HfHubClient
from arzlm.data.security.lockfile import SourceLock, load_source_lock
from arzlm.data.security.policy import (
    DenylistEntry,
    RemoteFileMeta,
    Verdict,
    classify_remote_file,
    is_fineweb_full_dump_path,
    load_denylist,
)
from arzlm.data.security.provenance import current_ledger
from arzlm.data.security.scan_local import maybe_scan_local_file
from arzlm.data.shards import sha256_file
from arzlm.paths import SECURITY_DIR, configure_hf_home

DEFAULT_DENYLIST = SECURITY_DIR / "denylist.json"


class SecurityError(RuntimeError):
    """Raised when a remote file must not be fetched or hashes do not match."""


def load_default_denylist(path: Path | None = None) -> list[DenylistEntry]:
    return load_denylist(Path(path) if path is not None else DEFAULT_DENYLIST)


def _client(client: HubClient | None) -> HubClient:
    return client if client is not None else HfHubClient()


def _resolve_lock(lock: SourceLock | None | object) -> SourceLock | None:
    if lock is _UNSET:
        return load_source_lock() if (SECURITY_DIR / "source-lock.json").is_file() else None
    return lock  # type: ignore[return-value]


def assert_allowed_prefix(dataset_id: str, prefix: str) -> None:
    assert_dataset_allowed(dataset_id)
    if dataset_id == FINEWEB_EDU.dataset_id:
        norm = prefix.replace("\\", "/")
        if not norm.startswith("sample/10BT"):
            raise SecurityError(
                "Refusing FineWeb-Edu prefix "
                f"{prefix!r}. ArzLM may only fetch sample/10BT/, never data/CC-MAIN-*."
            )


def evaluate_file(
    meta: RemoteFileMeta,
    *,
    denylist: list[DenylistEntry],
    lock: SourceLock | None = None,
) -> Verdict:
    lock_entry = lock.entry(meta.dataset_id, meta.path) if lock is not None else None
    if lock is not None:
        expected = lock.expected_revision(meta.dataset_id)
        if expected and meta.revision and expected != meta.revision:
            return Verdict(
                "mismatch",
                f"revision mismatch lock={expected} live={meta.revision}",
                meta,
            )
    return classify_remote_file(meta, denylist=denylist, lock_entry=lock_entry)


def assert_verdict_fetchable(verdict: Verdict) -> None:
    if verdict.kind == "approved":
        return
    path = verdict.meta.path
    if verdict.kind == "denied":
        raise SecurityError(f"DENIED {path}: {verdict.reason}")
    if verdict.kind == "mismatch":
        raise SecurityError(f"HASH/REVISION MISMATCH {path}: {verdict.reason}")
    raise SecurityError(f"UNRESOLVED {path}: {verdict.reason} (fail closed)")


def resolve_and_evaluate(
    dataset_id: str,
    revision: str,
    path: str,
    *,
    denylist: list[DenylistEntry] | None = None,
    lock: SourceLock | None = None,
    client: HubClient | None = None,
) -> tuple[RemoteFileMeta, Verdict]:
    assert_dataset_allowed(dataset_id)
    if is_fineweb_full_dump_path(dataset_id, path):
        meta = RemoteFileMeta(dataset_id=dataset_id, revision=revision, path=path)
        denylist = denylist if denylist is not None else load_default_denylist()
        verdict = classify_remote_file(meta, denylist=denylist, lock_entry=lock.entry(dataset_id, path) if lock else None)
        return meta, verdict
    hub = _client(client)
    resolved = hub.resolve_revision(dataset_id, revision)
    if lock is not None:
        expected = lock.expected_revision(dataset_id)
        if expected and resolved != expected:
            meta = RemoteFileMeta(dataset_id=dataset_id, revision=resolved, path=path)
            return meta, Verdict("mismatch", f"revision mismatch lock={expected} live={resolved}", meta)
    meta = hub.get_file(dataset_id, resolved, path)
    denylist = denylist if denylist is not None else load_default_denylist()
    return meta, evaluate_file(meta, denylist=denylist, lock=lock)


def assert_file_fetch_allowed(
    dataset_id: str,
    revision: str,
    path: str,
    *,
    denylist: list[DenylistEntry] | None = None,
    lock: SourceLock | None = None,
    client: HubClient | None = None,
) -> RemoteFileMeta:
    meta, verdict = resolve_and_evaluate(
        dataset_id,
        revision,
        path,
        denylist=denylist,
        lock=lock,
        client=client,
    )
    assert_verdict_fetchable(verdict)
    return meta


def list_fetchable_parquet(
    dataset_id: str,
    revision: str,
    prefix: str,
    *,
    denylist: list[DenylistEntry] | None = None,
    lock: SourceLock | None | object = _UNSET,
    client: HubClient | None = None,
    suffixes: tuple[str, ...] = (".parquet",),
) -> list[RemoteFileMeta]:
    """Enumerate prefix via Hub metadata and return only approved files.

    Denied files are excluded. Unresolved or mismatched files abort the list.
    """
    assert_allowed_prefix(dataset_id, prefix)
    hub = _client(client)
    resolved = hub.resolve_revision(dataset_id, revision)
    lock = _resolve_lock(lock)
    if lock is not None:
        expected = lock.expected_revision(dataset_id)
        if expected and resolved != expected:
            raise SecurityError(f"{dataset_id}: revision mismatch lock={expected} live={resolved}")
    denylist = denylist if denylist is not None else load_default_denylist()
    if lock is not None:
        locked_paths = [
            path
            for src in lock.all_locked_sources()
            if src.dataset_id == dataset_id
            for path in src.files
            if path.replace("\\", "/").startswith(prefix.replace("\\", "/"))
            and path.endswith(suffixes)
        ]
        if not locked_paths:
            return []
        approved_locked: list[RemoteFileMeta] = []
        for path in sorted(set(locked_paths)):
            meta = assert_file_fetch_allowed(
                dataset_id,
                resolved,
                path,
                denylist=denylist,
                lock=lock,
                client=client,
            )
            approved_locked.append(meta)
        return approved_locked
    metas = hub.list_files(dataset_id, resolved, prefix, suffixes=suffixes)
    if not metas:
        raise SecurityError(f"no files under {dataset_id} {prefix!r} @ {resolved}")
    approved: list[RemoteFileMeta] = []
    unresolved: list[Verdict] = []
    for meta in metas:
        verdict = evaluate_file(meta, denylist=denylist, lock=lock)
        if verdict.kind == "approved":
            approved.append(meta)
        elif verdict.kind == "denied":
            continue
        else:
            unresolved.append(verdict)
    if unresolved:
        sample = "; ".join(f"{v.meta.path}: {v.reason}" for v in unresolved[:5])
        raise SecurityError(
            f"{len(unresolved)} unresolved source file(s) under {dataset_id} {prefix!r}. "
            f"Fail closed. Examples: {sample}"
        )
    if not approved:
        raise SecurityError(f"no approved files remain under {dataset_id} {prefix!r} @ {resolved}")
    return approved


def gated_hub_download(
    dataset_id: str,
    filename: str,
    revision: str,
    *,
    denylist: list[DenylistEntry] | None = None,
    lock: SourceLock | None | object = _UNSET,
    client: HubClient | None = None,
    config: str | None = None,
    local_av: bool | None = None,
) -> Path:
    """Download one Hub file only after a passing security evaluation."""
    configure_hf_home()
    lock = _resolve_lock(lock)
    denylist = denylist if denylist is not None else load_default_denylist()
    meta = assert_file_fetch_allowed(
        dataset_id,
        revision,
        filename,
        denylist=denylist,
        lock=lock,
        client=client,
    )
    from huggingface_hub import hf_hub_download

    local = Path(
        hf_hub_download(
            dataset_id,
            filename=filename,
            repo_type="dataset",
            revision=meta.revision,
        )
    )
    digest = sha256_file(local)
    if meta.sha256 and digest.lower() != meta.sha256.lower():
        raise SecurityError(
            f"downloaded sha256 mismatch for {filename}: expected {meta.sha256}, got {digest}"
        )
    scan = maybe_scan_local_file(local, enabled=local_av)
    if scan.infected:
        raise SecurityError(f"local AV flagged {filename}: {scan.detail}")
    if (not scan.skipped) and "scanner error" in scan.detail:
        raise SecurityError(f"local AV error for {filename}: {scan.detail}")
    ledger = current_ledger()
    if ledger is not None:
        ledger.record(meta, config=config, local_sha256=digest, local_bytes=local.stat().st_size)
    return local
