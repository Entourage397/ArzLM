"""Pre-corpus source security audit. Metadata only; no file content downloads."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arzlm.data.catalog import (
    CORPUS_NAME,
    CORPUS_NAME_6B,
    FINEWEB_EDU,
    FINEMATH_4PLUS,
    PES2O,
    PES2O_S2ORC_TRAIN_SHARDS,
    STACK_EDU,
    CODE_LANGUAGE_WEIGHTS,
)
from arzlm.data.security.gate import evaluate_file, load_default_denylist
from arzlm.data.security.hub import HubClient, HfHubClient
from arzlm.data.security.lockfile import (
    LOCK_SCHEMA_VERSION,
    LockedSource,
    SourceLock,
    default_lock_policy,
    load_source_lock,
    lock_entry_from_meta,
    write_source_lock,
)
from arzlm.data.security.policy import (
    DenylistEntry,
    RemoteFileMeta,
    Verdict,
    classify_remote_file,
    load_denylist,
)
from arzlm.paths import SECURITY_DIR
from arzlm.training.report import utc_now

KNOWN_UNSAFE_FINEWEB_PATH = "data/CC-MAIN-2024-38/000_00041.parquet"


class SecurityAuditError(RuntimeError):
    def __init__(self, message: str, report: "SecurityAuditReport") -> None:
        super().__init__(message)
        self.report = report


@dataclass
class FileOutcome:
    domain: str
    verdict: Verdict


@dataclass
class DomainAudit:
    domain: str
    dataset_id: str
    config: str | None
    revision: str
    passed: bool
    approved: list[RemoteFileMeta] = field(default_factory=list)
    denied: list[Verdict] = field(default_factory=list)
    unresolved: list[Verdict] = field(default_factory=list)
    mismatched: list[Verdict] = field(default_factory=list)


@dataclass
class SecurityAuditReport:
    passed: bool
    checked_at: str
    domains: dict[str, DomainAudit]
    denylist_canaries: list[Verdict] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def all_verdicts(self, kind: str) -> list[Verdict]:
        out: list[Verdict] = []
        for domain in self.domains.values():
            bucket = {
                "approved": [Verdict("approved", "ok", m) for m in domain.approved],
                "denied": domain.denied,
                "unresolved": domain.unresolved,
                "mismatch": domain.mismatched,
            }[kind]
            out.extend(bucket)
        return out


def _source_specs() -> list[tuple[str, Any, str, tuple[str, ...]]]:
    """domain, spec, prefix, suffixes."""
    return [
        ("general", FINEWEB_EDU, "sample/10BT/", (".parquet",)),
        ("math", FINEMATH_4PLUS, "finemath-4plus/", (".parquet",)),
        ("science", PES2O, "data/v2/", (".json.gz",)),
    ]


def _pes2o_s2orc_paths() -> set[str]:
    return {f"data/v2/train-{i:05d}-of-00020.json.gz" for i in PES2O_S2ORC_TRAIN_SHARDS}


def _pes2o_s2ag_paths() -> set[str]:
    return {f"data/v2/train-{i:05d}-of-00020.json.gz" for i in range(0, 10)}


def _pes2o_allowed_paths() -> set[str]:
    return _pes2o_s2orc_paths() | _pes2o_s2ag_paths()


def _stack_prefixes() -> list[str]:
    return [f"{lang}/" for lang in CODE_LANGUAGE_WEIGHTS]


def _filter_candidates(domain: str, files: list[RemoteFileMeta]) -> list[RemoteFileMeta]:
    if domain != "science":
        return files
    allow = _pes2o_allowed_paths()
    return [m for m in files if m.path in allow]


def audit_domain(
    domain: str,
    dataset_id: str,
    config: str | None,
    revision: str,
    prefix: str,
    *,
    suffixes: tuple[str, ...],
    denylist: list[DenylistEntry],
    lock: SourceLock | None,
    client: HubClient,
    filter_candidates: bool = True,
) -> DomainAudit:
    resolved = client.resolve_revision(dataset_id, revision)
    audit = DomainAudit(
        domain=domain,
        dataset_id=dataset_id,
        config=config,
        revision=resolved,
        passed=False,
    )
    if lock is not None:
        expected = lock.expected_revision(dataset_id)
        if expected and resolved != expected:
            dummy = RemoteFileMeta(dataset_id=dataset_id, revision=resolved, path=prefix)
            audit.mismatched.append(
                Verdict("mismatch", f"revision mismatch lock={expected} live={resolved}", dummy)
            )
            return audit
    listed = client.list_files(dataset_id, resolved, prefix, suffixes=suffixes)
    files = _filter_candidates(domain, listed) if filter_candidates and domain == "science" else listed
    if not files:
        dummy = RemoteFileMeta(dataset_id=dataset_id, revision=resolved, path=prefix)
        audit.unresolved.append(Verdict("unresolved", "no candidate files listed from Hub metadata", dummy))
        return audit
    lock_src = None
    if lock is not None:
        for src in lock.all_locked_sources():
            if src.dataset_id == dataset_id and (src.domain == domain or not src.domain):
                lock_src = src
                break
    live_by_path = {m.path: m for m in files}
    if lock_src is not None:
        # Audit the files we will fetch. Unlocked live siblings (unscanned,
        # extra shards) are skipped rather than failing the lock.
        for locked_path in lock_src.files:
            meta = live_by_path.get(locked_path)
            if meta is None:
                dummy = RemoteFileMeta(dataset_id=dataset_id, revision=resolved, path=locked_path)
                audit.unresolved.append(
                    Verdict("unresolved", "locked file missing from live Hub listing", dummy)
                )
                continue
            verdict = evaluate_file(meta, denylist=denylist, lock=lock)
            if verdict.kind == "approved":
                audit.approved.append(meta)
            elif verdict.kind == "denied":
                audit.denied.append(verdict)
            elif verdict.kind == "mismatch":
                audit.mismatched.append(verdict)
            else:
                audit.unresolved.append(verdict)
        audit.passed = bool(audit.approved) and not audit.unresolved and not audit.mismatched
        return audit
    for meta in files:
        verdict = evaluate_file(meta, denylist=denylist, lock=lock)
        if verdict.kind == "approved":
            audit.approved.append(meta)
        elif verdict.kind == "denied":
            audit.denied.append(verdict)
        elif verdict.kind == "mismatch":
            audit.mismatched.append(verdict)
        else:
            audit.unresolved.append(verdict)
    audit.passed = bool(audit.approved) and not audit.unresolved and not audit.mismatched
    return audit


def verify_denylist_canaries(
    denylist: list[DenylistEntry],
    *,
    client: HubClient,
    lock: SourceLock | None,
) -> list[Verdict]:
    out: list[Verdict] = []
    for entry in denylist:
        try:
            meta = client.get_file(entry.dataset_id, entry.revision or "main", entry.path)
        except Exception as exc:  # metadata lookup failed — still deny the known path
            meta = RemoteFileMeta(
                dataset_id=entry.dataset_id,
                revision=entry.revision or "",
                path=entry.path,
                sha256=entry.sha256,
            )
            out.append(evaluate_file(meta, denylist=denylist, lock=lock))
            out[-1] = Verdict(
                out[-1].kind,
                f"{out[-1].reason} (hub lookup: {type(exc).__name__})",
                meta,
                denylist=True,
            )
            continue
        out.append(evaluate_file(meta, denylist=denylist, lock=lock))
    return out


def run_source_security_audit(
    *,
    client: HubClient | None = None,
    lock: SourceLock | None = None,
    denylist: list[DenylistEntry] | None = None,
    lock_path: Path | None = None,
    denylist_path: Path | None = None,
) -> SecurityAuditReport:
    hub = client if client is not None else HfHubClient()
    if denylist is None:
        denylist = load_denylist(denylist_path) if denylist_path else load_default_denylist()
    if lock is None:
        path = lock_path if lock_path is not None else (SECURITY_DIR / "source-lock.json")
        if path.is_file():
            lock = load_source_lock(path)
    domains: dict[str, DomainAudit] = {}
    for domain, spec, prefix, suffixes in _source_specs():
        if domain == "science" and lock is not None:
            sci = lock.sources.get("science")
            if sci is not None and sci.dataset_id != spec.dataset_id:
                fb_prefix = "" if (sci.prefix or "").startswith("(") else (sci.prefix or "")
                domains[domain] = audit_domain(
                    domain,
                    sci.dataset_id,
                    sci.config,
                    sci.revision,
                    fb_prefix,
                    suffixes=(".parquet", ".arrow", ".json", ".jsonl", ".csv", ".json.gz"),
                    denylist=denylist,
                    lock=lock,
                    client=hub,
                    filter_candidates=False,
                )
                continue
        domains[domain] = audit_domain(
            domain,
            spec.dataset_id,
            spec.config,
            spec.revision,
            prefix,
            suffixes=suffixes,
            denylist=denylist,
            lock=lock,
            client=hub,
        )
    stack_files: list[RemoteFileMeta] = []
    resolved_stack = hub.resolve_revision(STACK_EDU.dataset_id, STACK_EDU.revision)
    if lock is not None:
        expected = lock.expected_revision(STACK_EDU.dataset_id)
        stack_mismatch = expected and resolved_stack != expected
    else:
        stack_mismatch = False
    stack = DomainAudit(
        domain="code",
        dataset_id=STACK_EDU.dataset_id,
        config=None,
        revision=resolved_stack,
        passed=False,
    )
    if stack_mismatch:
        dummy = RemoteFileMeta(dataset_id=STACK_EDU.dataset_id, revision=resolved_stack, path="Python/")
        stack.mismatched.append(
            Verdict(
                "mismatch",
                f"revision mismatch lock={lock.expected_revision(STACK_EDU.dataset_id)} live={resolved_stack}",
                dummy,
            )
        )
    else:
        for prefix in _stack_prefixes():
            stack_files.extend(
                hub.list_files(STACK_EDU.dataset_id, resolved_stack, prefix, suffixes=(".parquet",))
            )
        if not stack_files:
            dummy = RemoteFileMeta(dataset_id=STACK_EDU.dataset_id, revision=resolved_stack, path="Python/")
            stack.unresolved.append(Verdict("unresolved", "no candidate files listed from Hub metadata", dummy))
        lock_src = lock.sources.get("code") if lock is not None else None
        live_by_path = {m.path: m for m in stack_files}
        if lock_src is not None:
            for locked_path in lock_src.files:
                meta = live_by_path.get(locked_path)
                if meta is None:
                    dummy = RemoteFileMeta(
                        dataset_id=STACK_EDU.dataset_id,
                        revision=resolved_stack,
                        path=locked_path,
                    )
                    stack.unresolved.append(
                        Verdict("unresolved", "locked file missing from live Hub listing", dummy)
                    )
                    continue
                verdict = evaluate_file(meta, denylist=denylist, lock=lock)
                if verdict.kind == "approved":
                    stack.approved.append(meta)
                elif verdict.kind == "denied":
                    stack.denied.append(verdict)
                elif verdict.kind == "mismatch":
                    stack.mismatched.append(verdict)
                else:
                    stack.unresolved.append(verdict)
        else:
            for meta in stack_files:
                verdict = evaluate_file(meta, denylist=denylist, lock=lock)
                if verdict.kind == "approved":
                    stack.approved.append(meta)
                elif verdict.kind == "denied":
                    stack.denied.append(verdict)
                elif verdict.kind == "mismatch":
                    stack.mismatched.append(verdict)
                else:
                    stack.unresolved.append(verdict)
        stack.passed = bool(stack.approved) and not stack.unresolved and not stack.mismatched
    domains["code"] = stack

    if lock is not None:
        for src in lock.supplements:
            extra = audit_domain(
                src.domain,
                src.dataset_id,
                src.config,
                src.revision,
                "" if (src.prefix or "").startswith("(") else (src.prefix or ""),
                suffixes=(".parquet", ".arrow", ".json", ".jsonl", ".csv", ".json.gz"),
                denylist=denylist,
                lock=lock,
                client=hub,
                filter_candidates=False,
            )
            parent = domains.get(src.domain)
            if parent is None:
                domains[src.domain] = extra
                continue
            parent.approved.extend(extra.approved)
            parent.denied.extend(extra.denied)
            parent.unresolved.extend(extra.unresolved)
            parent.mismatched.extend(extra.mismatched)
            parent.passed = parent.passed and extra.passed and not extra.unresolved and not extra.mismatched

    canaries = verify_denylist_canaries(denylist, client=hub, lock=lock)
    passed = all(d.passed for d in domains.values()) and all(v.kind == "denied" for v in canaries)
    return SecurityAuditReport(
        passed=passed,
        checked_at=utc_now(),
        domains=domains,
        denylist_canaries=canaries,
    )


def require_remote_security_audit(
    *,
    client: HubClient | None = None,
    lock: SourceLock | None = None,
    denylist: list[DenylistEntry] | None = None,
) -> SecurityAuditReport:
    report = run_source_security_audit(client=client, lock=lock, denylist=denylist)
    if not report.passed:
        raise SecurityAuditError(
            "Source security audit failed (fail closed). "
            "Fix denylist/unresolved files before corpus preparation.",
            report,
        )
    return report


def _label(domain: DomainAudit) -> str:
    name = {
        "general": "FineWeb-Edu",
        "math": "Math",
        "code": "Code",
        "science": "Science",
    }[domain.domain]
    if domain.passed:
        extra = f", with {len(domain.denied)} denied/unsafe files excluded" if domain.denied else ""
        return f"{name}: PASS{extra}"
    bits = []
    if domain.denied:
        bits.append(f"{len(domain.denied)} denied")
    if domain.unresolved:
        bits.append(f"{len(domain.unresolved)} unresolved")
    if domain.mismatched:
        bits.append(f"{len(domain.mismatched)} mismatched")
    return f"{name}: FAIL ({', '.join(bits) or 'unknown failure'})"


def format_audit_report(report: SecurityAuditReport) -> str:
    lines = ["SECURITY AUDIT"]
    for key in ("general", "math", "code", "science"):
        if key in report.domains:
            lines.append(_label(report.domains[key]))
    unsafe: list[str] = []
    unresolved: list[str] = []
    approved: list[str] = []
    for domain in report.domains.values():
        for v in domain.denied:
            unsafe.append(f"{v.meta.dataset_id}:{v.meta.path} ({v.reason})")
        for v in domain.unresolved + domain.mismatched:
            unresolved.append(f"{v.meta.dataset_id}:{v.meta.path} ({v.reason})")
        for meta in domain.approved:
            approved.append(f"{meta.dataset_id}:{meta.path}")
    for v in report.denylist_canaries:
        unsafe.append(f"denylist-canary {v.meta.dataset_id}:{v.meta.path} = {v.kind.upper()} ({v.reason})")
    lines.append("Unsafe files: " + (json.dumps(unsafe, indent=2) if unsafe else "[]"))
    lines.append("Unknown/unresolved files: " + (json.dumps(unresolved, indent=2) if unresolved else "[]"))
    lines.append("Approved physical files: " + (json.dumps(approved, indent=2) if approved else "[]"))
    lines.append(f"Result: {'PASS' if report.passed else 'FAIL'}")
    return "\n".join(lines)


SCIENCE_FALLBACKS: list[tuple[str, str, tuple[str, ...]]] = [
    ("ccdv/arxiv-summarization", "document/", (".parquet",)),
    ("CShorten/ML-ArXiv-Papers", "", (".csv", ".parquet")),
    ("armanc/scientific_papers", "", (".parquet", ".arrow", ".json", ".jsonl")),
]

GENERAL_SUPPLEMENTS: list[tuple[str, str, tuple[str, ...], int]] = [
    # 6GB packed only ~192M of the 550M general quota. Stay on Hub-safe shards.
    ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2/", (".parquet",), 24_000_000_000),
]

# 6B unique general needs ~3.465B packed tokens. FineWeb-Edu sample-10BT
# historically had one Hub-safe shard (~0.54GB). Expand within the already
# audited SmolLM-corpus family: remaining Cosmopedia-v2 plus FineWeb-Edu-dedup.
# Never fetch data/CC-MAIN-* and never weaken the denylist.
GENERAL_SUPPLEMENTS_6B: list[tuple[str, str, tuple[str, ...], int]] = [
    ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2/", (".parquet",), 130_000_000_000),
    ("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup/", (".parquet",), 50_000_000_000),
]

# FineWeb sample-10BT currently has one Hub-safe shard (~0.54GB). Below this
# compressed-byte floor we add Hub-safe educational supplements rather than
# fetching unscanned 2GB sample-10BT files.
FINEWEB_SAFE_BYTE_FLOOR = 4_000_000_000
SCIENCE_FALLBACK_MAX_BYTES = 2_500_000_000


def is_eval_split_path(path: str) -> bool:
    rel = path.replace("\\", "/").lower()
    name = rel.rsplit("/", 1)[-1]
    if name.startswith(("test-", "test_", "validation-", "validation_", "valid-", "valid_")):
        return True
    parts = rel.split("/")
    return any(part in {"test", "validation", "valid", "dev"} for part in parts[:-1])


def _approved_lock_files(
    files: list[RemoteFileMeta],
    denylist: list[DenylistEntry],
    *,
    max_bytes: int | None = None,
    skip_eval_splits: bool = True,
) -> dict[str, Any]:
    locked = {}
    total = 0
    for meta in sorted(files, key=lambda m: m.path):
        if skip_eval_splits and is_eval_split_path(meta.path):
            continue
        if classify_remote_file(meta, denylist=denylist).kind != "approved":
            continue
        size = int(meta.size or 0)
        if max_bytes is not None and locked and total + size > max_bytes:
            break
        locked[meta.path] = lock_entry_from_meta(meta)
        total += size
    return locked


def _try_science_fallbacks(hub: HubClient, denylist: list[DenylistEntry]) -> LockedSource | None:
    for dataset_id, prefix, suffixes in SCIENCE_FALLBACKS:
        try:
            resolved = hub.resolve_revision(dataset_id, "main")
            files = hub.list_files(dataset_id, resolved, prefix, suffixes=suffixes)
        except Exception:
            continue
        locked = _approved_lock_files(files, denylist, max_bytes=SCIENCE_FALLBACK_MAX_BYTES)
        if locked:
            return LockedSource(
                domain="science",
                dataset_id=dataset_id,
                config=None,
                revision=resolved,
                prefix=prefix or "(root)",
                license="see Hub dataset card; paper copyrights remain with authors",
                files=locked,
            )
    return None


def _general_supplements(
    hub: HubClient,
    denylist: list[DenylistEntry],
    fineweb_files: dict[str, Any],
    *,
    scale: str = "1b",
) -> list[LockedSource]:
    table = GENERAL_SUPPLEMENTS_6B if scale == "6b" else GENERAL_SUPPLEMENTS
    fineweb_bytes = sum(int(e.size or 0) for e in fineweb_files.values())
    # 1B: only supplement when FineWeb Hub-safe bytes are below the floor.
    # 6B: always take Hub-safe educational supplements; FineWeb alone cannot
    # supply 3.465B unique packed tokens from one safe shard.
    if scale != "6b" and fineweb_bytes >= FINEWEB_SAFE_BYTE_FLOOR:
        return []
    out: list[LockedSource] = []
    for dataset_id, prefix, suffixes, max_bytes in table:
        try:
            resolved = hub.resolve_revision(dataset_id, "main")
            files = hub.list_files(dataset_id, resolved, prefix, suffixes=suffixes)
        except Exception as exc:
            print(
                f"general supplement list failed {dataset_id} {prefix!r}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            continue
        print(
            f"general supplement {dataset_id} {prefix!r}: {len(files)} listed files",
            flush=True,
        )
        locked = _approved_lock_files(files, denylist, max_bytes=max_bytes)
        print(
            f"general supplement {dataset_id} {prefix!r}: {len(locked)} approved files",
            flush=True,
        )
        if not locked:
            continue
        out.append(
            LockedSource(
                domain="general",
                dataset_id=dataset_id,
                config=None,
                revision=resolved,
                prefix=prefix or "(root)",
                license="see Hub dataset card",
                files=locked,
            )
        )
    return out


def extend_general_supplements(lock: SourceLock, *, client: HubClient | None = None) -> SourceLock:
    """Replace general supplements using the current Hub-safe budget. Other domains stay pinned."""
    hub = client if client is not None else HfHubClient()
    denylist = load_default_denylist()
    general = lock.sources["general"]
    kept = [src for src in lock.supplements if src.domain != "general"]
    scale = "6b" if "6B" in str(lock.corpus) else "1b"
    lock.supplements = kept + _general_supplements(hub, denylist, general.files, scale=scale)
    lock.created_at = utc_now()
    return lock


def build_source_lock(
    *,
    client: HubClient | None = None,
    denylist: list[DenylistEntry] | None = None,
    scale: str = "1b",
) -> SourceLock:
    """Pin current catalog revisions and enumerate candidate files via Hub metadata."""
    if scale not in {"1b", "6b"}:
        raise ValueError(f"unknown source-lock scale {scale!r}")
    hub = client if client is not None else HfHubClient()
    denylist = denylist if denylist is not None else load_default_denylist()
    sources: dict[str, LockedSource] = {}
    for domain, spec, prefix, suffixes in _source_specs():
        resolved = hub.resolve_revision(spec.dataset_id, spec.revision)
        files = _filter_candidates(domain, hub.list_files(spec.dataset_id, resolved, prefix, suffixes=suffixes))
        if domain == "science":
            s2orc_files = [m for m in files if m.path in _pes2o_s2orc_paths()]
            locked = _approved_lock_files(s2orc_files, denylist)
            if not locked:
                s2ag_files = [m for m in files if m.path in _pes2o_s2ag_paths()]
                locked = _approved_lock_files(s2ag_files, denylist)
            if not locked:
                fb = _try_science_fallbacks(hub, denylist)
                if fb is not None:
                    sources[domain] = fb
                    continue
            sources[domain] = LockedSource(
                domain=domain,
                dataset_id=spec.dataset_id,
                config=spec.config,
                revision=resolved,
                prefix=prefix,
                license=spec.license,
                files=locked,
            )
            continue
        locked = _approved_lock_files(files, denylist)
        sources[domain] = LockedSource(
            domain=domain,
            dataset_id=spec.dataset_id,
            config=spec.config,
            revision=resolved,
            prefix=prefix,
            license=spec.license,
            files=locked,
        )
    resolved = hub.resolve_revision(STACK_EDU.dataset_id, STACK_EDU.revision)
    code_files: dict[str, Any] = {}
    for prefix in _stack_prefixes():
        for meta in hub.list_files(STACK_EDU.dataset_id, resolved, prefix, suffixes=(".parquet",)):
            verdict = classify_remote_file(meta, denylist=denylist)
            if verdict.kind == "approved":
                code_files[meta.path] = lock_entry_from_meta(meta)
    sources["code"] = LockedSource(
        domain="code",
        dataset_id=STACK_EDU.dataset_id,
        config=None,
        revision=resolved,
        prefix="(per-language parquet)",
        license=STACK_EDU.license,
        files=code_files,
    )
    supplements: list[LockedSource] = []
    general = sources.get("general")
    if general is not None:
        supplements.extend(_general_supplements(hub, denylist, general.files, scale=scale))
    return SourceLock(
        schema_version=LOCK_SCHEMA_VERSION,
        corpus=CORPUS_NAME_6B if scale == "6b" else CORPUS_NAME,
        created_at=utc_now(),
        policy=default_lock_policy(),
        sources=sources,
        supplements=supplements,
    )


def refresh_source_lock(
    path: Path | None = None,
    *,
    client: HubClient | None = None,
    scale: str = "1b",
) -> Path:
    lock = build_source_lock(client=client, scale=scale)
    return write_source_lock(lock, path)


def report_to_dict(report: SecurityAuditReport) -> dict[str, Any]:
    domains = {}
    for key, d in report.domains.items():
        domains[key] = {
            "dataset_id": d.dataset_id,
            "config": d.config,
            "revision": d.revision,
            "passed": d.passed,
            "approved": [m.path for m in d.approved],
            "denied": [{"path": v.meta.path, "reason": v.reason} for v in d.denied],
            "unresolved": [{"path": v.meta.path, "reason": v.reason} for v in d.unresolved],
            "mismatched": [{"path": v.meta.path, "reason": v.reason} for v in d.mismatched],
        }
    return {
        "passed": report.passed,
        "checked_at": report.checked_at,
        "domains": domains,
        "denylist_canaries": [
            {"path": v.meta.path, "kind": v.kind, "reason": v.reason} for v in report.denylist_canaries
        ],
    }
