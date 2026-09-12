"""Fail-closed source-file security for ArzLM corpus ingestion."""

from arzlm.data.security.audit import (
    SecurityAuditError,
    SecurityAuditReport,
    format_audit_report,
    require_remote_security_audit,
    run_source_security_audit,
)
from arzlm.data.security.gate import (
    SecurityError,
    assert_file_fetch_allowed,
    gated_hub_download,
    list_fetchable_parquet,
)
from arzlm.data.security.policy import Verdict, classify_remote_file

__all__ = [
    "SecurityAuditError",
    "SecurityAuditReport",
    "SecurityError",
    "Verdict",
    "assert_file_fetch_allowed",
    "classify_remote_file",
    "format_audit_report",
    "gated_hub_download",
    "list_fetchable_parquet",
    "require_remote_security_audit",
    "run_source_security_audit",
]
