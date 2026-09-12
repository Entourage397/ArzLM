"""Inert-text policy: documents are never executed, fetched, or compiled."""

from __future__ import annotations

from typing import Any

from arzlm.data.security.gate import SecurityError

FORBIDDEN_LOAD_FLAGS = ("trust_remote_code",)


def assert_inert_load_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Force trust_remote_code=False. Refuse an explicit True."""
    out = dict(kwargs)
    if out.get("trust_remote_code") is True:
        raise SecurityError("trust_remote_code=True is forbidden for ArzLM corpus sources")
    out["trust_remote_code"] = False
    return out


def load_dataset_inert(*args: Any, **kwargs: Any):
    from datasets import load_dataset

    return load_dataset(*args, **assert_inert_load_kwargs(kwargs))
