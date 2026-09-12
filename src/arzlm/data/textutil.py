"""UTF-8 / NFC sanitization shared by packing and the encode worker."""

from __future__ import annotations

import unicodedata
from typing import Any


def sanitize_document(text: str) -> str:
    """Make text safe for the Rust tokenizers NFC normalizer.

    Unpaired surrogates and invalid UTF-8 are replaced. NFC is applied in
    Python first so the HuggingFace normalizer sees an already-normalized
    string.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.encode("utf-8", "replace").decode("utf-8")
    return unicodedata.normalize("NFC", text)


def as_token_ids(ids: Any) -> list[int] | None:
    """Return uint16-safe token ids, or None if the payload is not a 1-D int list."""
    if ids is None:
        return None
    if hasattr(ids, "tolist") and not isinstance(ids, (list, tuple)):
        ids = ids.tolist()
    if isinstance(ids, bool) or not isinstance(ids, (int, list, tuple)):
        return None
    if isinstance(ids, int):
        ids = [ids]
    out: list[int] = []
    for tid in ids:
        if isinstance(tid, bool) or not isinstance(tid, int):
            try:
                if isinstance(tid, bool) or isinstance(tid, type):
                    return None
                tid = int(tid)
            except (TypeError, ValueError):
                return None
        if tid < 0 or tid > 65535:
            return None
        out.append(tid)
    return out


def encode_doc(tokenizer, text: str) -> list[int]:
    text = sanitize_document(text)
    if not text.strip():
        return []
    ids = tokenizer.encode(text, bos=False, eos=False)
    coerced = as_token_ids(ids)
    return [] if coerced is None else coerced
