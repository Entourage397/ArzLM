"""Exact content and URL hashing for cheap cross-source dedup."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from arzlm.data.textutil import sanitize_document


def normalize_for_hash(text: str) -> bytes:
    return sanitize_document(text).strip().encode("utf-8")


def content_sha256(text: str) -> str:
    return hashlib.sha256(normalize_for_hash(text)).hexdigest()


def normalize_url(url: str | None) -> str | None:
    if not url:
        return None
    raw = str(url).strip()
    if not raw:
        return None
    parts = urlsplit(raw)
    scheme = (parts.scheme or "http").lower()
    host = (parts.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((scheme, host, path, parts.query, ""))


def source_key(kind: str, value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    if kind == "url":
        norm = normalize_url(value)
        return f"url:{norm}" if norm else None
    return f"{kind}:{value}"


@dataclass
class DedupHit:
    kind: str  # content | url | id
    previous_domain: str
    previous_native_id: str | None


class DedupIndex:
    """Crash-safe exact-hash index. SQLite WAL, one row per unique document."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS content (
                h TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                native_id TEXT,
                split TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS keys (
                k TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                native_id TEXT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dups (
                h TEXT,
                pair TEXT NOT NULL,
                kind TEXT NOT NULL
            )
            """
        )
        self.conn.commit()
        self._pending = 0

    def close(self) -> None:
        self.flush()
        self.conn.close()

    def flush(self) -> None:
        self.conn.commit()
        self._pending = 0

    def lookup_content(self, digest: str) -> tuple[str, str | None] | None:
        row = self.conn.execute("SELECT domain, native_id FROM content WHERE h = ?", (digest,)).fetchone()
        if row is None:
            return None
        return str(row[0]), row[1]

    def lookup_key(self, key: str) -> tuple[str, str | None] | None:
        row = self.conn.execute("SELECT domain, native_id FROM keys WHERE k = ?", (key,)).fetchone()
        if row is None:
            return None
        return str(row[0]), row[1]

    def record_dup(self, digest: str, current_domain: str, previous_domain: str, kind: str) -> None:
        pair = f"{previous_domain}->{current_domain}"
        self.conn.execute("INSERT INTO dups (h, pair, kind) VALUES (?, ?, ?)", (digest, pair, kind))
        self._maybe_commit()

    def add(
        self,
        digest: str,
        domain: str,
        native_id: str | None,
        split: str,
        keys: list[str],
    ) -> None:
        self.conn.execute(
            "INSERT INTO content (h, domain, native_id, split) VALUES (?, ?, ?, ?)",
            (digest, domain, native_id, split),
        )
        for key in keys:
            self.conn.execute(
                "INSERT OR IGNORE INTO keys (k, domain, native_id) VALUES (?, ?, ?)",
                (key, domain, native_id),
            )
        self._maybe_commit()

    def _maybe_commit(self) -> None:
        self._pending += 1
        if self._pending >= 256:
            self.flush()

    def dup_counts(self) -> dict[str, int]:
        rows = self.conn.execute("SELECT pair, kind, COUNT(*) FROM dups GROUP BY pair, kind").fetchall()
        out: dict[str, int] = {}
        for pair, kind, n in rows:
            out[f"{kind}:{pair}"] = int(n)
        return out

    def counts(self) -> dict[str, int]:
        n_content = self.conn.execute("SELECT COUNT(*) FROM content").fetchone()[0]
        n_keys = self.conn.execute("SELECT COUNT(*) FROM keys").fetchone()[0]
        n_dups = self.conn.execute("SELECT COUNT(*) FROM dups").fetchone()[0]
        return {"content": int(n_content), "keys": int(n_keys), "dups": int(n_dups)}
