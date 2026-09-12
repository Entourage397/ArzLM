"""Software Heritage blob fetch via the official S3 content bucket.

Stack-Edu stores blob_id only. The Stack v2 / Stack-Edu cards document:

    s3.get_object(Bucket='softwareheritage', Key=f'content/{blob_id}')

Bulk archive dumps require a Software Heritage agreement. Individual GetObject
of listed blob IDs is the documented per-file method. No unofficial text mirror.

Returned bytes are decoded as inert training text. Never exec, compile, or pip-install them.
"""

from __future__ import annotations

import gzip
import io
import time
import urllib.error
import urllib.request

from arzlm.data.network import ByteCounter

SWH_BUCKET_HOST = "softwareheritage.s3.amazonaws.com"
DEFAULT_TIMEOUT_S = 30.0
MAX_RETRIES = 3


def swh_content_url(blob_id: str) -> str:
    blob_id = blob_id.strip()
    if not blob_id or "/" in blob_id or ".." in blob_id:
        raise ValueError(f"invalid SWH blob_id: {blob_id!r}")
    return f"https://{SWH_BUCKET_HOST}/content/{blob_id}"


def fetch_swh_text(
    blob_id: str,
    *,
    counter: ByteCounter | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    encoding_hint: str | None = None,
) -> str | None:
    """Return decoded source text, or None if the object is missing/unreadable."""
    url = swh_content_url(blob_id)
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, method="GET", headers={"User-Agent": "arzlm-stem/1.0"})
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read()
            if counter is not None:
                counter.add(len(raw), requests=1)
            if not raw:
                return None
            try:
                payload = gzip.decompress(raw)
            except OSError:
                payload = raw
            enc = encoding_hint or "utf-8"
            try:
                return payload.decode(enc, errors="replace")
            except LookupError:
                return payload.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if counter is not None:
                counter.add(0, requests=1)
            if exc.code in {404, 403}:
                return None
            last_exc = exc
            time.sleep(0.25 * (attempt + 1))
        except Exception as exc:  # noqa: BLE001 — network/gzip failures skip the file
            last_exc = exc
            time.sleep(0.25 * (attempt + 1))
    if last_exc is not None and counter is not None:
        counter.add(0, requests=1)
    return None


def gzip_bytes_to_text(raw: bytes, encoding: str = "utf-8") -> str:
    bio = io.BytesIO(raw)
    with gzip.GzipFile(fileobj=bio) as fh:
        return fh.read().decode(encoding, errors="replace")
