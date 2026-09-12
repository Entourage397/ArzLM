"""Persistent encode subprocess. FineWeb unicode panics die here, not in the packer."""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

_MAX_PAYLOAD = 32 * 1024 * 1024


def read_frame(stream) -> bytes | None:
    hdr = stream.read(4)
    if len(hdr) < 4:
        return None
    n = struct.unpack("<I", hdr)[0]
    if n > _MAX_PAYLOAD:
        raise RuntimeError(f"encode payload too large: {n}")
    data = bytearray()
    while len(data) < n:
        chunk = stream.read(n - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def write_frame(stream, payload: bytes) -> None:
    stream.write(struct.pack("<I", len(payload)))
    stream.write(payload)
    stream.flush()


def pack_ids(ids: list[int]) -> bytes:
    if not ids:
        return b""
    return struct.pack("<" + "I" * len(ids), *ids)


def unpack_ids(payload: bytes) -> list[int]:
    if not payload:
        return []
    if len(payload) % 4:
        raise RuntimeError("encode worker returned truncated ids")
    n = len(payload) // 4
    return list(struct.unpack("<" + "I" * n, payload))


def main() -> int:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer_dir = Path(sys.argv[1])
    from litgpt.tokenizer import Tokenizer

    from arzlm.data.textutil import encode_doc

    tokenizer = Tokenizer(tokenizer_dir)
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    while True:
        payload = read_frame(stdin)
        if payload is None:
            return 0
        text = payload.decode("utf-8", "replace")
        write_frame(stdout, pack_ids(encode_doc(tokenizer, text)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
