"""Streaming iterators for ArzLM-STEM sources. Remote I/O stops once quotas fill."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any
from urllib.request import Request, urlopen

from arzlm.data.catalog import (
    CODE_LANGUAGE_WEIGHTS,
    FINEWEB_EDU,
    FINEMATH_4PLUS,
    PERMISSIVE_LICENSE_TYPES,
    PERMISSIVE_SPDX,
    PES2O,
    PES2O_S2ORC_TRAIN_SHARDS,
    STACK_EDU,
    assert_dataset_allowed,
)
from arzlm.data.network import ByteCounter, CountingReader
from arzlm.data.security.gate import assert_file_fetch_allowed, gated_hub_download, list_fetchable_parquet
from arzlm.data.security.inert import load_dataset_inert
from arzlm.data.security.provenance import current_ledger
from arzlm.data.swh import fetch_swh_text
from arzlm.paths import configure_hf_home

DENIED_FINEWEB_MARKERS = (
    "data/CC-MAIN-2024-38/000_00041.parquet",
    "CC-MAIN-2024-38/000_00041",
)


def row_hits_denied_fineweb(*parts: object) -> bool:
    blob = " ".join(str(p or "") for p in parts).replace("\\", "/")
    return any(marker in blob for marker in DENIED_FINEWEB_MARKERS)


@dataclass
class SourceDoc:
    domain: str
    dataset_id: str
    native_id: str
    text: str
    source_keys: list[str] = field(default_factory=list)
    language: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _configure_hf() -> None:
    configure_hf_home()
    try:
        from huggingface_hub.hf_file_system import HfFileSystem

        clearer = getattr(HfFileSystem, "clear_instance_cache", None)
        if callable(clearer):
            clearer()
    except Exception:
        pass


def _load_streaming(dataset_id: str, config: str | None, revision: str, split: str = "train"):
    assert_dataset_allowed(dataset_id)
    _configure_hf()
    kwargs: dict[str, Any] = {
        "path": dataset_id,
        "split": split,
        "streaming": True,
        "revision": revision,
        "trust_remote_code": False,
    }
    if config is not None:
        kwargs["name"] = config
    return load_dataset_inert(**kwargs)


FINEWEB_PARQUET_PREFIX = "sample/10BT/"
FINEMATH_PARQUET_PREFIX = "finemath-4plus/"


def keep_hub_parquet_row(row: dict[str, Any]) -> bool:
    """FineWeb/FineMath rows need text; Stack-Edu rows are blob ids without text."""
    if "text" not in row:
        return True
    return bool(str(row.get("text") or "").strip())


def list_dataset_parquet(dataset_id: str, revision: str, prefix: str) -> list[str]:
    assert_dataset_allowed(dataset_id)
    _configure_hf()
    if prefix == FINEWEB_PARQUET_PREFIX and not prefix.startswith("sample/10BT"):
        raise ValueError("FineWeb-Edu parquet listing is sample/10BT only")
    metas = list_fetchable_parquet(dataset_id, revision, prefix, suffixes=(".parquet",))
    files = [m.path for m in metas]
    if prefix == FINEWEB_PARQUET_PREFIX:
        files = [f for f in files if f.startswith("sample/10BT/") and "/100BT/" not in f and "/350BT/" not in f]
    return sorted(files)


def iter_hub_parquet_rows(
    dataset_id: str,
    revision: str,
    prefix: str,
    *,
    skip: int = 0,
    columns: list[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Download official Hub parquet one file at a time. Stops when the caller stops.

    Avoids `datasets.load_dataset` / HfFileSystem dircache deepcopy failures.
    Parquet is opened with PyArrow as columnar data, never unpickled.
    """
    import pyarrow.parquet as pq

    files = list_dataset_parquet(dataset_id, revision, prefix)
    if not files:
        raise FileNotFoundError(f"no parquet under {dataset_id} {prefix!r} @ {revision}")
    yielded = 0
    for name in files:
        local = gated_hub_download(dataset_id, name, revision)
        pf = pq.ParquetFile(local)
        kwargs: dict[str, Any] = {"batch_size": 256}
        if columns:
            kwargs["columns"] = columns
        for batch in pf.iter_batches(**kwargs):
            table = batch.to_pydict()
            n = len(next(iter(table.values())))
            for j in range(n):
                row = {k: table[k][j] for k in table}
                if not keep_hub_parquet_row(row):
                    continue
                if yielded < skip:
                    yielded += 1
                    continue
                yielded += 1
                yield row


def iter_fineweb_edu(*, skip: int = 0) -> Iterator[SourceDoc]:
    spec = FINEWEB_EDU
    # Parquet-first: sample/10BT only (14 files). Do not use datasets streaming
    # (HfFileSystem dircache deepcopy can abort the packer).
    for i, row in enumerate(iter_hub_parquet_rows(spec.dataset_id, spec.revision, FINEWEB_PARQUET_PREFIX, skip=skip)):
        if row_hits_denied_fineweb(row.get("file_path"), row.get("dump"), row.get("id")):
            continue
        url = row.get("url")
        native = str(row.get("id") or url or f"fineweb:{i}")
        keys = []
        if url:
            keys.append(f"url:{url}")  # identity only; never fetched
        keys.append(f"id:{native}")
        yield SourceDoc(
            domain="general",
            dataset_id=spec.dataset_id,
            native_id=native,
            text=str(row.get("text") or ""),
            source_keys=keys,
            language=row.get("language"),
            extra={"dump": row.get("dump"), "score": row.get("score")},
        )


def iter_finemath_4plus(*, skip: int = 0) -> Iterator[SourceDoc]:
    spec = FINEMATH_4PLUS
    try:
        yield from _finemath_from_rows(
            iter_hub_parquet_rows(spec.dataset_id, spec.revision, FINEMATH_PARQUET_PREFIX, skip=skip),
            skip=0,
        )
        return
    except Exception as exc:
        print(f"finemath parquet failed ({exc!r}); trying datasets streaming", flush=True)
    try:
        ds = _load_streaming(spec.dataset_id, spec.config, spec.revision)
        yield from _finemath_from_rows(ds, skip=skip)
        return
    except Exception as exc:
        print(f"finemath datasets streaming failed ({exc!r}); using name-scan parquet fallback", flush=True)
    yield from _iter_finemath_parquet(skip=skip)


def _finemath_from_rows(rows, *, skip: int) -> Iterator[SourceDoc]:
    spec = FINEMATH_4PLUS
    yielded = 0
    for i, row in enumerate(rows):
        text = row.get("text") or ""
        if not str(text).strip():
            continue
        if yielded < skip:
            yielded += 1
            continue
        yielded += 1
        url = row.get("url")
        native = str(url or f"finemath:{i}")
        keys = [f"url:{url}"] if url else []
        yield SourceDoc(
            domain="math",
            dataset_id=spec.dataset_id,
            native_id=native,
            text=str(text),
            source_keys=keys,
            language=row.get("language"),
            extra={"int_score": row.get("int_score"), "crawl": row.get("crawl")},
        )


def _iter_finemath_parquet(*, skip: int) -> Iterator[SourceDoc]:
    spec = FINEMATH_4PLUS
    files = list_dataset_parquet(spec.dataset_id, spec.revision, FINEMATH_PARQUET_PREFIX)
    seen = 0
    import pyarrow.parquet as pq

    _configure_hf()
    for name in files:
        local = gated_hub_download(spec.dataset_id, name, spec.revision)
        pf = pq.ParquetFile(local)
        for batch in pf.iter_batches(batch_size=256, columns=None):
            rows = batch.to_pydict()
            n = len(next(iter(rows.values())))
            for j in range(n):
                if seen < skip:
                    seen += 1
                    continue
                seen += 1
                row = {k: rows[k][j] for k in rows}
                yield from _finemath_from_rows([row], skip=0)


def pes2o_shard_names() -> list[str]:
    return [f"data/v2/train-{i:05d}-of-00020.json.gz" for i in range(20)]


def is_s2orc_source(value: object) -> bool:
    """peS2o cards say `s2orc`; v2 jsonl uses `s2orc/train`."""
    return str(value or "").startswith("s2orc")


def pes2o_shard_url(filename: str) -> str:
    spec = PES2O
    from huggingface_hub import hf_hub_url

    return hf_hub_url(
        spec.dataset_id,
        filename=filename,
        repo_type="dataset",
        revision=spec.revision,
    )


def _iter_gzip_json_lines(fileobj, *, max_line: int = 16 * 1024 * 1024) -> Iterator[bytes]:
    """Read gzip jsonl without gzip.readline() buffering a multi-GB 'line'.

    A truncated HTTP body must not abort the whole science packer. Drop the
    incomplete tail of the current shard and let the caller continue.
    """
    try:
        gz_file = gzip.GzipFile(fileobj=fileobj)
    except OSError:
        return
    try:
        with gz_file as gz:
            buf = b""
            while True:
                try:
                    chunk = gz.read(1024 * 1024)
                except EOFError:
                    return
                if not chunk:
                    if buf.strip():
                        yield buf
                    return
                buf += chunk
                start = 0
                while True:
                    i = buf.find(b"\n", start)
                    if i < 0:
                        buf = buf[start:]
                        break
                    yield buf[start:i]
                    start = i + 1
                if len(buf) > max_line:
                    buf = b""
    except EOFError:
        return
    except OSError:
        return


def iter_pes2o_s2orc(
    *,
    counter: ByteCounter | None = None,
    skip: int = 0,
    shard_indices: list[int] | tuple[int, ...] | None = None,
    max_docs_per_shard: int | None = 15_000,
) -> Iterator[SourceDoc]:
    """Stream peS2o v2 json.gz one shard at a time. Keep s2orc full text.

    Does not use datasets.load_dataset (that would download all ~87GB).
    Default shards are 10-19 (full-text s2orc). Files 0-9 are s2ag abstracts.
    `skip` counts yielded s2orc documents for crash-safe resume.
    """
    spec = PES2O
    names = pes2o_shard_names()
    indices = list(PES2O_S2ORC_TRAIN_SHARDS if shard_indices is None else shard_indices)
    allow_abstracts = any(int(i) < 10 for i in indices)
    yielded = 0
    for shard_i in indices:
        filename = names[int(shard_i)]
        meta = assert_file_fetch_allowed(spec.dataset_id, spec.revision, filename)
        ledger = current_ledger()
        if ledger is not None:
            ledger.record(meta, config=spec.config)
        url = pes2o_shard_url(filename)
        req = Request(
            url,
            headers={"User-Agent": "arzlm-stem/1.0", "Accept-Encoding": "identity"},
        )
        from_this_shard = 0
        try:
            with urlopen(req, timeout=120) as resp:
                reader = CountingReader(resp, counter) if counter is not None else resp
                for blob in _iter_gzip_json_lines(reader):
                    if not blob.strip():
                        continue
                    try:
                        row = json.loads(blob.decode("utf-8", "replace"))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
                    if not isinstance(row, dict) or not str(row.get("text") or "").strip():
                        continue
                    if not allow_abstracts and not is_s2orc_source(row.get("source")):
                        continue
                    text = row.get("text") or ""
                    from_this_shard += 1
                    if yielded < skip:
                        yielded += 1
                        if max_docs_per_shard is not None and from_this_shard >= max_docs_per_shard:
                            break
                        continue
                    yielded += 1
                    native = str(row.get("id") or f"pes2o:{yielded}")
                    yield SourceDoc(
                        domain="science",
                        dataset_id=spec.dataset_id,
                        native_id=native,
                        text=str(text),
                        source_keys=[f"s2:{native}"],
                        extra={
                            "source": str(row.get("source") or ""),
                            "created": row.get("created"),
                            "version": row.get("version"),
                            "shard": filename,
                        },
                    )
                    if max_docs_per_shard is not None and from_this_shard >= max_docs_per_shard:
                        break
        except (EOFError, OSError, TimeoutError) as exc:
            print(
                f"peS2o shard {filename} unreadable ({type(exc).__name__}: {exc}); skipping remainder of shard",
                flush=True,
            )
            continue


def license_is_permissive(row: dict[str, Any]) -> bool:
    lt = str(row.get("license_type") or "").strip().lower()
    if lt in PERMISSIVE_LICENSE_TYPES:
        return True
    licenses = row.get("detected_licenses") or []
    if isinstance(licenses, str):
        licenses = [licenses]
    for lic in licenses:
        token = str(lic).strip().lower()
        if token in PERMISSIVE_SPDX:
            return True
    return False


def stack_edu_row_eligible(
    row: dict[str, Any],
    *,
    min_score: float = 4.0,
    max_score: float | None = None,
) -> bool:
    """Permissive Stack-Edu rows in an educational-score band.

    The Hub card keeps ``int_score`` in {3,4,5} after a threshold of 3.
    The first packing pass uses ``min_score=4``. A later fill pass may
    emit the unused ``int_score==3`` band without changing licenses.
    """
    if not license_is_permissive(row):
        return False
    score = row.get("int_score", row.get("score"))
    if score is None:
        return max_score is None and bool(str(row.get("blob_id") or "").strip())
    try:
        value = float(score)
    except (TypeError, ValueError):
        return False
    if value < min_score:
        return False
    if max_score is not None and value >= max_score:
        return False
    return bool(str(row.get("blob_id") or "").strip())


def iter_stack_edu_language(
    language: str,
    *,
    counter: ByteCounter | None = None,
    skip: int = 0,
    fetch_workers: int = 8,
    row_counter: list[int] | None = None,
    min_score: float = 4.0,
    max_score: float | None = None,
) -> Iterator[SourceDoc]:
    """Stream Stack-Edu IDs for one language and fetch bytes from Software Heritage.

    `skip` counts dataset rows (cheap parquet/metadata), not SWH fetches, so
    crash-resume does not re-download already-seen blobs.

    Stack-Edu source code is training text only. Never compile, install,
    execute, or follow commands contained in blobs or README samples.
    """
    if language not in CODE_LANGUAGE_WEIGHTS:
        raise ValueError(f"unsupported Stack-Edu language {language!r}")
    spec = STACK_EDU
    ds = iter_hub_parquet_rows(spec.dataset_id, spec.revision, f"{language}/")

    def _eligible(row: dict[str, Any]) -> bool:
        return stack_edu_row_eligible(row, min_score=min_score, max_score=max_score)

    def _work(row: dict[str, Any]) -> SourceDoc | None:
        if not _eligible(row):
            return None
        blob_id = str(row.get("blob_id") or "").strip()
        text = fetch_swh_text(
            blob_id,
            counter=counter,
            encoding_hint=row.get("src_encoding"),
        )
        if not text or not text.strip():
            return None
        path = row.get("path") or ""
        native = blob_id
        return SourceDoc(
            domain="code",
            dataset_id=spec.dataset_id,
            native_id=native,
            text=text,
            source_keys=[f"swh:{blob_id}"],
            language=language,
            extra={
                "repo_name": row.get("repo_name"),
                "path": path,
                "license_type": row.get("license_type"),
                "score": row.get("score"),
            },
        )

    n = 0

    def _mark() -> None:
        if row_counter is not None:
            row_counter[0] = n

    if fetch_workers <= 1:
        for row in ds:
            n += 1
            if n <= skip:
                _mark()
                continue
            _mark()
            doc = _work(dict(row))
            if doc is not None:
                yield doc
        return

    from concurrent.futures import ThreadPoolExecutor

    pending: list = []
    with ThreadPoolExecutor(max_workers=max(1, fetch_workers)) as pool:
        for row in ds:
            n += 1
            if n <= skip:
                _mark()
                continue
            _mark()
            row = dict(row)
            if not _eligible(row):
                continue
            pending.append(pool.submit(_work, row))
            if len(pending) >= fetch_workers:
                doc = pending.pop(0).result()
                if doc is not None:
                    yield doc
        for fut in pending:
            doc = fut.result()
            if doc is not None:
                yield doc


def iter_science_locked(
    *,
    counter: ByteCounter | None = None,
    skip: int = 0,
) -> Iterator[SourceDoc]:
    """Stream science text from whatever the source-lock actually pinned."""
    from arzlm.data.security.lockfile import load_source_lock

    lock = load_source_lock()
    return _iter_locked_domain(lock, "science", skip=skip, counter=counter)


def iter_general_locked(*, skip: int = 0, start_source: int = 0) -> Iterator[SourceDoc]:
    """FineWeb-Edu sample-10BT first, then Hub-safe educational supplements."""
    from arzlm.data.security.lockfile import load_source_lock

    lock = load_source_lock()
    return _iter_locked_domain(
        lock,
        "general",
        skip=skip,
        start_source=start_source,
        primary_factory=lambda: iter_fineweb_edu(skip=0),
    )


def _iter_locked_domain(
    lock,
    domain: str,
    *,
    skip: int,
    start_source: int = 0,
    primary_factory=None,
    counter: ByteCounter | None = None,
) -> Iterator[SourceDoc]:
    yielded = 0

    def _emit(doc: SourceDoc) -> Iterator[SourceDoc]:
        nonlocal yielded
        if yielded < skip:
            yielded += 1
            return
        yielded += 1
        yield doc

    sources = lock.sources_for_domain(domain)
    start_source = max(0, int(start_source))
    for i, src in enumerate(sources):
        if i < start_source:
            continue
        if i == 0 and primary_factory is not None and src.dataset_id == FINEWEB_EDU.dataset_id:
            stream = primary_factory()
        elif src.dataset_id == PES2O.dataset_id:
            indices = []
            for path in src.files:
                try:
                    indices.append(int(path.split("train-")[1].split("-")[0]))
                except (IndexError, ValueError):
                    continue
            stream = iter_pes2o_s2orc(
                counter=counter,
                skip=0,
                shard_indices=indices or None,
                max_docs_per_shard=None,
            )
        else:
            stream = iter_locked_text_table(src, skip=0)
        for doc in stream:
            if not isinstance(doc, SourceDoc):
                continue
            # Skip counts only inside the first consumed source. Later lock
            # sources (Cosmopedia, science fallbacks) must not be eaten by a
            # FineWeb stream_index from a previous resume.
            if i == start_source:
                yield from _emit(doc)
            else:
                yield doc


def text_table_column(names: list[str]) -> str:
    preferred = ("text", "article", "body", "article_text", "content", "abstract", "summary", "document")
    for name in preferred:
        if name in names:
            return name
    return names[0]


def iter_locked_text_table(src: Any, skip: int = 0) -> Iterator[SourceDoc]:
    import pyarrow.csv as pcsv
    import pyarrow.parquet as pq

    yielded = 0
    for path in sorted(src.files):
        rel = str(path).replace("\\", "/")
        lower = rel.lower()
        if not lower.endswith((".parquet", ".csv", ".json", ".jsonl")):
            continue
        local = gated_hub_download(src.dataset_id, path, src.revision)
        if lower.endswith(".parquet"):
            pf = pq.ParquetFile(local)
            names = list(pf.schema.names)
            key = text_table_column(names) if names else "text"
            extra_cols = [c for c in ("id", "metadata", "file_path", "url", "dump") if c in names and c != key]
            cols = [key, *extra_cols]
            row_index = 0
            for batch in pf.iter_batches(batch_size=256, columns=cols, use_threads=False):
                data = batch.to_pydict()
                n = len(data[key])
                for j in range(n):
                    text = str(data[key][j] or "")
                    extra_vals = {c: data[c][j] for c in extra_cols}
                    if row_hits_denied_fineweb(path, extra_vals.get("file_path"), extra_vals.get("metadata"), extra_vals.get("id")):
                        row_index += 1
                        continue
                    if not text.strip():
                        row_index += 1
                        continue
                    if yielded < skip:
                        yielded += 1
                        row_index += 1
                        continue
                    yielded += 1
                    native = f"{path}:{row_index}"
                    row_index += 1
                    yield SourceDoc(
                        domain=src.domain,
                        dataset_id=src.dataset_id,
                        native_id=native,
                        text=text,
                        source_keys=[f"id:{native}"],
                        extra={"path": path, **{k: extra_vals[k] for k in extra_vals if extra_vals[k] is not None}},
                    )
            continue
        if lower.endswith(".csv"):
            table = pcsv.read_csv(local)
            names = [str(n) for n in table.schema.names]
            if not names:
                continue
            key = text_table_column(names)
            column = table.column(key)
            for i, raw in enumerate(column.to_pylist()):
                text = str(raw or "")
                if not text.strip():
                    continue
                if yielded < skip:
                    yielded += 1
                    continue
                yielded += 1
                native = f"{path}:{i}"
                yield SourceDoc(
                    domain=src.domain,
                    dataset_id=src.dataset_id,
                    native_id=native,
                    text=text,
                    source_keys=[f"id:{native}"],
                    extra={"path": path},
                )
            continue
        if lower.endswith(".jsonl") or lower.endswith(".json"):
            with open(local, encoding="utf-8") as fh:
                payload = fh.read()
            blobs: list[Any]
            if lower.endswith(".jsonl"):
                blobs = [json.loads(line) for line in payload.splitlines() if line.strip()]
            else:
                loaded = json.loads(payload)
                blobs = loaded if isinstance(loaded, list) else [loaded]
            for i, row in enumerate(blobs):
                if not isinstance(row, dict):
                    continue
                key = text_table_column([str(k) for k in row]) if row else None
                text = str(row.get(key) or "") if key else ""
                if not text.strip():
                    continue
                if yielded < skip:
                    yielded += 1
                    continue
                yielded += 1
                native = f"{path}:{i}"
                yield SourceDoc(
                    domain=src.domain,
                    dataset_id=src.dataset_id,
                    native_id=native,
                    text=text,
                    source_keys=[f"id:{native}"],
                    extra={"path": path},
                )



def synthetic_stem_documents(domain: str, n_docs: int, seed: int) -> Iterator[SourceDoc]:
    """Offline STEM-ish documents for tests. Includes LaTeX, code, and prose."""
    import numpy as np

    rng = np.random.default_rng(seed)
    templates = {
        "general": (
            "Photosynthesis converts light energy into chemical energy in chloroplasts. "
            "Lesson {i} explains energy flow with a short definition for students."
        ),
        "math": (
            "Solve for x. Starting from $E = mc^2$ we rewrite "
            r"$\frac{{a}}{{b}} = \sum_{{i=1}}^{{n}} i$ and obtain "
            r"$\int_0^1 \sqrt{{x}} \, dx = \frac{{2}}{{3}}$. Worked example {i}."
        ),
        "code": (
            "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n"
            "# example {i}: x = a + b * (c - 1)\n"
        ),
        "science": (
            "We report the measured band gap of the semiconductor sample. "
            "The Hamiltonian H = p^2/2m + V(x) describes the electron. Paper {i}."
        ),
    }
    if domain not in templates:
        raise ValueError(domain)
    tpl = templates[domain]
    for i in range(n_docs):
        extra = "".join(chr(int(rng.integers(97, 123))) for _ in range(24))
        text = tpl.format(i=i) + " " + extra
        native = f"synthetic:{domain}:{i}"
        yield SourceDoc(
            domain=domain,
            dataset_id="synthetic",
            native_id=native,
            text=text,
            source_keys=[f"id:{native}"],
            language="Python" if domain == "code" else "en",
        )
