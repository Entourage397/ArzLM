# Dataset source security

File security (can this remote artifact be fetched and parsed as data?) is
separate from content quality (does the web text mention exploits, malware, or
URLs). This document covers **file security**. Training documents remain inert
text: they are never executed, compiled, installed, or used to fetch further
URLs.

FineWeb-Edu remains an approved high-quality source. ArzLM does **not** switch
datasets because one physical shard in the full crawl dump is scanner-flagged.
The planned corpus uses config `sample-10BT` only (`sample/10BT/*.parquet`).
The full `data/CC-MAIN-*` dump is out of scope and is refused by the ingest
gate.

## Incident: FineWeb-Edu `000_00041.parquet`

Observed 2026-09-08 against Hugging Face Hub metadata (no file content
downloaded).

| Field | Value |
| --- | --- |
| Dataset | `HuggingFaceFW/fineweb-edu` |
| Path | `data/CC-MAIN-2024-38/000_00041.parquet` |
| Revision (`main` / pinned) | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| SHA256 | `87753428a95294f19aac1f579d145c264ab55484d9fd205a413698821aaf1dc2` |
| Xet hash | `029d8216b2dff2f861b43b281e39e080b87d27f09aedbe5936a8a45da5e22c31` |
| Size | 802,227,208 bytes (~802 MB) |
| Hub UI | [Unsafe](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/main/data/CC-MAIN-2024-38/000_00041.parquet) |

### What Hugging Face's "Unsafe" badge means

Hugging Face runs every committed repository file through a malware scanner
([Hub malware scanning](https://huggingface.co/docs/hub/en/security-malware)).
Scanning is triggered at each commit. The UI badge is **not** by itself a
confirmed named virus, a confirmed executable, or a confirmation that the
dataset authors shipped a weaponized payload.

Authoritative per-file metadata is exposed by the Hub API when tree/path
requests set `expand=true`. `huggingface_hub.RepoFile.security` maps from
`securityFileStatus` and includes:

- `status`: `safe` / `unsafe` / `unscanned` / `queued` / …
- `avScan`: ClamAV result (`status`, `message`, scanner `version`)
- `pickleImportScan`: HF Picklescan result

Repo-level `dataset_info(..., securityStatus=true)` did **not** return
`security_repo_status` for this dataset at audit time. Per-file
`get_paths_info(..., expand=True)` did.

Live metadata for the flagged file:

- **ClamAV** (`1.5.2/27937`): `unsafe`, message `Hugging Face ClamAV detected 1 infection(s)`
- **Picklescan**: `unscanned` / UI: "not a pickle"
- **VirusTotal**: no report shown on the blob page
- **Threat / signature name**: not published in the API or UI

Classification of this signal:

| Hypothesis | Assessment |
| --- | --- |
| Known scanner flag | **Yes.** Hugging Face ClamAV marked the blob unsafe. |
| Confirmed executable malware | **Not established.** No signature name, no VirusTotal report, file is Parquet. |
| Possible false positive | **Possible.** ClamAV on a ~800MB web-crawl Parquet can match signatures inside **text columns** (EICAR-like strings, encoded samples, exploit write-ups). |
| Malware-related text inside training data | **Possible** and expected in web crawls. That is a content issue, not a reason to execute anything. |
| Unsafe serialization (pickle) | **No.** Picklescan: not a pickle. |

We treat the badge as a genuine security signal until proven otherwise. We do
**not** download, open, deserialize, or hash-scan the blob locally to "see
what is inside."

The file is hard-denied in `data/security/denylist.json` by dataset ID, path,
and SHA256, even if Hub later changes the badge.

## sample-10BT vs the full dump

`sample-10BT` is an official sampled configuration. Its physical files are
`sample/10BT/000_00000.parquet` … `013_00000.parquet` (14 files). The flagged
object lives under `data/CC-MAIN-2024-38/`, which is the full crawl dump.
Those are different Git LFS objects (different SHA256 values).

Historical ArzLM 10M/50M packs recorded `dataset_config: sample-10BT`. They
did not name this dump shard.

At audit time, ClamAV status of `sample/10BT/`:

- `000`–`012`: Hub `status=unscanned`, `av_scan.status=unscanned` (each ~2.15 GB)
- `013`: Hub `status=safe` (~541 MB)

Large files often stay `unscanned` (see Hub docs: missing ok/infected badge
can mean queued, in progress, or scan error). Unscanned is **not** the same
as unsafe. Fail-closed policy still refuses to fetch unscanned files unless
`reviewed_allow_unscanned` is set on that lock entry after explicit review.

## Acquisition check (this corpus-build attempt)

Checked **metadata, filenames, and small pointer files only**. No Parquet
was opened.

| Location | Result |
| --- | --- |
| WSL project `hf-cache/hub/datasets--HuggingFaceFW--fineweb-edu` | 60 KB: `refs/main` → `87f09149…`, README snapshot, `.no_exist` dataset-script markers. **No parquet, no `000_00041`, no matching SHA256.** |
| Windows project `hf-cache/` | empty / unused |
| User `~/.cache/huggingface/hub/datasets--HuggingFaceFW--fineweb-edu` | README + `.no_exist` only. **No parquet.** Not modified (not project-owned). |
| Filename search `*000_00041*` | none |
| SHA256 string search in project caches | none |

**Verdict: A. definitely not acquired.** The flagged remote object did not
enter the project-owned cache and did not contribute to existing packed
shards. A partial ArzLM-STEM-1B-v1 `train/general` shard exists from an
aborted pack that used `sample/10BT` (or FineWeb-Edu streaming under that
config), not `data/CC-MAIN-2024-38/000_00041.parquet`.

## Pipeline controls

1. **Denylist** `data/security/denylist.json` — hard deny by id/path/SHA256.
2. **Source lock** `data/security/source-lock.json` — pinned dataset id,
   revision, config/prefix, physical files, sizes, hashes, Hub security
   status at lock time. Analogous to a package lockfile. Production packs
   must not follow a floating `main`.
3. **Pre-fetch gate** — `get_paths_info` / `list_repo_tree(expand=True)`
   **without downloading content**. Unsafe/infected → never fetch. Missing or
   pending AV on a file that is not reviewed → stop (fail closed). Denied
   files in an otherwise clean prefix are excluded; unresolved files abort.
4. **CLI** `python -m arzlm data-security-audit` must pass before
   `prepare --source fineweb-edu` or `prepare-stem --source remote`.
5. **Formats** — Parquet/Arrow/JSONL/gzip text only. No `.pkl` / `.pt` /
   dataset scripts. `trust_remote_code=False` always. PyArrow reads values as
   data. Stack-Edu blobs are decoded text, never compiled or executed.
6. **Documents are inert** — URLs in FineWeb rows are identity keys only.
   No `requests`/`urlopen` of document URLs. No archive extraction or package
   install based on document text.
7. **Provenance** — each actually-read upstream file is appended to
   `state/upstream-files.jsonl` (dataset, revision, path, hash, size, Hub
   status, timestamp). Packed shards already have SHA256 sidecars.
8. **Local AV** — optional ClamAV (`ARZLM_LOCAL_AV=1`) on project-owned
   downloads after fetch, before parse. Disabled by default. Skip if
   `clamscan`/`clamdscan` is absent. Do not install system packages for this.
   A local positive is a hard failure, not a "probably fine" override. Do not
   scan `~/.cache/huggingface`.

## Specialist sources

The same gate applies to FineMath-4+, Stack-Edu language parquet, and peS2o
v2 s2orc shards (`data/v2/train-00010`–`00019`). Code is training text.

At lock time: Stack-Edu language parquet was ClamAV `safe`. FineMath-4+ was
`safe` or `queued` with `av_scan.status=safe`. peS2o s2orc shards were
`queued` with AV often still `unscanned` (fail closed until scan or review).

## Resume

Do not start ArzLM-STEM-1B-v1 packing until `python -m arzlm data-security-audit`
prints `Result: PASS` (or unresolved sample-10BT files are explicitly
reviewed in the lock). See the task report for the recommended next command.
