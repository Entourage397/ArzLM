# ArzLM-STEM-1B-v1

Local packed corpus for ArzLM-100M. Unique training target: **1,000,000,000** tokens.
Validation is extra and domain-held-out. Training never streams remote datasets.

## Mixture (starting target)

| Domain | Target share | Train tokens | Source |
| --- | ---: | ---: | --- |
| general | 55% | 550M | `HuggingFaceFW/fineweb-edu` `sample-10BT` |
| math | 20% | 200M | `HuggingFaceTB/finemath` `finemath-4plus` |
| code | 15% | 150M | `HuggingFaceTB/stack-edu` + Software Heritage S3 |
| science | 10% | 100M | `allenai/peS2o` `v2`, `source==s2orc` only |

Exact achieved counts live in `data/prepared/arzlm-stem-1b-v1/manifest.json` after the pack.

## Math decision

**Chosen: FineMath-4+.** Nemotron-CC-Math-4plus is the stronger published math crawl
(Karimi Mahabadi et al., 2025; NVIDIA Hub card) but is gated under the
**NVIDIA Data Agreement for Model Training** (internal training only) and was
cleaned with Phi-4. FineMath-4+ is public ODC-By-1.0, streams inline `text`,
and is decontaminated against GSM8k/MATH/MMLU/ARC. We do not mix both.

## Science

peS2o loader default is **v2** (`peS2o.py` `DEFAULT_CONFIG_NAME`). `data/v3/*.zst`
exists but is not in the datasets script; README still titles V2 as latest.
Hub files `train-00000..00009` (~1.5GB) are `s2ag/train` title/abstracts;
`train-00010..00019` (~7GB) are `s2orc/train` full text. The JSON `source`
field is `s2orc/train` / `s2ag/train`, not the bare card names. We stream only
s2orc shards and skip abstracts. Schema has no field-of-study; diversity is
rotating prefixes of s2orc shards, not exact FoS quotas. Do not
`load_dataset("allenai/peS2o")` (that pulls ~87GB).

## Code

Stack-Edu stores `blob_id` only. Content is fetched from
`softwareheritage` S3 `content/{blob_id}` (official Stack v2 recipe). No
unofficial text mirror. Permissive `license_type` / SPDX only. Language mix is
the prior in `arzlm.data.catalog.CODE_LANGUAGE_WEIGHTS` (Markdown capped at 2%).

## Dedup

Exact SHA-256 of NFC-stripped UTF-8 plus URL / blob / S2 keys. No MinHash.

## Storage

Little-endian **uint16** shards (`vocab=32000` fits). Format `arzlm-stem-v1`.

## Curriculum

YAML `mixture:` overrides resample domains without rebuilding shards.

## Tokenizer

Selected **`tokenizer/stem-v1`** (32,000 IDs, sha256 `a87cd2eecd8c7e5923b5a983630e2afdeb9e78df078a4af83c54c77f903f06cf`).
Held-out weighted chars/token (55/20/15/10): FineWeb-Edu tokenizer **4.031** vs STEM-mix 8M BPE **4.193** (+4.0%).
General English 4.754 → 4.542 chars/token (95.5% of current, above the 92% floor).
Code 1.946 → 3.274. Math 3.518 → 3.860. Science 4.210 → 4.318.
Convergence: 400k mix did not fill 32k (vocab 19,506, weighted 3.73); 2M mix 4.075; 8M 4.193.
Previous FineWeb-Edu tokenizer (`3132f2ffe4b80466…`) is kept as a historical artifact under `tokenizer/trained/`.
All serious training after this point starts from scratch.
