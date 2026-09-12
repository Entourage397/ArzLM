# Why Apache-2.0

This is not a casual default.

* **Original code.** `pyproject.toml` already declared Apache-2.0 for the
  ArzLM package before the 300M run. LitGPT (the GPT implementation we
  call) is Apache-2.0. That combination is a legitimate license for the
  source in this repository.
* **Vendored Muon.** KellerJordan/Muon is MIT, which is compatible with
  Apache-2.0. The MIT copyright remains in NOTICE and in the vendored file.
* **Weights.** The 300M checkpoint is trained on ODC-By datasets
  (FineWeb-Edu, FineMath, SmolLM Cosmopedia-v2, peS2o) plus permissive
  Software Heritage blobs from Stack-Edu. ODC-By requires attribution of
  the *database*; it does not by itself forbid releasing a trained model.
  HuggingFaceTB releases SmolLM / SmolLM2 (trained on the same family of
  sources) as Apache-2.0 with dataset cards. We follow that same pattern
  and keep NOTICE as the ODC-By attribution.
* **Not used.** NVIDIA Nemotron-CC-Math was excluded because its training
  agreement is incompatible with a public model release.

If a downstream user needs a different interpretation of Common Crawl
terms or Mixtral-generated Cosmopedia text, they should obtain their own
counsel. We do not claim that Apache-2.0 extinguishes third-party rights
in crawled pages or in individual source files.
