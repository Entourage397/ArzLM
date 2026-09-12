"""Probe official STEM sources: schema, streaming, one document, SWH fetch."""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from arzlm.data.catalog import FINEWEB_EDU, FINEMATH_4PLUS, MATH_DECISION, NEMOTRON_CC_MATH, PES2O, STACK_EDU
from arzlm.data.network import ByteCounter
from arzlm.data.sources import iter_fineweb_edu, iter_finemath_4plus, iter_pes2o_s2orc, iter_stack_edu_language
from arzlm.data.swh import fetch_swh_text, swh_content_url


def _summarize_doc(doc) -> dict:
    return {
        "dataset_id": doc.dataset_id,
        "native_id": doc.native_id[:80],
        "n_chars": len(doc.text),
        "preview": doc.text[:240].replace("\n", " "),
        "language": doc.language,
        "keys": doc.source_keys[:4],
    }


def main() -> int:
    report: dict = {
        "math_decision": MATH_DECISION,
        "nemotron_permitted": NEMOTRON_CC_MATH.pretraining_permitted,
        "sources": {
            "general": FINEWEB_EDU.__dict__,
            "math": FINEMATH_4PLUS.__dict__,
            "code": STACK_EDU.__dict__,
            "science": PES2O.__dict__,
        },
        "probes": {},
    }
    t0 = time.perf_counter()
    try:
        doc = next(iter_fineweb_edu())
        report["probes"]["fineweb_edu"] = _summarize_doc(doc)
    except Exception as exc:  # noqa: BLE001
        report["probes"]["fineweb_edu"] = {"error": repr(exc), "traceback": traceback.format_exc()}
    try:
        doc = next(iter_finemath_4plus())
        report["probes"]["finemath"] = _summarize_doc(doc)
    except Exception as exc:  # noqa: BLE001
        report["probes"]["finemath"] = {"error": repr(exc), "traceback": traceback.format_exc()}
    sci_counter = ByteCounter("pes2o")
    try:
        doc = next(iter_pes2o_s2orc(counter=sci_counter, shard_indices=[10], max_docs_per_shard=None))
        report["probes"]["pes2o"] = {**_summarize_doc(doc), "network": sci_counter.snapshot()}
        report["probes"]["pes2o"]["source_field"] = doc.extra.get("source")
    except Exception as exc:  # noqa: BLE001
        report["probes"]["pes2o"] = {"error": repr(exc), "traceback": traceback.format_exc(), "network": sci_counter.snapshot()}
    code_counter = ByteCounter("code")
    try:
        doc = next(iter_stack_edu_language("Python", counter=code_counter, fetch_workers=1))
        report["probes"]["stack_edu_python"] = {
            **_summarize_doc(doc),
            "network": code_counter.snapshot(),
            "swh_url": swh_content_url(doc.native_id),
        }
    except Exception as exc:  # noqa: BLE001
        report["probes"]["stack_edu_python"] = {"error": str(exc), "network": code_counter.snapshot()}
        # Unsigned S3 smoke with a fake id to record the error class
        missing = fetch_swh_text("0" * 40, counter=code_counter)
        report["probes"]["swh_missing"] = {"text_is_none": missing is None, "network": code_counter.snapshot()}
    report["elapsed_s"] = time.perf_counter() - t0
    text = json.dumps(report, indent=2, default=str)
    Path("/tmp/arzlm-stem-probe.json").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
