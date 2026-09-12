import json
from pathlib import Path

src = Path("/home/asus/arzlm/runs/wsl")
for name in ("linux-eager-bf16.json", "linux-compile-bf16.json"):
    d = json.loads((src / name).read_text())
    print("====", name, "====")
    print(
        "pytorch",
        d.get("pytorch_version"),
        "triton",
        d.get("triton_version"),
        "cuda",
        d.get("cuda_version"),
        "sdpa",
        d.get("sdpa_backend"),
    )
    for t in d["trials"]:
        num = t.get("numerical") or {}
        print(
            {
                "micro": t["micro_batch_size"],
                "tok_s": round(t["tokens_per_sec"], 1),
                "step_ms": round(t["step_time_s"] * 1000, 2),
                "alloc_gb": round(t["peak_allocated_vram_bytes"] / 1024**3, 3),
                "reserved_gb": round(t["peak_reserved_vram_bytes"] / 1024**3, 3),
                "compile_s": None if t.get("compile_time_s") is None else round(t["compile_time_s"], 3),
                "compile_fwd_s": None if t.get("compile_forward_s") is None else round(t["compile_forward_s"], 3),
                "loss": t.get("training_loss"),
                "finite": t.get("loss_finite"),
                "max_abs": num.get("max_abs_logit_diff"),
                "close_1e-1": num.get("close_atol_1e-1"),
                "eager_loss": num.get("eager_loss"),
                "compiled_loss": num.get("compiled_loss"),
            }
        )
