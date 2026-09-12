# WSL2 + official Linux PyTorch (torch.compile)

Windows native CUDA on this machine has no Triton. Do not install unofficial
Windows Triton packages. Measure compile on WSL2 with the official Linux CUDA
wheels instead.

Do not change Windows NVIDIA drivers.

## Layout

The repo and virtualenv must live on the Linux filesystem (`ext4` in
`/home/...`), not under `/mnt/c`. `/mnt/c` is 9p and is too slow for training
and for `torch.compile` kernel caches.

Default destination: `~/arzlm` inside the Ubuntu distro.

Architecture, packed FineWeb-Edu windows, tokenizer, and seed stay identical to
`baseline-v0`.

## One-time bootstrap

From PowerShell:

```powershell
wsl -d Ubuntu -- bash -lc 'cd /mnt/c/Users/ASUS/ArzLM && sed -i "s/\r$//" scripts/wsl/*.sh && bash scripts/wsl/bootstrap.sh'
```

This:

1. rsyncs the tree to `$HOME/arzlm` (excludes Windows `.venv`)
2. copies `data/prepared/tiny` and `data/prepared/10m`
3. installs Python 3.12 via `uv` (Ubuntu system Python may be 3.14)
4. installs `torch` from `https://download.pytorch.org/whl/cu128`
5. installs `build-essential` (Triton/Inductor needs `gcc` to compile CUDA utils)
6. installs ArzLM editable + pytest

## Benchmark

Same methodology as Windows: seq=1024, BF16, fused AdamW, seed 42, micro-batch
1/2/4. Compile runs use 20 warmup steps so Inductor compilation is not mixed
into the measured window.

```powershell
wsl -d Ubuntu -- bash -lc 'sed -i "s/\r$//" /mnt/c/Users/ASUS/ArzLM/scripts/wsl/*.sh; bash /home/asus/arzlm/scripts/wsl/run_compile_bench.sh'
```

Reports:

- `~/arzlm/runs/wsl/linux-eager-bf16.json`
- `~/arzlm/runs/wsl/linux-compile-bf16.json`

Copy them onto the Windows tree (gitignored `runs/` stays on Linux):

```powershell
wsl -d Ubuntu -- bash -lc 'bash /home/asus/arzlm/scripts/wsl/copy_results_to_windows.sh'
```

## Migration rule

Measured on this machine (seq=1024, seed 42, BF16, flash SDPA):

- Windows eager micro-2: **19,060 tok/s** (`baseline-v0`)
- Linux eager micro-2: **19,851 tok/s** (~+4%, below the 10% bar)
- Linux `torch.compile` micro-2: **25,289 tok/s** (~+33%)
- Linux `torch.compile` micro-4: **27,446 tok/s** (~+44%)

Move **training** to WSL2 + official Linux PyTorch + `torch.compile` + `gcc`.
Do not migrate for Linux eager alone. JSON reports live in `docs/wsl/results/`.

## Muon + torch.compile

Same pinned Linux wheels. CUDA graphs **segfault** with hybrid Muon; the trainer
disables them automatically (`inductor-no-cudagraphs`). Fastest stable Muon
setup is **model compile, micro-batch 4**. Compiled 10M selected **Muon lr=0.05**
(eager 10M had a late 0.075 crossover that did not reproduce). Do not compile
the train/optimizer step (unstable). JSON:

- `docs/wsl/results/linux-muon-eager-bf16.json`
- `docs/wsl/results/linux-muon-compile-bf16.json`
- `docs/wsl/results/linux-muon-ckpt-eager.json`
- `docs/wsl/results/linux-muon-ckpt-compile.json`

```powershell
wsl -d Ubuntu -- bash -lc 'python3 -c "from pathlib import Path; p=Path(\"/mnt/c/Users/ASUS/ArzLM/scripts/wsl/run_muon_compile_bench.sh\"); Path(\"/tmp/run_muon_compile_bench.sh\").write_bytes(p.read_bytes().replace(b\"\\r\\n\", b\"\\n\"))"; bash /tmp/run_muon_compile_bench.sh'
```

