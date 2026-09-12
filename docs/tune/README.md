# Optimizer sweep snapshots

Copied from WSL `~/arzlm/runs/tune/` and Windows `runs/tune/` so comparisons survive gitignore of `runs/`.

Primary metric: validation cross-entropy vs tokens.

## Windows eager 2M (original grid)

AdamW vs Muon lr=0.01/0.02/0.05. Winner **Muon 0.05** (val 6.306), but 0.05 was the upper bound.

## WSL expanded 2M bracket

`wsl-muon-jordan-lr{0.05,0.075,0.10}-2m-*`. Winner **0.05** (6.308). Did not run 0.15.

## WSL 10M confirmation

Eager (micro-2): `wsl-muon-10m-lr{0.05,0.075}-*`. 0.075 wins (5.209 vs 5.239) after a late crossover.

Compile + micro-4 (the 50M stack): `wsl-muon-10m-lr{0.05,0.075}-compile-*`. **0.05 wins** (5.153 vs 5.220) at every common checkpoint.

## WSL WD check

At lr=0.075: `wsl-muon-wd0p05-lr0p075-2m-*` and `wsl-muon-wd0p2-*`. wd=0.10 is the 2M 0.075 trial.

At lr=0.05: `wsl-muon-wd0p05-lr0p05-2m-*` and `wsl-muon-wd0p2-lr0p05-2m-*`. wd=0.10 is the 2M 0.05 LR-bracket trial.

Keep **wd=0.10**. See `docs/experiments.md` for tables and the 50M recommendation.
