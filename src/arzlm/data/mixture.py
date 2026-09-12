"""Deterministic domain-aware window sampling over local packed shards.

Physical 55/20/15/10 shards already match the default mix. YAML weight
overrides resample domains (with wrap) without rebuilding the corpus.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from arzlm.data.catalog import DOMAINS, MIXTURE_TARGET, STEM_FORMAT
from arzlm.data.packed import ConcatPackedDataset, PackedTokenDataset, read_meta

DOMAIN_IDS = {name: i for i, name in enumerate(DOMAINS)}
ID_TO_DOMAIN = {i: name for name, i in DOMAIN_IDS.items()}


def normalize_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    src = dict(MIXTURE_TARGET if weights is None else weights)
    extra = [k for k in src if k not in DOMAINS]
    if extra:
        raise ValueError(f"Unknown mixture domains: {extra}")
    missing = [d for d in DOMAINS if d not in src]
    if missing:
        raise ValueError(f"Mixture missing domains: {missing}")
    total = float(sum(src[d] for d in DOMAINS))
    if total <= 0:
        raise ValueError("mixture weights must sum to a positive value")
    return {d: float(src[d]) / total for d in DOMAINS}


def epoch_assignments(
    *,
    seed: int,
    epoch: int,
    lengths: Mapping[str, int],
    weights: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Return (n, 2) int32 array of (domain_id, local_index) for one epoch.

    When requested counts fit in each domain, every window is used once.
    Weight overrides that exceed a domain wrap the permuted local indices.
    """
    w = normalize_weights(weights)
    present = [d for d in DOMAINS if int(lengths.get(d, 0)) > 0]
    if not present:
        raise ValueError("no domain windows available")
    n_total = int(sum(int(lengths[d]) for d in present))
    w_present = {d: w[d] for d in present}
    z = sum(w_present.values())
    w_present = {d: w_present[d] / z for d in present}

    rng = np.random.default_rng(int(seed) + 1_000_003 * int(epoch) + 17)
    counts: dict[str, int] = {}
    allocated = 0
    for i, d in enumerate(present):
        if i == len(present) - 1:
            counts[d] = n_total - allocated
        else:
            counts[d] = int(round(w_present[d] * n_total))
            allocated += counts[d]

    domain_col: list[np.ndarray] = []
    local_col: list[np.ndarray] = []
    for d in present:
        n_d = int(lengths[d])
        k = counts[d]
        perm = rng.permutation(n_d)
        if k <= n_d:
            locals_ = perm[:k]
        else:
            reps = int(np.ceil(k / n_d))
            locals_ = np.tile(perm, reps)[:k]
        domain_col.append(np.full(k, DOMAIN_IDS[d], dtype=np.int32))
        local_col.append(np.asarray(locals_, dtype=np.int32))
    domains = np.concatenate(domain_col)
    locals_ = np.concatenate(local_col)
    order = rng.permutation(n_total)
    return np.stack([domains[order], locals_[order]], axis=1)


class MixtureWindowDataset(Dataset):
    """One training epoch over mixed domain windows. Local files only."""

    def __init__(
        self,
        domain_datasets: Mapping[str, Dataset],
        *,
        seed: int,
        epoch: int = 0,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        self.domain_names = [d for d in DOMAINS if d in domain_datasets and len(domain_datasets[d]) > 0]
        if not self.domain_names:
            raise ValueError("MixtureWindowDataset needs at least one non-empty domain")
        self.domain_datasets = {d: domain_datasets[d] for d in self.domain_names}
        self.lengths = {d: len(self.domain_datasets[d]) for d in self.domain_names}
        # Missing domains get zero length so epoch_assignments can skip them
        # after we pass only present keys.
        self.seed = int(seed)
        self.weights = normalize_weights(weights)
        self._epoch = int(epoch)
        self._index = epoch_assignments(
            seed=self.seed,
            epoch=self._epoch,
            lengths=self.lengths,
            weights=self.weights,
        )

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) == self._epoch:
            return
        self._epoch = int(epoch)
        self._index = epoch_assignments(
            seed=self.seed,
            epoch=self._epoch,
            lengths=self.lengths,
            weights=self.weights,
        )

    def __len__(self) -> int:
        return int(self._index.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0:
            idx += len(self)
        domain_id, local = self._index[idx]
        name = ID_TO_DOMAIN[int(domain_id)]
        return self.domain_datasets[name][int(local)]

    def domain_of(self, idx: int) -> str:
        return ID_TO_DOMAIN[int(self._index[idx, 0])]


class CyclingDataset(Dataset):
    """Map global sequence index -> (epoch, local) so resume can skip by index."""

    def __init__(self, inner: Dataset, *, seed: int, n_items: int) -> None:
        self.inner = inner
        self.n = len(inner)
        if self.n <= 0:
            raise ValueError("inner dataset is empty")
        self.seed = int(seed)
        self.n_items = int(n_items)
        if self.n_items <= 0:
            raise ValueError("n_items must be positive")
        self._epoch: int | None = None
        self._perm: np.ndarray | None = None

    def __len__(self) -> int:
        return self.n_items

    def _inner_index(self, i: int) -> int:
        epoch = i // self.n
        local = i % self.n
        setter = getattr(self.inner, "set_epoch", None)
        if setter is not None:
            setter(epoch)
            return local
        if self._epoch != epoch or self._perm is None:
            rng = np.random.default_rng(self.seed + 1_000_003 * epoch)
            self._perm = rng.permutation(self.n)
            self._epoch = epoch
        return int(self._perm[local])

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= self.n_items:
            raise IndexError(idx)
        inner_idx = self._inner_index(idx)
        return self.inner[inner_idx]


class OffsetSampler(Sampler[int]):
    """Sequential indices [start, stop). Used to resume at iter_num * batch_size."""

    def __init__(self, start: int, stop: int) -> None:
        if start < 0 or stop < start:
            raise ValueError(f"invalid sampler range {start}:{stop}")
        self.start = int(start)
        self.stop = int(stop)

    def __iter__(self):
        return iter(range(self.start, self.stop))

    def __len__(self) -> int:
        return self.stop - self.start


def is_stem_meta(meta: dict) -> bool:
    return meta.get("format") == STEM_FORMAT or "domains" in meta


def load_domain_dataset(
    data_dir: Path,
    domain: str,
    split: str,
    seq_length: int,
) -> ConcatPackedDataset | PackedTokenDataset:
    meta = read_meta(data_dir)
    if is_stem_meta(meta):
        spec = meta["domains"][domain][split]
        files = spec["files"]
        paths = [Path(data_dir) / f for f in files]
        dtype = meta.get("dtype", "uint16")
        if len(paths) == 1:
            return PackedTokenDataset(paths[0], seq_length=seq_length, dtype=dtype)
        return ConcatPackedDataset(paths, seq_length=seq_length, dtype=dtype)
    raise ValueError(f"{data_dir} is not a STEM corpus")


def load_stem_train_dataset(
    data_dir: str | Path,
    seq_length: int,
    *,
    seed: int,
    weights: Mapping[str, float] | None = None,
    n_items: int,
) -> CyclingDataset:
    data_dir = Path(data_dir)
    meta = read_meta(data_dir)
    if not is_stem_meta(meta):
        raise ValueError(f"{data_dir} is not {STEM_FORMAT}")
    domains = {}
    for name in DOMAINS:
        if name not in meta.get("domains", {}):
            continue
        files = meta["domains"][name]["train"]["files"]
        if not files:
            continue
        domains[name] = load_domain_dataset(data_dir, name, "train", seq_length)
    inner = MixtureWindowDataset(domains, seed=seed, epoch=0, weights=weights)
    return CyclingDataset(inner, seed=seed, n_items=n_items)
