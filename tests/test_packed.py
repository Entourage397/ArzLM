import numpy as np
import torch

from arzlm.data.packed import PackedTokenDataset, pack_documents


def test_pack_documents_inserts_eos_boundaries() -> None:
    packed = pack_documents([[10, 11, 12], [20, 21], []], eos_id=3)
    assert packed.dtype == np.uint16
    assert packed.tolist() == [10, 11, 12, 3, 20, 21, 3]


def test_packed_dataset_contiguous_windows(tmp_path) -> None:
    tokens = np.arange(20, dtype=np.uint16)
    path = tmp_path / "train.bin"
    tokens.tofile(path)
    ds = PackedTokenDataset(path, seq_length=3)
    # block = 4, 20 // 4 = 5 sequences
    assert len(ds) == 5
    first = ds[0].tolist()
    assert first == [0, 1, 2, 3]
    # Next-token alignment used by the trainer
    x = first[:-1]
    y = first[1:]
    assert y == [1, 2, 3]
    assert x == [0, 1, 2]
    last = ds[4].tolist()
    assert last == [16, 17, 18, 19]
    assert isinstance(ds[0], torch.Tensor)
    assert ds[0].dtype == torch.int64
