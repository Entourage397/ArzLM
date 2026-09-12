import torch
import pytest
from litgpt.tokenizer import Tokenizer

from arzlm.cloud.prompting import decode_token_ids, encode_prompt_batch
from arzlm.tokenizer.train import save_litgpt_tokenizer_files, train_bpe_tokenizer


def test_encode_prompt_batch_is_rank2_long(tmp_path) -> None:
    texts = [
        "Photosynthesis converts light energy into chemical energy in chloroplasts.",
        "Force equals mass times acceleration.",
        "Prime numbers are integers greater than one.",
    ]
    tokenizer = train_bpe_tokenizer(texts * 50, vocab_size=400, min_frequency=1)
    out = save_litgpt_tokenizer_files(tokenizer, tmp_path / "tok")
    cfg = (out / "tokenizer_config.json").read_text(encoding="utf-8")
    assert '"model_max_length": 2048' in cfg
    tok = Tokenizer(out)
    raw = tok.encode("Force equals mass times acceleration.", bos=False, eos=False)
    assert isinstance(raw, torch.Tensor)
    assert raw.ndim == 1
    with pytest.raises(TypeError, match="single element"):
        torch.tensor([raw], dtype=torch.long)
    batch = encode_prompt_batch(tok, "Force equals mass times acceleration.", bos=False, eos=False)
    assert batch.dtype == torch.long
    assert batch.shape == (1, int(raw.numel()))
    assert batch[0].tolist() == raw.to(dtype=torch.long).tolist()
    decoded = decode_token_ids(tok, batch)
    assert isinstance(decoded, str)
    assert "Force" in decoded or "force" in decoded.lower() or "mass" in decoded.lower()


def test_encode_does_not_prefix_bos_when_disabled(tmp_path) -> None:
    texts = ["Alpha beta gamma delta epsilon zeta."] * 80
    tokenizer = train_bpe_tokenizer(texts, vocab_size=300, min_frequency=1)
    out = save_litgpt_tokenizer_files(tokenizer, tmp_path / "tok", model_max_length=2048)
    tok = Tokenizer(out)
    assert tok.use_bos is False
    raw = tok.encode("Alpha beta gamma", bos=False, eos=False)
    ids = encode_prompt_batch(tok, "Alpha beta gamma", bos=False, eos=False)
    assert ids.shape == (1, int(raw.numel()))
    assert ids[0].tolist() == raw.to(dtype=torch.long).tolist()
