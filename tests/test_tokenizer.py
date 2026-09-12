import pytest
from litgpt.tokenizer import Tokenizer

from arzlm.tokenizer.train import BOS, EOS, save_litgpt_tokenizer_files, train_bpe_tokenizer


def test_trained_tokenizer_vocab_is_32000() -> None:
    from arzlm.paths import TOKENIZER_DIR

    tok_json = TOKENIZER_DIR / "tokenizer.json"
    if not tok_json.is_file():
        pytest.skip("no trained tokenizer in repo")
    lit = Tokenizer(TOKENIZER_DIR)
    assert lit.eos_id is not None
    assert int(lit.vocab_size) == 32000
    sample = "naïve café 日本語 \\frac{a}{b}\n    def foo():\n        return 1\n"
    ids = lit.encode(sample, bos=False, eos=False)
    decoded = lit.decode(ids)
    assert len(ids) > 0
    assert isinstance(decoded, str)


def test_training_pieces_split_on_documents_not_mid_string() -> None:
    from arzlm.tokenizer.train import _MAX_PIECE, _training_pieces

    text = "alpha doc\n\nbeta doc\n\n"
    pieces = list(_training_pieces([text]))
    assert pieces == ["alpha doc", "beta doc"]
    long_para = ("word " * 20_000).strip()
    assert len(long_para) > _MAX_PIECE
    capped = list(_training_pieces([long_para]))
    assert capped
    assert all(len(p) <= _MAX_PIECE for p in capped)
    assert "".join(capped).replace(" ", "") == long_para.replace(" ", "")


def test_train_bpe_from_files(tmp_path) -> None:
    from arzlm.tokenizer.train import train_bpe_from_files

    path = tmp_path / "train.txt"
    path.write_text("\n\n".join(["Force equals mass times acceleration."] * 80), encoding="utf-8")
    trained = train_bpe_from_files([path], vocab_size=400, min_frequency=1)
    assert trained.get_vocab_size(with_added_tokens=True) <= 400


def test_train_bpe_from_files_long_paragraph(tmp_path) -> None:
    from arzlm.tokenizer.train import train_bpe_from_files

    path = tmp_path / "long.txt"
    path.write_text(("Force equals mass times acceleration. " * 4000).strip() + "\n\n", encoding="utf-8")
    trained = train_bpe_from_files([path], vocab_size=400, min_frequency=1)
    assert trained.get_vocab_size(with_added_tokens=True) <= 400


def test_tokenizer_roundtrip(tmp_path) -> None:
    texts = [
        "Photosynthesis converts light energy into chemical energy in chloroplasts.",
        "Force equals mass times acceleration.",
        "Prime numbers are integers greater than one.",
    ]
    tokenizer = train_bpe_tokenizer(texts * 50, vocab_size=400, min_frequency=1)
    out = save_litgpt_tokenizer_files(tokenizer, tmp_path / "tok")
    lit = Tokenizer(out)
    sample = texts[0]
    ids = lit.encode(sample, bos=False, eos=False)
    decoded = lit.decode(ids)
    cfg = (out / "tokenizer_config.json").read_text(encoding="utf-8")
    assert '"model_max_length": 2048' in cfg
    assert sample.replace(" ", "") in decoded.replace(" ", "") or decoded.startswith("Photosynthesis")
    # Explicit EOS must be append-only via litgpt.Tokenizer.encode(eos=True)
    with_eos = lit.encode(sample, bos=False, eos=True).tolist()
    without = ids.tolist()
    assert with_eos[-1] == lit.eos_id
    assert with_eos[:-1] == without or with_eos == without + [lit.eos_id]
    assert lit.token_to_id(BOS) == lit.bos_id
    assert lit.token_to_id(EOS) == lit.eos_id
    assert lit.eos_id is not None
