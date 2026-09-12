from arzlm.data.prepare import IsolatedEncoder, _sanitize_document

from tests.helpers import write_mini_tokenizer


def test_sanitize_replaces_unpaired_surrogate() -> None:
    raw = "hello \ud800 world"
    out = _sanitize_document(raw)
    assert "\ud800" not in out
    assert "hello" in out
    assert _sanitize_document("   ").strip() == ""


def test_isolated_encoder_roundtrip_and_empty(tmp_path) -> None:
    tok_dir = write_mini_tokenizer(tmp_path)
    encoder = IsolatedEncoder(tok_dir, timeout_s=60.0)
    try:
        ids = encoder.encode("photosynthesis converts light energy into chemical energy.")
        assert ids is not None
        assert len(ids) > 0
        assert encoder.encode("   ") == []
        assert encoder.encode("") == []
        assert encoder.n_worker_restarts == 0
    finally:
        encoder.close()
