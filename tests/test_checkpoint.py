import torch

from arzlm.training.loop import make_dataloaders, run_training
from tests.helpers import cpu_experiment, write_random_packed


def test_checkpoint_resumes_data_position(tmp_path) -> None:
    """The next batch after resume is the next deterministic window, not a reshuffle from 0."""
    data_dir = write_random_packed(tmp_path, n_train=33 * 32, n_val=33 * 8, seq_length=32, seed=3)
    loader0, _, _ = make_dataloaders(
        data_dir, seq_length=32, batch_size=1, num_workers=0, seed=3, n_items=8, start_index=0
    )
    batches = [batch.clone() for batch in loader0]
    loader1, _, _ = make_dataloaders(
        data_dir, seq_length=32, batch_size=1, num_workers=0, seed=3, n_items=8, start_index=2
    )
    resumed = next(iter(loader1))
    assert torch.equal(resumed, batches[2])


def test_checkpoint_save_and_resume(tmp_path) -> None:
    exp = cpu_experiment(tmp_path / "a", seq_length=32, max_steps=2, seed=7)
    result = run_training(exp)
    ckpt = exp.out_dir / "final" / "lit_model.pth"
    assert ckpt.is_file()
    assert (ckpt.parent / "model_config.yaml").is_file()

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "model" in blob
    assert "optimizer" in blob
    assert blob["step_count"] == 2

    exp2 = cpu_experiment(tmp_path / "b", seq_length=32, max_steps=3, seed=7)
    exp2.out_dir = exp.out_dir
    exp2.data_dir = exp.data_dir
    exp2.tokenizer_dir = exp.tokenizer_dir
    exp2.resume = ckpt
    exp2.eval.final_validation = False
    resumed = run_training(exp2)
    assert resumed["steps"] >= 2
    assert resumed["finite"] is True
