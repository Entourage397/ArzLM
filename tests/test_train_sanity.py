import pytest
import torch

from arzlm.training.loop import run_training
from tests.helpers import cpu_experiment, gpu_available


def test_deterministic_mini_run(tmp_path) -> None:
    losses = []
    for i in range(2):
        exp = cpu_experiment(tmp_path / f"run{i}", seq_length=32, max_steps=2, seed=42)
        result = run_training(exp)
        assert result["finite"] is True
        losses.append(result["loss"])
    assert losses[0] == pytest.approx(losses[1], rel=0, abs=1e-6)


def test_no_nan_short_run(tmp_path) -> None:
    exp = cpu_experiment(tmp_path, seq_length=32, max_steps=3, seed=0)
    if gpu_available():
        exp.precision = "bf16-true"
        exp.compile = "false"
    result = run_training(exp)
    assert result["finite"] is True
    assert result["loss"] == result["loss"]  # not NaN
    assert abs(result["loss"]) != float("inf")


@pytest.mark.gpu
def test_gpu_smoke_if_available(tmp_path) -> None:
    if not gpu_available():
        pytest.skip("CUDA not available")
    exp = cpu_experiment(tmp_path, seq_length=32, max_steps=2, seed=1)
    exp.precision = "bf16-true"
    result = run_training(exp)
    assert result["finite"] is True
    assert torch.cuda.is_available()
