import pytest

from judge_bias.metrics import accuracy, cohen_kappa


def test_cohen_kappa_removes_chance_agreement() -> None:
    reference = ["A", "A", "B", "tie"]
    predicted = ["A", "B", "B", "tie"]

    assert accuracy(reference, predicted) == 0.75
    assert cohen_kappa(reference, predicted) == pytest.approx(7 / 11)


def test_kappa_handles_perfect_and_constant_predictions() -> None:
    assert cohen_kappa(["A", "B", "tie"], ["A", "B", "tie"]) == 1.0
    assert cohen_kappa(["A", "A"], ["A", "A"]) == 1.0
    assert cohen_kappa(["A", "B"], ["A", "A"]) == 0.0


def test_kappa_rejects_invalid_or_empty_labels() -> None:
    with pytest.raises(ValueError, match="empty|at least"):
        cohen_kappa([], [])
    with pytest.raises(ValueError, match="equal lengths"):
        cohen_kappa(["A"], [])
    with pytest.raises(ValueError, match="invalid labels"):
        cohen_kappa(["A"], ["C"])
