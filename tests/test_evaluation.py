from pathlib import Path

import pytest

from judge_bias.calibration import fit_calibration
from judge_bias.data import load_examples, split_examples
from judge_bias.evaluation import (
    calibrated_predictor,
    longer_answer_win_rate,
    self_preference_win_rate,
    summarize_predictions,
    swap_predictor,
)
from judge_bias.judge import ScriptedPairwiseJudge

DATA = Path(__file__).parents[1] / "data" / "toy_preferences.jsonl"


def test_local_dataset_calibrates_without_test_labels() -> None:
    train, test = split_examples(load_examples(DATA))
    judge = ScriptedPairwiseJudge(
        position_bias=0.65,
        verbosity_bias=0.25,
        self_preference_bias=0.7,
        tie_threshold=0.12,
    )
    calibration = fit_calibration(judge, train)
    predictor = calibrated_predictor(judge, calibration)
    predicted = [predictor(example) for example in test]
    result = summarize_predictions(judge, test, predictor, predicted)

    assert result["position_flip_rate"] == 0.0
    assert result["cohen_kappa"] >= 0.7
    assert calibration.verbosity_penalty > 0
    assert calibration.self_preference_penalty > 0


def test_descriptive_bias_rates_use_eligible_decisions() -> None:
    train, _ = split_examples(load_examples(DATA))
    predicted = ["A"] * len(train)

    assert 0 <= longer_answer_win_rate(train, predicted) <= 1
    assert self_preference_win_rate(train, predicted, "scripted-judge") == pytest.approx(
        4 / 7
    )


def test_swap_predictor_is_position_invariant() -> None:
    _, test = split_examples(load_examples(DATA))
    judge = ScriptedPairwiseJudge(position_bias=1.0)
    predictor = swap_predictor(judge)

    for example in test:
        original = predictor(example)
        swapped = predictor(example.swapped())
        expected = {"A": "B", "B": "A", "tie": "tie"}[original]
        assert swapped == expected
