import pytest

from judge_bias.calibration import fit_calibration, swap_margin
from judge_bias.evaluation import position_flip_rate, swap_predictor
from judge_bias.judge import Answer, PairExample, ScriptedPairwiseJudge


def make_example() -> PairExample:
    return PairExample(
        example_id="pair",
        question="Which is correct?",
        answer_a=Answer("short correct answer", quality=1.0, author="other"),
        answer_b=Answer(
            "a much longer answer which is incorrect and repeats filler words",
            quality=0.5,
            author="scripted-judge",
        ),
        reference="A",
    )


def test_swap_margin_cancels_only_the_position_term() -> None:
    judge = ScriptedPairwiseJudge(
        position_bias=2.0, verbosity_bias=0.0, self_preference_bias=0.0
    )
    example = make_example()

    assert judge.margin(example) == pytest.approx(4.5)
    assert swap_margin(judge, example) == pytest.approx(0.5)
    assert position_flip_rate([example], judge.predict) == 1.0
    assert position_flip_rate([example], swap_predictor(judge)) == 0.0


def test_calibration_finds_low_bias_perfect_training_predictions() -> None:
    judge = ScriptedPairwiseJudge(
        position_bias=0.8,
        verbosity_bias=0.4,
        self_preference_bias=0.7,
        tie_threshold=0.05,
    )
    examples = [
        PairExample(
            example_id=str(index),
            question="q",
            answer_a=Answer("a " * a_words, quality=a_quality, author=a_author),
            answer_b=Answer("b " * 2, quality=0.0, author="other"),
            reference=reference,
        )
        for index, (a_words, a_quality, a_author, reference) in enumerate(
            [
                (20, 0.2, "other", "A"),
                (2, -0.4, "scripted-judge", "B"),
                (15, -0.2, "scripted-judge", "B"),
                (3, 0.5, "other", "A"),
            ]
        )
    ]

    calibration = fit_calibration(judge, examples)

    assert calibration.train_kappa == 1.0
    assert calibration.verbosity_penalty >= 0
    assert calibration.self_preference_penalty >= 0
    assert [
        calibration.predict(judge, example) for example in examples
    ] == [example.reference for example in examples]
