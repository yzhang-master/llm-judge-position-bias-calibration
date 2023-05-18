from collections.abc import Callable

from judge_bias.calibration import Calibration
from judge_bias.judge import PairExample, ScriptedPairwiseJudge
from judge_bias.metrics import accuracy, cohen_kappa

Predictor = Callable[[PairExample], str]


def position_flip_rate(examples: list[PairExample], predictor: Predictor) -> float:
    if not examples:
        raise ValueError("at least one example is required")
    return sum(
        predictor(example) != predictor(example.swapped()) for example in examples
    ) / len(examples)


def longer_answer_win_rate(examples: list[PairExample], predicted: list[str]) -> float:
    eligible = [
        (example, prediction)
        for example, prediction in zip(examples, predicted, strict=True)
        if prediction != "tie"
        and example.answer_a.word_count != example.answer_b.word_count
    ]
    if not eligible:
        return 0.0
    wins = 0
    for example, prediction in eligible:
        longer = "A" if example.answer_a.word_count > example.answer_b.word_count else "B"
        wins += prediction == longer
    return wins / len(eligible)


def self_preference_win_rate(
    examples: list[PairExample],
    predicted: list[str],
    model_name: str,
) -> float:
    eligible = []
    for example, prediction in zip(examples, predicted, strict=True):
        self_a = example.answer_a.author == model_name
        self_b = example.answer_b.author == model_name
        if self_a != self_b and prediction != "tie":
            eligible.append((self_a, prediction))
    if not eligible:
        return 0.0
    return sum(prediction == ("A" if self_a else "B") for self_a, prediction in eligible) / len(
        eligible
    )


def classification_metrics(
    examples: list[PairExample], predicted: list[str]
) -> dict[str, float]:
    reference = [example.reference for example in examples]
    return {
        "accuracy": accuracy(reference, predicted),
        "cohen_kappa": cohen_kappa(reference, predicted),
    }


def summarize_predictions(
    judge: ScriptedPairwiseJudge,
    examples: list[PairExample],
    predictor: Predictor,
    predicted: list[str],
) -> dict[str, float]:
    return {
        **classification_metrics(examples, predicted),
        "position_flip_rate": position_flip_rate(examples, predictor),
        "longer_answer_win_rate": longer_answer_win_rate(examples, predicted),
        "self_preference_win_rate": self_preference_win_rate(
            examples, predicted, judge.model_name
        ),
    }


def raw_predictor(judge: ScriptedPairwiseJudge) -> Predictor:
    return judge.predict


def swap_predictor(judge: ScriptedPairwiseJudge) -> Predictor:
    def predict(example: PairExample) -> str:
        from judge_bias.calibration import swap_margin

        return judge.predict_margin(swap_margin(judge, example))

    return predict


def calibrated_predictor(
    judge: ScriptedPairwiseJudge, calibration: Calibration
) -> Predictor:
    def predict(example: PairExample) -> str:
        return calibration.predict(judge, example)

    return predict
