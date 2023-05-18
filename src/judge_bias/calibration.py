import math
from collections.abc import Iterable
from dataclasses import dataclass

from judge_bias.judge import PairExample, ScriptedPairwiseJudge
from judge_bias.metrics import accuracy, cohen_kappa


def swap_margin(judge: ScriptedPairwiseJudge, example: PairExample) -> float:
    return (judge.margin(example) - judge.margin(example.swapped())) / 2.0


def verbosity_delta(example: PairExample) -> float:
    return math.log1p(example.answer_a.word_count) - math.log1p(
        example.answer_b.word_count
    )


def self_preference_delta(example: PairExample, model_name: str) -> float:
    return float(example.answer_a.author == model_name) - float(
        example.answer_b.author == model_name
    )


@dataclass(frozen=True)
class Calibration:
    verbosity_penalty: float
    self_preference_penalty: float
    train_kappa: float

    def margin(
        self,
        judge: ScriptedPairwiseJudge,
        example: PairExample,
    ) -> float:
        return (
            swap_margin(judge, example)
            - self.verbosity_penalty * verbosity_delta(example)
            - self.self_preference_penalty
            * self_preference_delta(example, judge.model_name)
        )

    def predict(
        self,
        judge: ScriptedPairwiseJudge,
        example: PairExample,
    ) -> str:
        return judge.predict_margin(self.margin(judge, example))


def fit_calibration(
    judge: ScriptedPairwiseJudge,
    train_examples: list[PairExample],
    *,
    verbosity_grid: Iterable[float] = tuple(index / 20 for index in range(13)),
    self_preference_grid: Iterable[float] = tuple(index / 10 for index in range(13)),
) -> Calibration:
    if not train_examples:
        raise ValueError("calibration requires training examples")
    reference = [example.reference for example in train_examples]
    margins = [swap_margin(judge, example) for example in train_examples]
    verbosity = [verbosity_delta(example) for example in train_examples]
    self_preference = [
        self_preference_delta(example, judge.model_name) for example in train_examples
    ]
    best: tuple[float, float, float, float] | None = None
    best_calibration: Calibration | None = None
    for verbosity_penalty in verbosity_grid:
        for self_penalty in self_preference_grid:
            predicted = [
                judge.predict_margin(
                    margin - verbosity_penalty * verbose - self_penalty * self_bias
                )
                for margin, verbose, self_bias in zip(
                    margins, verbosity, self_preference, strict=True
                )
            ]
            kappa = cohen_kappa(reference, predicted)
            candidate = (
                kappa,
                accuracy(reference, predicted),
                -abs(verbosity_penalty),
                -abs(self_penalty),
            )
            if best is None or candidate > best:
                best = candidate
                best_calibration = Calibration(
                    verbosity_penalty=verbosity_penalty,
                    self_preference_penalty=self_penalty,
                    train_kappa=kappa,
                )
    if best_calibration is None:
        raise ValueError("calibration grids must not be empty")
    return best_calibration
