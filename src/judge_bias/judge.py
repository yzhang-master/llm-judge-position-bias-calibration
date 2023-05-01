import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Answer:
    text: str
    quality: float
    author: str = "unknown"

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass(frozen=True)
class PairExample:
    example_id: str
    question: str
    answer_a: Answer
    answer_b: Answer
    reference: str
    split: str = "test"

    def swapped(self) -> "PairExample":
        reference = {"A": "B", "B": "A", "tie": "tie"}[self.reference]
        return PairExample(
            example_id=self.example_id,
            question=self.question,
            answer_a=self.answer_b,
            answer_b=self.answer_a,
            reference=reference,
            split=self.split,
        )


class ScriptedPairwiseJudge:
    def __init__(
        self,
        *,
        model_name: str = "scripted-judge",
        position_bias: float = 0.0,
        verbosity_bias: float = 0.0,
        self_preference_bias: float = 0.0,
        tie_threshold: float = 0.1,
    ) -> None:
        if tie_threshold < 0:
            raise ValueError("tie_threshold must be nonnegative")
        self.model_name = model_name
        self.position_bias = position_bias
        self.verbosity_bias = verbosity_bias
        self.self_preference_bias = self_preference_bias
        self.tie_threshold = tie_threshold

    def score(self, answer: Answer, position: int) -> float:
        if position not in (0, 1):
            raise ValueError("position must be 0 or 1")
        position_term = self.position_bias * (1 if position == 0 else -1)
        verbosity_term = self.verbosity_bias * math.log1p(answer.word_count)
        self_term = self.self_preference_bias * (answer.author == self.model_name)
        return answer.quality + position_term + verbosity_term + self_term

    def margin(self, example: PairExample) -> float:
        return self.score(example.answer_a, 0) - self.score(example.answer_b, 1)

    def predict_margin(self, margin: float) -> str:
        if margin > self.tie_threshold:
            return "A"
        if margin < -self.tie_threshold:
            return "B"
        return "tie"

    def predict(self, example: PairExample) -> str:
        return self.predict_margin(self.margin(example))
