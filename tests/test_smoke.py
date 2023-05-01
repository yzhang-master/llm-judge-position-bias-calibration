from judge_bias import Answer, PairExample, ScriptedPairwiseJudge, cohen_kappa


def test_position_bias_changes_swapped_prediction() -> None:
    example = PairExample(
        example_id="smoke",
        question="Which answer is better?",
        answer_a=Answer("correct concise answer", quality=1.0),
        answer_b=Answer("incorrect answer", quality=0.8),
        reference="A",
    )
    judge = ScriptedPairwiseJudge(position_bias=0.5)

    assert judge.predict(example) == "A"
    assert judge.predict(example.swapped()) == "A"
    assert cohen_kappa(["A", "B"], ["A", "A"]) == 0.0
