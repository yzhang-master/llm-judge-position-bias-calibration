from judge_bias import Answer, PairExample, ScriptedPairwiseJudge

example = PairExample(
    example_id="smoke",
    question="Which answer is better?",
    answer_a=Answer("correct concise answer", quality=1.0),
    answer_b=Answer("incorrect answer", quality=0.8),
    reference="A",
)
judge = ScriptedPairwiseJudge(position_bias=0.5)
print({"original": judge.predict(example), "swapped": judge.predict(example.swapped())})
