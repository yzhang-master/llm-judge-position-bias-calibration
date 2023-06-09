import argparse
import json
from pathlib import Path

from judge_bias import (
    ScriptedPairwiseJudge,
    calibrated_predictor,
    fit_calibration,
    load_examples,
    raw_predictor,
    split_examples,
    summarize_predictions,
    swap_predictor,
)


def evaluate(data_path: Path) -> dict:
    train, test = split_examples(load_examples(data_path))
    judge = ScriptedPairwiseJudge(
        model_name="scripted-judge",
        position_bias=0.65,
        verbosity_bias=0.25,
        self_preference_bias=0.7,
        tie_threshold=0.12,
    )
    calibration = fit_calibration(judge, train)
    predictors = {
        "raw": raw_predictor(judge),
        "swap_augmented": swap_predictor(judge),
        "swap_plus_calibrated": calibrated_predictor(judge, calibration),
    }
    methods = {}
    for name, predictor in predictors.items():
        predicted = [predictor(example) for example in test]
        methods[name] = {
            **summarize_predictions(judge, test, predictor, predicted),
            "predictions": predicted,
        }
    return {
        "dataset": str(data_path),
        "train_examples": len(train),
        "test_examples": len(test),
        "label_source": "manual toy labels by repository author",
        "judge": {
            "position_bias": judge.position_bias,
            "verbosity_bias": judge.verbosity_bias,
            "self_preference_bias": judge.self_preference_bias,
            "tie_threshold": judge.tie_threshold,
        },
        "calibration": {
            "verbosity_penalty": calibration.verbosity_penalty,
            "self_preference_penalty": calibration.self_preference_penalty,
            "train_kappa": calibration.train_kappa,
        },
        "reference": [example.reference for example in test],
        "methods": methods,
    }


def main() -> None:
    root = Path(__file__).parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=root / "data" / "toy_preferences.jsonl")
    parser.add_argument("--output", type=Path, default=root / "results" / "local_metrics.json")
    args = parser.parse_args()

    result = evaluate(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
