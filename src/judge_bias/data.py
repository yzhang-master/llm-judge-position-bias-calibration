import json
from pathlib import Path

from judge_bias.judge import Answer, PairExample
from judge_bias.metrics import LABELS


def load_examples(path: str | Path) -> list[PairExample]:
    examples = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record["reference"] not in LABELS:
                raise ValueError(f"invalid reference on line {line_number}")
            if record["split"] not in {"train", "test"}:
                raise ValueError(f"invalid split on line {line_number}")
            examples.append(
                PairExample(
                    example_id=record["id"],
                    question=record["question"],
                    answer_a=Answer(**record["answer_a"]),
                    answer_b=Answer(**record["answer_b"]),
                    reference=record["reference"],
                    split=record["split"],
                )
            )
    if not examples:
        raise ValueError("dataset is empty")
    ids = [example.example_id for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("example ids must be unique")
    return examples


def split_examples(
    examples: list[PairExample],
) -> tuple[list[PairExample], list[PairExample]]:
    train = [example for example in examples if example.split == "train"]
    test = [example for example in examples if example.split == "test"]
    if not train or not test:
        raise ValueError("both train and test examples are required")
    return train, test
