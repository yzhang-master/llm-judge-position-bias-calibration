LABELS = frozenset({"A", "B", "tie"})


def _validate_labels(reference: list[str], predicted: list[str]) -> None:
    if len(reference) != len(predicted):
        raise ValueError("reference and predicted must have equal lengths")
    if not reference:
        raise ValueError("at least one label is required")
    invalid = (set(reference) | set(predicted)) - LABELS
    if invalid:
        raise ValueError(f"invalid labels: {sorted(invalid)}")


def accuracy(reference: list[str], predicted: list[str]) -> float:
    _validate_labels(reference, predicted)
    return sum(a == b for a, b in zip(reference, predicted, strict=True)) / len(reference)


def cohen_kappa(reference: list[str], predicted: list[str]) -> float:
    _validate_labels(reference, predicted)
    labels = sorted(LABELS)
    observed = accuracy(reference, predicted)
    expected = sum(
        reference.count(label) * predicted.count(label) for label in labels
    ) / len(reference) ** 2
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)
