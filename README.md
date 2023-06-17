# llm-judge-position-bias-calibration

> **Era:** 2023–2024 · Senior AI Engineer focus area (training and evaluation). Technique from 2023, reimplemented in 2026.

A dependency-free, deterministic harness for measuring pairwise judge position bias, reducing it with swap augmentation, and calibrating verbosity and self-preference terms. Cohen's kappa is computed against a small local set of manually labeled examples. **No LLM API, downloaded labels, or network access is used.**

## Problem

A pairwise judge can prefer whichever answer appears first, the longer answer, or an answer attributed to itself. Reporting accuracy alone hides these failure modes. A useful evaluation must swap answer order, measure the remaining descriptive biases, and compare predictions with reference labels without selecting calibration parameters on the test set.

## Method

Each `Answer` has text, an author tag, and a numeric `quality` visible only to the scripted judge. For answer positions zero and one, the judge computes

```text
score(answer, position) = quality
                        + position_bias × (+1 for first, -1 for second)
                        + verbosity_bias × log(1 + word_count)
                        + self_preference_bias × 1[author == judge_name]
```

The decision margin is `score(A) - score(B)`; margins inside ±0.12 are ties. These coefficients are deliberately injected biases, not estimates of how a real LLM behaves.

**Swap augmentation.** If `m` is the original margin and `m_swap` is the margin after exchanging the answers, evaluation uses

```text
m_symmetric = (m - m_swap) / 2
```

The position term cancels exactly. Swapped predictions are mapped back to the original answer identities before computing the position flip rate.

**Calibration.** `fit_calibration` uses only the 12 training examples. It grid-searches nonnegative verbosity and self-preference penalties, maximizing training Cohen's kappa and then accuracy. The selected penalties are subtracted from the symmetric margin. The 12 test examples remain untouched during fitting.

**Metrics.** Cohen's kappa is `κ = (p_observed - p_chance) / (1 - p_chance)`, where chance agreement comes from the reference and prediction marginals. The report also includes accuracy, position flip rate, longer-answer win rate among unequal-length decisive pairs, and self-win rate among decisive pairs with exactly one self-attributed answer. The last two are descriptive associations, not causal bias estimates.

## Results

Actual local CPU run: `PYTHONPATH=src python scripts/evaluate.py`. Full predictions and machine-readable values are in `results/local_metrics.json`.

| Held-out method (12 examples) | Accuracy | Cohen's κ | Position flips | Longer-answer wins | Self wins |
|---|---:|---:|---:|---:|---:|
| Raw, original order | 58.33% | 0.0000 | 83.33% | 18.18% | 57.14% |
| Swap augmented | 75.00% | 0.5500 | 0.00% | 18.18% | 42.86% |
| Swap + train-only calibration | 83.33% | 0.7143 | 0.00% | 20.00% | 33.33% |

Calibration selected a verbosity penalty of 0.15 and self-preference penalty of 0.70, reaching training κ = 1.0. Longer-answer win rate rises slightly after calibration because this small, quality-confounded descriptive statistic is not the fitting objective; it must not be read as proof of increased verbosity bias.

The end-to-end smoke is also reproducible:

```text
{'original': 'A', 'swapped': 'A'}
```

Here the raw judge chooses whichever answer occupies position A, demonstrating the intended failure.

## Reproduce

Python 3.10+ is sufficient; the package has no runtime dependencies.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
pytest
PYTHONPATH=src python scripts/smoke.py
PYTHONPATH=src python scripts/evaluate.py
```

`scripts/evaluate.py` rewrites `results/local_metrics.json`. Tests cover kappa arithmetic and invalid inputs, exact position cancellation, swapped-label invariance, calibration behavior, descriptive metric denominators, and the real local dataset.

## Limitations

- The 24 examples are simplistic toy questions. Their reference labels were manually authored by the repository author; they are **not** an independent human-agreement study, and no inter-annotator agreement is claimed.
- Twelve training and twelve test labels are far too few for reliable LLM-judge conclusions. Kappa can change sharply when one prediction changes.
- The scripted judge sees ground-truth-like numeric quality scores. This makes the arithmetic testable but does not model language understanding or a real LLM's token-level behavior.
- Self-preference uses explicit author tags, and verbosity uses whitespace word counts. Real judges may use subtler cues.
- Grid-search calibration optimizes coarse labels, so the true bias coefficients are not uniquely identifiable. Results do not establish causal debiasing or generalization to other datasets.
- Swap augmentation doubles judge evaluations and removes only antisymmetric position effects. It cannot repair an incorrect underlying quality assessment.

## References

- L. Zheng et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* 2023. arXiv:2306.05685
- P. Wang et al. *Large Language Models are not Fair Evaluators.* 2023. arXiv:2305.17926
- J. Cohen. *A Coefficient of Agreement for Nominal Scales.* Educational and Psychological Measurement, 1960

## License

MIT
