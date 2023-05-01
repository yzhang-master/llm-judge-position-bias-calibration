# llm-judge-position-bias-calibration

> **Era:** 2023–2024 · Senior AI Engineer focus area (training and evaluation). Technique from 2023, reimplemented in 2026.

Deterministic pairwise judging with explicit position, verbosity, and self-preference biases. This scaffold includes a meaningful position-swap smoke test; calibration, local evaluation data, and measured results are implemented in subsequent commits.

## Reproduce

```bash
pip install -e ".[dev]"
ruff check .
pytest
PYTHONPATH=src python scripts/smoke.py
```

No external model or network calls are used.

## License

MIT
