"""Classification metrics for session ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.metrics import f1_score


@dataclass
class SplitMetrics:
    accuracy: float
    macro_f1: float
    n: int
    per_class_recall: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "accuracy": round(float(self.accuracy), 4),
            "macro_f1": round(float(self.macro_f1), 4),
            "n": int(self.n),
            "per_class_recall": {k: round(float(v), 4) for k, v in self.per_class_recall.items()},
        }


def split_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: List[str],
) -> SplitMetrics:
    """Accuracy + macro-F1 + per-class recall."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n = int(len(y_true))
    acc = float(np.mean(y_true == y_pred)) if n else 0.0
    macro = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0, labels=list(range(len(labels))))
    )
    recall: Dict[str, float] = {}
    for i, lab in enumerate(labels):
        mask = y_true == i
        recall[lab] = float(np.mean(y_pred[mask] == i)) if np.any(mask) else 0.0
    return SplitMetrics(accuracy=acc, macro_f1=macro, n=n, per_class_recall=recall)
