"""Bootstrap confidence intervals for classification metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import f1_score


@dataclass
class BootstrapCI:
    mean: float
    low: float
    high: float
    n_bootstrap: int

    def to_dict(self) -> dict:
        return {
            "mean": round(float(self.mean), 4),
            "ci95_low": round(float(self.low), 4),
            "ci95_high": round(float(self.high), 4),
            "n_bootstrap": int(self.n_bootstrap),
        }


def bootstrap_accuracy_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> BootstrapCI:
    rng = np.random.default_rng(int(seed))
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n = len(y_true)
    if n == 0:
        return BootstrapCI(0.0, 0.0, 0.0, n_bootstrap)
    samples = []
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        samples.append(float(np.mean(y_true[idx] == y_pred[idx])))
    arr = np.asarray(samples, dtype=np.float64)
    return BootstrapCI(
        mean=float(np.mean(arr)),
        low=float(np.percentile(arr, 2.5)),
        high=float(np.percentile(arr, 97.5)),
        n_bootstrap=int(n_bootstrap),
    )


def bootstrap_macro_f1_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_classes: int,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> BootstrapCI:
    rng = np.random.default_rng(int(seed))
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n = len(y_true)
    labels = list(range(int(n_classes)))
    if n == 0:
        return BootstrapCI(0.0, 0.0, 0.0, n_bootstrap)
    samples = []
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        f1 = f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0, labels=labels)
        samples.append(float(f1))
    arr = np.asarray(samples, dtype=np.float64)
    return BootstrapCI(
        mean=float(np.mean(arr)),
        low=float(np.percentile(arr, 2.5)),
        high=float(np.percentile(arr, 97.5)),
        n_bootstrap=int(n_bootstrap),
    )


def bootstrap_rate_ci(
    scores: List[float],
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> BootstrapCI:
    """Bootstrap CI for scalar rates (PER, WER, accuracy)."""
    rng = np.random.default_rng(int(seed))
    arr_in = np.asarray(scores, dtype=np.float64)
    n = len(arr_in)
    if n == 0:
        return BootstrapCI(0.0, 0.0, 0.0, n_bootstrap)
    samples = [float(np.mean(arr_in[rng.integers(0, n, size=n)])) for _ in range(int(n_bootstrap))]
    arr = np.asarray(samples, dtype=np.float64)
    return BootstrapCI(
        mean=float(np.mean(arr)),
        low=float(np.percentile(arr, 2.5)),
        high=float(np.percentile(arr, 97.5)),
        n_bootstrap=int(n_bootstrap),
    )


def summarize_ctc_multiseed(
    runs: List[Dict[str, object]],
    *,
    split_key: str = "test_beam",
    n_bootstrap: int = 2000,
) -> Dict[str, object]:
    """Aggregate multi-seed CTC runs with bootstrap CI on pooled word correctness."""
    pers = [float(r[split_key]["per"]) for r in runs]  # type: ignore[index]
    wers = [float(r[split_key]["wer"]) for r in runs]  # type: ignore[index]
    hyp_pool: List[str] = []
    ref_pool: List[str] = []
    for r in runs:
        hyp_pool.extend(r.get("hyp_words_test", r.get("hyp_words_val", [])))  # type: ignore[arg-type]
        ref_pool.extend(r.get("ref_words_test", r.get("ref_words_val", [])))  # type: ignore[arg-type]
    correct = [1.0 if h == r else 0.0 for h, r in zip(hyp_pool, ref_pool)]
    return {
        "n_seeds": len(runs),
        "seeds": [int(r["seed"]) for r in runs],  # type: ignore[index]
        "split_key": split_key,
        f"{split_key}_per_mean": round(float(np.mean(pers)), 4),
        f"{split_key}_per_std": round(float(np.std(pers)), 4),
        f"{split_key}_wer_mean": round(float(np.mean(wers)), 4),
        f"{split_key}_wer_std": round(float(np.std(wers)), 4),
        "bootstrap_word_acc": bootstrap_rate_ci(correct, n_bootstrap=n_bootstrap).to_dict(),
        "runs": [{k: v for k, v in r.items() if not str(k).startswith("hyp_words") and not str(k).startswith("ref_words")} for r in runs],
    }


def summarize_multiseed(
    runs: List[Dict[str, object]],
    *,
    n_bootstrap: int = 2000,
) -> Dict[str, object]:
    """Aggregate multi-seed runs + bootstrap CI on pooled val predictions."""
    accs = [float(r["val"]["accuracy"]) for r in runs]  # type: ignore[index]
    f1s = [float(r["val"]["macro_f1"]) for r in runs]  # type: ignore[index]
    y_pool: List[int] = []
    p_pool: List[int] = []
    for r in runs:
        y_pool.extend(r["y_val"])  # type: ignore[arg-type]
        p_pool.extend(r["p_val"])  # type: ignore[arg-type]
    y_arr = np.asarray(y_pool, dtype=np.int64)
    p_arr = np.asarray(p_pool, dtype=np.int64)
    n_classes = int(runs[0]["n_classes"])  # type: ignore[index]
    return {
        "n_seeds": len(runs),
        "seeds": [int(r["seed"]) for r in runs],  # type: ignore[index]
        "val_acc_mean": round(float(np.mean(accs)), 4),
        "val_acc_std": round(float(np.std(accs)), 4),
        "val_f1_mean": round(float(np.mean(f1s)), 4),
        "val_f1_std": round(float(np.std(f1s)), 4),
        "bootstrap_acc": bootstrap_accuracy_ci(y_arr, p_arr, n_bootstrap=n_bootstrap).to_dict(),
        "bootstrap_macro_f1": bootstrap_macro_f1_ci(
            y_arr, p_arr, n_classes=n_classes, n_bootstrap=n_bootstrap
        ).to_dict(),
        "runs": [{k: v for k, v in r.items() if k not in ("y_val", "p_val")} for r in runs],
    }
