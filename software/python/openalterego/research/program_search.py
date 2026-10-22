"""Budgeted virtual-montage program search on recorded channels only."""

from __future__ import annotations

import itertools
import time
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from openalterego.research.virtual_montage import robust_recording_normalize


@dataclass(frozen=True, order=True)
class MontageProgram:
    channels: tuple[int, ...]
    normalization: str
    c: float = 1.0

    def __post_init__(self) -> None:
        if (len(self.channels) != 6 or len(set(self.channels)) != 6 or
                tuple(sorted(self.channels)) != self.channels or
                any(channel < 0 or channel >= 14 for channel in self.channels) or
                self.normalization not in ("none", "recording_robust") or
                self.c not in (0.1, 1.0, 10.0)):
            raise ValueError("invalid virtual montage program")

    @property
    def columns(self) -> list[int]:
        return [column for channel in self.channels
                for column in (channel * 2, channel * 2 + 1)]

    def record(self) -> dict:
        return asdict(self)


def program_count() -> int:
    return len(list(itertools.combinations(range(14), 6))) * 2 * 3


def normalize_batches(batches: list[tuple[np.ndarray, np.ndarray]],
                      normalization: str) -> tuple[np.ndarray, np.ndarray]:
    if normalization not in ("none", "recording_robust"):
        raise ValueError("unknown normalization")
    features = [robust_recording_normalize(x) if normalization == "recording_robust"
                else x for x, _ in batches]
    return np.concatenate(features), np.concatenate([y for _, y in batches])


def proxy_rank(train: tuple[np.ndarray, np.ndarray],
               validation: tuple[np.ndarray, np.ndarray],
               normalization: str, *, width: int = 6,
               batch_size: int = 64) -> list[tuple[MontageProgram, float]]:
    """Vectorized train-centroid score on validation; no classifier fit per subset."""
    x, y = train
    val_x, val_y = validation
    if (x.ndim != 2 or val_x.ndim != 2 or x.shape[1] != 28 or
            val_x.shape[1] != 28 or len(x) != len(y) or
            len(val_x) != len(val_y) or width != 6 or batch_size < 1):
        raise ValueError("expected aligned 14-channel event features")
    classes = np.unique(y)
    if not np.array_equal(classes, np.arange(9)) or not np.array_equal(
            np.unique(val_y), classes):
        raise ValueError("all nine event labels are required in train and validation")
    centers = np.stack([x[y == label].mean(axis=0) for label in classes])
    scale = np.maximum(x.std(axis=0), 0.1)
    delta = (val_x[:, None, :] - centers[None, :, :]) / scale[None, None, :]
    contribution = -(delta.reshape(len(val_x), len(classes), 14, 2) ** 2).sum(axis=3)
    subsets = np.asarray(list(itertools.combinations(range(14), width)), dtype=np.int8)
    scored = []
    for start in range(0, len(subsets), batch_size):
        block = subsets[start:start + batch_size]
        class_scores = contribution[:, :, block].sum(axis=-1)
        predicted = class_scores.argmax(axis=1)
        recall = np.stack([(predicted[val_y == label] == label).mean(axis=0)
                           for label in classes])
        for channels, score in zip(block, recall.mean(axis=0)):
            scored.append((MontageProgram(tuple(map(int, channels)),
                                           normalization), float(score)))
    return sorted(scored, key=lambda item: (-item[1], item[0].channels))


def ridge_proxy_rank_many(train: tuple[np.ndarray, np.ndarray],
                          validations: list[tuple[np.ndarray, np.ndarray]],
                          normalization: str, *, alpha: float = 10.0,
                          batch_size: int = 64) -> list[list[tuple[MontageProgram, float]]]:
    """Fit each subset once and score it on several recording batches."""
    x, y = train
    if (x.ndim != 2 or x.shape[1] != 28 or len(x) != len(y) or
            alpha <= 0 or batch_size < 1 or not validations or
            not np.array_equal(np.unique(y), np.arange(9)) or
            any(val_x.ndim != 2 or val_x.shape[1] != 28 or
                len(val_x) != len(val_y) or
                not np.array_equal(np.unique(val_y), np.arange(9))
                for val_x, val_y in validations)):
        raise ValueError("expected aligned, complete 14-channel event features")
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-6)
    design = np.column_stack(((x - mean) / scale, np.ones(len(x))))
    val_designs = [np.column_stack(((val_x - mean) / scale,
                                   np.ones(len(val_x))))
                   for val_x, _ in validations]
    class_counts = np.bincount(y, minlength=9)
    weights = len(y) / (9 * class_counts[y])
    onehot = np.eye(9)[y]
    gram = design.T @ (design * weights[:, None])
    cross = design.T @ (onehot * weights[:, None])
    subsets = np.asarray(list(itertools.combinations(range(14), 6)), dtype=np.int8)
    regularizer = np.eye(13) * alpha
    regularizer[-1, -1] = 0
    scored = [[] for _ in validations]
    for start in range(0, len(subsets), batch_size):
        block = subsets[start:start + batch_size]
        columns = np.column_stack((2 * block, 2 * block + 1)).reshape(len(block), 12)
        columns = np.column_stack((columns, np.full(len(block), 28)))
        local_gram = gram[columns[:, :, None], columns[:, None, :]]
        coefficients = np.linalg.solve(local_gram + regularizer, cross[columns])
        for index, ((_, val_y), val_design) in enumerate(zip(validations,
                                                              val_designs)):
            predictions = np.einsum("nbd,bde->nbe", val_design[:, columns],
                                    coefficients).argmax(axis=2)
            recall = np.stack([(predictions[val_y == label] == label).mean(axis=0)
                               for label in range(9)])
            for channels, score in zip(block, recall.mean(axis=0)):
                scored[index].append((MontageProgram(tuple(map(int, channels)),
                                                     normalization), float(score)))
    return [sorted(rows, key=lambda item: (-item[1], item[0].channels))
            for rows in scored]


def ridge_proxy_rank(train: tuple[np.ndarray, np.ndarray],
                     validation: tuple[np.ndarray, np.ndarray],
                     normalization: str, *, alpha: float = 10.0,
                     batch_size: int = 64) -> list[tuple[MontageProgram, float]]:
    """Solve all subset ridge classifiers from shared sufficient statistics."""
    return ridge_proxy_rank_many(train, [validation], normalization,
                                 alpha=alpha, batch_size=batch_size)[0]


def _fit_score(program: MontageProgram, train, validation) -> tuple[float, float, float]:
    train_x, train_y = train[program.normalization]
    val_x, val_y = validation[program.normalization]
    columns = program.columns
    started, cpu_started = time.perf_counter(), time.process_time()
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=program.c, max_iter=2000, class_weight="balanced", random_state=42))
    model.fit(train_x[:, columns], train_y)
    score = float(balanced_accuracy_score(val_y, model.predict(val_x[:, columns])))
    return score, time.perf_counter() - started, time.process_time() - cpu_started


def _select_pool(ranked: list[tuple[MontageProgram, float]], count: int,
                 rng: np.random.Generator) -> dict[str, list[MontageProgram]]:
    if count < 2 or count > len(ranked):
        raise ValueError("invalid stage-one budget")
    top = [program for program, _ in ranked[:count]]
    canonical = sorted(program for program, _ in ranked)
    random_indices = rng.choice(len(canonical), size=count, replace=False)
    random = [canonical[int(index)] for index in random_indices]
    return {"proxy": top, "random": random}


def search(train_batches, validation_batches, test_loader, *,
           stage_one_per_strategy: int = 16,
           finalists_per_strategy: int = 4,
           proxy_kind: str = "ridge",
           selection_mode: str = "one_way",
           seed: int = 42,
           on_selection: Callable[[dict], None] | None = None) -> dict:
    """Select on session 2, then call test_loader only after both choices freeze."""
    started, cpu_started = time.perf_counter(), time.process_time()
    data = {}
    for normalization in ("none", "recording_robust"):
        data[normalization] = (
            normalize_batches(train_batches, normalization),
            normalize_batches(validation_batches, normalization))
    train = {key: split[0] for key, split in data.items()}
    validation = {key: split[1] for key, split in data.items()}
    if finalists_per_strategy < 1 or finalists_per_strategy > stage_one_per_strategy:
        raise ValueError("invalid finalist budget")
    if proxy_kind not in ("centroid", "ridge"):
        raise ValueError("unknown proxy")
    if selection_mode not in ("one_way", "bidirectional"):
        raise ValueError("unknown selection mode")
    proxy = ridge_proxy_rank if proxy_kind == "ridge" else proxy_rank
    proxy_started, proxy_cpu_started = time.perf_counter(), time.process_time()
    proxy_rows = []
    for norm in data:
        forward = proxy(train[norm], validation[norm], norm)
        if selection_mode == "bidirectional":
            reverse = dict(proxy(validation[norm], train[norm], norm))
            proxy_rows.extend((program, min(score, reverse[program]))
                              for program, score in forward)
        else:
            proxy_rows.extend(forward)
    ranked = sorted(proxy_rows,
        key=lambda item: (-item[1], item[0].normalization, item[0].channels))
    proxy_seconds = time.perf_counter() - proxy_started
    proxy_cpu_seconds = time.process_time() - proxy_cpu_started
    pools = _select_pool(ranked, stage_one_per_strategy,
                         np.random.default_rng(seed))
    def evaluate(program):
        forward = _fit_score(program, train, validation)
        if selection_mode == "one_way":
            return forward[0], forward[1], forward[2], [forward[0]]
        reverse = _fit_score(program, validation, train)
        return (min(forward[0], reverse[0]), forward[1] + reverse[1],
                forward[2] + reverse[2], [forward[0], reverse[0]])

    strategies = {}
    for strategy, programs in pools.items():
        stage_one = []
        for program in programs:
            score, wall_s, cpu_s, directional = evaluate(program)
            stage_one.append({"program": program.record(),
                              "validation_balanced_accuracy": score,
                              "directional_balanced_accuracy": directional,
                              "fit_and_score_wall_s": wall_s,
                              "fit_and_score_cpu_s": cpu_s})
        best = sorted(stage_one, key=lambda row: -row["validation_balanced_accuracy"])
        stage_two = []
        for row in best[:finalists_per_strategy]:
            base = row["program"]
            for c in (0.1, 1.0, 10.0):
                program = MontageProgram(tuple(base["channels"]),
                                         base["normalization"], c)
                if c == 1.0:
                    score, wall_s, cpu_s = row["validation_balanced_accuracy"], 0.0, 0.0
                    directional = row["directional_balanced_accuracy"]
                    reused = True
                else:
                    score, wall_s, cpu_s, directional = evaluate(program)
                    reused = False
                stage_two.append({"program": program.record(),
                                  "validation_balanced_accuracy": score,
                                  "directional_balanced_accuracy": directional,
                                  "fit_and_score_wall_s": wall_s,
                                  "fit_and_score_cpu_s": cpu_s,
                                  "reused_stage_one": reused})
        chosen = max(stage_two, key=lambda row: row["validation_balanced_accuracy"])
        strategies[strategy] = {"stage_one": stage_one, "stage_two": stage_two,
                                "choice": chosen["program"],
                                "validation_balanced_accuracy": chosen["validation_balanced_accuracy"],
                                "full_fits_for_selection": (stage_one_per_strategy +
                                2 * finalists_per_strategy) * (2 if selection_mode ==
                                "bidirectional" else 1),
                                "directional_fits_per_program": 2 if selection_mode ==
                                "bidirectional" else 1}
    selection = {"schema": "openalterego.virtual-program-selection.v1",
                 "proxy_kind": proxy_kind, "selection_mode": selection_mode,
                 "seed": seed,
                 "stage_one_per_strategy": stage_one_per_strategy,
                 "finalists_per_strategy": finalists_per_strategy,
                 "choices": {name: {"program": row["choice"],
                                     "validation_balanced_accuracy": row[
                                         "validation_balanced_accuracy"]}
                             for name, row in strategies.items()}}
    if on_selection is not None:
        on_selection(selection)
    test_batches = test_loader()
    test = {normalization: normalize_batches(test_batches, normalization)
            for normalization in data}
    final_train = ({norm: normalize_batches(train_batches + validation_batches,
                                            norm) for norm in data}
                   if selection_mode == "bidirectional" else train)
    for outcome in strategies.values():
        chosen = outcome["choice"]
        program = MontageProgram(tuple(chosen["channels"]),
                                 chosen["normalization"], chosen["c"])
        train_x, train_y = final_train[program.normalization]
        test_x, test_y = test[program.normalization]
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=program.c, max_iter=2000, class_weight="balanced", random_state=42))
        model.fit(train_x[:, program.columns], train_y)
        outcome["test_balanced_accuracy"] = float(balanced_accuracy_score(
            test_y, model.predict(test_x[:, program.columns])))
    return {"schema": "openalterego.virtual-program-search.v5",
            "program_count": program_count(),
            "unique_proxy_structures": len(ranked),
            "proxy_kind": proxy_kind,
            "split": ("cross-fit sessions 1 and 2 in both directions; refit on both; "
                      "reveal chosen designs on session 3" if selection_mode ==
                      "bidirectional" else "train session 1; select session 2; "
                      "reveal chosen designs on session 3"),
            "scope": "Recorded channel subsets only; no unobserved electrode or device physics prediction.",
            "seed": seed, "selection_mode": selection_mode,
            "proxy_seconds": proxy_seconds,
            "proxy_cpu_seconds": proxy_cpu_seconds,
            "total_seconds": time.perf_counter() - started,
            "total_cpu_seconds": time.process_time() - cpu_started,
            "stage_one_per_strategy": stage_one_per_strategy,
            "finalists_per_strategy": finalists_per_strategy,
            "selection": selection,
            "strategies": strategies}
