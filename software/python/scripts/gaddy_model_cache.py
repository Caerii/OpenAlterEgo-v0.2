"""Local, content-addressed prediction cache for fixed Gaddy model fits."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import sklearn


def _update_array(digest, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values)
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(memoryview(array).cast("B"))


def prediction_cache_path(name: str, model, training_x: np.ndarray,
                          training_y: np.ndarray, development_x: list[np.ndarray],
                          method: str, cache_dir: Path) -> Path:
    root = Path(__file__).resolve().parents[3]
    resolved_cache = cache_dir.resolve()
    if resolved_cache.is_relative_to(root):
        raise ValueError("model prediction cache must stay outside the repository")
    if method not in ("predict", "predict_proba"):
        raise ValueError(f"unsupported prediction method: {method}")
    digest = hashlib.sha256()
    digest.update(b"openalterego.gaddy-model-prediction-cache.v1\0")
    digest.update(sklearn.__version__.encode())
    digest.update(f"{type(model).__module__}.{type(model).__qualname__}:{method}".encode())
    digest.update(json.dumps(model.get_params(deep=True), sort_keys=True,
                             default=str).encode())
    _update_array(digest, training_x)
    _update_array(digest, training_y)
    for values in development_x:
        _update_array(digest, values)
    return resolved_cache / f"{name}-{digest.hexdigest()[:24]}.npz"


def fit_predict_cached(name: str, model, training_x: np.ndarray,
                       training_y: np.ndarray, development_x: list[np.ndarray],
                       method: str = "predict", cache_dir: Path | None = None
                       ) -> tuple[list[np.ndarray], bool]:
    cache_file = (prediction_cache_path(name, model, training_x, training_y,
                                        development_x, method, cache_dir)
                  if cache_dir is not None else None)
    lengths = [len(values) for values in development_x]
    offsets = np.r_[0, np.cumsum(lengths)]
    if cache_file is not None and cache_file.exists():
        with np.load(cache_file, allow_pickle=False) as cached:
            predicted = cached["predicted"]
            cached_offsets = cached["offsets"]
        if (predicted.ndim != 1 or not np.array_equal(offsets, cached_offsets)
                or not np.isfinite(predicted).all()):
            raise ValueError(f"invalid model prediction cache: {cache_file}")
        return [predicted[offsets[i]:offsets[i + 1]]
                for i in range(len(lengths))], True
    model.fit(training_x, training_y)
    if method == "predict_proba":
        predictions = [model.predict_proba(values)[:, 1] for values in development_x]
    else:
        predictions = [model.predict(values) for values in development_x]
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_file.with_name(f"{cache_file.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as output:
            np.savez(output, predicted=np.concatenate(predictions), offsets=offsets)
        os.replace(temporary, cache_file)
    return predictions, False


def fit_dictionary_cached(dictionary, fit_x: np.ndarray,
                          train_x: list[np.ndarray], dev_x: list[np.ndarray],
                          cache_dir: Path | None = None
                          ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], bool]:
    cache_file = None
    train_offsets = np.r_[0, np.cumsum([len(row) for row in train_x])]
    dev_offsets = np.r_[0, np.cumsum([len(row) for row in dev_x])]
    if cache_dir is not None:
        root = Path(__file__).resolve().parents[3]
        resolved_cache = cache_dir.resolve()
        if resolved_cache.is_relative_to(root):
            raise ValueError("dictionary cache must stay outside the repository")
        digest = hashlib.sha256()
        digest.update(b"openalterego.gaddy-dictionary-cache.v1\0")
        digest.update(sklearn.__version__.encode())
        digest.update(f"{type(dictionary).__module__}.{type(dictionary).__qualname__}".encode())
        digest.update(json.dumps(dictionary.get_params(deep=True), sort_keys=True,
                                 default=str).encode())
        _update_array(digest, fit_x)
        for row in train_x + dev_x:
            _update_array(digest, row)
        cache_file = resolved_cache / f"dictionary-{digest.hexdigest()[:24]}.npz"
    if cache_file is not None and cache_file.exists():
        with np.load(cache_file, allow_pickle=False) as cached:
            components = cached["components"]
            train_codes = cached["train_codes"]
            dev_codes = cached["dev_codes"]
            cached_train_offsets = cached["train_offsets"]
            cached_dev_offsets = cached["dev_offsets"]
        n_components = dictionary.get_params()["n_components"]
        if (components.shape != (n_components, fit_x.shape[1]) or
                train_codes.shape != (train_offsets[-1], n_components) or
                dev_codes.shape != (dev_offsets[-1], n_components) or
                not np.array_equal(train_offsets, cached_train_offsets) or
                not np.array_equal(dev_offsets, cached_dev_offsets) or
                not np.isfinite(components).all() or
                not np.isfinite(train_codes).all() or
                not np.isfinite(dev_codes).all()):
            raise ValueError(f"invalid dictionary cache: {cache_file}")
        train = [train_codes[train_offsets[i]:train_offsets[i + 1]]
                 for i in range(len(train_x))]
        dev = [dev_codes[dev_offsets[i]:dev_offsets[i + 1]]
               for i in range(len(dev_x))]
        return components, train, dev, True
    dictionary.fit(fit_x)
    train = [dictionary.transform(row) for row in train_x]
    dev = [dictionary.transform(row) for row in dev_x]
    components = dictionary.components_.copy()
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_file.with_name(f"{cache_file.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as output:
            np.savez(output, components=components,
                     train_codes=np.concatenate(train), dev_codes=np.concatenate(dev),
                     train_offsets=train_offsets, dev_offsets=dev_offsets)
        os.replace(temporary, cache_file)
    return components, train, dev, False
