from pathlib import Path

import numpy as np
import pytest

from scripts.gaddy_model_cache import (fit_dictionary_cached, fit_predict_cached,
                                       prediction_cache_path)


class CountingModel:
    fits = 0

    def __init__(self, offset=0.0):
        self.offset = offset

    def get_params(self, deep=True):
        return {"offset": self.offset}

    def fit(self, x, y):
        type(self).fits += 1
        self.mean = float(np.mean(y))
        return self

    def predict(self, x):
        return np.full(len(x), self.mean + self.offset)


class CountingDictionary:
    fits = 0

    def get_params(self, deep=True):
        return {"n_components": 2}

    def fit(self, x):
        type(self).fits += 1
        self.components_ = np.eye(2)
        return self

    def transform(self, x):
        return x @ self.components_.T


def test_fixed_fit_prediction_cache_skips_fit_and_preserves_boundaries(tmp_path):
    CountingModel.fits = 0
    x = np.arange(12, dtype=float).reshape(6, 2)
    y = np.arange(6, dtype=float)
    dev = [x[:2], x[2:5]]
    first, hit = fit_predict_cached("control", CountingModel(), x, y, dev,
                                    cache_dir=tmp_path)
    assert not hit and CountingModel.fits == 1
    second, hit = fit_predict_cached("control", CountingModel(), x, y, dev,
                                     cache_dir=tmp_path)
    assert hit and CountingModel.fits == 1
    assert [len(row) for row in second] == [2, 3]
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    _, hit = fit_predict_cached("control", CountingModel(offset=1), x, y, dev,
                                cache_dir=tmp_path)
    assert not hit and CountingModel.fits == 2
    _, hit = fit_predict_cached("control", CountingModel(), x, y + 1, dev,
                                cache_dir=tmp_path)
    assert not hit and CountingModel.fits == 3


def test_model_cache_rejects_repository_location_and_corruption(tmp_path):
    x = np.ones((2, 2))
    y = np.ones(2)
    dev = [x]
    root = Path(__file__).resolve().parents[3]
    with pytest.raises(ValueError, match="outside the repository"):
        prediction_cache_path("test", CountingModel(), x, y, dev,
                              "predict", root / "cache")
    cache_file = prediction_cache_path("test", CountingModel(), x, y, dev,
                                       "predict", tmp_path)
    np.savez(cache_file, predicted=np.ones(3), offsets=np.array([0, 3]))
    with pytest.raises(ValueError, match="invalid model prediction cache"):
        fit_predict_cached("test", CountingModel(), x, y, dev,
                           cache_dir=tmp_path)


def test_dictionary_cache_preserves_codes_and_skips_fit(tmp_path):
    CountingDictionary.fits = 0
    train = [np.ones((3, 2)), np.full((2, 2), 2.)]
    dev = [np.full((4, 2), 3.)]
    first = fit_dictionary_cached(CountingDictionary(), train[0], train, dev,
                                  cache_dir=tmp_path)
    second = fit_dictionary_cached(CountingDictionary(), train[0], train, dev,
                                   cache_dir=tmp_path)
    assert not first[3] and second[3] and CountingDictionary.fits == 1
    assert [len(row) for row in second[1]] == [3, 2]
    assert np.array_equal(first[0], second[0])
    assert all(np.array_equal(a, b) for a, b in zip(first[1], second[1]))
    original = next(tmp_path.glob("dictionary-*.npz"))
    changed = [train[0], np.full((2, 2), 4.)]
    assert not fit_dictionary_cached(CountingDictionary(), train[0], changed, dev,
                                     cache_dir=tmp_path)[3]
    cached_files = sorted(tmp_path.glob("dictionary-*.npz"))
    assert len(cached_files) == 2
    with np.load(original, allow_pickle=False) as cached:
        values = {name: cached[name] for name in cached.files}
    values["train_offsets"] = np.array([0, 5])
    np.savez(original, **values)
    with pytest.raises(ValueError, match="invalid dictionary cache"):
        fit_dictionary_cached(CountingDictionary(), train[0], train, dev,
                              cache_dir=tmp_path)
