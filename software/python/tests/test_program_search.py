import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from openalterego.research import program_search
from scripts.silentwear_program_search import register_selection


def _batch(seed):
    rng = np.random.default_rng(seed)
    labels = np.tile(np.arange(9), 5)
    features = rng.normal(0, 0.1, (len(labels), 28)).astype(np.float32)
    features[:, 0] += labels
    return features, labels


def test_program_grammar_and_proxy_cover_recorded_six_channel_space():
    assert program_search.program_count() == 18018
    with pytest.raises(ValueError):
        program_search.MontageProgram((0, 1, 2, 3, 4, 14), "none")
    ranked = program_search.proxy_rank(_batch(1), _batch(2), "none")
    assert len(ranked) == 3003
    assert 0 in ranked[0][0].channels
    assert ranked[0][1] >= ranked[-1][1]


def test_batched_ridge_proxy_matches_one_subset_fit():
    train, validation = _batch(1), _batch(2)
    ranked = program_search.ridge_proxy_rank(train, validation, "none")
    assert len(ranked) == 3003
    chosen = next(score for program, score in ranked if program.channels ==
                  (0, 1, 2, 3, 4, 5))
    x, y = train
    val_x, val_y = validation
    columns = list(range(12))
    scaler = StandardScaler().fit(x[:, columns])
    weights = len(y) / (9 * np.bincount(y)[y])
    model = Ridge(alpha=10).fit(scaler.transform(x[:, columns]),
                                np.eye(9)[y], sample_weight=weights)
    predicted = model.predict(scaler.transform(val_x[:, columns])).argmax(axis=1)
    assert chosen == pytest.approx(balanced_accuracy_score(val_y, predicted))


def test_random_budget_baseline_does_not_depend_on_proxy_ranking():
    ranked = program_search.proxy_rank(_batch(1), _batch(2), "none")
    first = program_search._select_pool(ranked, 8, np.random.default_rng(42))
    reversed_rank = program_search._select_pool(list(reversed(ranked)), 8,
                                                 np.random.default_rng(42))
    assert first["random"] == reversed_rank["random"]
    assert first["proxy"] != reversed_rank["proxy"]


def test_search_holds_test_loader_until_both_strategies_are_selected(monkeypatch):
    loaded = False
    frozen = False
    original = program_search._fit_score

    def guarded_fit(*args):
        assert not loaded
        return original(*args)

    def test_loader():
        nonlocal loaded
        assert frozen
        loaded = True
        return [_batch(3)]

    def freeze(selection):
        nonlocal frozen
        assert not loaded
        assert set(selection["choices"]) == {"proxy", "random"}
        frozen = True

    monkeypatch.setattr(program_search, "_fit_score", guarded_fit)
    result = program_search.search([_batch(1)], [_batch(2)], test_loader,
                                   stage_one_per_strategy=2,
                                   finalists_per_strategy=1,
                                   on_selection=freeze)
    assert loaded and frozen
    assert result["unique_proxy_structures"] == 6006
    assert result["program_count"] == 18018
    assert set(result["strategies"]) == {"proxy", "random"}
    assert all(row["full_fits_for_selection"] == 4 for row in
               result["strategies"].values())


def test_selection_record_is_exclusive_and_reconciles_same_choice(tmp_path):
    path = tmp_path / "selection.json"
    selection = {"choices": {"proxy": {"channels": [0, 1, 2, 3, 4, 5]}}}
    digest = register_selection(path, selection, "S01", "manifest-a")
    assert register_selection(path, selection, "S01", "manifest-a") == digest
    with pytest.raises(ValueError, match="differs"):
        register_selection(path, selection, "S01", "manifest-b")
