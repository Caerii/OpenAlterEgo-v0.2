import numpy as np
import pytest

from openalterego.research.somach_transfer import _match_moments, benchmark, features_from_csv


def test_csv_feature_import(tmp_path):
    path = tmp_path / "sample.csv"
    rows = ["Timestamp,CH1,CH2,Label,Phase"]
    rows += [f"{index * 4},{2000 + index % 7},{1900 + index % 9},UP,Phase_3_Mouthing" for index in range(100)]
    path.write_text("\n".join(rows), encoding="utf-8")
    features, label = features_from_csv(path)
    assert features.shape == (10,)
    assert np.isfinite(features).all()
    assert label == "UP"


def test_rejects_mixed_labels(tmp_path):
    path = tmp_path / "mixed.csv"
    rows = ["Timestamp,CH1,CH2,Label,Phase"]
    rows += [f"{index * 4},2000,1900,{'UP' if index < 70 else 'DOWN'},Phase_3_Mouthing" for index in range(100)]
    path.write_text("\n".join(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="labels"):
        features_from_csv(path)


def test_cross_configuration_report_has_shared_b_holdout():
    rng = np.random.default_rng(5)
    labels = np.repeat(["DOWN", "LEFT", "NOISE", "RIGHT", "SILENCE", "UP"], 20)
    a = rng.normal(size=(len(labels), 10))
    b = rng.normal(size=(len(labels), 10))
    a[:, 0] += np.repeat(np.arange(6), 20)
    b[:, 0] += np.repeat(np.arange(6), 20)
    report = benchmark((a, labels), (b, labels))
    assert report["samples"]["b_test"] == 24
    assert set(report["a_plus_b_calibration_by_shots_per_class"]) == {"1", "3", "5", "10"}
    assert 0 <= report["a_to_b_unlabeled_mean_shift_accuracy"] <= 1


def test_unlabeled_moment_match_uses_target_distribution():
    source = np.asarray([[1., 3.], [2., 5.], [3., 7.]])
    target = source * 2 + 10
    transformed = _match_moments(source, target, scale=True)
    assert transformed.mean(axis=0) == pytest.approx(target.mean(axis=0))
    assert transformed.std(axis=0) == pytest.approx(target.std(axis=0))
