import numpy as np
import pytest

from openalterego.research.virtual_montage import (
    benchmark,
    feature_drift,
    fixed_subsets,
    load_silentwear_events,
    robust_recording_normalize,
)


def test_recording_normalization_uses_features_only():
    samples = np.asarray([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=np.float32)
    normalized = robust_recording_normalize(samples)
    assert np.median(normalized, axis=0) == pytest.approx([0, 0])
    assert np.isfinite(normalized).all()


def test_feature_drift_shares_channel_offset():
    samples = np.zeros((5, 28), dtype=np.float32)
    drifted = feature_drift(samples, sigma=0.4, rng=np.random.default_rng(3))
    assert np.array_equal(drifted[:, 0::2], drifted[:, 1::2])
    assert np.count_nonzero(drifted) > 0
    assert all(len(subset) == 6 for subset in fixed_subsets().values())


def test_benchmark_reports_choice_regret_and_held_out_counts():
    rng = np.random.default_rng(9)
    labels = np.tile(np.arange(9), 6)

    def session():
        features = rng.normal(0, 0.25, size=(len(labels), 28))
        features[:, 0] += labels
        return features.astype(np.float32), labels.copy()

    report = benchmark(session(), session(), session(), drift_draws=2)
    assert report["event_counts"] == {"train": 54, "validation": 54, "test": 54}
    assert report["test_decision_regret"] >= 0
    assert set(report["choices"]) == {"validation_accuracy", "validation_drift_robustness"}


def test_silentwear_loader_reads_recorded_layout(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "example.h5"
    with h5py.File(path, "w") as output:
        group = output.create_group("emg")
        channels = [f"Ch_{i}_filt".encode() for i in list(range(11)) + [13, 14, 15]]
        group.create_dataset("block1_items", data=channels)
        group.create_dataset("block1_values", data=np.ones((1500, 14), dtype=np.float32))
        group.create_dataset("block2_items", data=[b"Label_int"])
        group.create_dataset("block2_values", data=np.repeat([[0], [1], [2]], 500, axis=0))
    features, labels, names = load_silentwear_events(path, normalization="recording_robust")
    assert features.shape == (3, 28)
    assert labels.tolist() == [0, 1, 2]
    assert len(names) == 14
