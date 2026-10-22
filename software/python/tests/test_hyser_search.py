from pathlib import Path

import numpy as np

from openalterego.research.hyser_features import read_wfdb_16, site_coordinates, trial_features
from openalterego.research.hyser_search import feasible, prepare_folds, score_designs


def test_dense_screen_scores_batched_and_scalar_designs_identically():
    rng = np.random.default_rng(42)
    labels = np.repeat(np.arange(1, 35), 4)
    features = rng.normal(size=(len(labels), 512)).astype(np.float32)
    folds = prepare_folds(features, labels)
    designs = np.array([[0, 16, 64], [1, 80, 192]], dtype=np.int16)
    together = score_designs(folds, designs)
    separate = np.array([score_designs(folds, row[None, :])[0] for row in designs])
    np.testing.assert_allclose(together, separate, atol=1e-12)


def test_grid_spacing_uses_site_labels_not_flat_indices():
    sites = tuple(f"{grid}-{row}-{column}" for grid in ("ED", "EP", "FD", "FP")
                  for row in range(1, 9) for column in range(1, 9))
    coordinates = site_coordinates(sites)
    assert not feasible((0, 1), coordinates)
    assert feasible((0, 2, 64), coordinates)
    assert not feasible((0, 2, 4), coordinates, max_per_grid=2)
    assert feasible((0, 2, 64), coordinates, max_per_grid=2)


def test_wfdb_reader_checks_interleaving_and_header_checksum(tmp_path: Path):
    base = tmp_path / "dynamic_preprocess_sample1"
    sites = [f"{grid}-{row}-{column}" for grid in ("ED", "EP", "FD", "FP")
             for row in range(1, 9) for column in range(1, 9)]
    raw = np.tile(np.arange(256, dtype=np.int16), (4, 1))
    raw.tofile(base.with_suffix(".dat"))
    header = [f"{base.name} 256 2048 4"]
    header += [f"{base.name}.dat 16 2(0)/V 0 0 {index} {4 * index} 0 {site}"
               for index, site in enumerate(sites)]
    base.with_suffix(".hea").write_text("\n".join(header) + "\n", encoding="ascii")
    signal, names = read_wfdb_16(base)
    assert names == tuple(sites)
    assert signal.shape == (4, 256)
    assert signal[0, 2] == 1.0
    assert trial_features(signal).shape == (512,)
    header[1] = header[1].replace(" 0 0 ED-1-1", " 1 0 ED-1-1")
    base.with_suffix(".hea").write_text("\n".join(header) + "\n", encoding="ascii")
    import pytest
    with pytest.raises(ValueError, match="checksum"):
        read_wfdb_16(base)
