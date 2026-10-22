import json

import numpy as np
import pytest

from scripts.gaddy_locked_range_audit import bootstrap_session
from scripts.gaddy_locked_trial_range import valid_ids


def test_query_manifest_range_and_valid_trial_order(tmp_path):
    files = {f"{trial}_info.json": {} for trial in range(100, 200)}
    (tmp_path / "manifest.json").write_text(json.dumps({
        "start": 100, "count": 100, "files": files}))
    for trial in range(100, 200):
        (tmp_path / f"{trial}_info.json").write_text(json.dumps({
            "sentence_index": -1 if trial == 103 else trial}))
    ids = valid_ids(tmp_path)
    assert len(ids) == 99
    assert ids[:5] == [100, 101, 102, 104, 105]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["start"] = 99
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="locked ID range"):
        valid_ids(tmp_path)


def test_bootstrap_preserves_paired_utterance_frame_weight():
    session = {"evaluation_frame_counts": [1, 3], "results": {
        "selected": {"utterance_mae": [0.0, 2.0]},
        "fixed": {"utterance_mae": [1.0, 3.0]},
    }}
    samples = bootstrap_session(session, 20, np.random.default_rng(42))
    assert np.allclose(samples["fixed"] - samples["selected"], 1.0)
    assert set(samples["selected"]).issubset({0.0, 1.5, 2.0})
