import numpy as np
import pytest

from openalterego.research.cross_subject_prior import (additive_channel_prior,
                                                       leave_one_out_prior)


def _programs():
    return [("none", (0, 1, 2, 3, 4, 5)),
            ("recording_robust", (0, 2, 5, 8, 11, 13))]


def test_leave_one_out_excludes_target_future_from_prior_choice():
    programs = _programs()
    personal = {subject: np.array([0.8, 0.2]) for subject in
                ("S01", "S02", "S03")}
    future = {"S01": np.array([0.1, 0.9]),
              "S02": np.array([0.9, 0.1]),
              "S03": np.array([0.9, 0.1])}
    first = leave_one_out_prior(programs, personal, future)
    future["S01"] = np.array([0.9, 0.1])
    second = leave_one_out_prior(programs, personal, future)
    assert first["S01"]["methods"]["donor_raw"]["program"] == ["none", [0, 1, 2, 3, 4, 5]]
    assert first["S01"]["methods"]["donor_raw"]["program"] == second["S01"]["methods"]["donor_raw"]["program"]


def test_additive_prior_requires_aligned_scores():
    with pytest.raises(ValueError):
        additive_channel_prior(_programs(), np.array([0.1]))
