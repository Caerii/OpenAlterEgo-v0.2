import numpy as np

from scripts.emg_uka_cross_session_rules import choose, rank_score
from scripts.emg_uka_session_program import CHANNELS


def test_session_variation_can_change_a_program_decision():
    scores = np.zeros((len(CHANNELS), 7))
    scores[0] = 0.5
    scores[1] = [0.9, 0.9, 0.9, 0.9, 0.9, 0.0, 0.0]
    assert choose(scores, "mean") == 1
    assert choose(scores, "mean_minus_std") == 0
    assert choose(scores, "worst_two_mean") == 0
    assert choose(scores, "minimum") == 0
    assert rank_score(scores, "mean")[1] > rank_score(scores, "mean")[0]
