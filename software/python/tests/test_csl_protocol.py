"""Check the published CSL block split before any raw corpus is opened."""

import itertools

import numpy as np
import pytest

from openalterego.research.compositional_search import beam_search
from openalterego.research.csl_protocol import (
    CONTACT_PAIRS, CslTrial, feasible_recorded_channels,
    minimum_contacts_to_complete, split_session)


def session_rows(mode="silent"):
    rows = []
    layout = ((1, "train", 250, "train"),
              (2, "train", 20, "train"),
              (3, "dev", 30, "dev"),
              (5, "eval", 20, "eval"),
              (7, "eval", 20, "eval"))
    for block, subset, count, prefix in layout:
        for index in range(count):
            text_id = (f"train-{index}" if block == 2 else
                       f"{prefix}-{index}")
            rows.append(CslTrial(
                "speaker1-silent", 1, mode, block, subset, text_id,
                f"block{block}-{subset}-{index}",
                "audible" if subset == "train" else mode))
    return rows


def test_chained_derivation_accounts_for_contacts_and_channels():
    assert len(CONTACT_PAIRS) == 38
    assert len({contact for pair in CONTACT_PAIRS for contact in pair}) == 40
    assert CONTACT_PAIRS[30] == (30, 31)
    assert CONTACT_PAIRS[31] == (32, 33)
    assert feasible_recorded_channels((0, 1, 2, 3, 4, 5),
                                      max_contacts=7, max_afe_channels=6)
    assert not feasible_recorded_channels((0, 2, 4, 6, 8, 10),
                                          max_contacts=7, max_afe_channels=6)
    assert minimum_contacts_to_complete((0, 1, 2), 6) == 7
    assert minimum_contacts_to_complete((0, 31), 6) == 8
    assert minimum_contacts_to_complete((0, 2, 4, 6, 8), 6) > 8


def test_completion_lower_bound_matches_small_exact_enumeration():
    for partial in ((0,), (0, 2), (0, 31)):
        exact = min(len({contact for channel in program
                         for contact in CONTACT_PAIRS[channel]})
                    for program in itertools.combinations(range(38), 3)
                    if set(partial).issubset(program))
        assert minimum_contacts_to_complete(partial, 3) == exact


def test_beam_keeps_a_contact_feasible_completion():
    options = tuple(tuple((channel,) for channel in range(38))
                    for _ in range(6))
    feasible = lambda program: feasible_recorded_channels(
        program, max_contacts=8, max_afe_channels=6)
    complete = lambda program, remaining: (
        minimum_contacts_to_complete(program, len(program) + remaining) <= 8)
    score = lambda programs: programs.sum(axis=1).astype(np.float64)
    chosen = beam_search(options, score, beam_size=4, feasible=feasible,
                         can_complete=complete)
    assert len(chosen.channels) == 6
    assert feasible(chosen.channels)


def test_complete_silent_session_has_temporal_and_text_safe_roles():
    split = split_session(session_rows())
    assert (len(split.fit), len(split.selection), len(split.final)) == (
        270, 30, 40)
    assert {row.block for row in split.fit} == {1, 2}
    assert {row.block for row in split.selection} == {3}
    assert {row.block for row in split.final} == {5, 7}
    assert {row.speech_mode for row in split.final} == {"silent"}


def test_final_text_in_fitting_is_rejected():
    rows = session_rows()
    rows[0] = CslTrial(**(rows[0].__dict__ | {"text_id": "eval-0"}))
    with pytest.raises(ValueError, match="sentence identity leaks"):
        split_session(rows)


def test_missing_block_and_mixed_session_are_rejected():
    rows = session_rows()
    with pytest.raises(ValueError, match="incomplete CSL session"):
        split_session(rows[:-1])
    rows[-1] = CslTrial(**(rows[-1].__dict__ | {"session_id": "other"}))
    with pytest.raises(ValueError, match="mixed speakers or sessions"):
        split_session(rows)
