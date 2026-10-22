"""Prospective CSL-EMG_Array topology and leakage-safe evaluation roles.

These rules follow the published acquisition protocol. A corpus adapter must
map the downloaded JSON metadata into CslTrial rows before using this module.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable


# Nominal published derivation, zero-indexed. Verify channel order against
# downloaded corpus metadata before mapping any array column to these pairs.
CONTACT_PAIRS = tuple((i, i + 1) for i in range(31)) + tuple(
    (i, i + 1) for i in range(32, 39))


def feasible_recorded_channels(channels: tuple[int, ...], *,
                               max_contacts: int, max_afe_channels: int
                               ) -> bool:
    """Check physical budgets for a subset of nominal recorded channels."""
    if (not channels or len(channels) > max_afe_channels or
            max_contacts < 2 or len(set(channels)) != len(channels) or
            any(channel < 0 or channel >= len(CONTACT_PAIRS)
                for channel in channels)):
        return False
    return len({contact for channel in channels
                for contact in CONTACT_PAIRS[channel]}) <= max_contacts


@lru_cache(maxsize=65536)
def minimum_contacts_to_complete(channels: tuple[int, ...],
                                 target_channels: int) -> int:
    """Exact minimum contact count when extending a partial chain program.

    Each contiguous run of selected differential edges needs one more
    contact than its number of channels. A small dynamic program finds the
    least-contact completion while keeping every channel already selected.
    """
    if (channels != tuple(sorted(set(channels))) or
            any(channel < 0 or channel >= len(CONTACT_PAIRS)
                for channel in channels) or
            target_channels < len(channels) or
            target_channels > len(CONTACT_PAIRS)):
        raise ValueError("invalid partial CSL program")
    required = set(channels)
    # (selected edges, previous edge selected) -> smallest contact count.
    states = {(0, False): 0}
    for channel in range(len(CONTACT_PAIRS)):
        following = {}
        for (count, previous), cost in states.items():
            adjacent = previous and channel != 31
            if channel not in required:
                key = (count, False)
                following[key] = min(following.get(key, float("inf")), cost)
            if count < target_channels:
                key = (count + 1, True)
                next_cost = cost + 1 + (not adjacent)
                following[key] = min(following.get(key, float("inf")),
                                     next_cost)
        states = following
    return min(cost for (count, _), cost in states.items()
               if count == target_channels)


@dataclass(frozen=True)
class CslTrial:
    session_id: str
    speaker_id: int
    session_mode: str
    block: int
    subset: str
    text_id: str
    utterance_id: str
    speech_mode: str


@dataclass(frozen=True)
class CslSplit:
    fit: tuple[CslTrial, ...]
    selection: tuple[CslTrial, ...]
    final: tuple[CslTrial, ...]


def _role(row: CslTrial) -> str | None:
    if (row.block == 1 and row.subset == "train") or (
            row.block == 2 and row.subset == "train"):
        return "fit"
    if row.block == 3 and row.subset == "dev":
        return "selection"
    if row.block in (5, 7) and row.subset == "eval":
        return "final"
    return None


def split_session(rows: Iterable[CslTrial], *, require_complete: bool = True
                  ) -> CslSplit:
    """Freeze a within-session online split with disjoint sentence identities.

    The role assignment uses block 3 development data for selecting a program
    and blocks 5/7 evaluation data only for final inspection. Later adaptation
    blocks 4/6 are deliberately excluded from this first fixed-model assay.
    """
    rows = tuple(rows)
    if not rows:
        raise ValueError("empty CSL session")
    sessions = {(row.session_id, row.speaker_id, row.session_mode)
                for row in rows}
    if len(sessions) != 1:
        raise ValueError("mixed speakers or sessions")
    if any(not row.session_id or not row.utterance_id or not row.text_id or
           row.speaker_id not in range(1, 9) or
           row.session_mode not in ("audible", "silent") or
           row.block not in range(0, 8) or
           row.subset not in ("train", "dev", "eval") or
           row.speech_mode not in ("audible", "silent")
           for row in rows):
        raise ValueError("invalid CSL trial metadata")
    if len({row.utterance_id for row in rows}) != len(rows):
        raise ValueError("duplicate utterance identity")
    if any((row.speech_mode != "audible" if row.subset == "train" else
            row.speech_mode != row.session_mode)
           for row in rows if row.block in (1, 2, 3, 5, 7)):
        raise ValueError("speech mode disagrees with published block protocol")
    by_role = {role: tuple(row for row in rows if _role(row) == role)
               for role in ("fit", "selection", "final")}
    if any(not group for group in by_role.values()):
        raise ValueError("fit, selection and final roles must be present")
    texts = {role: {row.text_id for row in group}
             for role, group in by_role.items()}
    if (texts["fit"] & texts["selection"] or
            texts["fit"] & texts["final"] or
            texts["selection"] & texts["final"]):
        raise ValueError("sentence identity leaks across evaluation roles")
    if require_complete:
        expected = {(1, "train"): 250, (2, "train"): 20,
                    (3, "dev"): 30, (5, "eval"): 20,
                    (7, "eval"): 20}
        for (block, subset), count in expected.items():
            actual = sum(row.block == block and row.subset == subset
                         for row in rows)
            if actual != count:
                raise ValueError(
                    f"incomplete CSL session: block {block} {subset}: "
                    f"expected {count}, got {actual}")
    return CslSplit(**by_role)
