"""Gowda small-vocab (124-word, 4-word trials) biophysical scenario."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...ml.ctc.trial_lm import MONTHS, WEEKDAYS
from ...ml.phonology.gowda_lexicon import build_lexicon
from ..biophysical.stream import BiophysicalSimStreamConfig
from ..dataset import DatasetConfig
from ..stream import ScenarioConfig, ScriptedWordEvent, SimStreamConfig

GOWDA_WORD_DURATION_S = 2.0
GOWDA_FS_HZ = 5000
GOWDA_CHANNELS = 31
_LABELS_PATH = Path(__file__).with_name("gowda_sv_labels.json")


def load_gowda_labels(path: Optional[Path] = None) -> List[str]:
    """Load 124-word Gowda small-vocab label list."""
    p = Path(path) if path is not None else _LABELS_PATH
    if p.is_file():
        raw = json.loads(p.read_text(encoding="utf-8"))
        return sorted(str(x) for x in raw)
    return gowda_small_vocab_labels_fallback()


def gowda_small_vocab_labels_fallback() -> List[str]:
    """Minimal fallback if JSON missing (weekdays + months only — not full 124)."""
    return sorted(set(WEEKDAYS) | set(MONTHS))


def gowda_small_vocab_labels() -> List[str]:
    return load_gowda_labels()


def _slot_pools(labels: Sequence[str]) -> List[List[str]]:
    labs = set(str(x) for x in labels)
    slot0 = sorted(labs & set(WEEKDAYS))
    slot1 = sorted(labs & set(MONTHS))
    rest = sorted(labs - set(slot0) - set(slot1))
    years = sorted(x for x in rest if str(x).startswith("nineteen_") or str(x).startswith("two_thousand"))
    ordinals = sorted(x for x in rest if x not in years)
    pools = [slot0, slot1, ordinals, years]
    if not all(pools):
        raise ValueError("incomplete Gowda slot pools; check gowda_sv_labels.json")
    return pools


def build_gowda_phone_lexicon(labels: Sequence[str]) -> Dict[str, Tuple[str, ...]]:
    lex = build_lexicon(list(labels))
    return {str(w): tuple(str(p) for p in ph) for w, ph in lex.items()}


def build_gowda_schedule(
    n_trials: int,
    *,
    seed: int = 1337,
    labels: Optional[Sequence[str]] = None,
    word_duration_s: float = GOWDA_WORD_DURATION_S,
) -> Tuple[ScriptedWordEvent, ...]:
    """Build ``n_trials`` four-word sentences (weekday, month, ordinal, year)."""
    labs = list(labels or load_gowda_labels())
    pools = _slot_pools(labs)
    rng = np.random.default_rng(int(seed))
    out: List[ScriptedWordEvent] = []
    for tid in range(int(n_trials)):
        for wi in range(4):
            word = str(rng.choice(pools[wi]))
            out.append(
                ScriptedWordEvent(
                    label=word,
                    duration_s=float(word_duration_s),
                    trial_id=int(tid),
                    word_idx=int(wi),
                )
            )
    return tuple(out)


def estimate_gowda_duration_s(
    n_trials: int,
    *,
    word_duration_s: float = GOWDA_WORD_DURATION_S,
    inter_word_gap_s: Tuple[float, float] = (0.10, 0.20),
    inter_trial_gap_s: Tuple[float, float] = (0.40, 0.80),
) -> float:
    """Upper-bound session length for scripted Gowda trials."""
    word_gap = float(inter_word_gap_s[1])
    trial_gap = float(inter_trial_gap_s[1])
    per_trial = 4.0 * float(word_duration_s) + 3.0 * word_gap + trial_gap
    return float(n_trials) * per_trial + 8.0


def build_gowda_scenario(
    n_trials: int,
    *,
    seed: int = 1337,
    labels: Optional[Sequence[str]] = None,
) -> ScenarioConfig:
    labs = list(labels or load_gowda_labels())
    schedule = build_gowda_schedule(int(n_trials), seed=int(seed), labels=labs)
    phone_lex = build_gowda_phone_lexicon(labs)
    return ScenarioConfig(
        labels=labs,
        p_event=1.0,
        event_duration_s=(GOWDA_WORD_DURATION_S, GOWDA_WORD_DURATION_S),
        gap_duration_s=(0.10, 0.20),
        drive_mode="phoneme",
        phone_lexicon=phone_lex,
        scripted_schedule=schedule,
        inter_word_gap_s=(0.08, 0.15),
        inter_trial_gap_s=(0.40, 0.80),
    )


def build_gowda_dataset_config(
    out_dir: Path,
    *,
    n_trials: int = 500,
    seed: int = 1337,
    realism: str = "tang",
    snr_target_db: Optional[float] = 18.9,
    snr_motion_target_db: Optional[float] = 12.7,
    phone_templates_path: Optional[str] = None,
    coarticulation_enabled: bool = True,
    coarticulation_overlap_fraction: float = 0.28,
    coarticulation_min_overlap_ms: float = 10.0,
) -> DatasetConfig:
    """DatasetConfig for Gowda-shaped biophysical corpus (31 ch @ 5 kHz)."""
    scenario = build_gowda_scenario(int(n_trials), seed=int(seed))
    duration_s = estimate_gowda_duration_s(int(n_trials))
    sim_cfg = SimStreamConfig(
        fs_hz=GOWDA_FS_HZ,
        channels=GOWDA_CHANNELS,
        chunk_ms=50,
        seed=int(seed),
        scenario=scenario,
        emg_paradigm="semg_literature_clamped",
        realtime_clock=False,
        realism_preset=str(realism),  # type: ignore[arg-type]
    )
    bcfg = BiophysicalSimStreamConfig(
        fs_hz=GOWDA_FS_HZ,
        channels=GOWDA_CHANNELS,
        chunk_ms=50,
        seed=int(seed),
        scenario=scenario,
        realtime_clock=False,
        emg_paradigm="semg_literature_clamped",
        montage_name="gowda_31ch",
        realism_preset=str(realism),  # type: ignore[arg-type]
        phone_templates_path=str(phone_templates_path) if phone_templates_path else None,
        coarticulation_enabled=bool(coarticulation_enabled),
        coarticulation_overlap_fraction=float(coarticulation_overlap_fraction),
        coarticulation_min_overlap_ms=float(coarticulation_min_overlap_ms),
    )
    return DatasetConfig(
        out_dir=Path(out_dir),
        duration_s=float(duration_s),
        config=sim_cfg,
        sim_engine="biophysical",
        biophysical=bcfg,
        snr_target_db=float(snr_target_db) if snr_target_db is not None else None,
        snr_motion_target_db=float(snr_motion_target_db) if snr_motion_target_db is not None else None,
        montage_name="gowda_31ch",
        realism_preset=str(realism),
        phone_templates_path=str(phone_templates_path) if phone_templates_path else None,
        min_event_s=0.5,
    )


def build_gowda_sim_config(
    *,
    n_trials: int = 8,
    seed: int = 1337,
    realism: str = "tang",
    realtime_clock: bool = True,
) -> "SimConfig":
    """SimConfig for Gowda-shaped realtime streaming (31 ch @ 5 kHz)."""
    from ...acquisition.simulate import SimConfig

    scenario = build_gowda_scenario(int(n_trials), seed=int(seed))
    return SimConfig(
        labels=list(scenario.labels),
        fs_hz=GOWDA_FS_HZ,
        channels=GOWDA_CHANNELS,
        chunk_ms=50,
        seed=int(seed),
        sim_engine="biophysical",
        realism_preset=str(realism),
        montage_name="gowda_31ch",
        realtime_clock=bool(realtime_clock),
        scenario=scenario,
    )
