"""EMG preprocessing configuration shared by training, calibration, and serving."""

from __future__ import annotations

import logging
from typing import Literal, Optional

from .filters import get_filter_spec_for_mode
from .online import OnlinePreprocessor

log = logging.getLogger("openalterego.dsp.emg_config")

EmgMode = Literal["standard", "clinical", "wide", "gowda"]


def resolve_emg_mode_for_serve(
    *,
    checkpoint_emg_mode: Optional[str],
    profile_preprocessing_mode: Optional[str],
) -> EmgMode:
    """Pick bandpass preset for realtime preprocessing.

    Checkpoints from ``train`` / ``calibrate`` include ``emg_mode`` when available. If present, it
    wins over the user profile so the frontend matches **how the model was trained**. When the
    profile disagrees, we log a warning (profile may be stale).
    """
    valid: tuple[str, ...] = ("standard", "clinical", "wide", "gowda")
    ck = str(checkpoint_emg_mode).strip() if checkpoint_emg_mode is not None else ""
    if ck in valid:
        if (
            profile_preprocessing_mode is not None
            and str(profile_preprocessing_mode).strip() != ck
        ):
            log.warning(
                "checkpoint emg_mode=%r != profile preprocessing_mode=%r; using checkpoint",
                ck,
                profile_preprocessing_mode,
            )
        return ck
    if profile_preprocessing_mode is not None and str(profile_preprocessing_mode).strip() in valid:
        return str(profile_preprocessing_mode).strip()
    if profile_preprocessing_mode is not None and str(profile_preprocessing_mode).strip():
        log.warning("unknown profile preprocessing_mode=%r; falling back to standard", profile_preprocessing_mode)
    return "standard"


def build_online_preprocessor(
    *,
    fs_hz: float,
    channels: int,
    emg_mode: EmgMode,
    ema_alpha: float = 0.01,
    motion_gate: bool = False,
    motion_threshold: float = 0.35,
    motion_attenuation: float = 0.15,
) -> OnlinePreprocessor:
    """Create :class:`OnlinePreprocessor` matching ``preprocess_streaming(..., mode=emg_mode)``."""
    try:
        spec = get_filter_spec_for_mode(emg_mode, fs_hz, notch_hz=60.0)
    except ValueError as e:
        log.warning("EMG mode %r invalid for fs_hz=%s: %s; using standard.", emg_mode, fs_hz, e)
        spec = get_filter_spec_for_mode("standard", fs_hz, notch_hz=60.0)
    return OnlinePreprocessor(
        fs_hz=float(fs_hz),
        channels=int(channels),
        bandpass_hz=spec.bandpass_hz,
        bandpass_order=int(spec.bandpass_order),
        notch_hz=spec.notch_hz,
        notch_harmonics=spec.notch_harmonics,
        ema_alpha=float(ema_alpha),
        motion_gate=bool(motion_gate),
        motion_threshold=float(motion_threshold),
        motion_attenuation=float(motion_attenuation),
    )


def validate_emg_wide_fs(fs_hz: float) -> None:
    """Raise ValueError if *wide* mode cannot be used at this sample rate."""
    get_filter_spec_for_mode("wide", float(fs_hz), notch_hz=60.0)


def validate_emg_gowda_fs(fs_hz: float) -> None:
    """Raise ValueError if *gowda* (80–1000 Hz) mode cannot be used at this sample rate."""
    get_filter_spec_for_mode("gowda", float(fs_hz), notch_hz=60.0)


def emg_signal_band_hz_for_quality(emg_mode: EmgMode, fs_hz: float) -> tuple[float, float]:
    """Signal band for :func:`dsp.quality.assess_signal_quality` / :class:`OnlineQualityMonitor`.

    High cutoff is clamped below Nyquist so Welch SNR is defined at the current sample rate.
    """
    fs = float(fs_hz)
    margin = 5.0
    hi_cap = max(margin * 2.0, fs / 2.0 - margin)
    if emg_mode == "clinical":
        return (0.5, min(8.0, hi_cap))
    if emg_mode == "wide":
        return (20.0, min(450.0, hi_cap))
    if emg_mode == "gowda":
        return (80.0, min(1000.0, hi_cap))
    return (1.0, min(50.0, hi_cap))
