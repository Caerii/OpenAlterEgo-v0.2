"""Literature-grounded conventions for *synthetic* sEMG / silent-speech streams.

This is **not** a biophysical forward model (no MUAP trains, no volume conduction PDEs). It encodes
**spectral targets**, **amplitude scales**, and **simple stochastic structure** that match what
wearable / silent-speech EMG papers typically assume for *processing* and *reported* conditions.

References (order-of-magnitude guidance)
----------------------------------------
- **Kapur et al. (2018)** — AlterEgo, IUI: silent-speech systems often emphasize a **low-frequency
  envelope / articulatory** band on the order of **~1–50 Hz** at moderate sampling rates.
- **Wang et al. (2021)** — npj Flexible Electronics: decoding pipeline uses a **wide sEMG band**
  (paper cites ~**20–500 Hz** filtering; text notes most useful information roughly **~15–450 Hz**).
- **Tang et al. (2024)** — IEEE TIM (headphone-integrated EMG): **20–450 Hz** band in processing;
  reports **~18.9 dB vs ~12.7 dB** SNR static vs motion — motivates **correlated low-frequency
  drift** as a stand-in for motion / electrode instability (not a literal impedance model).
- **Lai et al. (2023)** — knowledge-distillation EMG work: similar **tens–hundreds of Hz** decoding
  bands in preprocessing.

**Nyquist:** energy above **f_s/2** cannot be represented. A **full 20–450 Hz** *synthesis* target
requires **f_s ≳ 920 Hz** (with margin). At **250 Hz**, only content below **~125 Hz** exists; the
``semg_literature_clamped`` paradigm still uses a **20 Hz high-pass** mindset (like Tang/Wang) but
the **upper** band is **clamped to Nyquist**, which is the physically honest choice.
"""

from __future__ import annotations

from typing import Optional, Tuple

# Short string stored in dataset meta / chunk meta for traceability
LITERATURE_MODEL_VERSION = "openalterego_emg_sim_v1"

PARADIGM_SEMG_CLAMPED = "semg_literature_clamped"
PARADIGM_ALTEREGO = "alterego_envelope"
PARADIGM_SEMG_FULL = "semg_literature_full"

VALID_PARADIGMS = frozenset({PARADIGM_SEMG_CLAMPED, PARADIGM_ALTEREGO, PARADIGM_SEMG_FULL})

# Default AR(1) innovation scale (× noise_uV); Tang-style motion/LF correlation analog
DEFAULT_AR1_INNOVATION_SCALE = 0.42


def resolve_sim_token_band(
    fs_hz: int,
    paradigm: str,
    token_band_hz: Optional[Tuple[float, float]],
) -> Tuple[float, float]:
    """Return ``(low_hz, high_hz)`` for token carrier bandpass, clamped to Nyquist.

    If ``token_band_hz`` is set, it overrides the paradigm (still Nyquist-clamped).
    """
    nyq = float(fs_hz) / 2.0
    if paradigm not in VALID_PARADIGMS:
        raise ValueError(f"unknown emg_paradigm {paradigm!r}; expected one of {sorted(VALID_PARADIGMS)}")

    if token_band_hz is not None:
        low, high = float(token_band_hz[0]), float(token_band_hz[1])
        high = min(high, nyq - 10.0)
        if high <= low:
            high = min(50.0, nyq - 5.0)
            low = min(low, high - 10.0)
        low = max(0.5, low)
        return (low, high)

    if paradigm == PARADIGM_ALTEREGO:
        hi = min(50.0, nyq - 5.0)
        return (1.0, max(1.0, hi))

    if paradigm == PARADIGM_SEMG_FULL:
        if fs_hz < 920:
            raise ValueError(
                f"emg_paradigm={PARADIGM_SEMG_FULL!r} requires fs_hz >= 920 to synthesize up to ~450 Hz "
                f"(got fs_hz={fs_hz}). Use {PARADIGM_SEMG_CLAMPED!r} for lower rates."
            )
        return (20.0, min(450.0, nyq - 10.0))

    # PARADIGM_SEMG_CLAMPED — Tang/Wang-style lower cutoff, upper cutoff honest to Nyquist
    return (20.0, min(450.0, nyq - 10.0))
