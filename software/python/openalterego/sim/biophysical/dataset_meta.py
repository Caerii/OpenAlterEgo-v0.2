"""Serialize biophysical config into dataset ``meta.json`` (single place to add fields)."""

from __future__ import annotations

from typing import Any, Dict


def biophysical_block_for_meta(bcfg: Any) -> Dict[str, Any]:
    """Nested ``biophysical`` object written by :func:`openalterego.sim.dataset.generate_dataset`."""
    return {
        "baseline_firing_rate_hz": float(bcfg.baseline_firing_rate_hz),
        "token_firing_rate_hz": float(bcfg.token_firing_rate_hz),
        "muap_duration_ms": float(bcfg.muap_duration_ms),
        "muap_amplitude_uV": float(bcfg.muap_amplitude_uV),
        "electrode_noise_uV": float(bcfg.electrode_noise_uV),
        "use_motor_unit_pool": bool(bcfg.use_motor_unit_pool),
        "n_motor_units": int(bcfg.n_motor_units) if bcfg.use_motor_unit_pool else 0,
        "off_label_rate_scale": float(bcfg.off_label_rate_scale),
        "spike_jitter_std_s": float(bcfg.spike_jitter_std_s),
        "ar1_phi": float(bcfg.ar1_phi),
        "ar1_innovation_scale": float(bcfg.ar1_innovation_scale),
        "line_noise_uV": float(bcfg.line_noise_uV),
        "noise_scale": float(bcfg.noise_scale),
        "volume_mix_mode": str(bcfg.volume_mix_mode),
        "band_limit_output": bool(bcfg.band_limit_output),
        "muap_spatial_spread": bool(bcfg.muap_spatial_spread),
        "use_forward_pickup": bool(bcfg.use_forward_pickup),
        "n_muscle_sources": int(bcfg.n_muscle_sources),
        "use_conduction_delays": bool(bcfg.use_conduction_delays),
        "use_recruitment": bool(bcfg.use_recruitment),
        "use_physiological_pool": bool(getattr(bcfg, "use_physiological_pool", False)),
        "refractory_ms": float(getattr(bcfg, "refractory_ms", 0.0)),
        "use_batch_synthesis": bool(getattr(bcfg, "use_batch_synthesis", True)),
        "synth_mode": str(getattr(bcfg, "synth_mode", "fast")),
        "drive_mode": str(bcfg.scenario.drive_mode),
        "realism_preset": str(getattr(bcfg, "realism_preset", "off")),
        "phone_templates_path": getattr(bcfg, "phone_templates_path", None),
        "coarticulation_enabled": bool(getattr(bcfg, "coarticulation_enabled", True)),
        "coarticulation_overlap_fraction": float(
            getattr(bcfg, "coarticulation_overlap_fraction", 0.28)
        ),
        "coarticulation_min_overlap_ms": float(
            getattr(bcfg, "coarticulation_min_overlap_ms", 10.0)
        ),
    }
