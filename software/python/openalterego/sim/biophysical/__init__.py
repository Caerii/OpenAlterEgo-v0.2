"""Biophysically motivated synthetic sEMG (motor pool, pickup, recruitment, conduction delays).

**v5 (motor pool path)** adds physiological fiber types (S/FR/FF), refractory periods, batch
vectorized MUAP superposition, and rank-based Henneman recruitment. Earlier v4 added discrete
1D muscle-source pickup, conduction delays, and sensor realism layers. Still **not** a
finite-element volume conductor or articulatory speech synthesizer.

Use :class:`BiophysicalSimStream` for ``FrameChunk``-compatible output, or
:class:`~openalterego.sim.stream.SimStream` for band-noise tokens.
"""

from .conduction import channel_delays_from_pickup
from .dataset_meta import biophysical_block_for_meta
from .forward_model import green_pickup_matrix, motor_unit_pickup_weights
from .motor_injection import MotorPoolInjector
from .motor_pool import (
    firing_rates_baseline_segment,
    firing_rates_token_segment,
    init_motor_unit_layer,
)
from .muap import bipolar_muap_template, stretch_muap_template
from .pool import add_muap_spikes
from .recruitment import recruitment_rate_multipliers
from .sensor_pipeline import SensorNoiseState
from .stream import BiophysicalSimStream, BiophysicalSimStreamConfig
from .versions import BIOPHYS_MODEL_VERSION, BIOPHYS_MODEL_VERSION_LEGACY

__all__ = [
    "BIOPHYS_MODEL_VERSION",
    "BIOPHYS_MODEL_VERSION_LEGACY",
    "BiophysicalSimStream",
    "BiophysicalSimStreamConfig",
    "MotorPoolInjector",
    "SensorNoiseState",
    "add_muap_spikes",
    "biophysical_block_for_meta",
    "bipolar_muap_template",
    "channel_delays_from_pickup",
    "firing_rates_baseline_segment",
    "firing_rates_token_segment",
    "green_pickup_matrix",
    "init_motor_unit_layer",
    "motor_unit_pickup_weights",
    "recruitment_rate_multipliers",
    "stretch_muap_template",
]
