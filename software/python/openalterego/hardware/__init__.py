"""Hardware specification DSL (``.oae.json``) — validate, resolve, and simulate acquisition stacks.

Example::

    from openalterego.hardware import load_spec, validate_spec, resolve_sim_config

    spec = load_spec("presets/v0_openbci")
    issues = validate_spec(spec)
    sim_cfg = resolve_sim_config(spec)
"""

from .load import load_spec, list_presets
from .resolve import (
    ResolvedHardware,
    resolve_all,
    resolve_sim_config,
    resolve_virtual_ble_spec,
)
from .schema import HardwareSpec
from .validate import ValidationIssue, validate_spec

__all__ = [
    "HardwareSpec",
    "ResolvedHardware",
    "ValidationIssue",
    "list_presets",
    "load_spec",
    "resolve_all",
    "resolve_sim_config",
    "resolve_virtual_ble_spec",
    "validate_spec",
]
