"""Literature-aware validation for hardware specs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

from ..dsp.emg_config import validate_emg_wide_fs
from ..sim.literature import PARADIGM_SEMG_FULL, VALID_PARADIGMS, resolve_sim_token_band
from .montages import get_montage
from .schema import HardwareSpec

Severity = Literal["error", "warning", "info"]


@dataclass(frozen=True)
class ValidationIssue:
    severity: Severity
    code: str
    message: str

    def format(self) -> str:
        return f"[{self.severity.upper()}] {self.code}: {self.message}"


def _resolve_emg_paradigm(spec: HardwareSpec) -> str:
    if spec.sim.emg_paradigm:
        return str(spec.sim.emg_paradigm)
    mode = spec.preprocess.mode
    if mode == "standard" or mode == "clinical":
        return "alterego_envelope"
    if spec.afe.fs_hz >= 920:
        return PARADIGM_SEMG_FULL
    return "semg_literature_clamped"


def validate_spec(spec: HardwareSpec) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    # Montage exists
    try:
        montage = get_montage(spec.electrodes.montage)
    except ValueError as e:
        issues.append(ValidationIssue("error", "montage.unknown", str(e)))
        montage = None

    if montage is not None:
        if spec.afe.channels != montage.channels:
            issues.append(
                ValidationIssue(
                    "warning",
                    "channels.montage_mismatch",
                    f"afe.channels={spec.afe.channels} != montage {montage.name} "
                    f"default {montage.channels}",
                )
            )

    paradigm = _resolve_emg_paradigm(spec)
    if paradigm not in VALID_PARADIGMS:
        issues.append(
            ValidationIssue(
                "error",
                "sim.paradigm_invalid",
                f"emg_paradigm {paradigm!r} not in {sorted(VALID_PARADIGMS)}",
            )
        )
    else:
        try:
            resolve_sim_token_band(spec.afe.fs_hz, paradigm, None)
        except ValueError as e:
            issues.append(ValidationIssue("error", "sim.paradigm_fs", str(e)))

    if spec.preprocess.mode == "wide":
        try:
            validate_emg_wide_fs(float(spec.afe.fs_hz))
        except ValueError as e:
            issues.append(
                ValidationIssue(
                    "warning",
                    "preprocess.wide_fs_low",
                    f"{e} — literature uses 500–1000 Hz for wide EMG (Wang/Tang/Lai)",
                )
            )

    if spec.preprocess.mode == "clinical" and spec.electrodes.reference != "earlobe":
        issues.append(
            ValidationIssue(
                "info",
                "clinical.ref",
                "Kapur 2020 clinical paper uses earlobe reference/bias",
            )
        )

    if spec.electrodes.type.startswith("dry") and spec.afe.gain >= 24:
        issues.append(
            ValidationIssue(
                "warning",
                "afe.gain_dry",
                f"gain={spec.afe.gain} may clip with dry electrodes; SilentWear uses gain 6 @ 500 Hz",
            )
        )

    if spec.tier == "v0" and spec.electrodes.type != "wet_ag_agcl":
        issues.append(
            ValidationIssue(
                "info",
                "tier.v0_electrodes",
                "V0 benchtop typically uses wet Ag/AgCl for fastest SNR convergence",
            )
        )

    if spec.ble.transport == "ble":
        bytes_per_pkt = 20 + spec.ble.frames_per_packet * spec.afe.channels * 2
        if bytes_per_pkt > spec.ble.mtu_payload_bytes:
            issues.append(
                ValidationIssue(
                    "error",
                    "ble.mtu",
                    f"packet size {bytes_per_pkt} B exceeds mtu_payload_bytes={spec.ble.mtu_payload_bytes}",
                )
            )
        throughput = spec.afe.fs_hz * spec.afe.channels * 2
        pkt_rate = spec.afe.fs_hz / spec.ble.frames_per_packet
        issues.append(
            ValidationIssue(
                "info",
                "ble.budget",
                f"~{throughput/1000:.1f} kB/s raw int16; ~{pkt_rate:.0f} packets/s @ "
                f"{spec.ble.frames_per_packet} frames/packet",
            )
        )

    if spec.link.loss_prob > 0.05:
        issues.append(
            ValidationIssue(
                "warning",
                "link.loss_high",
                f"loss_prob={spec.link.loss_prob} — expect sample gaps in OA v1 stream",
            )
        )

    # Literature SNR targets (simulation guidance)
    if spec.sim.realism == "off":
        issues.append(
            ValidationIssue(
                "info",
                "sim.realism_off",
                "realism=off yields optimistic SNR vs Tang 2025 motion benchmarks",
            )
        )

    return issues


def has_errors(issues: List[ValidationIssue]) -> bool:
    return any(i.severity == "error" for i in issues)
