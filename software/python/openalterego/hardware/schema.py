"""Pydantic schema for OpenAlterEgo hardware DSL (``.oae.json``)."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

Tier = Literal["v0", "v1", "v2"]
ElectrodeType = Literal["wet_ag_agcl", "dry_textile", "dry_metal", "tattoo_flex"]
ReferenceSite = Literal["earlobe", "wrist", "differential_only", "neck_ground"]
PreprocessMode = Literal["standard", "clinical", "wide"]
PacketFormat = Literal["oa_v1", "raw_i16"]
SimEngine = Literal["heuristic", "biophysical"]
RealismPreset = Literal["off", "wearable", "tang", "field"]


class AfeConfig(BaseModel):
    """Analog front-end parameters (maps to ADS1299-class devices)."""

    part: str = "ads1299"
    channels: int = Field(8, ge=1, le=32)
    fs_hz: int = Field(250, ge=100, le=16000)
    gain: float = Field(24.0, gt=0)
    adc_bits: int = Field(16, ge=12, le=24, description="Bits on the wire (OA v1 int16)")
    internal_adc_bits: int = Field(24, ge=16, le=32)
    vref_v: float = Field(2.4, gt=0)


class ElectrodeConfig(BaseModel):
    type: ElectrodeType = "wet_ag_agcl"
    montage: str = "alterego_8ch"
    reference: ReferenceSite = "earlobe"


    @field_validator("montage")
    @classmethod
    def _montage_known(cls, v: str) -> str:
        from .montages import get_montage

        get_montage(v)
        return v


class PreprocessConfig(BaseModel):
    mode: PreprocessMode = "standard"
    notch_hz: float = 60.0
    notch_harmonics: bool = True


class BleConfig(BaseModel):
    transport: Literal["ble", "usb"] = "ble"
    packet_format: PacketFormat = "oa_v1"
    frames_per_packet: int = Field(12, ge=1, le=256)
    device_name: str = "OpenAlterEgo"
    mtu_payload_bytes: int = Field(244, ge=20, le=512)


class LinkSimConfig(BaseModel):
    """Virtual BLE link impairments (stress-test path)."""

    loss_prob: float = Field(0.0, ge=0.0, le=1.0)
    jitter_ms: float = Field(0.0, ge=0.0)
    extra_latency_ms: float = Field(0.0, ge=0.0)
    seed: int = 123


class SimBinding(BaseModel):
    """How this hardware spec binds to the synthetic EMG generator."""

    engine: SimEngine = "biophysical"
    realism: RealismPreset = "tang"
    emg_paradigm: Optional[str] = None
    seed: int = 1337
    chunk_ms: int = Field(40, ge=10, le=500)
    noise_uV: float = Field(22.0, ge=0.0)
    drift_uV_per_s: float = Field(18.0, ge=0.0)
    line_noise_uV: float = Field(0.0, ge=0.0)
    mains_freq_hz: float = 60.0
    crosstalk: float = Field(0.12, ge=0.0, le=1.0)
    ar1_phi: float = Field(0.97, ge=0.0, le=0.999)
    ar1_innovation_scale: float = Field(0.42, ge=0.0)
    noise_scale: float = Field(1.0, gt=0.0, le=8.0)
    snr_target_static_db: Optional[float] = Field(
        None, description="Auto-tune noise_scale to approach this session SNR (dB)"
    )
    labels: List[str] = Field(
        default_factory=lambda: ["yes", "no", "left", "right", "select", "cancel"]
    )


class HardwareSpec(BaseModel):
    """Root document for ``.oae.json`` hardware DSL."""

    schema_version: int = 1
    name: str
    tier: Tier
    description: str = ""
    literature_refs: List[str] = Field(default_factory=list)
    afe: AfeConfig = Field(default_factory=AfeConfig)
    electrodes: ElectrodeConfig = Field(default_factory=ElectrodeConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    ble: BleConfig = Field(default_factory=BleConfig)
    link: LinkSimConfig = Field(default_factory=LinkSimConfig)
    sim: SimBinding = Field(default_factory=SimBinding)

    @field_validator("schema_version")
    @classmethod
    def _schema_v1_only(cls, v: int) -> int:
        if v != 1:
            raise ValueError(f"unsupported schema_version {v}; only 1 is supported")
        return v

    @model_validator(mode="after")
    def _coerce_channel_count_from_montage(self) -> "HardwareSpec":
        from .montages import get_montage

        m = get_montage(self.electrodes.montage)
        if self.afe.channels != m.channels:
            # Allow explicit override only if channels still <= montage max sites
            if self.afe.channels > m.channels:
                raise ValueError(
                    f"afe.channels={self.afe.channels} exceeds montage {self.electrodes.montage!r} "
                    f"({m.channels} sites)"
                )
        return self

    def model_dump_public(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")
