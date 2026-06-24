"""Wire protocol for OpenAlterEgo websocket messages.

Design goals:
- Dead simple JSON.
- Forward compatible (include a `type` field).
- Unity-friendly (flat objects, no fancy stuff).

Messages you can expect:

Token (server -> client):
    {
      "type": "token",
      "token": "yes",
      "confidence": 0.92,
      "t": 1730000000.123,
      "seq": 12345,
      "source": "sim",
      "meta": {"latency_ms": 34.2, "snr_db_per_channel": [12.1, 11.0], "weak_channels": [3]}
    }

Control (client -> server):
    {"type":"control","cmd":"ping"}
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class FinalTranscriptMessage(BaseModel):
    type: Literal["final_transcript"] = "final_transcript"
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    alternatives: list[str] = Field(default_factory=list)
    utterance_id: str = ""
    t: float = 0.0
    seq: int = 0
    source: str = "unknown"
    meta: Dict[str, Any] = Field(default_factory=dict)


class PartialTranscriptMessage(BaseModel):
    type: Literal["partial_transcript"] = "partial_transcript"
    text: str = ""
    utterance_id: str = ""
    t: float = 0.0
    seq: int = 0
    source: str = "unknown"
    meta: Dict[str, Any] = Field(default_factory=dict)


class TokenMessage(BaseModel):
    type: Literal["token"] = "token"
    token: str
    confidence: float = Field(ge=0.0, le=1.0)
    t: float
    seq: int = 0
    source: str = "unknown"
    meta: Dict[str, Any] = Field(default_factory=dict)


class StatusMessage(BaseModel):
    type: Literal["status"] = "status"
    t: float
    status: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class ControlMessage(BaseModel):
    type: Literal["control"] = "control"
    cmd: str
    args: Dict[str, Any] = Field(default_factory=dict)
