"""User profile dataclass for personalization."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

PreprocessingMode = Literal["standard", "clinical", "wide"]


@dataclass(frozen=True)
class UserProfile:
    """Per-user configuration and metadata.
    
    This dataclass stores all user-specific settings needed for personalized
    silent speech recognition. It's immutable (frozen) to prevent accidental
    modifications.
    
    Attributes
    ----------
    user_id:
        Unique identifier for the user
    created_at:
        Unix timestamp when profile was created
    model_path:
        Path to user's trained model file (None if not trained)
    confidence_threshold:
        Per-user confidence threshold for predictions (default 0.70)
    preprocessing_mode:
        Preprocessing mode: "standard" (1-50 Hz), "clinical" (0.5-8 Hz), or "wide" (20-450 Hz)
    window_ms:
        Inference window size in milliseconds (default 600)
    stride_ms:
        Inference stride in milliseconds (default 120)
    calibration_date:
        Unix timestamp of last calibration (None if never calibrated)
    calibration_samples:
        Number of samples collected during calibration
    baseline_snr:
        Baseline signal-to-noise ratio from calibration (None if not measured)
    """
    
    user_id: str
    created_at: float = field(default_factory=time.time)
    model_path: Optional[Path] = None
    confidence_threshold: float = 0.70
    preprocessing_mode: PreprocessingMode = "standard"
    window_ms: int = 600
    stride_ms: int = 120
    
    # Calibration metadata
    calibration_date: Optional[float] = None
    calibration_samples: int = 0
    baseline_snr: Optional[float] = None
    
    def __post_init__(self) -> None:
        """Validate profile data."""
        if not self.user_id or not isinstance(self.user_id, str):
            raise ValueError("user_id must be a non-empty string")
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}")
        if self.window_ms <= 0:
            raise ValueError(f"window_ms must be > 0, got {self.window_ms}")
        if self.stride_ms <= 0:
            raise ValueError(f"stride_ms must be > 0, got {self.stride_ms}")
        if self.stride_ms >= self.window_ms:
            raise ValueError(f"stride_ms ({self.stride_ms}) must be < window_ms ({self.window_ms})")
        
        # Convert model_path to Path if it's a string
        if self.model_path is not None and isinstance(self.model_path, str):
            object.__setattr__(self, "model_path", Path(self.model_path))
    
    def to_dict(self) -> dict:
        """Convert profile to dictionary (for JSON serialization)."""
        d = {
            "user_id": self.user_id,
            "created_at": self.created_at,
            "confidence_threshold": self.confidence_threshold,
            "preprocessing_mode": self.preprocessing_mode,
            "window_ms": self.window_ms,
            "stride_ms": self.stride_ms,
            "calibration_date": self.calibration_date,
            "calibration_samples": self.calibration_samples,
            "baseline_snr": self.baseline_snr,
        }
        if self.model_path is not None:
            d["model_path"] = str(self.model_path)
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> UserProfile:
        """Create profile from dictionary."""
        return cls(
            user_id=d["user_id"],
            created_at=d.get("created_at", time.time()),
            model_path=Path(d["model_path"]) if d.get("model_path") else None,
            confidence_threshold=d.get("confidence_threshold", 0.70),
            preprocessing_mode=d.get("preprocessing_mode", "standard"),
            window_ms=d.get("window_ms", 600),
            stride_ms=d.get("stride_ms", 120),
            calibration_date=d.get("calibration_date"),
            calibration_samples=d.get("calibration_samples", 0),
            baseline_snr=d.get("baseline_snr"),
        )
