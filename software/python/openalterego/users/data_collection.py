"""Data collection utilities with quality validation.

This module provides helpers for collecting calibration data with real-time
quality checks and metadata collection.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..dsp.quality import OnlineQualityMonitor, SignalQualityMetrics, assess_signal_quality
from .manager import UserManager
from .profile import UserProfile, PreprocessingMode


@dataclass
class SessionMetadata:
    """Metadata for a data collection session.
    
    Attributes
    ----------
    user_id:
        User identifier
    session_id:
        Unique session identifier
    fs_hz:
        Sampling rate
    channels:
        Number of channels
    electrode_placement:
        Notes on electrode placement (optional)
    electrode_types:
        Types of electrodes used (optional)
    collection_date:
        Unix timestamp when collection started
    duration_s:
        Total duration of collection
    preprocessing_mode:
        Intended preprocessing mode for this data
    quality_metrics:
        Signal quality metrics computed during collection
    notes:
        Additional notes or observations
    """
    user_id: str
    session_id: str
    fs_hz: int
    channels: int
    collection_date: float = field(default_factory=time.time)
    duration_s: float = 0.0
    preprocessing_mode: PreprocessingMode = "standard"
    electrode_placement: Optional[str] = None
    electrode_types: Optional[str] = None
    quality_metrics: Optional[Dict[str, float]] = None
    notes: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "fs_hz": self.fs_hz,
            "channels": self.channels,
            "collection_date": self.collection_date,
            "duration_s": self.duration_s,
            "preprocessing_mode": self.preprocessing_mode,
            "electrode_placement": self.electrode_placement,
            "electrode_types": self.electrode_types,
            "quality_metrics": self.quality_metrics,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> SessionMetadata:
        """Create from dictionary."""
        return cls(
            user_id=d["user_id"],
            session_id=d["session_id"],
            fs_hz=int(d["fs_hz"]),
            channels=int(d["channels"]),
            collection_date=d.get("collection_date", time.time()),
            duration_s=d.get("duration_s", 0.0),
            preprocessing_mode=d.get("preprocessing_mode", "standard"),
            electrode_placement=d.get("electrode_placement"),
            electrode_types=d.get("electrode_types"),
            quality_metrics=d.get("quality_metrics"),
            notes=d.get("notes"),
        )


class DataCollectionSession:
    """Manages a data collection session with quality monitoring.
    
    Tracks signals, events, and quality metrics in real-time.
    """
    
    def __init__(
        self,
        user_id: str,
        fs_hz: int,
        channels: int,
        session_id: Optional[str] = None,
        preprocessing_mode: PreprocessingMode = "standard",
    ):
        """Initialize a data collection session.
        
        Parameters
        ----------
        user_id:
            User identifier
        fs_hz:
            Sampling rate
        channels:
            Number of channels
        session_id:
            Unique session ID (generated if None)
        preprocessing_mode:
            Intended preprocessing mode for this data
        """
        self.user_id = user_id
        self.fs_hz = fs_hz
        self.channels = channels
        self.session_id = session_id or f"session_{int(time.time())}"
        self.preprocessing_mode = preprocessing_mode
        
        # Storage
        self.signals: List[np.ndarray] = []
        self.events: List[Dict[str, int | str]] = []
        self.start_time = time.time()
        
        # Quality monitoring
        window_samples = int(fs_hz * 4.0)  # 4 second window
        self.quality_monitor = OnlineQualityMonitor(
            fs_hz=float(fs_hz),
            window_samples=window_samples,
            signal_band_hz=self._get_signal_band(),
            noise_band_hz=(0.5, 5.0),
            low_freq_cutoff_hz=5.0,
        )
        
        # Quality history
        self.quality_history: List[SignalQualityMetrics] = []
        self.warnings: List[str] = []
    
    def _get_signal_band(self) -> tuple[float, float]:
        """Get signal band based on preprocessing mode."""
        if self.preprocessing_mode == "wide":
            return (20.0, 450.0)
        elif self.preprocessing_mode == "clinical":
            return (0.5, 8.0)
        else:  # standard
            return (1.0, 50.0)
    
    def add_chunk(self, signals: np.ndarray) -> SignalQualityMetrics:
        """Add a chunk of signals and return quality metrics.
        
        Parameters
        ----------
        signals:
            Signal chunk (time, channels)
            
        Returns
        -------
        metrics:
            Current signal quality metrics
        """
        if signals.ndim != 2 or signals.shape[1] != self.channels:
            raise ValueError(
                f"signals must have shape (time, {self.channels}), got {signals.shape}"
            )
        
        # Store signals
        self.signals.append(signals.astype(np.float32, copy=False))
        
        # Update quality monitor
        metrics = self.quality_monitor.update(signals)
        self.quality_history.append(metrics)
        
        # Check for quality issues
        if metrics.snr_db is not None and metrics.snr_db < 10.0:
            warning = f"Low SNR detected: {metrics.snr_db:.2f} dB (recommended: >15 dB)"
            if warning not in self.warnings:
                self.warnings.append(warning)
        
        if metrics.motion_index > 0.3:
            warning = f"High motion artifacts: {metrics.motion_index:.3f} (recommended: <0.2)"
            if warning not in self.warnings:
                self.warnings.append(warning)
        
        return metrics
    
    def add_event(self, start_sample: int, end_sample: int, label: str) -> None:
        """Add an event (token) to the session.
        
        Parameters
        ----------
        start_sample:
            Start sample index
        end_sample:
            End sample index
        label:
            Token label
        """
        self.events.append({
            "start_sample": int(start_sample),
            "end_sample": int(end_sample),
            "label": str(label),
        })
    
    def get_total_samples(self) -> int:
        """Get total number of samples collected."""
        return sum(s.shape[0] for s in self.signals)
    
    def finalize(self) -> tuple[np.ndarray, pd.DataFrame, SessionMetadata]:
        """Finalize session and return data.
        
        Returns
        -------
        signals:
            Concatenated signals (time, channels)
        events:
            Events DataFrame
        metadata:
            Session metadata
        """
        # Concatenate all signals
        if not self.signals:
            raise ValueError("No signals collected")
        
        signals = np.concatenate(self.signals, axis=0).astype(np.float32)
        
        # Create events DataFrame (BLE sessions may have no events yet)
        events = pd.DataFrame(self.events)
        if events.empty:
            events = pd.DataFrame(columns=["start_sample", "end_sample", "label"])
        
        # Compute final quality metrics
        final_quality = assess_signal_quality(
            signals,
            fs_hz=float(self.fs_hz),
            signal_band_hz=self._get_signal_band(),
            noise_band_hz=(0.5, 5.0),
            low_freq_cutoff_hz=5.0,
            axis=0,
        )
        
        # Create metadata
        duration_s = (time.time() - self.start_time)
        metadata = SessionMetadata(
            user_id=self.user_id,
            session_id=self.session_id,
            fs_hz=self.fs_hz,
            channels=self.channels,
            collection_date=self.start_time,
            duration_s=duration_s,
            preprocessing_mode=self.preprocessing_mode,
            quality_metrics={
                "snr_db": final_quality.snr_db,
                "motion_index": float(final_quality.motion_index),
                "baseline_wander": float(final_quality.baseline_wander),
                "signal_power": float(final_quality.signal_power),
                "noise_power": float(final_quality.noise_power),
            },
            notes="; ".join(self.warnings) if self.warnings else None,
        )
        
        return signals, events, metadata
    
    def save(self, output_dir: Path, session_extra: Optional[Dict[str, Any]] = None) -> Path:
        """Save session data to directory.
        
        Parameters
        ----------
        output_dir:
            Directory to save to
            
        Returns
        -------
        output_dir:
            The output directory (for convenience)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        signals, events, metadata = self.finalize()
        
        # Save signals
        signals_path = output_dir / "signals.npy"
        np.save(signals_path, signals)
        
        # Save events
        events_path = output_dir / "events.csv"
        events.to_csv(events_path, index=False)
        
        # Save metadata
        metadata_path = output_dir / "session.json"
        meta_dict = metadata.to_dict()
        if session_extra:
            meta_dict.update(session_extra)
        with open(metadata_path, "w") as f:
            json.dump(meta_dict, f, indent=2)
        
        return output_dir
