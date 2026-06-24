"""User calibration workflow.

Calibration computes per-user thresholds, baseline SNR, and trains a personalized model.
Based on research findings that per-user personalization is essential for accuracy.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..dsp.filters import get_filter_spec_for_mode, preprocess_streaming, PreprocessingMode
from ..ml.data_split import stratified_train_val_indices
from ..dsp.quality import assess_signal_quality
from .defaults import default_users_dir
from .manager import UserManager
from .profile import UserProfile

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader


@dataclass
class CalibrationConfig:
    """Configuration for user calibration.
    
    Attributes
    ----------
    min_samples_per_token:
        Minimum samples required per token (default: 50)
    val_split:
        Validation split ratio (default: 0.2)
    epochs:
        Number of training epochs (default: 25)
    lr:
        Learning rate (default: 1e-3)
    batch_size:
        Batch size for training (default: 32)
    segment_ms:
        Segment window size in milliseconds (default: 600)
    seed:
        Random seed for reproducibility (default: 1337)
    strict_motion:
        If True, abort when motion_index exceeds motion_reject_above.
    motion_reject_above:
        Reject calibration when assessed motion_index is above this (strict_motion only).
    """
    min_samples_per_token: int = 50
    val_split: float = 0.2
    epochs: int = 25
    lr: float = 1e-3
    batch_size: int = 32
    segment_ms: int = 600
    seed: int = 1337
    strict_motion: bool = False
    motion_reject_above: float = 0.35
    stride_ms: Optional[int] = None
    """Decode stride to store on profile (default: keep existing profile stride_ms)."""
    stratified_split: bool = True
    """If True, split val with per-label shares (recommended)."""
    show_progress: bool = True
    """Print phase messages and epoch progress during training."""
    arch: str = "se_resnet"
    """Model architecture: cnn or se_resnet."""


@dataclass
class CalibrationReport:
    """Calibration results and metrics.
    
    Attributes
    ----------
    user_id:
        User identifier
    calibration_date:
        Unix timestamp of calibration
    tokens:
        List of tokens in vocabulary
    samples_per_token:
        Number of samples per token
    confidence_threshold:
        Computed per-user confidence threshold
    baseline_snr:
        Baseline SNR in dB
    motion_index:
        Motion artifact index (0-1, higher = more motion)
    validation_accuracy:
        Validation accuracy after training
    validation_loss:
        Validation loss after training
    warnings:
        List of warning messages (e.g., insufficient samples)
    """
    user_id: str
    calibration_date: float
    tokens: List[str]
    samples_per_token: Dict[str, int]
    confidence_threshold: float
    baseline_snr: Optional[float]
    motion_index: float
    validation_accuracy: float
    validation_loss: float
    warnings: List[str]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "user_id": self.user_id,
            "calibration_date": self.calibration_date,
            "tokens": self.tokens,
            "samples_per_token": self.samples_per_token,
            "confidence_threshold": self.confidence_threshold,
            "baseline_snr": self.baseline_snr,
            "motion_index": self.motion_index,
            "validation_accuracy": self.validation_accuracy,
            "validation_loss": self.validation_loss,
            "warnings": self.warnings,
        }
    
    def to_string(self) -> str:
        """Generate human-readable report."""
        lines = [
            f"Calibration Report for User: {self.user_id}",
            f"Date: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.calibration_date))}",
            "",
            f"Vocabulary: {len(self.tokens)} tokens",
            f"  {', '.join(self.tokens)}",
            "",
            "Samples per token:",
        ]
        for token, count in self.samples_per_token.items():
            status = "✓" if count >= 50 else "⚠"
            lines.append(f"  {status} {token}: {count} samples")
        
        lines.extend([
            "",
            f"Confidence Threshold: {self.confidence_threshold:.3f}",
            f"Baseline SNR: {self.baseline_snr:.2f} dB" if self.baseline_snr is not None else "Baseline SNR: Not measured",
            f"Motion Index: {self.motion_index:.3f} (lower is better)",
            "",
            f"Validation Accuracy: {self.validation_accuracy:.1%}",
            f"Validation Loss: {self.validation_loss:.4f}",
        ])
        
        if self.warnings:
            lines.extend([
                "",
                "Warnings:",
            ])
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")
        
        return "\n".join(lines)


def compute_confidence_threshold(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    labels: List[str],
) -> float:
    """Compute per-user confidence threshold from validation predictions.
    
    Uses mean - 2*std of correct predictions, clipped to [0.5, 0.95].
    
    Parameters
    ----------
    model:
        Trained model
    val_loader:
        Validation data loader
    device:
        Device (CPU or CUDA)
    labels:
        List of label strings
        
    Returns
    -------
    threshold:
        Confidence threshold in [0.5, 0.95]
    """
    import torch

    model.eval()
    correct_confidences: List[float] = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            
            pred_idx = torch.argmax(probs, dim=1)
            correct_mask = pred_idx == y
            
            # Get confidence of correct predictions
            correct_probs = probs[correct_mask]
            if len(correct_probs) > 0:
                max_probs = torch.max(correct_probs, dim=1)[0]
                correct_confidences.extend(max_probs.cpu().numpy().tolist())
    
    if len(correct_confidences) == 0:
        # Fallback if no correct predictions
        return 0.70
    
    confidences = np.array(correct_confidences)
    mean_conf = float(np.mean(confidences))
    std_conf = float(np.std(confidences))
    
    # Threshold = mean - 2*std (captures ~95% of correct predictions)
    threshold = mean_conf - 2.0 * std_conf
    
    # Clip to reasonable range
    threshold = max(0.5, min(0.95, threshold))
    
    return float(threshold)


def calibrate_user(
    user_id: str,
    data_dir: Path,
    fs_hz: int,
    user_manager: Optional[UserManager] = None,
    config: Optional[CalibrationConfig] = None,
    preprocessing_mode: Optional[PreprocessingMode] = None,
) -> Tuple[UserProfile, CalibrationReport]:
    """Calibrate a user from calibration data.
    
    This function:
    1. Loads calibration data (signals.npy + events.csv)
    2. Runs preprocessing using user's preferred mode
    3. Trains a personalized model
    4. Computes per-user confidence threshold
    5. Calculates baseline SNR and motion artifacts
    6. Generates calibration report
    7. Updates user profile
    
    Parameters
    ----------
    user_id:
        User identifier
    data_dir:
        Directory containing signals.npy and events.csv
    fs_hz:
        Sampling rate in Hz
    user_manager:
        UserManager instance. If None, uses :func:`default_users_dir`.
    config:
        Calibration configuration (uses defaults if None)
    preprocessing_mode:
        Preprocessing mode (uses user's preference if None)
        
    Returns
    -------
    profile:
        Updated user profile with calibration results
    report:
        Calibration report with metrics
    """
    if config is None:
        config = CalibrationConfig()
    
    if user_manager is None:
        user_manager = UserManager(default_users_dir())
    
    # Load or create user profile
    profile = user_manager.load_profile(user_id)
    if profile is None:
        profile = UserProfile(user_id=user_id)
    
    # Use provided preprocessing mode or user's preference
    if preprocessing_mode is None:
        preprocessing_mode = profile.preprocessing_mode
    
    # Load calibration data
    signals_path = data_dir / "signals.npy"
    events_path = data_dir / "events.csv"
    
    if not signals_path.exists():
        raise FileNotFoundError(f"signals.npy not found in {data_dir}")
    if not events_path.exists():
        raise FileNotFoundError(f"events.csv not found in {data_dir}")

    show = bool(config.show_progress)
    if show:
        print(f"[openalterego] calibrate: loading {data_dir} for user {user_id!r}")
    
    signals = np.load(signals_path).astype(np.float32)
    events = pd.read_csv(events_path)
    
    # Validate data
    if signals.ndim != 2:
        raise ValueError(f"signals must be 2D (time, channels), got shape {signals.shape}")
    if signals.shape[0] < 100:
        raise ValueError(f"signals too short: {signals.shape[0]} samples")
    
    # Get unique labels
    labels = sorted(list({str(x) for x in events["label"].unique()}))
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    
    # Count samples per token
    samples_per_token: Dict[str, int] = {}
    for label in labels:
        count = len(events[events["label"] == label])
        samples_per_token[label] = count
    
    # Validate minimum samples
    warnings: List[str] = []
    for token, count in samples_per_token.items():
        if count < config.min_samples_per_token:
            warnings.append(
                f"Token '{token}' has only {count} samples (recommended: {config.min_samples_per_token}+)"
            )
    
    # Assess signal quality on full signal
    # Use appropriate frequency bands based on preprocessing mode
    if preprocessing_mode == "wide":
        signal_band = (20.0, 450.0)
    elif preprocessing_mode == "clinical":
        signal_band = (0.5, 8.0)
    else:  # standard
        signal_band = (1.0, 50.0)
    
    quality_metrics = assess_signal_quality(
        signals,
        fs_hz=float(fs_hz),
        signal_band_hz=signal_band,
        noise_band_hz=(0.5, 5.0),
        low_freq_cutoff_hz=5.0,
        axis=0,
    )
    
    baseline_snr = quality_metrics.snr_db
    motion_index = quality_metrics.motion_index

    if config.strict_motion and float(motion_index) > float(config.motion_reject_above):
        raise ValueError(
            f"strict_motion: motion_index={float(motion_index):.3f} exceeds "
            f"motion_reject_above={float(config.motion_reject_above):.3f}"
        )

    if motion_index > 0.3:
        warnings.append(f"High motion artifact index: {motion_index:.3f} (consider re-calibration with less movement)")

    if show:
        snr_txt = f"{baseline_snr:.1f} dB" if baseline_snr is not None else "n/a"
        print(
            f"[openalterego] calibrate: quality snr={snr_txt} motion_index={motion_index:.3f} "
            f"mode={preprocessing_mode}"
        )
    
    # Preprocess signals (streaming/causal path to match realtime OnlinePreprocessor + serve)
    if preprocessing_mode not in ("standard", "clinical", "wide"):
        raise ValueError(f"unsupported preprocessing_mode: {preprocessing_mode}")
    if preprocessing_mode == "wide":
        get_filter_spec_for_mode("wide", float(fs_hz), notch_hz=60.0)

    preprocessed = preprocess_streaming(
        signals.astype(np.float32, copy=False),
        fs_hz=float(fs_hz),
        channels=int(signals.shape[1]),
        mode=preprocessing_mode,
        rectify_signals=False,
        ema_alpha=0.01,
        notch_hz=60.0,
        chunk_samples=128,
    )
    
    if config.stratified_split:
        tr_idx, val_idx = stratified_train_val_indices(
            events["label"].astype(str).values,
            float(config.val_split),
            int(config.seed),
        )
    else:
        np.random.seed(config.seed)
        idx = np.arange(len(events))
        np.random.shuffle(idx)
        n_val = int(len(idx) * config.val_split)
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]

    tr_events = events.iloc[tr_idx].reset_index(drop=True)
    val_events = events.iloc[val_idx].reset_index(drop=True)
    if len(val_events) == 0 and float(config.val_split) > 0.0 and len(tr_events) > 1:
        val_events = tr_events.iloc[-1:].reset_index(drop=True)
        tr_events = tr_events.iloc[:-1].reset_index(drop=True)
    
    # Create datasets using preprocessed signals
    # We'll create segments directly since signals are already preprocessed
    from dataclasses import dataclass as _dataclass
    
    @_dataclass
    class Example:
        x: np.ndarray  # (channels, time)
        y: int
    
    def create_segments(preprocessed_signals, events_df, label_to_id, fs_hz, segment_ms, seed):
        """Create segments from preprocessed signals."""
        rng = np.random.default_rng(int(seed))
        segment_samples = max(8, int(fs_hz * int(segment_ms) / 1000))
        items = []
        
        for _, row in events_df.iterrows():
            s = int(row["start_sample"])
            e = int(row["end_sample"])
            label = str(row["label"])
            if label not in label_to_id:
                continue
            y = int(label_to_id[label])
            seg = preprocessed_signals[s:e, :]  # (time, ch)
            if seg.shape[0] < 8:
                continue
            
            x = seg.T  # (ch, time)
            # Crop or pad
            ch, t = x.shape
            n = segment_samples
            if t == n:
                x_seg = x.astype(np.float32, copy=False)
            elif t > n:
                # Random crop
                start = int(rng.integers(0, t - n + 1))
                x_seg = x[:, start : start + n].astype(np.float32, copy=False)
            else:
                # Pad
                x_seg = np.zeros((ch, n), dtype=np.float32)
                x_seg[:, :t] = x.astype(np.float32, copy=False)
            
            items.append(Example(x=x_seg, y=y))
        
        return items
    
    tr_items = create_segments(preprocessed, tr_events, label_to_id, fs_hz, config.segment_ms, config.seed)
    val_items = create_segments(preprocessed, val_events, label_to_id, fs_hz, config.segment_ms, config.seed + 1)

    if show:
        print(
            f"[openalterego] calibrate: {len(tr_items)} train / {len(val_items)} val segments, "
            f"{len(labels)} tokens"
        )
    
    import torch
    from torch.utils.data import DataLoader

    from ..ml.model import create_model
    from ..ml.train import evaluate, fit_epochs

    # Create simple dataset wrapper
    class SimpleDataset:
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            ex = self.items[idx]
            x = torch.from_numpy(ex.x)
            y = torch.tensor(ex.y, dtype=torch.long)
            return x, y
    
    ds_tr = SimpleDataset(tr_items)
    ds_val = SimpleDataset(val_items)
    
    if len(ds_tr) == 0:
        raise ValueError("No training examples after preprocessing")
    if len(ds_val) == 0:
        raise ValueError("No validation examples after preprocessing")
    
    # Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels = int(signals.shape[1])
    model = create_model(str(config.arch), channels=channels, classes=len(labels))
    model.to(device)
    
    tr_loader = DataLoader(ds_tr, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=config.batch_size, shuffle=False)

    user_dir = user_manager.get_user_dir(user_id)
    model_path = user_dir / "model.pt"

    _, val_loss, val_acc = fit_epochs(
        model,
        tr_loader,
        val_loader,
        device,
        epochs=int(config.epochs),
        lr=float(config.lr),
        show_progress=show,
        save_best_path=model_path,
        extra_ckpt={
            "labels": labels,
            "fs": fs_hz,
            "channels": channels,
            "preprocess_mode": "streaming",
            "emg_mode": str(preprocessing_mode),
            "segment_ms": int(config.segment_ms),
            "user_id": user_id,
            "arch": str(config.arch),
        },
    )
    
    # Compute confidence threshold
    confidence_threshold = compute_confidence_threshold(model, val_loader, device, labels)

    if show:
        print(
            f"[openalterego] calibrate: val_acc={val_acc:.1%} threshold={confidence_threshold:.3f} "
            f"-> {model_path}"
        )
    
    # Save model (fit_epochs already wrote best checkpoint)

    stride_ms_out = int(config.stride_ms) if config.stride_ms is not None else int(profile.stride_ms)
    profile = UserProfile(
        user_id=profile.user_id,
        created_at=profile.created_at,
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        preprocessing_mode=preprocessing_mode,
        window_ms=config.segment_ms,
        stride_ms=stride_ms_out,
        calibration_date=time.time(),
        calibration_samples=len(events),
        baseline_snr=baseline_snr,
    )
    
    user_manager.save_profile(profile)
    
    # Generate report
    report = CalibrationReport(
        user_id=user_id,
        calibration_date=profile.calibration_date,
        tokens=labels,
        samples_per_token=samples_per_token,
        confidence_threshold=confidence_threshold,
        baseline_snr=baseline_snr,
        motion_index=motion_index,
        validation_accuracy=val_acc,
        validation_loss=val_loss,
        warnings=warnings,
    )
    
    # Save report
    report_path = user_dir / "calibration_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    
    report_txt_path = user_dir / "calibration_report.txt"
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(report.to_string())
    
    return profile, report
