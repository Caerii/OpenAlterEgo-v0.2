"""Training script (baseline, realtime-friendly).

Input session folder:
- signals.npy : float32 array (time, channels)
- events.csv  : start_sample,end_sample,label

Example:
    # generate synthetic dataset
    openalterego sim-dataset --out ./sim_session --minutes 2
    # train (streaming-compatible preprocessing)
    openalterego train --data ./sim_session --fs 250 --preprocess-mode streaming --segment-ms 600
    # serve
    openalterego serve --source sim --model ./sim_session/model.pt

Notes
-----
- This is a *closed vocabulary*, per-user baseline.
- Use --segment-ms that matches your realtime window (server --window-ms).
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import create_model, default_arch
from .data_split import resolve_train_val_indices, stratified_train_val_indices
from .device import configure_cuda_for_training, resolve_device
from .segment_cache import ensure_segment_arrays
from .training_perf import build_train_dataloader, maybe_compile_model, resolve_use_amp
from ..dsp.emg_config import EmgMode, validate_emg_gowda_fs, validate_emg_wide_fs
from ..dsp.filters import preprocess_basic, preprocess_streaming
from ..dsp.preprocess_cache import cache_paths, ensure_preprocessed_signals, is_cache_valid


PreprocessMode = Literal["offline", "streaming", "none"]


@dataclass
class Example:
    x: np.ndarray  # (channels, time)
    y: int


class StackedSegmentDataset(Dataset):
    """Pre-materialized ``(N, C, T)`` segments for fast ``DataLoader`` + worker prefetch."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        if len(X) != len(y):
            raise ValueError("X and y length mismatch")
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.X[idx]), torch.tensor(int(self.y[idx]), dtype=torch.long)


class SegmentDataset(StackedSegmentDataset):
    """Build stacked segments from events (optionally via disk cache)."""

    def __init__(
        self,
        signals: np.ndarray,
        events: pd.DataFrame,
        label_to_id: Dict[str, int],
        fs_hz: int,
        segment_ms: int,
        preprocess_mode: PreprocessMode = "offline",
        emg_mode: EmgMode = "standard",
        seed: int = 1337,
        channel_indices: Optional[List[int]] = None,
        *,
        session_dir: Optional[Path] = None,
        split_tag: str = "tr",
        use_segment_cache: bool = True,
        rebuild_segment_cache: bool = False,
        show_progress: bool = False,
        cache_preprocess_mode: Optional[str] = None,
        cache_emg_mode: Optional[str] = None,
        per_event_preprocess: bool = False,
    ):
        prepared = signals
        if per_event_preprocess:
            prepared = np.asarray(signals, dtype=np.float32)
        elif preprocess_mode == "offline":
            prepared = preprocess_basic(
                signals,
                fs_hz=int(fs_hz),
                mode=emg_mode,
                rectify_signals=False,
                normalize_mode="zscore",
            )
        elif preprocess_mode == "streaming":
            prepared = preprocess_streaming(
                signals,
                fs_hz=int(fs_hz),
                channels=int(signals.shape[1]),
                rectify_signals=False,
                ema_alpha=0.01,
                mode=emg_mode,
            )
        elif preprocess_mode == "none":
            prepared = np.asarray(signals, dtype=np.float32)
        else:
            raise ValueError(f"unknown preprocess_mode: {preprocess_mode}")

        pp_mode = str(cache_preprocess_mode or ("per_event" if per_event_preprocess else preprocess_mode))
        emg = str(cache_emg_mode or emg_mode)
        X: np.ndarray
        y: np.ndarray
        if session_dir is not None and (pp_mode != "none" or per_event_preprocess):
            X, y, _ = ensure_segment_arrays(
                np.asarray(prepared, dtype=np.float32),
                events,
                label_to_id,
                session_dir,
                preprocess_mode=pp_mode,
                emg_mode=emg,  # type: ignore[arg-type]
                fs_hz=int(fs_hz),
                segment_ms=int(segment_ms),
                seed=int(seed),
                split_tag=str(split_tag),
                channel_indices=channel_indices,
                use_cache=bool(use_segment_cache),
                rebuild=bool(rebuild_segment_cache),
                show_progress=bool(show_progress),
                per_event_preprocess=bool(per_event_preprocess),
            )
        else:
            from .segment_cache import build_segment_arrays

            X, y = build_segment_arrays(
                np.asarray(prepared, dtype=np.float32),
                events,
                label_to_id,
                fs_hz=int(fs_hz),
                segment_ms=int(segment_ms),
                seed=int(seed),
                channel_indices=channel_indices,
                per_event_preprocess=bool(per_event_preprocess),
                preprocess_emg_mode=emg,
            )
        super().__init__(X, y)


@torch.no_grad()
def evaluate(model: torch.nn.Module, dl: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for x, y in dl:
        x = x.to(device, non_blocking=device.type == "cuda")
        y = y.to(device, non_blocking=device.type == "cuda")
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        n += x.size(0)
    if n == 0:
        return 0.0, 0.0
    return total_loss / n, correct / n


def fit_epochs(
    model: torch.nn.Module,
    tr_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    show_progress: bool = True,
    save_best_path: Path | None = None,
    extra_ckpt: Dict[str, object] | None = None,
    teachers: Optional[List[torch.nn.Module]] = None,
    distill_alpha: float = 0.5,
    distill_temperature: float = 4.0,
    use_amp: bool = False,
    grad_accum_steps: int = 1,
    epoch_profiles: Optional[List["EpochTiming"]] = None,
) -> Tuple[float, float, float]:
    """Train for ``epochs``; optionally checkpoint best val accuracy.

    When ``teachers`` is set, adds KL distillation loss against the mean teacher logits.

    Returns (best_val_acc, final_val_loss, final_val_acc).
    """
    from .training_perf import EpochTiming, cuda_synchronize

    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    best_val = -1.0
    final_loss = 0.0
    final_acc = 0.0
    alpha = float(distill_alpha)
    temp = max(1e-3, float(distill_temperature))
    use_distill = teachers is not None and len(teachers) > 0
    amp_enabled = bool(use_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    accum = max(1, int(grad_accum_steps))

    for epoch in range(int(epochs)):
        model.train()
        total = 0.0
        correct = 0
        n = 0
        opt.zero_grad(set_to_none=True)
        batch_iter = tr_loader
        if show_progress:
            batch_iter = tqdm(tr_loader, desc=f"epoch {epoch + 1}/{epochs}")
        ep_timing = EpochTiming() if epoch_profiles is not None else None
        t_batch = time.perf_counter()
        for step, (x, y) in enumerate(batch_iter):
            if ep_timing is not None:
                ep_timing.data_wait_s += time.perf_counter() - t_batch
            t_compute = time.perf_counter()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                if use_distill:
                    with torch.no_grad():
                        t_logits = torch.stack([t(x) for t in teachers], dim=0).mean(dim=0)
                    kd = F.kl_div(
                        F.log_softmax(logits / temp, dim=1),
                        F.softmax(t_logits / temp, dim=1),
                        reduction="batchmean",
                    ) * (temp * temp)
                    loss = alpha * loss + (1.0 - alpha) * kd
                loss = loss / float(accum)
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0 or (step + 1) == len(tr_loader):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            cuda_synchronize(device)
            if ep_timing is not None:
                ep_timing.compute_s += time.perf_counter() - t_compute
            total += float(loss.item()) * float(accum) * x.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            n += x.size(0)
            if show_progress and hasattr(batch_iter, "set_postfix"):
                batch_iter.set_postfix(loss=total / max(n, 1), acc=correct / max(n, 1))
            t_batch = time.perf_counter()

        t_val = time.perf_counter()
        val_loss, val_acc = evaluate(model, val_loader, device=device)
        cuda_synchronize(device)
        if ep_timing is not None:
            ep_timing.val_s = time.perf_counter() - t_val
            epoch_profiles.append(ep_timing)
        final_loss, final_acc = val_loss, val_acc
        if show_progress:
            print(f"[val] loss={val_loss:.4f} acc={val_acc:.3f} (n={len(val_loader.dataset)})")

        if val_acc >= best_val:
            best_val = float(val_acc)
            if save_best_path is not None:
                ckpt: Dict[str, object] = {"state_dict": model.state_dict()}
                if extra_ckpt:
                    ckpt.update(extra_ckpt)
                torch.save(ckpt, save_best_path)
                if show_progress:
                    print(f"saved best: {save_best_path} (val_acc={best_val:.3f})")

    return best_val, final_loss, final_acc


def _parse_channel_indices(raw: str, n_channels: int) -> Optional[List[int]]:
    text = str(raw or "").strip()
    if not text:
        return None
    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0 or idx >= n_channels:
            raise ValueError(f"channel index {idx} out of range [0, {n_channels})")
        out.append(idx)
    return out or None


def _train_one_model(
    *,
    signals: np.ndarray,
    tr_events: pd.DataFrame,
    val_events: pd.DataFrame,
    label_to_id: Dict[str, int],
    labels: List[str],
    fs_hz: int,
    segment_ms: int,
    preprocess_mode: str,
    emg_mode: EmgMode,
    seed: int,
    arch: str,
    channel_indices: Optional[List[int]],
    batch_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
    save_best_path: Path,
    show: bool,
    teachers: Optional[List[torch.nn.Module]] = None,
    distill_alpha: float = 0.5,
    distill_temperature: float = 4.0,
    user_id: str = "",
    use_amp: bool = False,
    session_dir: Optional[Path] = None,
    use_preprocess_cache: bool = True,
    rebuild_preprocess_cache: bool = False,
    split_tag_train: str = "tr",
    split_tag_val: str = "val",
    use_segment_cache: bool = True,
    rebuild_segment_cache: bool = False,
    num_workers: int = -1,
    compile_model: bool = False,
    grad_accum_steps: int = 1,
    per_event_preprocess: bool = False,
) -> float:
    n_ch = len(channel_indices) if channel_indices is not None else int(signals.shape[1])
    prepared = signals
    cache_hit = False
    pp_mode = "per_event" if per_event_preprocess else str(preprocess_mode)
    if per_event_preprocess:
        prepared = np.asarray(signals, dtype=np.float32)
        if show:
            print(f"[openalterego] preprocess: per-event offline emg_mode={emg_mode}")
    elif pp_mode != "none":
        if session_dir is None:
            raise ValueError("session_dir required when preprocessing is enabled")
        _npy, meta_path = cache_paths(session_dir, pp_mode, str(emg_mode), int(fs_hz))
        raw_path = session_dir / "signals.npy"
        will_hit = (
            bool(use_preprocess_cache)
            and not bool(rebuild_preprocess_cache)
            and is_cache_valid(
                meta_path,
                raw_path,
                preprocess_mode=pp_mode,
                emg_mode=str(emg_mode),
                fs_hz=int(fs_hz),
            )
        )
        prepared, cache_hit = ensure_preprocessed_signals(
            signals,
            session_dir,
            preprocess_mode=preprocess_mode,  # type: ignore[arg-type]
            emg_mode=emg_mode,
            fs_hz=int(fs_hz),
            use_cache=bool(use_preprocess_cache),
            rebuild=bool(rebuild_preprocess_cache),
            show_progress=show and not will_hit,
        )
        if show:
            note = "cache hit" if cache_hit else "computed"
            print(f"[openalterego] preprocess: {note} shape={prepared.shape}")
    ds_tr = SegmentDataset(
        prepared,
        tr_events,
        label_to_id,
        fs_hz=int(fs_hz),
        segment_ms=int(segment_ms),
        preprocess_mode="none",
        emg_mode=emg_mode,
        seed=int(seed),
        channel_indices=channel_indices,
        session_dir=session_dir,
        split_tag=str(split_tag_train),
        use_segment_cache=bool(use_segment_cache),
        rebuild_segment_cache=bool(rebuild_segment_cache),
        show_progress=show,
        cache_preprocess_mode=pp_mode,
        cache_emg_mode=str(emg_mode),
        per_event_preprocess=bool(per_event_preprocess),
    )
    ds_val = SegmentDataset(
        prepared,
        val_events,
        label_to_id,
        fs_hz=int(fs_hz),
        segment_ms=int(segment_ms),
        preprocess_mode="none",
        emg_mode=emg_mode,
        seed=int(seed) + 1,
        channel_indices=channel_indices,
        session_dir=session_dir,
        split_tag=str(split_tag_val),
        use_segment_cache=bool(use_segment_cache),
        rebuild_segment_cache=bool(rebuild_segment_cache),
        show_progress=False,
        cache_preprocess_mode=pp_mode,
        cache_emg_mode=str(emg_mode),
        per_event_preprocess=bool(per_event_preprocess),
    )
    if len(ds_tr) == 0:
        raise SystemExit("No training segments found. Check events.csv or preprocessing settings.")
    dl_tr = build_train_dataloader(
        ds_tr, batch_size=int(batch_size), device=device, num_workers=int(num_workers), shuffle=True
    )
    dl_val = build_train_dataloader(
        ds_val, batch_size=int(batch_size), device=device, num_workers=int(num_workers), shuffle=False
    )
    model = create_model(str(arch), channels=n_ch, classes=len(labels)).to(device)
    model = maybe_compile_model(model, enabled=bool(compile_model))
    extra = {
        "labels": labels,
        "fs": int(fs_hz),
        "channels": n_ch,
        "preprocess_mode": str(preprocess_mode),
        "emg_mode": str(emg_mode),
        "segment_ms": int(segment_ms),
        "arch": str(arch),
    }
    if channel_indices is not None:
        extra["channel_indices"] = list(channel_indices)
    if user_id:
        extra["user_id"] = str(user_id)
    best_val, _, _ = fit_epochs(
        model,
        dl_tr,
        dl_val,
        device,
        epochs=int(epochs),
        lr=float(lr),
        show_progress=show,
        save_best_path=save_best_path,
        extra_ckpt=extra,
        teachers=teachers,
        distill_alpha=float(distill_alpha),
        distill_temperature=float(distill_temperature),
        use_amp=bool(use_amp),
        grad_accum_steps=int(grad_accum_steps),
    )
    return float(best_val)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to session folder")
    ap.add_argument("--fs", type=int, required=True, help="Sampling rate (Hz)")
    ap.add_argument("--segment-ms", type=int, default=None, help="Segment length ms (default: 600, or from user profile)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument(
        "--no-stratified-split",
        action="store_true",
        help="Use random row shuffle instead of per-label stratified val split (legacy behavior)",
    )
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--preprocess-mode", type=str, default="offline", choices=["offline", "streaming", "none"])
    ap.add_argument(
        "--emg-mode",
        type=str,
        default="",
        choices=["", "standard", "clinical", "wide", "gowda"],
        help="Bandpass preset (standard/clinical/wide/gowda). Default: user profile if --user-id else standard.",
    )
    ap.add_argument("--user-id", type=str, default="", help="If set, load/save UserProfile under --users-dir")
    ap.add_argument("--users-dir", type=str, default="", help="User data root (default: ./users)")
    ap.add_argument(
        "--touch-calibration-date",
        action="store_true",
        help="When saving the user profile, set calibration_date to now (does not change calibration_samples / SNR)",
    )
    ap.add_argument("--quiet", action="store_true", help="Disable epoch progress bars")
    ap.add_argument(
        "--arch",
        type=str,
        default=default_arch(),
        choices=["cnn", "se_resnet"],
        help="Model architecture (default: se_resnet)",
    )
    ap.add_argument(
        "--split-by",
        type=str,
        default="auto",
        choices=["auto", "event", "group", "gowda"],
        help="Train/val split: auto=group when trial_id present; gowda=370/30 sentence split",
    )
    ap.add_argument(
        "--channels",
        type=str,
        default="",
        help="Comma-separated channel indices to keep (e.g. 0,2,5). Overrides --top-channels.",
    )
    ap.add_argument(
        "--top-channels",
        type=int,
        default=0,
        help="Keep top-N channels from --channel-importance JSON",
    )
    ap.add_argument(
        "--channel-importance",
        type=str,
        default="",
        help="JSON from analyze channel-importance (used with --top-channels)",
    )
    ap.add_argument(
        "--teacher-checkpoints",
        type=str,
        default="",
        help="Comma-separated teacher model.pt paths for knowledge distillation",
    )
    ap.add_argument("--distill-alpha", type=float, default=0.5, help="Weight on hard CE vs KD loss")
    ap.add_argument("--distill-temperature", type=float, default=4.0)
    ap.add_argument(
        "--ensemble-count",
        type=int,
        default=0,
        help="Train N teacher models (seeds seed..seed+N-1) before optional distillation",
    )
    ap.add_argument(
        "--distill-student",
        action="store_true",
        help="After ensemble teachers, train distilled student to model.pt",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device (default: auto = CUDA if available)",
    )
    ap.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision (AMP is enabled by default on CUDA)",
    )
    ap.add_argument(
        "--no-preprocess-cache",
        action="store_true",
        help="Recompute preprocessing every run (do not read/write session cache)",
    )
    ap.add_argument(
        "--rebuild-preprocess-cache",
        action="store_true",
        help="Force rebuild preprocess_cache/ under the session folder",
    )
    ap.add_argument(
        "--no-segment-cache",
        action="store_true",
        help="Rebuild training tensors from events every run (no segments/ cache)",
    )
    ap.add_argument(
        "--rebuild-segment-cache",
        action="store_true",
        help="Force rebuild sessions/<name>/segments/*.npz",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="DataLoader workers (-1 = auto, 0 = main process only)",
    )
    ap.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile(model) on CUDA when available",
    )
    ap.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch = batch_size * steps)",
    )
    ap.add_argument(
        "--per-event-preprocess",
        action="store_true",
        help="Bandpass+z-score each word span independently (paper-style; no streaming state bleed)",
    )
    ap.add_argument(
        "--mmap-signals",
        action="store_true",
        help="Memory-map signals.npy instead of loading full array into RAM",
    )
    args = ap.parse_args()

    from ..users.defaults import default_users_dir
    from ..users.manager import UserManager
    from ..users.profile import UserProfile

    users_dir = Path(args.users_dir) if args.users_dir else default_users_dir()
    profile = None
    mgr: UserManager | None = None
    if args.user_id:
        mgr = UserManager(users_dir)
        profile = mgr.load_profile(args.user_id)
        if profile is None:
            raise SystemExit(
                f"Unknown user {args.user_id!r}. Create with: openalterego user create --user-id {args.user_id}"
            )

    if args.emg_mode:
        emg_mode: EmgMode = args.emg_mode  # type: ignore[assignment]
    elif profile is not None:
        emg_mode = profile.preprocessing_mode
    else:
        emg_mode = "standard"

    if emg_mode == "wide":
        validate_emg_wide_fs(float(args.fs))
    if emg_mode == "gowda":
        validate_emg_gowda_fs(float(args.fs))

    segment_ms = int(args.segment_ms) if args.segment_ms is not None else (int(profile.window_ms) if profile else 600)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data)
    sig_path = data_dir / "signals.npy"
    if bool(args.mmap_signals):
        signals = np.load(sig_path, mmap_mode="r")
    else:
        signals = np.load(sig_path)
    events = pd.read_csv(data_dir / "events.csv")

    labels = sorted(list({str(x) for x in events["label"].unique()}))
    label_to_id = {lab: i for i, lab in enumerate(labels)}

    groups = None
    if "trial_id" in events.columns and args.split_by in ("auto", "group", "gowda"):
        groups = events["trial_id"].astype(int).values

    if args.no_stratified_split:
        idx = np.arange(len(events))
        np.random.shuffle(idx)
        n_val = int(len(idx) * float(args.val_split))
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]
    else:
        tr_idx, val_idx = resolve_train_val_indices(
            events["label"].astype(str).values,
            float(args.val_split),
            int(args.seed),
            split_by=str(args.split_by),
            groups=groups,
        )

    tr_events = events.iloc[tr_idx].reset_index(drop=True)
    val_events = events.iloc[val_idx].reset_index(drop=True)
    if len(val_events) == 0 and float(args.val_split) > 0.0 and len(tr_events) > 1:
        val_events = tr_events.iloc[-1:].reset_index(drop=True)
        tr_events = tr_events.iloc[:-1].reset_index(drop=True)

    channel_indices = _parse_channel_indices(args.channels, int(signals.shape[1]))
    if channel_indices is None and int(args.top_channels) > 0:
        import json

        ci_path = Path(args.channel_importance) if args.channel_importance else (data_dir / "channel_importance.json")
        if not ci_path.is_file():
            raise SystemExit(f"--top-channels requires {ci_path}")
        ci = json.loads(ci_path.read_text(encoding="utf-8"))
        channel_indices = [int(x) for x in ci["top_channels"][: int(args.top_channels)]]

    device = resolve_device(str(args.device))
    configure_cuda_for_training(device)
    use_amp = resolve_use_amp(device, no_amp=bool(args.no_amp))
    show = not bool(args.quiet)
    out = (mgr.get_user_dir(args.user_id) / "model.pt") if args.user_id and mgr else (data_dir / "model.pt")

    split_note = str(args.split_by) if args.split_by != "auto" else (
        "group" if groups is not None else "event"
    )
    if show:
        ch_note = f" channels={len(channel_indices)}" if channel_indices else ""
        dev_note = str(device.type)
        if device.type == "cuda":
            dev_note = f"cuda ({torch.cuda.get_device_name(device)})"
        amp_note = ""
        if device.type == "cuda":
            amp_note = " amp=off" if args.no_amp else " amp=on"
        nw = int(args.num_workers)
        perf_note = ""
        if nw != 0:
            from .training_perf import default_num_workers

            perf_note = f" workers={default_num_workers(nw)}"
        if args.compile:
            perf_note += " compile=on"
        if int(args.grad_accum_steps) > 1:
            perf_note += f" accum={int(args.grad_accum_steps)}"
        print(
            f"[openalterego] train: arch={args.arch} split={split_note} "
            f"{len(tr_events)} train / {len(val_events)} val events, "
            f"{len(labels)} labels, emg_mode={emg_mode}, device={dev_note}{amp_note}{perf_note}{ch_note}"
        )

    teacher_paths: List[Path] = []
    if args.teacher_checkpoints.strip():
        teacher_paths = [Path(p.strip()) for p in args.teacher_checkpoints.split(",") if p.strip()]

    ensemble_count = int(args.ensemble_count)
    if ensemble_count > 0:
        teachers_dir = data_dir / "teachers"
        teachers_dir.mkdir(parents=True, exist_ok=True)
        teacher_paths = []
        for i in range(ensemble_count):
            t_seed = int(args.seed) + i
            random.seed(t_seed)
            np.random.seed(t_seed)
            torch.manual_seed(t_seed)
            t_path = teachers_dir / f"teacher_{i}.pt"
            if show:
                print(f"[openalterego] ensemble teacher {i + 1}/{ensemble_count} seed={t_seed}")
            _train_one_model(
                signals=signals,
                tr_events=tr_events,
                val_events=val_events,
                label_to_id=label_to_id,
                labels=labels,
                fs_hz=int(args.fs),
                segment_ms=segment_ms,
                preprocess_mode=str(args.preprocess_mode),
                emg_mode=emg_mode,
                seed=t_seed,
                arch=str(args.arch),
                channel_indices=channel_indices,
                batch_size=int(args.batch_size),
                epochs=int(args.epochs),
                lr=float(args.lr),
                device=device,
                save_best_path=t_path,
                show=show,
                user_id=str(args.user_id),
                use_amp=use_amp,
                session_dir=data_dir,
                use_preprocess_cache=not bool(args.no_preprocess_cache),
                rebuild_preprocess_cache=bool(args.rebuild_preprocess_cache),
                use_segment_cache=not bool(args.no_segment_cache),
                rebuild_segment_cache=bool(args.rebuild_segment_cache),
                num_workers=int(args.num_workers),
                compile_model=bool(args.compile),
                grad_accum_steps=int(args.grad_accum_steps),
                per_event_preprocess=bool(args.per_event_preprocess),
            )
            teacher_paths.append(t_path)

    teachers = None
    if teacher_paths:
        from .analysis import load_teacher_models

        teachers = load_teacher_models(teacher_paths, device)
        if show:
            print(f"[openalterego] distillation from {len(teachers)} teacher(s)")

    if ensemble_count > 0 and not args.distill_student and not args.teacher_checkpoints.strip():
        if show:
            print(f"[openalterego] ensemble done ({ensemble_count} teachers in {data_dir / 'teachers'})")
        return

    best_val = _train_one_model(
        signals=signals,
        tr_events=tr_events,
        val_events=val_events,
        label_to_id=label_to_id,
        labels=labels,
        fs_hz=int(args.fs),
        segment_ms=segment_ms,
        preprocess_mode=str(args.preprocess_mode),
        emg_mode=emg_mode,
        seed=int(args.seed),
        arch=str(args.arch),
        channel_indices=channel_indices,
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        device=device,
        save_best_path=out,
        show=show,
        teachers=teachers,
        distill_alpha=float(args.distill_alpha),
        distill_temperature=float(args.distill_temperature),
        user_id=str(args.user_id),
        use_amp=use_amp,
        session_dir=data_dir,
        use_preprocess_cache=not bool(args.no_preprocess_cache),
        rebuild_preprocess_cache=bool(args.rebuild_preprocess_cache),
        use_segment_cache=not bool(args.no_segment_cache),
        rebuild_segment_cache=bool(args.rebuild_segment_cache),
        num_workers=int(args.num_workers),
        compile_model=bool(args.compile),
        grad_accum_steps=int(args.grad_accum_steps),
        per_event_preprocess=bool(args.per_event_preprocess),
    )

    if args.user_id and profile is not None and mgr is not None:
        cal_date = float(time.time()) if args.touch_calibration_date else profile.calibration_date
        updated = UserProfile(
            user_id=profile.user_id,
            created_at=profile.created_at,
            model_path=out,
            confidence_threshold=profile.confidence_threshold,
            preprocessing_mode=emg_mode,
            window_ms=segment_ms,
            stride_ms=profile.stride_ms,
            calibration_date=cal_date,
            calibration_samples=profile.calibration_samples,
            baseline_snr=profile.baseline_snr,
        )
        mgr.save_profile(updated)
        print(f"[openalterego] updated user profile: {mgr.get_profile_path(args.user_id)}")

    print(f"done. best val acc={best_val:.3f}")


if __name__ == "__main__":
    main()
