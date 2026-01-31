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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import OpenAlterEgoCNN
from ..dsp.filters import preprocess_basic, preprocess_streaming


PreprocessMode = Literal["offline", "streaming", "none"]


@dataclass
class Example:
    x: np.ndarray  # (channels, time)
    y: int


class SegmentDataset(Dataset):
    def __init__(
        self,
        signals: np.ndarray,  # (time, channels)
        events: pd.DataFrame,
        label_to_id: Dict[str, int],
        fs_hz: int,
        segment_ms: int,
        preprocess_mode: PreprocessMode = "offline",
        seed: int = 1337,
    ):
        self.fs_hz = int(fs_hz)
        self.label_to_id = label_to_id
        self.rng = np.random.default_rng(int(seed))
        self.segment_samples = max(8, int(self.fs_hz * int(segment_ms) / 1000))

        if preprocess_mode == "offline":
            signals = preprocess_basic(signals, fs_hz=self.fs_hz, rectify_signals=False, normalize_mode="zscore")
        elif preprocess_mode == "streaming":
            signals = preprocess_streaming(
                signals,
                fs_hz=self.fs_hz,
                channels=int(signals.shape[1]),
                rectify_signals=False,
                ema_alpha=0.01,
            )
        elif preprocess_mode == "none":
            signals = signals.astype(np.float32, copy=False)
        else:
            raise ValueError(f"unknown preprocess_mode: {preprocess_mode}")

        self.signals = signals.astype(np.float32, copy=False)

        self.items: List[Example] = []
        for _, row in events.iterrows():
            s = int(row["start_sample"])
            e = int(row["end_sample"])
            label = str(row["label"])
            if label not in label_to_id:
                continue
            y = int(label_to_id[label])
            seg = self.signals[s:e, :]  # (time, ch)
            if seg.shape[0] < 8:
                continue

            x = seg.T  # (ch, time)
            x = self._crop_or_pad(x)
            self.items.append(Example(x=x, y=y))

    def _crop_or_pad(self, x: np.ndarray) -> np.ndarray:
        ch, t = x.shape
        n = self.segment_samples
        if t == n:
            return x.astype(np.float32, copy=False)
        if t > n:
            # Random crop for augmentation.
            start = int(self.rng.integers(0, t - n + 1))
            return x[:, start : start + n].astype(np.float32, copy=False)
        # pad
        pad = n - t
        out = np.zeros((ch, n), dtype=np.float32)
        out[:, :t] = x.astype(np.float32, copy=False)
        return out

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ex = self.items[idx]
        x = torch.from_numpy(ex.x)  # (ch, time)
        y = torch.tensor(ex.y, dtype=torch.long)
        return x, y


@torch.no_grad()
def evaluate(model: torch.nn.Module, dl: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        n += x.size(0)
    if n == 0:
        return 0.0, 0.0
    return total_loss / n, correct / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to session folder")
    ap.add_argument("--fs", type=int, required=True, help="Sampling rate (Hz)")
    ap.add_argument("--segment-ms", type=int, default=600, help="Crop/pad each segment to this length")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--preprocess-mode", type=str, default="offline", choices=["offline", "streaming", "none"])
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data)
    signals = np.load(data_dir / "signals.npy")
    events = pd.read_csv(data_dir / "events.csv")

    labels = sorted(list({str(x) for x in events["label"].unique()}))
    label_to_id = {lab: i for i, lab in enumerate(labels)}

    # split by event rows
    idx = np.arange(len(events))
    np.random.shuffle(idx)
    n_val = int(len(idx) * float(args.val_split))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    tr_events = events.iloc[tr_idx].reset_index(drop=True)
    val_events = events.iloc[val_idx].reset_index(drop=True)

    ds_tr = SegmentDataset(
        signals,
        tr_events,
        label_to_id,
        fs_hz=int(args.fs),
        segment_ms=int(args.segment_ms),
        preprocess_mode=args.preprocess_mode,
        seed=int(args.seed),
    )
    ds_val = SegmentDataset(
        signals,
        val_events,
        label_to_id,
        fs_hz=int(args.fs),
        segment_ms=int(args.segment_ms),
        preprocess_mode=args.preprocess_mode,
        seed=int(args.seed) + 1,
    )

    if len(ds_tr) == 0:
        raise SystemExit("No training segments found. Check events.csv or preprocessing settings.")

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OpenAlterEgoCNN(channels=int(signals.shape[1]), classes=len(labels)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = -1.0
    out = data_dir / "model.pt"

    for epoch in range(int(args.epochs)):
        model.train()
        total = 0.0
        correct = 0
        n = 0
        pbar = tqdm(dl_tr, desc=f"epoch {epoch+1}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device)  # (B, ch, time)
            y = y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item()) * x.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            n += x.size(0)
            pbar.set_postfix(loss=total / max(n, 1), acc=correct / max(n, 1))

        val_loss, val_acc = evaluate(model, dl_val, device=device)
        print(f"[val] loss={val_loss:.4f} acc={val_acc:.3f} (n={len(ds_val)})")

        if val_acc >= best_val:
            best_val = float(val_acc)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "labels": labels,
                    "fs": int(args.fs),
                    "channels": int(signals.shape[1]),
                    "preprocess_mode": str(args.preprocess_mode),
                    "segment_ms": int(args.segment_ms),
                },
                out,
            )
            print(f"saved best: {out} (val_acc={best_val:.3f})")

    print(f"done. best val acc={best_val:.3f}")


if __name__ == "__main__":
    main()
