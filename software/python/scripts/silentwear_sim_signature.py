"""Compare one public training recording with a matching-shape synthetic probe."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from openalterego.research.signal_signature import event_windows, signal_signature
from openalterego.research.virtual_montage import sha256_file
from openalterego.sim.biophysical.stream import BiophysicalSimStream, BiophysicalSimStreamConfig
from openalterego.sim.stream import ScenarioConfig, SimStream, SimStreamConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True, help="SilentWear session-1 silent HDF5 only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=180)
    parser.add_argument("--engine", choices=("heuristic", "biophysical"), default="biophysical")
    args = parser.parse_args()
    if ("sess_1_batch_" not in args.recording.name or not set(args.recording.parts).intersection({"S01", "S02", "S03"})
            or args.seconds < 30):
        parser.error("use an S01-S03 session-1 training batch and at least 30 synthetic seconds")
    with h5py.File(args.recording, "r") as source:
        group = source["emg"]
        expected = [f"Ch_{i}_filt".encode() for i in list(range(11)) + [13, 14, 15]]
        if list(group["block1_items"][:]) != expected or list(group["block2_items"][:]) != [b"Label_int"]:
            parser.error("unexpected publisher channel/label layout")
        real = np.asarray(group["block1_values"][:], dtype=np.float32)
        real_labels = np.asarray(group["block2_values"][:, 0], dtype=np.int64)
    real_windows, real_classes = event_windows(real, real_labels)
    real_stats = signal_signature(real_windows, real_classes)

    scenario = ScenarioConfig(labels=[str(i) for i in range(1, 9)],
                              gap_duration_s=(0.85, 1.15), event_duration_s=(0.85, 1.15), p_event=1.0)
    if args.engine == "biophysical":
        cfg = BiophysicalSimStreamConfig(fs_hz=500, channels=14, seed=1337, realtime_clock=False,
                                        scenario=scenario)
        sim = BiophysicalSimStream(cfg)
    else:
        cfg = SimStreamConfig(fs_hz=500, channels=14, seed=1337, realtime_clock=False,
                              scenario=scenario)
        sim = SimStream(cfg)
    chunks, labels = [], []
    for _ in range(args.seconds * cfg.fs_hz // sim.chunk_samples):
        chunk = sim.next_chunk()
        chunks.append(chunk.samples)
        label = chunk.meta["sim_active_label"]
        active = int(chunk.meta["sim_injected_samples"])
        labels.extend(([int(label)] * active if label is not None else []) + [0] * (len(chunk.samples) - active))
    synthetic_windows, synthetic_classes = event_windows(np.concatenate(chunks), np.asarray(labels))
    synthetic_stats = signal_signature(synthetic_windows, synthetic_classes)
    ratios = {key: synthetic_stats[key] / max(real_stats[key], 1e-12)
              for key in ("median_centered_rms", "median_waveform_length", "command_to_rest_rms_ratio",
                          "median_absolute_channel_correlation")}
    report = {"schema": "openalterego.signal-signature.v1", "input": {"file": args.recording.name,
              "sha256": sha256_file(args.recording)}, "simulator": {"engine": args.engine, "seed": cfg.seed,
              "seconds": args.seconds, "sample_rate_hz": 500, "channels": 14}, "real": real_stats,
              "synthetic": synthetic_stats, "synthetic_to_real_ratios": ratios,
              "scope": "Exploratory signal-shape diagnostic on one training recording; no anatomical or device-physics validation."}
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    print(json.dumps(ratios, indent=2))


if __name__ == "__main__":
    main()
