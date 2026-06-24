"""Short sim-source pipeline run (real checkpoint + StreamingClassifier)."""

from __future__ import annotations

import pytest
import torch

from openalterego.acquisition.simulate import SimConfig, stream_simulated_chunks
from openalterego.api.server import WebSocketHub, _aiter_from_generator, run_pipeline
from openalterego.ml.model import OpenAlterEgoCNN
from openalterego.runtime.streaming import StreamDecodeConfig


@pytest.mark.asyncio
async def test_run_pipeline_sim_exits_after_n_chunks(tmp_path: Path) -> None:
    labels = ["yes", "no", "left", "right", "select", "cancel"]
    m = OpenAlterEgoCNN(channels=8, classes=len(labels))
    ckpt = tmp_path / "m.pt"
    torch.save(
        {
            "state_dict": m.state_dict(),
            "labels": labels,
            "fs": 250,
            "channels": 8,
            "preprocess_mode": "streaming",
            "emg_mode": "standard",
            "segment_ms": 600,
        },
        ckpt,
    )

    hub = WebSocketHub()
    sim = SimConfig(fs_hz=250, channels=8, realtime_clock=False, labels=labels)
    achunks = _aiter_from_generator(stream_simulated_chunks(sim))

    async def first_n(ait, n: int):
        i = 0
        async for c in ait:
            yield c
            i += 1
            if i >= n:
                break

    await run_pipeline(
        hub=hub,
        source_name="sim",
        chunks=first_n(achunks, 24),
        model_path=str(ckpt),
        decode_cfg=StreamDecodeConfig(
            stable_n=1,
            min_confidence=0.0,
            window_ms=600,
            stride_ms=200,
        ),
        online_preproc=True,
        user_profile=None,
        online_quality=True,
        channel_quality_meta=True,
        weak_channel_deficit_db=6.0,
    )
