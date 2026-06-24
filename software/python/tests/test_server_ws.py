"""Async WebSocket smoke tests for the realtime server (no BLE / no model file)."""

from __future__ import annotations

import asyncio
import contextlib
import json

import pytest
import websockets

from openalterego.api.server import (
    WebSocketHub,
    _aiter_from_generator,
    run_demo_tokens,
    run_pipeline_open_speech,
)
from openalterego.acquisition.simulate import stream_simulated_chunks


@pytest.mark.asyncio
async def test_websocket_demo_emits_token_json() -> None:
    hub = WebSocketHub()
    async with websockets.serve(hub.handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        uri = f"ws://127.0.0.1:{port}"
        async with websockets.connect(uri) as ws:
            demo = asyncio.create_task(run_demo_tokens(hub, period_s=0.05))
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(raw)
                assert data["type"] == "token"
                assert "token" in data
                assert "confidence" in data
            finally:
                demo.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await demo


@pytest.mark.asyncio
async def test_websocket_control_ping_pong() -> None:
    hub = WebSocketHub()
    async with websockets.serve(hub.handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        uri = f"ws://127.0.0.1:{port}"
        async with websockets.connect(uri) as ws:
            await ws.send(json.dumps({"type": "control", "cmd": "ping"}))
            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            data = json.loads(raw)
            assert data["type"] == "status"
            assert data["status"] == "pong"
            assert "t" in data


@pytest.mark.asyncio
async def test_websocket_ptt_control_queued() -> None:
    hub = WebSocketHub()
    async with websockets.serve(hub.handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        uri = f"ws://127.0.0.1:{port}"
        async with websockets.connect(uri) as ws:
            await ws.send(json.dumps({"type": "control", "cmd": "ptt_start"}))
            await asyncio.sleep(0.05)
            assert not hub.ptt_events.empty()
            assert hub.ptt_events.get_nowait() == "ptt_start"


@pytest.mark.asyncio
async def test_open_speech_ptt_emits_final_transcript() -> None:
    from pathlib import Path

    ckpt = Path("sessions/gowda_sv_full/ablations/ctc_spd_v3_diag_delta_seed1337.pt")
    sess = Path("sessions/gowda_sv_full")
    if not ckpt.is_file():
        pytest.skip("phase5 checkpoint not present")

    from openalterego.sim.scenarios.gowda_small_vocab import build_gowda_sim_config

    hub = WebSocketHub()
    sim_cfg = build_gowda_sim_config(n_trials=1, seed=1337, realism="off", realtime_clock=False)
    chunks = _aiter_from_generator(stream_simulated_chunks(sim_cfg))

    async with websockets.serve(hub.handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        uri = f"ws://127.0.0.1:{port}"
        pipe = asyncio.create_task(
            run_pipeline_open_speech(
                hub=hub,
                source_name="sim",
                chunks=chunks,
                ctc_checkpoint=str(ckpt),
                ctc_session=str(sess),
                device_preferred="cpu",
            )
        )
        try:
            async with websockets.connect(uri) as ws:
                await ws.send(json.dumps({"type": "control", "cmd": "ptt_start"}))
                await asyncio.sleep(0.2)
                await ws.send(json.dumps({"type": "control", "cmd": "ptt_end"}))
                deadline = asyncio.get_event_loop().time() + 120.0
                while asyncio.get_event_loop().time() < deadline:
                    raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
                    data = json.loads(raw)
                    if data.get("type") == "final_transcript":
                        assert "text" in data
                        assert data["confidence"] >= 0.0
                        return
                pytest.fail("no final_transcript received")
        finally:
            pipe.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pipe
