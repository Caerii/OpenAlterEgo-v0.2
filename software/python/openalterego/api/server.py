"""Websocket server for OpenAlterEgo realtime output.

Design goal: Unity / XR / smartglasses can connect and receive JSON messages like:

    {"type":"token","token":"yes","confidence":0.92,"t":1730000000.123,"seq":1234,"source":"sim"}

Quick start (pure demo tokens):
    python -m openalterego.api.server --source demo

Synthetic signal -> virtual BLE -> streaming ML:
    # 1) generate a synthetic dataset
    python -m openalterego.cli sim-dataset --out ./sim_session --minutes 2
    # 2) train a small model
    python -m openalterego.ml.train --data ./sim_session --fs 250 --preprocess-mode streaming
    # 3) serve realtime
    python -m openalterego.api.server --source sim --model ./sim_session/model.pt

Notes
-----
This is a starter server. It's intentionally small and hackable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

import websockets
from websockets.server import WebSocketServerProtocol

from ..acquisition.ble_client import BleSpec, stream_ble_chunks
from ..acquisition.simulate import SimConfig, stream_simulated_chunks
from ..acquisition.virtual import VirtualBleSpec, stream_virtual_ble_chunks
from ..core.types import FrameChunk, TokenEvent
from ..dsp.online import OnlinePreprocessor
from ..ml.infer import load_model
from ..runtime.streaming import StreamDecodeConfig, StreamingClassifier
from .protocol import ControlMessage, StatusMessage, TokenMessage


log = logging.getLogger("openalterego.server")


class WebSocketHub:
    def __init__(self):
        self.clients: Set[WebSocketServerProtocol] = set()

    async def broadcast(self, msg: Dict[str, Any]) -> None:
        if not self.clients:
            return
        s = json.dumps(msg)
        dead = []
        send_tasks = []
        for ws in list(self.clients):
            send_tasks.append(ws.send(s))
        results = await asyncio.gather(*send_tasks, return_exceptions=True)
        for ws, r in zip(list(self.clients), results):
            if isinstance(r, Exception):
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    async def handler(self, ws: WebSocketServerProtocol):
        self.clients.add(ws)
        try:
            async for raw in ws:
                # best-effort: parse control messages, otherwise ignore
                try:
                    data = json.loads(raw)
                except Exception:
                    continue
                if isinstance(data, dict) and data.get("type") == "control":
                    try:
                        msg = ControlMessage.model_validate(data)
                    except Exception:
                        continue
                    if msg.cmd == "ping":
                        await ws.send(StatusMessage(t=time.time(), status="pong").model_dump_json())
        finally:
            self.clients.discard(ws)


async def _source_iter(args) -> asyncio.Queue:
    raise NotImplementedError


async def run_pipeline(
    *,
    hub: WebSocketHub,
    source_name: str,
    chunks,
    model_path: str,
    decode_cfg: StreamDecodeConfig,
    online_preproc: bool = True,
) -> None:
    """Consume chunks, run inference, broadcast tokens."""
    lm = load_model(model_path)
    if lm.preprocess_mode not in ("streaming", "none"):
        log.warning(
            "Model checkpoint preprocess_mode=%s. For best realtime results, train with --preprocess-mode streaming.",
            lm.preprocess_mode,
        )

    if online_preproc:
        pre = OnlinePreprocessor(fs_hz=lm.fs, channels=lm.channels)
    else:
        pre = None

    clf = StreamingClassifier(model=lm, cfg=decode_cfg, source_name=source_name)

    await hub.broadcast(StatusMessage(t=time.time(), status="pipeline_started", meta={"source": source_name}).model_dump())

    async for chunk in chunks:
        if pre is not None:
            y = pre.process(chunk.samples)
            chunk = FrameChunk(samples=y, fs_hz=chunk.fs_hz, t0=chunk.t0, seq0=chunk.seq0, meta=chunk.meta)

        events = clf.push(chunk)
        for ev in events:
            msg = TokenMessage(
                token=ev.token,
                confidence=ev.confidence,
                t=ev.t,
                seq=ev.seq,
                source=ev.source,
                meta=ev.meta,
            ).model_dump()
            await hub.broadcast(msg)


async def run_demo_tokens(hub: WebSocketHub, *, period_s: float = 1.5) -> None:
    tokens = ["yes", "no", "left", "right", "select", "cancel"]
    i = 0
    while True:
        tok = tokens[i % len(tokens)]
        i += 1
        msg = TokenMessage(token=tok, confidence=1.0, t=time.time(), seq=i, source="demo").model_dump()
        await hub.broadcast(msg)
        await asyncio.sleep(period_s)


async def main_async() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)

    ap.add_argument("--source", type=str, default="demo", choices=["demo", "sim", "virtual_ble", "ble"])
    ap.add_argument("--model", type=str, default="", help="Path to model.pt (required for sim/virtual_ble/ble)")

    # decode config
    ap.add_argument("--window-ms", type=int, default=600)
    ap.add_argument("--stride-ms", type=int, default=120)
    ap.add_argument("--min-confidence", type=float, default=0.70)
    ap.add_argument("--stable-n", type=int, default=3)

    # sim source knobs
    ap.add_argument("--labels", type=str, default="yes,no,left,right,select,cancel")
    ap.add_argument("--seed", type=int, default=1337)

    # ble knobs
    ap.add_argument("--device-name", type=str, default="")
    ap.add_argument("--data-uuid", type=str, default="")
    ap.add_argument("--packet-format", type=str, default="raw_i16", choices=["raw_i16", "oa_v1"])

    # virtual_ble knobs (loss/jitter)
    ap.add_argument("--loss", type=float, default=0.0)
    ap.add_argument("--jitter-ms", type=float, default=0.0)
    ap.add_argument("--extra-latency-ms", type=float, default=0.0)

    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    hub = WebSocketHub()

    decode_cfg = StreamDecodeConfig(
        window_ms=int(args.window_ms),
        stride_ms=int(args.stride_ms),
        min_confidence=float(args.min_confidence),
        stable_n=int(args.stable_n),
    )

    async with websockets.serve(hub.handler, args.host, args.port):
        log.info("OpenAlterEgo server on ws://%s:%s (source=%s)", args.host, args.port, args.source)

        if args.source == "demo":
            await run_demo_tokens(hub)
            return

        if not args.model:
            raise SystemExit("--model is required when --source is not demo")

        if args.source == "sim":
            sim_cfg = SimConfig(labels=[s.strip() for s in args.labels.split(",") if s.strip()], seed=int(args.seed))
            chunks = _aiter_from_generator(stream_simulated_chunks(sim_cfg))
            await run_pipeline(hub=hub, source_name="sim", chunks=chunks, model_path=args.model, decode_cfg=decode_cfg)
            return

        if args.source == "virtual_ble":
            vs = VirtualBleSpec()
            vs.channels = 8
            vs.fs_hz = 250
            vs.link.loss_prob = float(args.loss)
            vs.link.jitter_ms = float(args.jitter_ms)
            vs.link.extra_latency_ms = float(args.extra_latency_ms)
            vs.sim.seed = int(args.seed)
            chunks = stream_virtual_ble_chunks(vs)
            await run_pipeline(
                hub=hub,
                source_name="virtual_ble",
                chunks=chunks,
                model_path=args.model,
                decode_cfg=decode_cfg,
            )
            return

        if args.source == "ble":
            if not args.device_name or not args.data_uuid:
                raise SystemExit("--device-name and --data-uuid are required for --source ble")
            bs = BleSpec(
                device_name=str(args.device_name),
                data_char_uuid=str(args.data_uuid),
                fs_hz=250,
                channels=8,
                packet_format=str(args.packet_format),
            )
            chunks = stream_ble_chunks(bs)
            await run_pipeline(hub=hub, source_name="ble", chunks=chunks, model_path=args.model, decode_cfg=decode_cfg)
            return


def _aiter_from_generator(gen):
    """Wrap a blocking generator into an async iterator (runs in default loop)."""
    async def _aiter():
        loop = asyncio.get_running_loop()
        it = iter(gen)
        while True:
            # pull next item in a thread so we don't block the event loop
            item = await loop.run_in_executor(None, lambda: next(it))
            yield item
    return _aiter()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
