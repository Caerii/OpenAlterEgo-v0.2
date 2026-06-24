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
from pathlib import Path
from typing import Any, Dict, Optional, Set

import numpy as np
import websockets
from websockets.server import ServerConnection

from ..acquisition.ble_client import BleSpec, stream_ble_chunks
from ..acquisition.simulate import SimConfig, stream_simulated_chunks
from ..acquisition.virtual import VirtualBleSpec, stream_virtual_ble_chunks
from ..core.types import FrameChunk, TokenEvent
from ..dsp.emg_config import (
    build_online_preprocessor,
    emg_signal_band_hz_for_quality,
    resolve_emg_mode_for_serve,
)
from ..dsp.quality import OnlineQualityMonitor, weak_channel_indices
from ..ml.infer import LoadedModel, load_model
from ..runtime.streaming import StreamDecodeConfig, StreamingClassifier
from ..users.defaults import default_users_dir
from ..users.manager import UserManager
from ..users.profile import UserProfile
from ..users.recalibration import RecalibrationMonitor
from .protocol import ControlMessage, FinalTranscriptMessage, StatusMessage, TokenMessage


log = logging.getLogger("openalterego.server")


class WebSocketHub:
    def __init__(self):
        self.clients: Set[ServerConnection] = set()
        self.ptt_events: asyncio.Queue[str] = asyncio.Queue()

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

    async def handler(self, ws: ServerConnection):
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
                    elif msg.cmd in ("ptt_start", "ptt_end"):
                        await self.ptt_events.put(str(msg.cmd))
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
    loaded_model: Optional[LoadedModel] = None,
    decode_cfg: StreamDecodeConfig,
    online_preproc: bool = True,
    user_profile: Optional[UserProfile] = None,
    online_quality: bool = False,
    quality_every_n_chunks: int = 1,
    quality_warn_db: float = 5.0,
    quality_status_interval_s: float = 30.0,
    channel_quality_meta: bool = False,
    weak_channel_deficit_db: float = 6.0,
    weak_channel_warn: bool = False,
    weak_channel_status_interval_s: float = 15.0,
    motion_gate: bool = False,
    motion_threshold: float = 0.35,
    motion_attenuation: float = 0.15,
) -> None:
    """Consume chunks, run inference, broadcast tokens."""
    lm = loaded_model if loaded_model is not None else load_model(model_path)
    if lm.preprocess_mode not in ("streaming", "none"):
        log.warning(
            "Model checkpoint preprocess_mode=%s. For best realtime results, train with --preprocess-mode streaming.",
            lm.preprocess_mode,
        )
    if lm.user_id:
        log.info("Loaded checkpoint user_id=%s", lm.user_id)
    log.info(
        "Loaded model: fs=%d ch=%d labels=%d preprocess_mode=%s emg_mode=%s",
        lm.fs,
        lm.channels,
        len(lm.labels),
        lm.preprocess_mode,
        lm.emg_mode if lm.emg_mode is not None else "(legacy ckpt)",
    )

    serve_emg = resolve_emg_mode_for_serve(
        checkpoint_emg_mode=lm.emg_mode,
        profile_preprocessing_mode=user_profile.preprocessing_mode if user_profile else None,
    )
    if online_preproc:
        if lm.fs <= 0:
            raise ValueError("invalid model fs")
        pre = build_online_preprocessor(
            fs_hz=float(lm.fs),
            channels=lm.channels,
            emg_mode=serve_emg,
            motion_gate=bool(motion_gate),
            motion_threshold=float(motion_threshold),
            motion_attenuation=float(motion_attenuation),
        )
        if motion_gate:
            log.info(
                "Motion gate enabled (threshold=%.2f attenuation=%.2f)",
                motion_threshold,
                motion_attenuation,
            )
    else:
        pre = None

    qmon: Optional[OnlineQualityMonitor] = None
    if online_quality:
        sig_lo, sig_hi = emg_signal_band_hz_for_quality(serve_emg, float(lm.fs))
        qmon = OnlineQualityMonitor(
            fs_hz=float(lm.fs),
            window_samples=min(max(int(lm.fs * 2), 256), 2048),
            signal_band_hz=(sig_lo, sig_hi),
            per_channel=bool(channel_quality_meta),
        )
        log.info("Online quality monitor enabled (signal band %.1f–%.1f Hz)", sig_lo, sig_hi)

    clf = StreamingClassifier(model=lm, cfg=decode_cfg, source_name=source_name)

    recal_mon: Optional[RecalibrationMonitor] = None
    if online_quality and decode_cfg.baseline_snr_db is not None:
        recal_mon = RecalibrationMonitor.from_baseline(
            float(decode_cfg.baseline_snr_db),
            warn_db=float(quality_warn_db),
            cooldown_s=float(quality_status_interval_s),
        )

    await hub.broadcast(StatusMessage(t=time.time(), status="pipeline_started", meta={"source": source_name}).model_dump())

    last_quality_warn_t = 0.0
    last_weak_channel_t = 0.0
    q_chunk_i = 0
    last_qm = None
    q_every = max(1, int(quality_every_n_chunks))
    async for chunk in chunks:
        y = chunk.samples
        if pre is not None:
            y = pre.process(chunk.samples)
            meta_pre = {
                "motion_index": float(pre.last_motion_index),
                "motion_gated": bool(pre.last_motion_gated),
            }
        else:
            meta_pre = {}

        meta = dict(chunk.meta)
        meta.update(meta_pre)
        if qmon is not None:
            q_chunk_i += 1
            if (q_chunk_i - 1) % q_every == 0:
                last_qm = qmon.update(y)
            if last_qm is not None and last_qm.snr_db is not None:
                meta["snr_db"] = float(last_qm.snr_db)
            if (
                channel_quality_meta
                and last_qm is not None
                and last_qm.snr_db_per_channel is not None
            ):
                sdb = last_qm.snr_db_per_channel
                meta["snr_db_per_channel"] = [float(v) for v in np.asarray(sdb).tolist()]
                meta["weak_channels"] = weak_channel_indices(
                    np.asarray(sdb, dtype=np.float64),
                    deficit_db=float(weak_channel_deficit_db),
                )
            baseline = decode_cfg.baseline_snr_db
            qm = last_qm
            if recal_mon is not None and qm is not None and qm.snr_db is not None:
                st = recal_mon.update(float(qm.snr_db), motion_index=float(qm.motion_index))
                meta.update(st.to_meta())
                if st.should_broadcast:
                    await hub.broadcast(
                        StatusMessage(
                            t=time.time(),
                            status="quality_warning",
                            meta={
                                **st.to_meta(),
                                "baseline_snr_db": float(baseline) if baseline is not None else None,
                                "status_detail": "recalibration_monitor",
                            }
                            | (
                                {
                                    "weak_channels": weak_channel_indices(
                                        np.asarray(qm.snr_db_per_channel, dtype=np.float64),
                                        deficit_db=float(weak_channel_deficit_db),
                                    ),
                                }
                                if channel_quality_meta and qm.snr_db_per_channel is not None
                                else {}
                            ),
                        ).model_dump()
                    )
            elif baseline is not None and qm is not None and qm.snr_db is not None:
                if float(qm.snr_db) < float(baseline) - float(quality_warn_db):
                    now = time.time()
                    if now - last_quality_warn_t >= float(quality_status_interval_s):
                        await hub.broadcast(
                            StatusMessage(
                                t=now,
                                status="quality_warning",
                                meta=(
                                    {
                                        "snr_db": float(qm.snr_db),
                                        "baseline_snr_db": float(baseline),
                                        "re_calibration_suggested": True,
                                        "motion_index": float(qm.motion_index),
                                    }
                                    | (
                                        {
                                            "weak_channels": weak_channel_indices(
                                                np.asarray(qm.snr_db_per_channel, dtype=np.float64),
                                                deficit_db=float(weak_channel_deficit_db),
                                            ),
                                        }
                                        if channel_quality_meta
                                        and qm.snr_db_per_channel is not None
                                        else {}
                                    )
                                ),
                            ).model_dump()
                        )
                        last_quality_warn_t = now

            if (
                weak_channel_warn
                and channel_quality_meta
                and last_qm is not None
                and last_qm.snr_db_per_channel is not None
            ):
                weak = weak_channel_indices(
                    np.asarray(last_qm.snr_db_per_channel, dtype=np.float64),
                    deficit_db=float(weak_channel_deficit_db),
                )
                if weak:
                    now = time.time()
                    if now - last_weak_channel_t >= float(weak_channel_status_interval_s):
                        sdb = last_qm.snr_db_per_channel
                        await hub.broadcast(
                            StatusMessage(
                                t=now,
                                status="channel_quality",
                                meta={
                                    "weak_channels": weak,
                                    "snr_db_per_channel": [float(v) for v in np.asarray(sdb).tolist()],
                                    "median_snr_db": float(np.median(sdb[np.isfinite(sdb)]))
                                    if np.any(np.isfinite(sdb))
                                    else None,
                                },
                            ).model_dump()
                        )
                        last_weak_channel_t = now

        chunk = FrameChunk(samples=y, fs_hz=chunk.fs_hz, t0=chunk.t0, seq0=chunk.seq0, meta=meta)

        events = clf.push(chunk)
        for ev in events:
            tok_meta = dict(ev.meta)
            if (
                channel_quality_meta
                and last_qm is not None
                and last_qm.snr_db_per_channel is not None
            ):
                sdb = last_qm.snr_db_per_channel
                tok_meta["snr_db_per_channel"] = [float(v) for v in np.asarray(sdb).tolist()]
                tok_meta["weak_channels"] = weak_channel_indices(
                    np.asarray(sdb, dtype=np.float64),
                    deficit_db=float(weak_channel_deficit_db),
                )
            msg = TokenMessage(
                token=ev.token,
                confidence=ev.confidence,
                t=ev.t,
                seq=ev.seq,
                source=ev.source,
                meta=tok_meta,
            ).model_dump()
            await hub.broadcast(msg)


async def run_pipeline_open_speech(
    *,
    hub: WebSocketHub,
    source_name: str,
    chunks,
    ctc_checkpoint: str,
    ctc_session: str,
    fs_hz: int = 5000,
    channels: int = 31,
    feature_mode: str = "diag_delta",
    ctc_decode_mode: str = "trial",
    lm_weight: float = 0.0,
    device_preferred: str = "auto",
) -> None:
    """PTT open-speech: buffer utterance, SPD+CTC decode, broadcast final_transcript."""
    from ..runtime.ctc_streaming import build_streaming_ctc_decoder

    decoder = build_streaming_ctc_decoder(
        ctc_checkpoint,
        ctc_session,
        device_preferred=str(device_preferred),
        fs_hz=int(fs_hz),
        channels=int(channels),
        feature_mode=str(feature_mode),
        decode_mode=str(ctc_decode_mode),
        lm_weight=float(lm_weight),
        source_name=str(source_name),
    )
    log.info(
        "Open-speech CTC decoder: checkpoint=%s session=%s mode=%s",
        ctc_checkpoint,
        ctc_session,
        ctc_decode_mode,
    )
    await hub.broadcast(
        StatusMessage(
            t=time.time(),
            status="pipeline_started",
            meta={"source": source_name, "decode_mode": "open_speech"},
        ).model_dump()
    )
    seq = 0
    async for chunk in chunks:
        while not hub.ptt_events.empty():
            cmd = hub.ptt_events.get_nowait()
            if cmd == "ptt_start":
                decoder.on_ptt_start()
                await hub.broadcast(
                    StatusMessage(t=time.time(), status="ptt_recording", meta={"cmd": cmd}).model_dump()
                )
            elif cmd == "ptt_end":
                result = decoder.on_ptt_end()
                if result is not None:
                    seq += 1
                    await hub.broadcast(
                        FinalTranscriptMessage(
                            text=result.text,
                            confidence=float(result.confidence),
                            alternatives=list(result.words),
                            utterance_id=result.utterance_id,
                            t=time.time(),
                            seq=int(seq),
                            source=str(source_name),
                            meta=dict(result.meta),
                        ).model_dump()
                    )
        decoder.feed_chunk(np.asarray(chunk.samples, dtype=np.float32))


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
    ap.add_argument(
        "--decode-mode",
        type=str,
        default="commands",
        choices=["commands", "open_speech"],
        help="commands: CNN token stream; open_speech: PTT SPD+CTC transcripts",
    )
    ap.add_argument("--model", type=str, default="", help="Path to model.pt (overrides user profile model_path)")
    ap.add_argument(
        "--ctc-checkpoint",
        type=str,
        default="",
        help="SPD+CTC checkpoint for --decode-mode open_speech",
    )
    ap.add_argument(
        "--ctc-session",
        type=str,
        default="",
        help="Session dir for lexicon/LM/SPD basis (e.g. ./sessions/gowda_sv_full)",
    )
    ap.add_argument("--ctc-feature-mode", type=str, default="diag_delta")
    ap.add_argument("--ctc-decode-mode", type=str, default="trial", choices=["trial", "word"])
    ap.add_argument("--ctc-lm-weight", type=float, default=0.0)
    ap.add_argument("--ctc-device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--user-id", type=str, default="", help="Load model and decode settings from user profile")
    ap.add_argument("--users-dir", type=str, default="", help="User data root (default: ./users)")

    # decode config (use None to apply defaults from user profile when --user-id is set)
    ap.add_argument("--window-ms", type=int, default=None, help="Default: 600, or user profile window_ms")
    ap.add_argument("--stride-ms", type=int, default=None, help="Default: 120, or user profile stride_ms")
    ap.add_argument("--min-confidence", type=float, default=None, help="Default: 0.7, or user confidence_threshold")
    ap.add_argument("--stable-n", type=int, default=3)
    ap.add_argument(
        "--adaptive-threshold",
        action="store_true",
        help="EMA-adjust confidence gate from recent prediction confidences",
    )
    ap.add_argument("--threshold-alpha", type=float, default=0.1, help="EMA blend factor for adaptive threshold (0,1]")
    ap.add_argument(
        "--baseline-snr-db",
        type=float,
        default=None,
        help="Raise gate when chunk meta snr_db is below this (default: user profile baseline_snr if set)",
    )
    ap.add_argument(
        "--no-snr-gate",
        action="store_true",
        help="Disable SNR-based gate lift even if profile has baseline_snr",
    )
    ap.add_argument(
        "--snr-deficit-scale",
        type=float,
        default=None,
        help="Gate increase per 1 dB SNR below baseline (default: StreamDecodeConfig.snr_deficit_scale)",
    )
    ap.add_argument(
        "--online-quality",
        action="store_true",
        help="Sliding-window SNR on preprocessed samples; sets FrameChunk.meta snr_db for SNR gate / clients",
    )
    ap.add_argument(
        "--quality-warn-db",
        type=float,
        default=5.0,
        help="With baseline_snr_db set, emit quality_warning status when SNR is this far below baseline",
    )
    ap.add_argument(
        "--quality-status-interval",
        type=float,
        default=30.0,
        help="Minimum seconds between quality_warning broadcasts (rate limit)",
    )
    ap.add_argument(
        "--quality-every-n-chunks",
        type=int,
        default=1,
        help="Update online quality (SNR meta) every N chunks; reuse last estimate between updates",
    )
    ap.add_argument(
        "--latency-log-every-windows",
        type=int,
        default=0,
        help="Log mean classifier window inference time every N windows (0 disables)",
    )
    ap.add_argument(
        "--channel-quality-meta",
        action="store_true",
        help="With --online-quality, add per-channel SNR and weak channel indices to chunk/token meta",
    )
    ap.add_argument(
        "--weak-channel-deficit-db",
        type=float,
        default=6.0,
        help="Channel is weak if SNR is this many dB below median (with --channel-quality-meta)",
    )
    ap.add_argument(
        "--weak-channel-warn",
        action="store_true",
        help="Broadcast channel_quality status when weak channels are detected (needs --channel-quality-meta)",
    )
    ap.add_argument(
        "--weak-channel-status-interval",
        type=float,
        default=15.0,
        help="Minimum seconds between channel_quality broadcasts",
    )
    ap.add_argument(
        "--motion-gate",
        action="store_true",
        help="Attenuate preprocessed samples during high motion_index chunks",
    )
    ap.add_argument("--motion-threshold", type=float, default=0.35, help="Motion gate threshold (0–1)")
    ap.add_argument(
        "--motion-attenuation",
        type=float,
        default=0.15,
        help="Output scale during motion gate (0=mute, 1=pass-through)",
    )

    # sim source knobs
    ap.add_argument("--hw-spec", type=str, default="", help="Hardware DSL preset or .oae.json path")
    ap.add_argument("--labels", type=str, default="yes,no,left,right,select,cancel")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--sim-engine",
        type=str,
        default="heuristic",
        choices=["heuristic", "biophysical"],
        help="Synthetic source: band-noise heuristic vs MUAP + Poisson pool (see openalterego.sim.biophysical)",
    )
    ap.add_argument(
        "--sim-realism",
        type=str,
        default="",
        choices=["", "off", "wearable", "tang", "field"],
        help="Override sim sensor realism preset (--source sim); empty = engine defaults",
    )
    ap.add_argument(
        "--sim-scenario",
        type=str,
        default="",
        choices=["", "gowda_sv"],
        help="Named sim scenario (gowda_sv = 31ch @ 5kHz scripted trials for open-speech)",
    )
    ap.add_argument(
        "--sim-trials",
        type=int,
        default=8,
        help="gowda_sv streaming: number of 4-word trials to simulate (default 8)",
    )
    ap.add_argument(
        "--abstain-entropy-norm-max",
        type=float,
        default=None,
        help="Abstain (clear stabilizer) when normalized softmax entropy exceeds this [0,1]",
    )
    ap.add_argument(
        "--abstain-min-margin",
        type=float,
        default=None,
        help="Abstain when p_top1 - p_top2 is below this (e.g. 0.15)",
    )

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

    users_dir = Path(args.users_dir) if args.users_dir else default_users_dir()
    profile: Optional[UserProfile] = None
    if args.user_id:
        um = UserManager(users_dir)
        profile = um.load_profile(args.user_id)
        if profile is None:
            if str(args.model).strip():
                log.warning(
                    "Unknown user %r; continuing with --model only (no profile, defaults for decode/EMG).",
                    args.user_id,
                )
                profile = None
            else:
                raise SystemExit(
                    f"Unknown user {args.user_id!r}. Create with: openalterego user create --user-id {args.user_id}"
                    " or pass --model to serve without a profile."
                )

    model_path = str(args.model).strip() if args.model else ""
    if profile is not None and profile.model_path is not None:
        if not model_path:
            model_path = str(profile.model_path)
    elif not model_path:
        model_path = ""

    w_ms = int(args.window_ms) if args.window_ms is not None else (int(profile.window_ms) if profile else 600)
    s_ms = int(args.stride_ms) if args.stride_ms is not None else (int(profile.stride_ms) if profile else 120)
    min_c = float(args.min_confidence) if args.min_confidence is not None else (float(profile.confidence_threshold) if profile else 0.70)

    baseline_snr_db: Optional[float] = None
    if not args.no_snr_gate:
        baseline_snr_db = args.baseline_snr_db
        if baseline_snr_db is None and profile is not None and profile.baseline_snr is not None:
            baseline_snr_db = float(profile.baseline_snr)

    decode_kw: Dict[str, Any] = dict(
        window_ms=w_ms,
        stride_ms=s_ms,
        min_confidence=min_c,
        stable_n=int(args.stable_n),
        adaptive_threshold=bool(args.adaptive_threshold),
        threshold_alpha=float(args.threshold_alpha),
        baseline_snr_db=baseline_snr_db,
    )
    if args.snr_deficit_scale is not None:
        decode_kw["snr_deficit_scale"] = float(args.snr_deficit_scale)
    if int(args.latency_log_every_windows) > 0:
        decode_kw["latency_log_every_windows"] = int(args.latency_log_every_windows)
    decode_cfg = StreamDecodeConfig(**decode_kw)

    if args.adaptive_threshold or baseline_snr_db is not None:
        log.info(
            "decode: adaptive_threshold=%s threshold_alpha=%s baseline_snr_db=%s",
            decode_cfg.adaptive_threshold,
            decode_cfg.threshold_alpha,
            decode_cfg.baseline_snr_db,
        )

    pipeline_kw = dict(
        online_quality=bool(args.online_quality),
        quality_every_n_chunks=max(1, int(args.quality_every_n_chunks)),
        quality_warn_db=float(args.quality_warn_db),
        quality_status_interval_s=float(args.quality_status_interval),
        channel_quality_meta=bool(args.channel_quality_meta),
        weak_channel_deficit_db=float(args.weak_channel_deficit_db),
        weak_channel_warn=bool(args.weak_channel_warn),
        weak_channel_status_interval_s=float(args.weak_channel_status_interval),
        motion_gate=bool(args.motion_gate),
        motion_threshold=float(args.motion_threshold),
        motion_attenuation=float(args.motion_attenuation),
    )

    if args.weak_channel_warn and not args.channel_quality_meta:
        log.warning("--weak-channel-warn has no effect without --channel-quality-meta")
    if args.channel_quality_meta and not args.online_quality:
        log.warning("--channel-quality-meta has no effect without --online-quality")

    async with websockets.serve(hub.handler, args.host, args.port):
        log.info("OpenAlterEgo server on ws://%s:%s (source=%s)", args.host, args.port, args.source)

        if args.source == "demo":
            await run_demo_tokens(hub)
            return

        if str(args.decode_mode) == "open_speech":
            ctc_ckpt = str(args.ctc_checkpoint).strip()
            ctc_sess = str(args.ctc_session).strip()
            if not ctc_ckpt or not ctc_sess:
                raise SystemExit("--ctc-checkpoint and --ctc-session are required for --decode-mode open_speech")

            open_kw = dict(
                hub=hub,
                source_name=str(args.source),
                ctc_checkpoint=ctc_ckpt,
                ctc_session=ctc_sess,
                fs_hz=5000,
                channels=31,
                feature_mode=str(args.ctc_feature_mode),
                ctc_decode_mode=str(args.ctc_decode_mode),
                lm_weight=float(args.ctc_lm_weight),
                device_preferred=str(args.ctc_device),
            )

            if args.source == "sim":
                from ..sim.scenarios.gowda_small_vocab import build_gowda_sim_config

                realism = str(args.sim_realism).strip() or "tang"
                sim_cfg = build_gowda_sim_config(
                    n_trials=int(args.sim_trials),
                    seed=int(args.seed),
                    realism=realism,
                    realtime_clock=True,
                )
                chunks = _aiter_from_generator(stream_simulated_chunks(sim_cfg))
                await run_pipeline_open_speech(chunks=chunks, **open_kw)
                return

            if args.source == "virtual_ble":
                from ..hardware.bind import load_hw_spec_optional, virtual_ble_from_hw

                hw = load_hw_spec_optional(str(args.hw_spec))
                if hw is not None:
                    vs = virtual_ble_from_hw(
                        hw,
                        seed=int(args.seed),
                        loss_prob=float(args.loss),
                        jitter_ms=float(args.jitter_ms),
                        extra_latency_ms=float(args.extra_latency_ms),
                    )
                else:
                    vs = VirtualBleSpec()
                    vs.channels = 8
                    vs.fs_hz = 250
                    vs.link.loss_prob = float(args.loss)
                    vs.link.jitter_ms = float(args.jitter_ms)
                    vs.link.extra_latency_ms = float(args.extra_latency_ms)
                    vs.sim.seed = int(args.seed)
                chunks = stream_virtual_ble_chunks(vs)
                open_kw["fs_hz"] = int(vs.fs_hz)
                open_kw["channels"] = int(vs.channels)
                await run_pipeline_open_speech(chunks=chunks, **open_kw)
                return

            if args.source == "ble":
                from ..hardware.bind import ble_afe_from_hw, load_hw_spec_optional

                if not args.device_name or not args.data_uuid:
                    raise SystemExit("--device-name and --data-uuid are required for --source ble")
                hw = load_hw_spec_optional(str(args.hw_spec))
                if hw is not None:
                    bs = BleSpec(
                        device_name=str(args.device_name or hw.ble.device_name),
                        data_char_uuid=str(args.data_uuid),
                        fs_hz=int(hw.afe.fs_hz),
                        channels=int(hw.afe.channels),
                        packet_format=str(hw.ble.packet_format),
                        afe=ble_afe_from_hw(hw),
                    )
                else:
                    bs = BleSpec(
                        device_name=str(args.device_name),
                        data_char_uuid=str(args.data_uuid),
                        fs_hz=250,
                        channels=8,
                        packet_format=str(args.packet_format),
                    )
                chunks = stream_ble_chunks(bs)
                open_kw["fs_hz"] = int(bs.fs_hz)
                open_kw["channels"] = int(bs.channels)
                await run_pipeline_open_speech(chunks=chunks, **open_kw)
                return

            raise SystemExit(f"--decode-mode open_speech not supported for source={args.source!r}")

        if not model_path:
            raise SystemExit("--model is required when --source is not demo (or use --user-id with a saved model)")

        if args.source == "sim":
            from ..hardware.bind import load_hw_spec_optional, merge_sim_config

            lm_sim = load_model(model_path)
            labels_sim = [s.strip() for s in args.labels.split(",") if s.strip()]
            if set(labels_sim) != set(lm_sim.labels):
                log.warning(
                    "Sim --labels differ from model checkpoint labels: sim=%s model=%s",
                    sorted(labels_sim),
                    sorted(lm_sim.labels),
                )
            hw = load_hw_spec_optional(str(args.hw_spec))
            if hw is not None:
                sim_cfg = merge_sim_config(
                    hw,
                    labels=labels_sim,
                    seed=int(args.seed),
                    realtime_clock=False,
                    sim_engine=str(args.sim_engine),
                    realism_preset=str(args.sim_realism),
                )
                if int(sim_cfg.fs_hz) != int(lm_sim.fs) or int(sim_cfg.channels) != int(lm_sim.channels):
                    log.warning(
                        "hw-spec fs/ch=%s/%s differs from model fs/ch=%s/%s; using hw-spec for stream",
                        sim_cfg.fs_hz,
                        sim_cfg.channels,
                        lm_sim.fs,
                        lm_sim.channels,
                    )
            else:
                sim_cfg = SimConfig(
                    labels=labels_sim,
                    fs_hz=int(lm_sim.fs),
                    channels=int(lm_sim.channels),
                    seed=int(args.seed),
                    sim_engine=str(args.sim_engine),
                    realtime_clock=False,
                )
                rsim = str(args.sim_realism).strip()
                if rsim:
                    sim_cfg.realism_preset = rsim
            chunks = _aiter_from_generator(stream_simulated_chunks(sim_cfg))
            await run_pipeline(
                hub=hub,
                source_name="sim",
                chunks=chunks,
                model_path=model_path,
                loaded_model=lm_sim,
                decode_cfg=decode_cfg,
                user_profile=profile,
                **pipeline_kw,
            )
            return

        if args.source == "virtual_ble":
            from ..hardware.bind import load_hw_spec_optional, virtual_ble_from_hw

            hw = load_hw_spec_optional(str(args.hw_spec))
            if hw is not None:
                vs = virtual_ble_from_hw(
                    hw,
                    seed=int(args.seed),
                    loss_prob=float(args.loss),
                    jitter_ms=float(args.jitter_ms),
                    extra_latency_ms=float(args.extra_latency_ms),
                )
            else:
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
                model_path=model_path,
                decode_cfg=decode_cfg,
                user_profile=profile,
                **pipeline_kw,
            )
            return

        if args.source == "ble":
            from ..hardware.bind import ble_afe_from_hw, load_hw_spec_optional

            if not args.device_name or not args.data_uuid:
                raise SystemExit("--device-name and --data-uuid are required for --source ble")
            hw = load_hw_spec_optional(str(args.hw_spec))
            if hw is not None:
                bs = BleSpec(
                    device_name=str(args.device_name or hw.ble.device_name),
                    data_char_uuid=str(args.data_uuid),
                    fs_hz=int(hw.afe.fs_hz),
                    channels=int(hw.afe.channels),
                    packet_format=str(hw.ble.packet_format),
                    afe=ble_afe_from_hw(hw),
                )
            else:
                bs = BleSpec(
                    device_name=str(args.device_name),
                    data_char_uuid=str(args.data_uuid),
                    fs_hz=250,
                    channels=8,
                    packet_format=str(args.packet_format),
                )
            chunks = stream_ble_chunks(bs)
            await run_pipeline(
                hub=hub,
                source_name="ble",
                chunks=chunks,
                model_path=model_path,
                decode_cfg=decode_cfg,
                user_profile=profile,
                **pipeline_kw,
            )
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
