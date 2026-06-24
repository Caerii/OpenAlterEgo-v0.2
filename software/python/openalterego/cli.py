"""OpenAlterEgo command line interface.

This is meant to be the *fast path* for:
- generating synthetic datasets
- training a baseline model
- serving realtime websocket output
- running a glasses/display stub
- user profiles and calibration

You can also run modules directly (python -m openalterego.api.server, etc.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from . import __version__
from .sim.dataset import DatasetConfig, generate_dataset, generate_dataset_shards
from .sim.literature import DEFAULT_AR1_INNOVATION_SCALE, VALID_PARADIGMS
from .sim.stream import ScenarioConfig, SimStreamConfig
from .users.defaults import default_users_dir
from .users.manager import UserManager
from .users.profile import UserProfile


def _cmd_sim_dataset(argv: List[str]) -> int:
    from .hardware.bind import add_hw_spec_argument, dataset_config_from_hw, load_hw_spec_optional

    ap = argparse.ArgumentParser(prog="openalterego sim-dataset")
    ap.add_argument("--out", type=str, required=True, help="Output directory (session folder)")
    add_hw_spec_argument(ap)
    ap.add_argument("--minutes", type=float, default=2.0)
    ap.add_argument("--fs", type=int, default=250)
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--labels", type=str, default="yes,no,left,right,select,cancel")
    ap.add_argument("--p-event", type=float, default=0.65)
    ap.add_argument(
        "--emg-paradigm",
        type=str,
        default="semg_literature_clamped",
        choices=sorted(VALID_PARADIGMS),
        help="Literature-aligned token passband (see openalterego.sim.literature); use semg_literature_full only if --fs >= 920",
    )
    ap.add_argument(
        "--no-ar1",
        action="store_true",
        help="Disable AR(1) correlated LF noise (default follows Tang-style motion-ish correlation model)",
    )
    ap.add_argument("--line-noise-uV", type=float, default=0.0, help="Optional 50/60 Hz hum amplitude (µV)")
    ap.add_argument("--mains-freq-hz", type=float, default=60.0)
    ap.add_argument(
        "--sim-engine",
        type=str,
        default="biophysical",
        choices=["heuristic", "biophysical"],
        help="heuristic: fast band-limited tokens; biophysical: MUAP motor pool (recommended)",
    )
    ap.add_argument(
        "--noise-scale",
        type=float,
        default=1.0,
        help="Biophysical only: scales sensor noise (use with --snr-target-db for auto-tune)",
    )
    ap.add_argument(
        "--snr-target-db",
        type=float,
        default=None,
        help="Biophysical: auto-tune noise_scale toward this session SNR (e.g. 18.9 Tang static)",
    )
    ap.add_argument(
        "--snr-motion-target-db",
        type=float,
        default=None,
        help="Biophysical: also tune motion-burst amplitude toward motion SNR (e.g. 12.7 Tang)",
    )
    ap.add_argument(
        "--shards",
        type=int,
        default=1,
        help="Generate N parallel session shards under --out/shard_XXX",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Parallel workers when --shards > 1",
    )
    ap.add_argument(
        "--realism",
        type=str,
        default="tang",
        choices=["off", "wearable", "tang", "field"],
        help="Sensor/motion/frontend realism preset (tang = Tang 2025 wearable targets)",
    )
    ap.add_argument(
        "--auto-chunk",
        action="store_true",
        help="Tune chunk_ms for throughput at the chosen fs (biophysical)",
    )
    ap.add_argument(
        "--drive-mode",
        type=str,
        default="word",
        choices=["word", "phoneme"],
        help="Biophysical motor pool: word = one synergy class per command; phoneme = phone inventory + phonemes.csv",
    )
    ap.add_argument(
        "--phone-lexicon",
        type=str,
        default="",
        help="Optional JSON {\"word\": [\"P\", \"H\"], ...} merged over defaults (biophysical phoneme drive)",
    )
    ap.add_argument(
        "--scenario",
        type=str,
        default="",
        choices=["", "gowda_sv"],
        help="Named scenario (gowda_sv = 124-word 4-trial sentences, 31ch @ 5kHz)",
    )
    ap.add_argument(
        "--trials",
        type=int,
        default=500,
        help="Gowda scenario: number of 4-word trials (default 500 for official split)",
    )
    ap.add_argument(
        "--phone-templates",
        type=str,
        default="",
        help="JSON per-phone templates from analyze fit-phone-templates (Gowda phoneme drive)",
    )
    ap.add_argument(
        "--no-coarticulation",
        action="store_true",
        help="Gowda scenario: disable raised-cosine phone overlap at boundaries",
    )
    args = ap.parse_args(argv)

    if str(args.scenario).strip() == "gowda_sv":
        from .sim.scenarios.gowda_small_vocab import build_gowda_dataset_config

        snr_default = None if str(args.realism) == "off" else 18.9
        snr_motion_default = None if str(args.realism) == "off" else 12.7
        ds = build_gowda_dataset_config(
            Path(args.out),
            n_trials=int(args.trials),
            seed=int(args.seed),
            realism=str(args.realism),
            snr_target_db=float(args.snr_target_db) if args.snr_target_db is not None else snr_default,
            snr_motion_target_db=(
                float(args.snr_motion_target_db) if args.snr_motion_target_db is not None else snr_motion_default
            ),
            phone_templates_path=str(args.phone_templates).strip() or None,
            coarticulation_enabled=not bool(args.no_coarticulation),
        )
        if int(args.shards) > 1:
            paths = generate_dataset_shards(ds, n_shards=int(args.shards), workers=int(args.workers))
            print(f"[openalterego] wrote {len(paths)} Gowda scenario shards under {args.out}")
            for p in paths:
                print(f"  - {p}")
            return 0
        out = generate_dataset(ds)
        print(f"[openalterego] wrote Gowda scenario dataset: {out}")
        print(f"  - signals.npy, events.csv (trial_id, word_idx), meta.json, phonemes.csv")
        return 0

    labels = [s.strip() for s in str(args.labels).split(",") if s.strip()]
    phone_lexicon = None
    lex_path = str(args.phone_lexicon).strip()
    if lex_path:
        from .sim.phonology import load_user_lexicon_overlay

        try:
            phone_lexicon = load_user_lexicon_overlay(Path(lex_path))
        except (OSError, json.JSONDecodeError, ValueError) as e:
            print(f"[openalterego] --phone-lexicon: {e}", file=sys.stderr)
            return 2
    sc = ScenarioConfig(
        labels=labels,
        p_event=float(args.p_event),
        drive_mode=str(args.drive_mode),
        phone_lexicon=phone_lexicon,
    )

    hw = load_hw_spec_optional(str(args.hw_spec))
    chunk_ms = 40
    if bool(args.auto_chunk) and str(args.sim_engine) == "biophysical":
        from .sim.biophysical.benchmark import recommend_chunk_ms

        fs_eff = int(hw.afe.fs_hz) if hw is not None else int(args.fs)
        ch_eff = int(hw.afe.channels) if hw is not None else int(args.channels)
        chunk_ms, t = recommend_chunk_ms(fs_eff, channels=ch_eff, target_realtime_factor=25.0)
        print(f"[openalterego] auto-chunk: chunk_ms={chunk_ms} ({t.realtime_factor:.0f}x realtime)")
    if hw is not None:
        ds = dataset_config_from_hw(
            hw,
            out_dir=str(args.out),
            duration_s=float(args.minutes) * 60.0,
            scenario=sc,
            seed=int(args.seed),
            sim_engine=str(args.sim_engine),
            realism_preset=str(args.realism),
            biophysical_noise_scale=float(args.noise_scale),
            snr_target_db=float(args.snr_target_db) if args.snr_target_db is not None else None,
            chunk_ms=int(chunk_ms) if bool(args.auto_chunk) else None,
        )
        if int(args.shards) > 1:
            ds.snr_motion_target_db = (
                float(args.snr_motion_target_db) if args.snr_motion_target_db is not None else None
            )
            paths = generate_dataset_shards(ds, n_shards=int(args.shards), workers=int(args.workers))
            print(f"[openalterego] wrote {len(paths)} shards under {args.out}")
            for p in paths:
                print(f"  - {p}")
            return 0
        ds.snr_motion_target_db = (
            float(args.snr_motion_target_db) if args.snr_motion_target_db is not None else None
        )
        out = generate_dataset(ds)
        meta_path = Path(out) / "meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            from .hardware.bind import hw_metadata_dict

            meta.update(hw_metadata_dict(hw))
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[openalterego] wrote dataset (hw-spec={hw.name}): {out}")
        print(f"  - signals.npy")
        print(f"  - events.csv")
        print(f"  - meta.json")
        return 0

    ar1_scale = 0.0 if args.no_ar1 else DEFAULT_AR1_INNOVATION_SCALE
    sim_cfg = SimStreamConfig(
        fs_hz=int(args.fs),
        channels=int(args.channels),
        chunk_ms=int(chunk_ms),
        seed=int(args.seed),
        scenario=sc,
        emg_paradigm=str(args.emg_paradigm),
        ar1_innovation_scale=float(ar1_scale),
        line_noise_uV=float(args.line_noise_uV),
        mains_freq_hz=float(args.mains_freq_hz),
        realtime_clock=False,
    )
    ds = DatasetConfig(
        out_dir=Path(args.out),
        duration_s=float(args.minutes) * 60.0,
        config=sim_cfg,
        sim_engine=str(args.sim_engine),
        biophysical_noise_scale=float(args.noise_scale),
        snr_target_db=float(args.snr_target_db) if args.snr_target_db is not None else None,
        snr_motion_target_db=float(args.snr_motion_target_db) if args.snr_motion_target_db is not None else None,
        realism_preset=str(args.realism),
    )
    if int(args.shards) > 1:
        paths = generate_dataset_shards(ds, n_shards=int(args.shards), workers=int(args.workers))
        print(f"[openalterego] wrote {len(paths)} shards under {args.out}")
        for p in paths:
            print(f"  - {p}")
        return 0
    out = generate_dataset(ds)
    print(f"[openalterego] wrote dataset: {out}")
    print(f"  - signals.npy")
    print(f"  - events.csv")
    print(f"  - meta.json")
    return 0


def _users_dir_path(s: str) -> Path:
    return Path(s) if s else default_users_dir()


def _cmd_user(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="openalterego user")
    ap.add_argument("--users-dir", type=str, default="", help="User data root (default: ./users)")
    sub = ap.add_subparsers(dest="user_cmd", required=True)

    p_create = sub.add_parser("create", help="Create a new user profile")
    p_create.add_argument("--user-id", type=str, required=True)

    sub.add_parser("list", help="List user ids that have a profile")

    p_show = sub.add_parser("show", help="Show profile fields")
    p_show.add_argument("--user-id", type=str, required=True)
    p_show.add_argument("--json", action="store_true", help="Print JSON only")

    p_del = sub.add_parser("delete", help="Delete user directory and all data")
    p_del.add_argument("--user-id", type=str, required=True)
    p_del.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")

    p_onboard = sub.add_parser("onboard", help="Open-speech onboarding: per-user SPD basis + optional CTC adapter")
    p_onboard.add_argument("--user-id", type=str, required=True)
    p_onboard.add_argument("--session", type=str, required=True, help="Gowda-shaped session dir (signals + events)")
    p_onboard.add_argument("--checkpoint", type=str, default="", help="Base CTC checkpoint to copy or fine-tune")
    p_onboard.add_argument("--adapter-epochs", type=int, default=0, help="Fine-tune adapter epochs (0 = copy only)")
    p_onboard.add_argument("--no-fit-basis", action="store_true", help="Skip per-user SPD basis fit")
    p_onboard.add_argument("--feature-mode", type=str, default="diag_delta")
    p_onboard.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p_onboard.add_argument("--seed", type=int, default=1337)

    p_check = sub.add_parser("check-quality", help="Compare session SNR to profile baseline (re-cal hint)")
    p_check.add_argument("--user-id", type=str, required=True)
    p_check.add_argument("--data", type=str, required=True, help="Session folder with signals.npy")
    p_check.add_argument("--fs", type=int, default=0, help="Sample rate (0 = read meta.json)")
    p_check.add_argument("--warn-db", type=float, default=3.0, help="Re-cal if SNR below baseline by this margin")
    p_check.add_argument("--json", action="store_true")

    args = ap.parse_args(argv)
    users_dir = _users_dir_path(args.users_dir)
    mgr = UserManager(users_dir)

    if args.user_cmd == "create":
        if mgr.user_exists(args.user_id):
            print(f"[openalterego] user already exists: {args.user_id!r}", file=sys.stderr)
            return 1
        mgr.save_profile(UserProfile(user_id=args.user_id))
        print(f"[openalterego] created user {args.user_id!r} at {mgr.get_user_dir(args.user_id)}")
        return 0
    if args.user_cmd == "list":
        for uid in mgr.list_users():
            print(uid)
        return 0
    if args.user_cmd == "show":
        p = mgr.load_profile(args.user_id)
        if p is None:
            print(f"[openalterego] unknown user {args.user_id!r}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(p.to_dict(), indent=2))
        else:
            print(f"user_id:              {p.user_id}")
            print(f"model_path:           {p.model_path}")
            print(f"confidence_threshold: {p.confidence_threshold}")
            print(f"preprocessing_mode:   {p.preprocessing_mode}")
            print(f"window_ms:            {p.window_ms}")
            print(f"stride_ms:            {p.stride_ms}")
            print(f"calibration_date:     {p.calibration_date}")
            print(f"calibration_samples:  {p.calibration_samples}")
            print(f"baseline_snr:         {p.baseline_snr}")
        return 0
    if args.user_cmd == "delete":
        if not mgr.user_exists(args.user_id):
            print(f"[openalterego] unknown user {args.user_id!r}", file=sys.stderr)
            return 1
        target = mgr.get_user_dir(args.user_id)
        if not args.yes:
            try:
                ans = input(f"Delete entire directory {target}? [y/N] ")
            except EOFError:
                ans = ""
            if ans.strip().lower() not in ("y", "yes"):
                print("Aborted.")
                return 1
        mgr.delete_user(args.user_id)
        print(f"[openalterego] deleted user {args.user_id!r}")
        return 0

    if args.user_cmd == "onboard":
        from pathlib import Path

        from .users.onboarding import onboard_open_speech

        ckpt = str(args.checkpoint).strip()
        report = onboard_open_speech(
            args.user_id,
            Path(args.session),
            users_dir=users_dir,
            base_checkpoint=Path(ckpt) if ckpt else None,
            fit_basis=not bool(args.no_fit_basis),
            adapter_epochs=int(args.adapter_epochs),
            feature_mode=str(args.feature_mode),
            seed=int(args.seed),
            device_preferred=str(args.device),
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.user_cmd == "check-quality":
        from .dsp.quality import assess_signal_quality
        from .dsp.emg_config import emg_signal_band_hz_for_quality
        from .users.recalibration import assess_session_recalibration

        p = mgr.load_profile(args.user_id)
        if p is None:
            print(f"[openalterego] unknown user {args.user_id!r}", file=sys.stderr)
            return 1
        session = Path(args.data)
        sig = session / "signals.npy"
        if not sig.is_file():
            print(f"[openalterego] missing {sig}", file=sys.stderr)
            return 1
        import numpy as np

        x = np.load(sig)
        fs = int(args.fs)
        if fs <= 0:
            meta = session / "meta.json"
            if meta.is_file():
                fs = int(json.loads(meta.read_text(encoding="utf-8")).get("fs_hz", 250))
            else:
                fs = 250
        band = emg_signal_band_hz_for_quality(str(p.preprocessing_mode), float(fs))
        qm = assess_signal_quality(x, fs_hz=float(fs), signal_band_hz=band, per_channel=False)
        st = assess_session_recalibration(
            session_snr_db=qm.snr_db,
            baseline_snr_db=p.baseline_snr,
            motion_index=float(qm.motion_index),
            warn_db=float(args.warn_db),
        )
        out = {
            "user_id": args.user_id,
            "session": str(session),
            "fs_hz": fs,
            **st.to_meta(),
            "baseline_snr_db": p.baseline_snr,
        }
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            print(f"[openalterego] session SNR={qm.snr_db} baseline={p.baseline_snr} motion={qm.motion_index:.3f}")
            print(f"  re_calibration_suggested={st.re_calibration_suggested} deficit={st.snr_deficit_db:.1f} dB")
            if st.reasons:
                print(f"  reasons: {', '.join(st.reasons)}")
        return 0 if not st.re_calibration_suggested else 2

    return 1


def _cmd_calibrate(argv: List[str]) -> int:
    from .users.calibration import CalibrationConfig, calibrate_user

    ap = argparse.ArgumentParser(prog="openalterego calibrate")
    ap.add_argument("--user-id", type=str, required=True)
    ap.add_argument("--data", type=str, required=True, help="Session folder with signals.npy and events.csv")
    ap.add_argument("--fs", type=int, required=True)
    ap.add_argument("--users-dir", type=str, default="")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--segment-ms", type=int, default=600)
    ap.add_argument("--min-samples-per-token", type=int, default=50)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument(
        "--preprocessing-mode",
        type=str,
        default="",
        choices=["", "standard", "clinical", "wide"],
        help="Override profile preprocessing mode (default: use profile)",
    )
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--strict-motion",
        action="store_true",
        help="Abort if assessed motion_index exceeds --motion-reject-above",
    )
    ap.add_argument("--motion-reject-above", type=float, default=0.35)
    ap.add_argument(
        "--stride-ms",
        type=int,
        default=None,
        help="Store this decode stride on the user profile after calibration (default: keep profile value)",
    )
    ap.add_argument(
        "--no-stratified-split",
        action="store_true",
        help="Random val split instead of per-label stratified split",
    )
    ap.add_argument("--quiet", action="store_true", help="Disable training progress output")
    ap.add_argument(
        "--arch",
        type=str,
        default="se_resnet",
        choices=["cnn", "se_resnet"],
        help="Model architecture (default: se_resnet)",
    )
    args = ap.parse_args(argv)

    users_dir = _users_dir_path(args.users_dir)
    mgr = UserManager(users_dir)
    cfg = CalibrationConfig(
        min_samples_per_token=int(args.min_samples_per_token),
        val_split=float(args.val_split),
        epochs=int(args.epochs),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        segment_ms=int(args.segment_ms),
        seed=int(args.seed),
        strict_motion=bool(args.strict_motion),
        motion_reject_above=float(args.motion_reject_above),
        stride_ms=int(args.stride_ms) if args.stride_ms is not None else None,
        stratified_split=not bool(args.no_stratified_split),
        show_progress=not bool(args.quiet),
        arch=str(args.arch),
    )
    pm = args.preprocessing_mode or None
    _, report = calibrate_user(
        args.user_id,
        Path(args.data),
        int(args.fs),
        user_manager=mgr,
        config=cfg,
        preprocessing_mode=pm,
    )
    print(report.to_string())
    return 0


def _cmd_collect(argv: List[str]) -> int:
    import asyncio

    from .users.collect import collect_from_ble_async, collect_from_sim

    ap = argparse.ArgumentParser(prog="openalterego collect")
    ap.add_argument("--users-dir", type=str, default="", help="Reserved for future use")
    sub = ap.add_subparsers(dest="src", required=True)

    p_sim = sub.add_parser("sim", help="Record from the synthetic stream (labels from simulator)")
    from .hardware.bind import add_hw_spec_argument

    add_hw_spec_argument(p_sim)
    p_sim.add_argument("--out", type=str, required=True, help="Session output directory")
    p_sim.add_argument("--user-id", type=str, required=True)
    p_sim.add_argument("--seconds", type=float, default=120.0)
    p_sim.add_argument("--fs", type=int, default=250)
    p_sim.add_argument("--channels", type=int, default=8)
    p_sim.add_argument("--seed", type=int, default=1337)
    p_sim.add_argument("--labels", type=str, default="yes,no,left,right,select,cancel")
    p_sim.add_argument("--p-event", type=float, default=0.65)
    p_sim.add_argument(
        "--preprocessing-mode",
        type=str,
        default="standard",
        choices=["standard", "clinical", "wide"],
    )
    p_sim.add_argument(
        "--realtime-clock",
        action="store_true",
        help="Sleep between chunks (wall clock); default runs as fast as possible",
    )
    p_sim.add_argument(
        "--emg-paradigm",
        type=str,
        default="semg_literature_clamped",
        choices=sorted(VALID_PARADIGMS),
        help="Synthetic EMG band model (see openalterego.sim.literature)",
    )
    p_sim.add_argument("--no-ar1", action="store_true", help="Disable correlated LF noise component")
    p_sim.add_argument("--line-noise-uV", type=float, default=0.0)
    p_sim.add_argument("--mains-freq-hz", type=float, default=60.0)

    p_ble = sub.add_parser("ble", help="Record from BLE until duration elapses (label events offline)")
    p_ble.add_argument("--out", type=str, required=True)
    p_ble.add_argument("--user-id", type=str, required=True)
    p_ble.add_argument("--seconds", type=float, required=True)
    p_ble.add_argument("--device-name", type=str, required=True)
    p_ble.add_argument("--data-uuid", type=str, required=True)
    p_ble.add_argument("--fs", type=int, default=250)
    p_ble.add_argument("--channels", type=int, default=8)
    p_ble.add_argument("--packet-format", type=str, default="raw_i16", choices=["raw_i16", "oa_v1"])
    p_ble.add_argument("--scale-uV-per-count", type=float, default=1.0)
    p_ble.add_argument(
        "--preprocessing-mode",
        type=str,
        default="standard",
        choices=["standard", "clinical", "wide"],
    )

    p_label = sub.add_parser("label-events", help="Write events.csv from a markers CSV (BLE post-labeling)")
    p_label.add_argument("--session", type=str, required=True, help="Session folder with signals.npy")
    p_label.add_argument("--markers", type=str, required=True, help="markers.csv (time_s/label or sample/label, ...)")
    p_label.add_argument("--fs", type=int, default=0, help="Sample rate (0 = read session meta.json)")
    p_label.add_argument("--min-duration-s", type=float, default=0.12)

    args = ap.parse_args(argv)

    if args.src == "sim":
        from .hardware.bind import load_hw_spec_optional
        from .users.collect import collect_from_hw_spec

        labels = [s.strip() for s in str(args.labels).split(",") if s.strip()]
        hw = load_hw_spec_optional(str(getattr(args, "hw_spec", "")))
        if hw is not None:
            out = collect_from_hw_spec(
                spec=hw,
                output_dir=Path(args.out),
                user_id=str(args.user_id),
                duration_s=float(args.seconds),
                labels=labels,
                p_event=float(args.p_event),
                seed=int(args.seed),
                realtime_clock=bool(args.realtime_clock),
            )
            print(f"[openalterego] collect sim (hw-spec={hw.name}) wrote {out}")
            return 0
        out = collect_from_sim(
            output_dir=Path(args.out),
            user_id=str(args.user_id),
            duration_s=float(args.seconds),
            fs_hz=int(args.fs),
            channels=int(args.channels),
            seed=int(args.seed),
            labels=labels,
            p_event=float(args.p_event),
            preprocessing_mode=args.preprocessing_mode,  # type: ignore[arg-type]
            realtime_clock=bool(args.realtime_clock),
            emg_paradigm=str(args.emg_paradigm),
            ar1_innovation_scale=0.0 if args.no_ar1 else None,
            line_noise_uV=float(args.line_noise_uV),
            mains_freq_hz=float(args.mains_freq_hz),
        )
        print(f"[openalterego] collect sim wrote {out}")
        return 0

    if args.src == "label-events":
        from .users.labeling import label_session_from_markers

        fs = int(args.fs) if int(args.fs) > 0 else None
        out = label_session_from_markers(
            args.session,
            args.markers,
            fs_hz=fs,
            min_duration_s=float(args.min_duration_s),
        )
        print(f"[openalterego] wrote {out}")
        return 0

    out = asyncio.run(
        collect_from_ble_async(
            output_dir=Path(args.out),
            user_id=str(args.user_id),
            max_seconds=float(args.seconds),
            device_name=str(args.device_name),
            data_char_uuid=str(args.data_uuid),
            fs_hz=int(args.fs),
            channels=int(args.channels),
            packet_format=str(args.packet_format),
            preprocessing_mode=args.preprocessing_mode,  # type: ignore[arg-type]
            scale_uV_per_count=float(args.scale_uV_per_count),
        )
    )
    print(f"[openalterego] collect ble wrote {out}")
    return 0


def _cmd_hw(argv: List[str]) -> int:
    from .hardware.load import export_spec_json, list_presets, load_spec
    from .hardware.montages import list_montage_names
    from .hardware.presets import get_preset_dict
    from .hardware.resolve import resolved_to_jsonable, resolve_all
    from .hardware.runner import (
        run_chunk_simulation,
        run_virtual_ble_simulation_sync,
        simulate_report_dict,
    )
    from .hardware.validate import has_errors, validate_spec

    ap = argparse.ArgumentParser(prog="openalterego hw")
    sub = ap.add_subparsers(dest="hw_cmd", required=True)

    p_list = sub.add_parser("list", help="List built-in presets and montages")
    p_list.add_argument("--json", action="store_true")

    p_show = sub.add_parser("show", help="Print a spec (preset name or .oae.json path)")
    p_show.add_argument("spec", type=str, help="Preset name or path to .oae.json")
    p_show.add_argument("--json", action="store_true")

    p_val = sub.add_parser("validate", help="Validate spec against literature constraints")
    p_val.add_argument("spec", type=str)

    p_res = sub.add_parser("resolve", help="Resolve spec to sim/BLE runtime bindings")
    p_res.add_argument("spec", type=str)
    p_res.add_argument("--json", action="store_true")

    p_sim = sub.add_parser("simulate", help="Run short bound simulation")
    p_sim.add_argument("spec", type=str)
    p_sim.add_argument("--seconds", type=float, default=5.0)
    p_sim.add_argument(
        "--path",
        type=str,
        default="chunks",
        choices=["chunks", "virtual_ble", "both"],
        help="chunks=direct FrameChunk sim; virtual_ble=OA v1 byte path",
    )
    p_sim.add_argument("--json", action="store_true")

    p_run = sub.add_parser("run", help="Validate + smoke-sim + collect session from hw spec")
    p_run.add_argument("spec", type=str)
    p_run.add_argument("--out", type=str, required=True, help="Session output directory")
    p_run.add_argument("--user-id", type=str, default="hw_user")
    p_run.add_argument("--seconds", type=float, default=60.0)
    p_run.add_argument("--seed", type=int, default=1337)
    p_run.add_argument("--smoke-seconds", type=float, default=2.0, help="Duration for pre-collect simulation")
    p_run.add_argument("--skip-smoke", action="store_true")
    p_run.add_argument("--json", action="store_true")

    p_exp = sub.add_parser("export", help="Write preset or spec to .oae.json")
    p_exp.add_argument("spec", type=str, help="Preset name")
    p_exp.add_argument("-o", "--out", type=str, required=True)

    args = ap.parse_args(argv)

    if args.hw_cmd == "list":
        data = {"presets": list_presets(), "montages": list_montage_names()}
        if args.json:
            print(json.dumps(data, indent=2))
        else:
            print("Presets:", ", ".join(data["presets"]))
            print("Montages:", ", ".join(data["montages"]))
        return 0

    if args.hw_cmd == "export":
        name = str(args.spec).strip()
        if name in list_presets():
            payload = get_preset_dict(name)
        else:
            payload = load_spec(name).model_dump_public()
        from .hardware.schema import HardwareSpec

        export_spec_json(HardwareSpec.model_validate(payload), args.out)
        print(f"[openalterego] hw export wrote {args.out}")
        return 0

    spec = load_spec(args.spec)

    if args.hw_cmd == "show":
        if args.json:
            print(json.dumps(spec.model_dump_public(), indent=2))
        else:
            print(f"name: {spec.name}  tier: {spec.tier}")
            print(spec.description)
            print(f"afe: {spec.afe.channels}ch @ {spec.afe.fs_hz} Hz  gain={spec.afe.gain}")
            print(f"electrodes: {spec.electrodes.type}  montage={spec.electrodes.montage}")
            print(f"preprocess: {spec.preprocess.mode}  sim: {spec.sim.engine}/{spec.sim.realism}")
        return 0

    if args.hw_cmd == "validate":
        issues = validate_spec(spec)
        for iss in issues:
            print(iss.format())
        if has_errors(issues):
            print(f"[openalterego] validate: {len(issues)} issue(s), FAILED", file=sys.stderr)
            return 1
        print(f"[openalterego] validate: {len(issues)} issue(s), OK")
        return 0

    if args.hw_cmd == "resolve":
        resolved = resolve_all(spec)
        if args.json:
            print(json.dumps(resolved_to_jsonable(resolved), indent=2))
        else:
            r = resolved_to_jsonable(resolved)
            print(f"resolved: {r['name']} ({r['tier']})")
            print(f"  emg_paradigm: {r['emg_paradigm']}")
            print(f"  preprocess: {r['preprocess_mode']}")
            print(f"  sim: {r['sim']['engine']} realism={r['sim']['realism']}")
            print(f"  afe: fs={r['afe']['fs_hz']} ch={spec.afe.channels} uV/count={r['afe']['uV_per_count']:.4f}")
        return 0

    if args.hw_cmd == "simulate":
        out: dict = {"spec": spec.name}
        if args.path in ("chunks", "both"):
            _, rep = run_chunk_simulation(spec, duration_s=float(args.seconds))
            out["chunks"] = simulate_report_dict(rep)
        if args.path in ("virtual_ble", "both"):
            rep = run_virtual_ble_simulation_sync(spec, duration_s=float(args.seconds))
            out["virtual_ble"] = simulate_report_dict(rep)
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            for key, rep in out.items():
                if key == "spec":
                    continue
                print(f"[{key}] duration={rep['duration_s']}s samples={rep['n_samples']}")
                if rep.get("mean_snr_db") is not None:
                    print(f"  mean_snr_db={rep['mean_snr_db']} motion={rep.get('mean_motion_index')}")
                if rep.get("packets_parsed"):
                    print(f"  packets parsed={rep['packets_parsed']} lost={rep['packets_lost']}")
        return 0

    if args.hw_cmd == "run":
        from .users.collect import collect_from_hw_spec

        issues = validate_spec(spec)
        if has_errors(issues):
            for iss in issues:
                print(iss.format(), file=sys.stderr)
            return 1
        result: dict = {"spec": spec.name, "tier": spec.tier, "steps": []}
        if not args.skip_smoke:
            _, rep = run_chunk_simulation(spec, duration_s=float(args.smoke_seconds))
            result["smoke"] = simulate_report_dict(rep)
            result["steps"].append("smoke_sim")
        out_path = collect_from_hw_spec(
            spec=spec,
            output_dir=Path(args.out),
            user_id=str(args.user_id),
            duration_s=float(args.seconds),
            seed=int(args.seed),
        )
        result["collect_out"] = str(out_path)
        result["steps"].append("collect")
        result["next"] = [
            f"openalterego calibrate --user-id {args.user_id} --data {out_path} --fs {spec.afe.fs_hz}",
            f"openalterego train --user-id {args.user_id} --data {out_path} --fs {spec.afe.fs_hz} --emg-mode {spec.preprocess.mode}",
            f"openalterego serve --source sim --user-id {args.user_id} --hw-spec {args.spec}",
        ]
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"[openalterego] hw run OK: {out_path}")
            for line in result["next"]:
                print(f"  next: {line}")
        return 0

    ap.print_help()
    return 2


def _delegate_module(module_main, argv: List[str]) -> int:
    """Run another module's main() with a temporary sys.argv."""
    old = sys.argv[:]
    try:
        sys.argv = [old[0]] + argv
        module_main()
        return 0
    finally:
        sys.argv = old


def _cmd_sim_benchmark(argv: List[str]) -> int:
    import json

    from .sim.biophysical.benchmark import benchmark_chunk, recommend_chunk_ms, run_extended_scaling_sweep, run_scaling_sweep
    from .sim.biophysical.accel_backend import active_backend_label

    ap = argparse.ArgumentParser(prog="openalterego sim-benchmark")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--fs", type=int, default=0, help="Single-point fs (0 = full sweep)")
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--motor-units", type=int, default=48)
    ap.add_argument("--chunk-ms", type=int, default=40)
    ap.add_argument(
        "--synth-mode",
        type=str,
        default="fast",
        choices=["fast", "batch", "legacy", "numba", "rust"],
    )
    ap.add_argument("--extended", action="store_true", help="High-fs sweep up to 4 kHz / 192 MU")
    ap.add_argument("--tune-chunk", action="store_true", help="Recommend chunk_ms for target realtime")
    ap.add_argument("--target-rt", type=float, default=25.0, help="Target realtime factor for --tune-chunk")
    args = ap.parse_args(argv)

    if args.tune_chunk:
        fs = int(args.fs) if int(args.fs) > 0 else 1000
        ms, t = recommend_chunk_ms(
            fs,
            channels=int(args.channels),
            n_motor_units=int(args.motor_units),
            synth_mode=str(args.synth_mode),
            target_realtime_factor=float(args.target_rt),
        )
        out = {"recommended_chunk_ms": ms, "timing": t.to_dict()}
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            print(f"[openalterego] recommend chunk_ms={ms} @ fs={fs} Hz")
            print(f"  realtime_factor={t.realtime_factor:.1f}x  {t.samples_per_s/1e6:.2f} Msamp/s")
        return 0

    if int(args.fs) > 0:
        t = benchmark_chunk(
            fs_hz=int(args.fs),
            channels=int(args.channels),
            n_motor_units=int(args.motor_units),
            chunk_ms=int(args.chunk_ms),
            synth_mode=str(args.synth_mode),
        )
        if args.json:
            print(json.dumps(t.to_dict(), indent=2))
        else:
            print(f"[openalterego] fs={t.fs_hz} ch={t.channels} mu={t.n_motor_units} mode={t.synth_mode}")
            print(f"  {t.ms_per_chunk:.2f} ms/chunk  {t.realtime_factor:.1f}x realtime  {t.samples_per_s/1e6:.2f} Msamp/s")
        return 0

    rep = run_extended_scaling_sweep(synth_mode=str(args.synth_mode), chunk_ms=int(args.chunk_ms)) if args.extended else run_scaling_sweep(synth_mode=str(args.synth_mode), chunk_ms=int(args.chunk_ms))
    if args.json:
        print(json.dumps(rep.to_dict(), indent=2))
    else:
        label = "extended scaling sweep" if args.extended else "scaling sweep"
        print(f"[openalterego] {label} (mode={args.synth_mode}, backend={active_backend_label()}):")
        for t in rep.timings:
            print(
                f"  fs={t.fs_hz:4d} ch={t.channels:2d} mu={t.n_motor_units:3d} chunk={t.chunk_ms:3d}ms "
                f"-> {t.realtime_factor:5.1f}x RT  {t.samples_per_s/1e6:.2f} Msamp/s"
            )
        for note in rep.notes:
            print(f"  note: {note}")
        print(f"  recommended chunk_ms: {rep.recommended_chunk_ms}")
    return 0


def _cmd_window_sweep(argv: List[str]) -> int:
    import json

    from .runtime.window_sweep import run_window_sweep

    ap = argparse.ArgumentParser(prog="openalterego window-sweep")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--session", type=str, default="", help="Labeled session for event accuracy")
    ap.add_argument("--windows", type=str, default="400,600,900,1200,1500")
    ap.add_argument("--stride-ms", type=int, default=120)
    ap.add_argument("--n-chunks", type=int, default=60)
    ap.add_argument("--target-latency-p95-ms", type=float, default=500.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    windows = [int(x.strip()) for x in str(args.windows).split(",") if x.strip()]
    session = Path(args.session) if str(args.session).strip() else None
    rep = run_window_sweep(
        model_path=str(args.model),
        session_dir=session,
        window_values_ms=windows,
        stride_ms=int(args.stride_ms),
        n_latency_chunks=int(args.n_chunks),
        target_latency_p95_ms=float(args.target_latency_p95_ms),
    )
    if args.json:
        print(json.dumps(rep.to_dict(), indent=2))
    else:
        print(f"[openalterego] window sweep fs={rep.fs_hz} ch={rep.channels}")
        for row in rep.rows:
            acc = "n/a" if row.event_accuracy is None else f"{row.event_accuracy:.3f}"
            print(
                f"  w={row.window_ms:4d}ms  p95={row.latency_p95_ms:6.1f}ms  "
                f"acc={acc}  n={row.n_events}"
            )
        for note in rep.notes:
            print(f"  note: {note}")
        print(f"  recommended: {rep.recommended_window_ms} ms")
    return 0


def _cmd_decode_utterance(argv: List[str]) -> int:
    import json

    from .ml.ctc.decode_utterance import decode_session_trial

    ap = argparse.ArgumentParser(prog="openalterego decode-utterance")
    ap.add_argument("--session", type=str, required=True)
    ap.add_argument("--trial-id", type=int, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--lm-weight", type=float, default=0.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    result = decode_session_trial(
        Path(args.session),
        int(args.trial_id),
        Path(args.checkpoint),
        device_preferred=str(args.device),
        lm_weight=float(args.lm_weight),
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        tid = result["trial_id"]
        acc = result["word_acc"]
        print(f"[openalterego] trial {tid} word_acc={acc:.1%}")
        for w in result["words"]:
            mark = "OK" if w["correct"] else "ERR"
            print(f"  [{mark}] {w['ref']} -> {w['hyp']}")
    return 0


def _cmd_analyze(argv: List[str]) -> int:
    import json

    from .ml.analysis import run_channel_importance

    ap = argparse.ArgumentParser(prog="openalterego analyze")
    sub = ap.add_subparsers(dest="analyze_cmd", required=True)

    p_ci = sub.add_parser("channel-importance", help="Rank EMG channels from SE gates + input gradients")
    p_ci.add_argument("--session", type=str, required=True)
    p_ci.add_argument("--model", type=str, required=True)
    p_ci.add_argument("--segment-ms", type=int, default=0)
    p_ci.add_argument("--top-k", type=int, default=16)
    p_ci.add_argument("--out", type=str, default="", help="Write JSON report (default: session/channel_importance.json)")
    p_ci.add_argument("--json", action="store_true")

    p_ab = sub.add_parser("gowda-ablation", help="Phase-1 Gowda preprocess/window ablation matrix")
    p_ab.add_argument("--data", type=str, required=True)
    p_ab.add_argument("--fs", type=int, default=5000)
    p_ab.add_argument("--epochs", type=int, default=30)
    p_ab.add_argument("--batch-size", type=int, default=64)
    p_ab.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p_ab.add_argument("--out", type=str, default="")

    p_p2 = sub.add_parser("gowda-phase2", help="Phase 2: weekday ceiling, multi-seed CI, CTC")
    p_p2.add_argument("--data", type=str, required=True)
    p_p2.add_argument("--fs", type=int, default=5000)
    p_p2.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p_p2.add_argument("--out", type=str, default="")
    p_p2.add_argument("--only", type=str, default="all", choices=["all", "weekday", "multiseed", "ctc"])
    p_p2.add_argument("--seeds", type=str, default="1337,1338,1339,1340,1341")

    p_p3 = sub.add_parser("gowda-phase3", help="Phase 3: full vocab import + SPD σ(τ) CTC")
    p_p3.add_argument("--data", type=str, required=True)
    p_p3.add_argument("--fs", type=int, default=5000)
    p_p3.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p_p3.add_argument("--out", type=str, default="")
    p_p3.add_argument("--epochs", type=int, default=50)
    p_p3.add_argument("--seed", type=int, default=1337)
    p_p3.add_argument("--import-only", action="store_true")
    p_p3.add_argument("--skip-import", action="store_true")
    p_p3.add_argument("--skip-raw-baseline", action="store_true")
    p_p3.add_argument("--download-dir", type=str, default="")
    p_p3.add_argument("--only", type=str, default="all", choices=["all", "import", "spd", "raw_ctc"])

    p_p4 = sub.add_parser("gowda-phase4", help="Phase 4: beam decode, test split, enhanced SPD v2")
    p_p4.add_argument("--data", type=str, required=True)
    p_p4.add_argument("--fs", type=int, default=5000)
    p_p4.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p_p4.add_argument("--out", type=str, default="")
    p_p4.add_argument("--epochs", type=int, default=60)
    p_p4.add_argument("--seed", type=int, default=1337)
    p_p4.add_argument("--seeds", type=str, default="1337,1338,1339")
    p_p4.add_argument("--checkpoint", type=str, default="")
    p_p4.add_argument("--beam-width", type=int, default=50)
    p_p4.add_argument("--only", type=str, default="all", choices=["all", "decode", "train", "multiseed"])
    p_p4.add_argument("--skip-multiseed", action="store_true")
    p_p4.add_argument("--no-upper-tri", action="store_true")

    p_p5 = sub.add_parser("gowda-phase5", help="Phase 5: efficient diag-delta SPD + Viterbi decode")
    p_p5.add_argument("--data", type=str, required=True)
    p_p5.add_argument("--fs", type=int, default=5000)
    p_p5.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p_p5.add_argument("--out", type=str, default="")
    p_p5.add_argument("--epochs", type=int, default=80)
    p_p5.add_argument("--seed", type=int, default=1337)
    p_p5.add_argument("--feature-mode", type=str, default="diag_delta", choices=["diag", "diag_delta", "upper_tri"])

    p_p6 = sub.add_parser("gowda-phase6", help="Phase 6: error analysis + trial-context LM decode")
    p_p6.add_argument("--data", type=str, required=True)
    p_p6.add_argument("--checkpoint", type=str, default="")
    p_p6.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p_p6.add_argument("--out", type=str, default="")
    p_p6.add_argument("--only", type=str, default="all", choices=["all", "errors", "trial_lm"])

    p_st = sub.add_parser("sim-transfer", help="Train sim corpus, evaluate on real Gowda test (sim2real harness)")
    p_st.add_argument("--sim", type=str, required=True)
    p_st.add_argument("--real", type=str, required=True)
    p_st.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p_st.add_argument("--out", type=str, default="")
    p_st.add_argument("--seed", type=int, default=1337)
    p_st.add_argument("--pretrain-epochs", type=int, default=30)
    p_st.add_argument("--finetune-epochs", type=int, default=15)
    p_st.add_argument("--anchor-epochs", type=int, default=15)
    p_st.add_argument("--no-anchor", action="store_true", help="Skip sim-pretrain → real anchor finetune")
    p_st.add_argument("--feature-mode", type=str, default="diag_delta")
    p_st.add_argument("--real-fracs", type=str, default="0,0.1,0.5,1.0")
    p_st.add_argument("--decode-mode", type=str, default="trial_lm")

    p_ra = sub.add_parser(
        "sim-realism",
        help="Realism preset + SNR ablations (probe stats vs real; optional sim-transfer)",
    )
    p_ra.add_argument("--real", type=str, required=True)
    p_ra.add_argument("--out", type=str, default="")
    p_ra.add_argument("--seed", type=int, default=1337)
    p_ra.add_argument("--probe-trials", type=int, default=8)
    p_ra.add_argument("--probe-only", action="store_true")
    p_ra.add_argument("--transfer", action="store_true")
    p_ra.add_argument("--top-k", type=int, default=0)
    p_ra.add_argument("--trials", type=int, default=100)
    p_ra.add_argument("--pretrain-epochs", type=int, default=30)
    p_ra.add_argument("--anchor-epochs", type=int, default=15)
    p_ra.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p_ra.add_argument("--feature-mode", type=str, default="diag_delta")
    p_ra.add_argument("--decode-mode", type=str, default="trial_lm")
    p_ra.add_argument("--variants", type=str, default="")
    p_ra.add_argument("--force-regen", action="store_true")

    p_fpt = sub.add_parser("fit-phone-templates", help="Fit per-phone EMG templates from Gowda session")
    p_fpt.add_argument("--session", type=str, required=True)
    p_fpt.add_argument("--out", type=str, default="")
    p_fpt.add_argument("--split", type=str, default="train", choices=["train", "all"])
    p_fpt.add_argument("--max-per-phone", type=int, default=400)
    p_fpt.add_argument("--seed", type=int, default=1337)
    p_fpt.add_argument(
        "--align",
        type=str,
        default="pseudo",
        choices=["pseudo", "corpus_duration"],
        help="Within-word phone segmentation for template fitting",
    )

    p_ps = sub.add_parser("phone-separability", help="Phone SPD cluster separability (real vs sim)")
    p_ps.add_argument("--session", type=str, required=True, help="Real Gowda session")
    p_ps.add_argument("--sim", type=str, default="")
    p_ps.add_argument("--out", type=str, default="")
    p_ps.add_argument("--max-events", type=int, default=400)
    p_ps.add_argument("--seed", type=int, default=1337)

    p_m1 = sub.add_parser(
        "m1-grid",
        help="M1 phoneme-synth ablation grid: generate corpora + sim-only/anchor transfer",
    )
    p_m1.add_argument("--real", type=str, required=True)
    p_m1.add_argument("--corpus-root", type=str, default="./corpus/m1_grid")
    p_m1.add_argument("--tags", type=str, default="", help="Comma subset of grid variant tags")
    p_m1.add_argument("--pretrain-epochs", type=int, default=30)
    p_m1.add_argument("--anchor-epochs", type=int, default=15)
    p_m1.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p_m1.add_argument("--force-regen", action="store_true")
    p_m1.add_argument("--skip-transfer", action="store_true", help="Generate corpora only")

    args = ap.parse_args(argv)

    if args.analyze_cmd == "sim-transfer":
        from pathlib import Path

        from .ml.eval.sim_transfer import run_sim_transfer

        fracs = [float(x.strip()) for x in str(args.real_fracs).split(",") if x.strip()]
        report = run_sim_transfer(
            Path(args.sim),
            Path(args.real),
            real_fracs=fracs,
            pretrain_epochs=int(args.pretrain_epochs),
            finetune_epochs=int(args.finetune_epochs),
            anchor_epochs=int(args.anchor_epochs),
            anchor_after_sim=not bool(args.no_anchor),
            seed=int(args.seed),
            device_preferred=str(args.device),
            feature_mode=str(args.feature_mode),
            decode_mode=str(args.decode_mode),
        )
        out = Path(args.out) if args.out else Path(args.real) / "ablations" / "sim_transfer_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[openalterego] sim-transfer report -> {out}")
        return 0

    if args.analyze_cmd == "sim-realism":
        from pathlib import Path

        from .ml.eval.sim_realism_ablation import run_realism_ablation
        from .sim.metrics.realism_match import parse_variant_tags

        tags = [x.strip() for x in str(args.variants).split(",") if x.strip()]
        variants = parse_variant_tags(tags if tags else None)
        out = Path(args.out) if args.out else Path(args.real) / "ablations" / "realism_ablation"
        run_realism_ablation(
            Path(args.real),
            out_dir=out,
            variants=variants,
            probe_trials=int(args.probe_trials),
            transfer_trials=int(args.trials),
            run_transfer=bool(args.transfer) and not bool(args.probe_only),
            top_k_transfer=int(args.top_k) if int(args.top_k) > 0 else None,
            pretrain_epochs=int(args.pretrain_epochs),
            anchor_epochs=int(args.anchor_epochs),
            seed=int(args.seed),
            device_preferred=str(args.device),
            feature_mode=str(args.feature_mode),
            decode_mode=str(args.decode_mode),
            force_regen=bool(args.force_regen),
        )
        return 0

    if args.analyze_cmd == "fit-phone-templates":
        from pathlib import Path

        from .ml.phonology.phone_templates import fit_and_save_phone_templates

        session = Path(args.session)
        out = Path(args.out) if args.out else session / "phone_templates.json"
        store = fit_and_save_phone_templates(
            session,
            out,
            split=str(args.split),
            max_segments_per_phone=int(args.max_per_phone),
            seed=int(args.seed),
            align_mode=str(args.align),  # type: ignore[arg-type]
        )
        print(f"[openalterego] fit-phone-templates: {len(store.phones)} phones -> {out}")
        return 0

    if args.analyze_cmd == "m1-grid":
        from pathlib import Path

        from .ml.eval.m1_transfer_grid import run_m1_transfer_grid

        tags = [x.strip() for x in str(args.tags).split(",") if x.strip()]
        run_m1_transfer_grid(
            Path(args.real),
            corpus_root=Path(args.corpus_root),
            pretrain_epochs=int(args.pretrain_epochs),
            anchor_epochs=int(args.anchor_epochs),
            device_preferred=str(args.device),
            force_regen=bool(args.force_regen),
            skip_transfer=bool(args.skip_transfer),
            tags=tags or None,
        )
        return 0

    if args.analyze_cmd == "phone-separability":
        from pathlib import Path

        from .ml.eval.phone_separability import run_phone_separability

        session = Path(args.session)
        sim = Path(args.sim) if args.sim else None
        report = run_phone_separability(session, sim, max_events=int(args.max_events), seed=int(args.seed))
        out = Path(args.out) if args.out else session / "ablations" / "phone_separability_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[openalterego] phone-separability -> {out}")
        if report.get("sim"):
            print(
                f"[openalterego] real between/within={report['real'].get('between_over_within')} "
                f"sim={report['sim'].get('between_over_within')}"
            )
        return 0

    if args.analyze_cmd == "gowda-phase6":
        from pathlib import Path

        from .ml.eval.gowda_phase6 import run_error_analysis_baseline, run_phase6_all, run_trial_lm_decode

        data_dir = Path(args.data)
        ckpt = Path(args.checkpoint) if str(args.checkpoint).strip() else data_dir / "ablations" / "ctc_spd_v3_diag_delta_seed1337.pt"
        if args.only == "errors":
            report = {"phase": 6, "sections": {"baseline_errors": run_error_analysis_baseline(data_dir, ckpt, device_preferred=str(args.device))}}
        elif args.only == "trial_lm":
            report = {"phase": 6, "sections": {"trial_lm": run_trial_lm_decode(data_dir, ckpt, device_preferred=str(args.device))}}
        else:
            report = run_phase6_all(data_dir, ckpt, device_preferred=str(args.device))
        out = Path(args.out) if args.out else data_dir / "ablations" / "phase6_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[openalterego] phase6 report -> {out}")
        return 0

    if args.analyze_cmd == "gowda-phase5":
        from pathlib import Path

        from .ml.eval.gowda_phase5 import run_spd_v3_efficient

        data_dir = Path(args.data)
        report = {
            "phase": 5,
            "sections": {
                "spd_v3": run_spd_v3_efficient(
                    data_dir,
                    fs_hz=int(args.fs),
                    epochs=int(args.epochs),
                    seed=int(args.seed),
                    device_preferred=str(args.device),
                    feature_mode=str(args.feature_mode),
                )
            },
        }
        out = Path(args.out) if args.out else data_dir / "ablations" / "phase5_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[openalterego] phase5 report -> {out}")
        return 0

    if args.analyze_cmd == "gowda-phase4":
        from pathlib import Path

        from .ml.eval.gowda_phase4 import (
            run_decode_ablation,
            run_multiseed_spd,
            run_phase4_all,
            run_spd_v2_train,
        )

        data_dir = Path(args.data)
        seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
        ckpt = Path(args.checkpoint) if str(args.checkpoint).strip() else data_dir / "ablations" / "ctc_spd.pt"
        if args.only == "decode":
            report = {
                "sections": {
                    "decode_ablation": run_decode_ablation(
                        data_dir, ckpt, device_preferred=str(args.device), beam_width=int(args.beam_width)
                    )
                }
            }
        elif args.only == "train":
            report = {
                "sections": {
                    "spd_v2_single": run_spd_v2_train(
                        data_dir,
                        fs_hz=int(args.fs),
                        epochs=int(args.epochs),
                        seed=int(args.seed),
                        device_preferred=str(args.device),
                        use_upper_tri=not bool(args.no_upper_tri),
                    )
                }
            }
        elif args.only == "multiseed":
            report = {
                "sections": {
                    "multiseed_spd_v2": run_multiseed_spd(
                        data_dir,
                        seeds=seeds,
                        fs_hz=int(args.fs),
                        epochs=int(args.epochs),
                        device_preferred=str(args.device),
                        use_upper_tri=not bool(args.no_upper_tri),
                    )
                }
            }
        else:
            report = run_phase4_all(
                data_dir,
                fs_hz=int(args.fs),
                device_preferred=str(args.device),
                seeds=seeds,
                legacy_checkpoint=ckpt if ckpt.is_file() else None,
                skip_multiseed=bool(args.skip_multiseed),
                epochs=int(args.epochs),
            )
        out = Path(args.out) if args.out else data_dir / "ablations" / "phase4_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[openalterego] phase4 report -> {out}")
        return 0

    if args.analyze_cmd == "gowda-phase3":
        from pathlib import Path

        from .ml.eval.gowda_phase3 import (
            import_full_small_vocab,
            run_phase3_all,
            run_raw_ctc_baseline,
            run_spd_ctc,
        )

        data_dir = Path(args.data)
        dl_dir = Path(args.download_dir) if str(args.download_dir).strip() else None
        if args.only == "import" or args.import_only:
            report = {"sections": {"import": import_full_small_vocab(data_dir, download_dir=dl_dir, fs_hz=int(args.fs))}}
        elif args.only == "spd":
            report = {
                "sections": {
                    "spd_ctc": run_spd_ctc(
                        data_dir,
                        fs_hz=int(args.fs),
                        device_preferred=str(args.device),
                        epochs=int(args.epochs),
                        seed=int(args.seed),
                    )
                }
            }
        elif args.only == "raw_ctc":
            report = {
                "sections": {
                    "raw_ctc_baseline": run_raw_ctc_baseline(
                        data_dir,
                        fs_hz=int(args.fs),
                        device_preferred=str(args.device),
                        epochs=int(args.epochs),
                        seed=int(args.seed),
                    )
                }
            }
        else:
            report = run_phase3_all(
                data_dir,
                fs_hz=int(args.fs),
                device_preferred=str(args.device),
                skip_import=bool(args.skip_import),
                skip_raw_baseline=bool(args.skip_raw_baseline),
                download_dir=dl_dir,
                epochs=int(args.epochs),
                seed=int(args.seed),
            )
        out = Path(args.out) if args.out else data_dir / "ablations" / "phase3_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[openalterego] phase3 report -> {out}")
        return 0

    if args.analyze_cmd == "gowda-phase2":
        from pathlib import Path

        from .ml.eval.gowda_phase2 import run_ctc_path, run_multiseed_bootstrap, run_phase2_all, run_weekday_ceiling

        data_dir = Path(args.data)
        seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
        if args.only == "weekday":
            report = {"sections": {"weekday_ceiling": run_weekday_ceiling(data_dir, fs_hz=int(args.fs), device_preferred=str(args.device))}}
        elif args.only == "multiseed":
            report = {
                "sections": {
                    "multiseed_bootstrap": run_multiseed_bootstrap(
                        data_dir, fs_hz=int(args.fs), seeds=seeds, device_preferred=str(args.device)
                    )
                }
            }
        elif args.only == "ctc":
            report = {"sections": {"ctc_phoneme": run_ctc_path(data_dir, fs_hz=int(args.fs), device_preferred=str(args.device))}}
        else:
            report = run_phase2_all(data_dir, fs_hz=int(args.fs), device_preferred=str(args.device), seeds=seeds)
        out = Path(args.out) if args.out else data_dir / "ablations" / "phase2_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {out}")
        return 0

    if args.analyze_cmd == "gowda-ablation":
        from pathlib import Path

        from .ml.eval.gowda_ablation import run_phase1_matrix

        data_dir = Path(args.data)
        report = run_phase1_matrix(
            data_dir,
            fs_hz=int(args.fs),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            device_preferred=str(args.device),
        )
        out = Path(args.out) if args.out else data_dir / "ablations" / "phase1_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("\n=== Phase 1 ablation summary ===")
        print(f"{'config':28s}  train_acc  val_acc  val_F1  gap")
        for row in report["rows"]:
            c = row["config"]["name"]
            print(
                f"{c:28s}  {row['train']['accuracy']:8.3f}  "
                f"{row['val']['accuracy']:7.3f}  {row['val']['macro_f1']:6.3f}  {row['train_val_gap']:+.3f}"
            )
        print(f"\nBest (macro-F1): {report['best_by_val_macro_f1']}")
        print(f"Wrote {out}")
        return 0

    if args.analyze_cmd == "channel-importance":
        seg = int(args.segment_ms) if int(args.segment_ms) > 0 else None
        rep = run_channel_importance(
            Path(args.session),
            Path(args.model),
            segment_ms=seg,
            top_k=int(args.top_k),
        )
        out_path = Path(args.out) if args.out else Path(args.session) / "channel_importance.json"
        out_path.write_text(json.dumps(rep.to_dict(), indent=2), encoding="utf-8")
        if args.json:
            print(json.dumps(rep.to_dict(), indent=2))
        else:
            print(f"[openalterego] channel importance -> {out_path}")
            print(f"  top-{len(rep.top_channels)} channels: {rep.top_channels}")
            for note in rep.notes:
                print(f"  note: {note}")
        return 0

    ap.print_help()
    return 2


def _cmd_dataset(argv: List[str]) -> int:
    import json

    from .ml.datasets.gaddy import convert_gaddy_raw_dir, import_gaddy_session
    from .ml.datasets.gowda import import_gowda_nato_subject, import_gowda_small_vocab, write_dataset_catalog

    ap = argparse.ArgumentParser(prog="openalterego dataset")
    sub = ap.add_subparsers(dest="dataset_cmd", required=True)

    p_cat = sub.add_parser("catalog", help="Write JSON catalog of public EMG datasets")
    p_cat.add_argument("--out", type=str, default="datasets/catalog.json")

    p_gaddy = sub.add_parser("import-gaddy", help="Download/convert Gaddy Zenodo silent-speech EMG")
    p_gaddy.add_argument("--out", type=str, required=True, help="Output session folder")
    p_gaddy.add_argument("--raw", type=str, default="", help="Already-extracted raw directory")
    p_gaddy.add_argument("--download-dir", type=str, default="", help="Where to store Zenodo tar")
    p_gaddy.add_argument("--max-samples", type=int, default=40)
    p_gaddy.add_argument("--top-labels", type=int, default=0, help="Keep only top-N frequent labels")
    p_gaddy.add_argument("--min-samples-per-label", type=int, default=1)
    p_gaddy.add_argument("--skip-download", action="store_true")
    p_gaddy.add_argument("--vocal-too", action="store_true", help="Include vocalized (not silent-only)")
    p_gaddy.add_argument(
        "--label-mode",
        type=str,
        default="first_word",
        choices=["first_word", "sentence"],
    )
    p_gaddy.add_argument("--json", action="store_true")

    p_gowda = sub.add_parser("import-gowda", help="Convert Gowda small-vocab (OSF download or local npy)")
    p_gowda.add_argument("--out", type=str, required=True)
    p_gowda.add_argument("--download", action="store_true", help="Fetch dataSmallVocab.npy + labels from OSF")
    p_gowda.add_argument("--download-dir", type=str, default="", help="Where to store OSF npy files")
    p_gowda.add_argument("--data-npy", type=str, default="")
    p_gowda.add_argument("--labels-npy", type=str, default="")
    p_gowda.add_argument("--fs", type=float, default=5000.0)
    p_gowda.add_argument("--max-segments", type=int, default=0)
    p_gowda.add_argument("--top-labels", type=int, default=0, help="Keep top-N labels by count (0 = all)")
    p_gowda.add_argument("--full-vocab", action="store_true", help="Import all labels (same as --top-labels 0)")
    p_gowda.add_argument("--min-samples-per-label", type=int, default=1)
    p_gowda.add_argument("--json", action="store_true")

    p_nato = sub.add_parser("import-gowda-nato", help="Convert a Gowda Subject N NATO-words folder")
    p_nato.add_argument("--subject-dir", type=str, required=True)
    p_nato.add_argument("--out", type=str, required=True)
    p_nato.add_argument("--fs", type=float, default=5000.0)
    p_nato.add_argument("--json", action="store_true")

    p_ab = sub.add_parser("ab-preprocess", help="A/B standard vs wide on an imported session")
    p_ab.add_argument("--session", type=str, required=True)
    p_ab.add_argument("--modes", type=str, default="standard,wide")
    p_ab.add_argument("--epochs", type=int, default=12)
    p_ab.add_argument("--segment-ms", type=int, default=600)
    p_ab.add_argument("--arch", type=str, default="se_resnet", choices=["cnn", "se_resnet"])
    p_ab.add_argument("--min-samples-per-label", type=int, default=2)
    p_ab.add_argument("--json", action="store_true")

    p_cache = sub.add_parser(
        "cache-preprocess",
        help="Build session preprocess_cache/ (streaming/offline × emg_mode)",
    )
    p_cache.add_argument("--session", type=str, required=True)
    p_cache.add_argument("--preprocess-mode", type=str, default="streaming", choices=["offline", "streaming"])
    p_cache.add_argument("--emg-mode", type=str, default="wide", choices=["standard", "clinical", "wide"])
    p_cache.add_argument("--fs", type=float, default=0, help="Override fs_hz (default: meta.json)")
    p_cache.add_argument("--rebuild", action="store_true")
    p_cache.add_argument("--json", action="store_true")

    args = ap.parse_args(argv)

    if args.dataset_cmd == "catalog":
        out = write_dataset_catalog(args.out)
        print(f"[openalterego] wrote dataset catalog: {out}")
        return 0

    if args.dataset_cmd == "import-gaddy":
        raw = Path(args.raw) if str(args.raw).strip() else None
        dl = Path(args.download_dir) if str(args.download_dir).strip() else None

        def _progress(msg: str) -> None:
            print(f"[openalterego] {msg}")

        if raw is not None and raw.is_dir():
            rep = convert_gaddy_raw_dir(
                raw,
                Path(args.out),
                silent_only=not bool(args.vocal_too),
                label_mode=str(args.label_mode),
                max_samples=int(args.max_samples) if int(args.max_samples) > 0 else None,
                top_labels=int(args.top_labels) if int(args.top_labels) > 0 else None,
                min_samples_per_label=int(args.min_samples_per_label),
            )
        else:
            rep = import_gaddy_session(
                Path(args.out),
                raw_dir=raw,
                download_dir=dl,
                max_samples=int(args.max_samples) if int(args.max_samples) > 0 else None,
                silent_only=not bool(args.vocal_too),
                label_mode=str(args.label_mode),
                skip_download=bool(args.skip_download),
                progress_cb=_progress,
            )
            # re-convert with label filters if requested (import writes unfiltered)
            if int(args.top_labels) > 0 or int(args.min_samples_per_label) > 1:
                raw_guess = dl / "raw" if dl else Path(args.out).parent / "gaddy_download" / "raw"
                if raw_guess.is_dir():
                    rep = convert_gaddy_raw_dir(
                        raw_guess,
                        Path(args.out),
                        silent_only=not bool(args.vocal_too),
                        label_mode=str(args.label_mode),
                        max_samples=int(args.max_samples) if int(args.max_samples) > 0 else None,
                        top_labels=int(args.top_labels) if int(args.top_labels) > 0 else None,
                        min_samples_per_label=int(args.min_samples_per_label),
                    )
        if args.json:
            print(json.dumps(rep.to_dict(), indent=2))
        else:
            print(f"[openalterego] Gaddy session -> {rep.out_dir}")
            print(f"  events={rep.n_events} skipped={rep.n_skipped} duration={rep.duration_s:.1f}s")
            print(f"  labels ({len(rep.labels)}): {', '.join(rep.labels[:12])}{'...' if len(rep.labels) > 12 else ''}")
        return 0

    if args.dataset_cmd == "import-gowda":
        from .ml.datasets.gowda import import_gowda_small_vocab, import_gowda_small_vocab_from_osf

        def _progress(msg: str) -> None:
            print(f"[openalterego] {msg}")

        top_n = None if bool(args.full_vocab) else (int(args.top_labels) if int(args.top_labels) > 0 else None)
        if bool(args.download):
            rep = import_gowda_small_vocab_from_osf(
                Path(args.out),
                download_dir=Path(args.download_dir) if str(args.download_dir).strip() else None,
                fs_hz=float(args.fs),
                max_segments=int(args.max_segments) if int(args.max_segments) > 0 else None,
                top_labels=top_n,
                min_samples_per_label=int(args.min_samples_per_label),
                progress_cb=_progress,
            )
        else:
            if not str(args.data_npy).strip() or not str(args.labels_npy).strip():
                ap.error("import-gowda requires --data-npy and --labels-npy, or --download")
            rep = import_gowda_small_vocab(
                args.data_npy,
                args.labels_npy,
                args.out,
                fs_hz=float(args.fs),
                max_segments=int(args.max_segments) if int(args.max_segments) > 0 else None,
                top_labels=top_n,
                min_samples_per_label=int(args.min_samples_per_label),
            )
        if args.json:
            print(json.dumps(rep.to_dict(), indent=2))
        else:
            print(f"[openalterego] Gowda session -> {rep.out_dir}")
            print(f"  events={rep.n_events} ch={rep.channels} fs={rep.fs_hz} duration={rep.duration_s:.1f}s")
            print(
                "  next: openalterego dataset cache-preprocess "
                f"--session {rep.out_dir} --emg-mode wide --preprocess-mode streaming"
            )
        return 0

    if args.dataset_cmd == "import-gowda-nato":
        rep = import_gowda_nato_subject(args.subject_dir, args.out, fs_hz=float(args.fs))
        if args.json:
            print(json.dumps(rep.to_dict(), indent=2))
        else:
            print(f"[openalterego] Gowda NATO session -> {rep.out_dir} events={rep.n_events}")
        return 0

    if args.dataset_cmd == "cache-preprocess":
        from .dsp.preprocess_cache import build_session_preprocess_cache

        fs = float(args.fs) if float(args.fs) > 0 else None
        rep = build_session_preprocess_cache(
            Path(args.session),
            preprocess_mode=str(args.preprocess_mode),  # type: ignore[arg-type]
            emg_mode=str(args.emg_mode),  # type: ignore[arg-type]
            fs_hz=fs,
            rebuild=bool(args.rebuild),
            show_progress=not bool(args.json),
        )
        if args.json:
            print(json.dumps(rep.to_dict(), indent=2))
        else:
            print(f"[openalterego] preprocess cache session={rep.session_dir}")
            print(f"  path={rep.cache_path} shape={rep.shape} built={rep.built} elapsed={rep.elapsed_s:.1f}s")
            for note in rep.notes:
                print(f"  note: {note}")
        return 0

    if args.dataset_cmd == "ab-preprocess":
        from .ml.eval.session_ab import run_session_preprocess_ab

        modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]
        rep = run_session_preprocess_ab(
            Path(args.session),
            emg_modes=modes,
            segment_ms=int(args.segment_ms),
            epochs=int(args.epochs),
            arch=str(args.arch),
            min_samples_per_label=int(args.min_samples_per_label),
            show_progress=not bool(args.json),
        )
        if args.json:
            print(json.dumps(rep.to_dict(), indent=2))
        else:
            print(f"[openalterego] A/B preprocess session={rep.session_dir} fs={rep.fs_hz} labels={rep.n_labels}")
            for row in rep.rows:
                snr = "n/a" if row.snr_db is None else f"{row.snr_db:.1f}dB"
                print(
                    f"  {row.emg_mode:8s}  snr={snr}  motion={row.motion_index:.3f}  "
                    f"val_acc={row.val_acc:.3f}  train_acc={row.train_acc:.3f}  n={row.n_train}/{row.n_val}"
                )
            for note in rep.notes:
                print(f"  note: {note}")
        return 0

    ap.print_help()
    return 2


def _cmd_latency_bench(argv: List[str]) -> int:
    import json

    from .runtime.latency_benchmark import run_latency_benchmark

    ap = argparse.ArgumentParser(prog="openalterego latency-bench")
    ap.add_argument("--model", type=str, required=True, help="Path to model.pt")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--n-chunks", type=int, default=200)
    ap.add_argument("--window-ms", type=int, default=600)
    ap.add_argument("--stride-ms", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sim-engine", type=str, default="heuristic", choices=["heuristic", "biophysical"])
    ap.add_argument("--motion-gate", action="store_true")
    ap.add_argument("--motion-threshold", type=float, default=0.35)
    args = ap.parse_args(argv)

    rep = run_latency_benchmark(
        model_path=str(args.model),
        n_chunks=int(args.n_chunks),
        window_ms=int(args.window_ms),
        stride_ms=int(args.stride_ms),
        seed=int(args.seed),
        sim_engine=str(args.sim_engine),
        motion_gate=bool(args.motion_gate),
        motion_threshold=float(args.motion_threshold),
    )
    if args.json:
        print(json.dumps(rep.to_dict(), indent=2))
    else:
        print(f"[openalterego] latency bench fs={rep.fs_hz} ch={rep.channels} n={rep.n_chunks}")
        print(f"  preprocess  p50={rep.preprocess.p50_ms:.2f} p95={rep.preprocess.p95_ms:.2f} ms")
        print(f"  inference   p50={rep.inference_window.p50_ms:.2f} p95={rep.inference_window.p95_ms:.2f} ms")
        print(f"  chunk_push  p50={rep.chunk_push.p50_ms:.2f} p95={rep.chunk_push.p95_ms:.2f} ms")
        for note in rep.notes:
            print(f"  note: {note}")
    return 0


def main(argv: List[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] == "hw":
        raise SystemExit(_cmd_hw(argv[1:]))
    if argv and argv[0] == "user":
        raise SystemExit(_cmd_user(argv[1:]))
    if argv and argv[0] == "calibrate":
        raise SystemExit(_cmd_calibrate(argv[1:]))
    if argv and argv[0] == "collect":
        raise SystemExit(_cmd_collect(argv[1:]))
    if argv and argv[0] == "dataset":
        raise SystemExit(_cmd_dataset(argv[1:]))
    if argv and argv[0] == "analyze":
        raise SystemExit(_cmd_analyze(argv[1:]))
    if argv and argv[0] == "decode-utterance":
        raise SystemExit(_cmd_decode_utterance(argv[1:]))

    ap = argparse.ArgumentParser(prog="openalterego")
    ap.add_argument("--version", action="store_true", help="Print version and exit")

    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("sim-dataset", help="Generate a synthetic dataset session folder")
    sub.add_parser("sim-benchmark", help="Benchmark biophysical synthesis throughput and scaling")
    sub.add_parser("latency-bench", help="Measure preprocess/inference latency percentiles (p50/p95)")
    sub.add_parser("window-sweep", help="Sweep window_ms vs latency and event accuracy")

    sub.add_parser("serve", help="Run the websocket server (delegates to openalterego.api.server)")

    sub.add_parser("collect", help="Record a session folder from sim or BLE (see: openalterego collect -h)")

    sub.add_parser("hw", help="Hardware DSL: validate, resolve, simulate (see: openalterego hw -h)")

    sub.add_parser("train", help="Train a baseline model (delegates to openalterego.ml.train)")

    sub.add_parser("train-benchmark", help="Benchmark training phases and bottlenecks")

    sub.add_parser("dataset", help="Import public EMG datasets (Gaddy, Gowda)")

    sub.add_parser("analyze", help="Model analysis (channel importance, Gowda phases, sim-transfer)")

    sub.add_parser(
        "decode-utterance",
        help="Offline SPD+CTC decode for one Gowda trial (4 words)",
    )

    sub.add_parser("glasses", help="Run the smart-glasses display stub (terminal)")

    ns, rest = ap.parse_known_args(argv)

    if ns.version:
        print(__version__)
        raise SystemExit(0)

    if ns.cmd == "sim-dataset":
        raise SystemExit(_cmd_sim_dataset(rest))

    if ns.cmd == "sim-benchmark":
        raise SystemExit(_cmd_sim_benchmark(rest))

    if ns.cmd == "latency-bench":
        raise SystemExit(_cmd_latency_bench(rest))

    if ns.cmd == "window-sweep":
        raise SystemExit(_cmd_window_sweep(rest))

    if ns.cmd == "collect":
        raise SystemExit(_cmd_collect(rest))

    if ns.cmd == "hw":
        raise SystemExit(_cmd_hw(rest))

    if ns.cmd == "serve":
        from .api.server import main as server_main

        raise SystemExit(_delegate_module(server_main, rest))

    if ns.cmd == "train":
        from .ml.train import main as train_main

        raise SystemExit(_delegate_module(train_main, rest))

    if ns.cmd == "train-benchmark":
        from .ml.train_benchmark import main as train_benchmark_main

        raise SystemExit(_delegate_module(train_benchmark_main, rest))

    if ns.cmd == "dataset":
        raise SystemExit(_cmd_dataset(rest))

    if ns.cmd == "analyze":
        raise SystemExit(_cmd_analyze(rest))

    if ns.cmd == "decode-utterance":
        raise SystemExit(_cmd_decode_utterance(rest))

    if ns.cmd == "glasses":
        from .clients.glasses_stub import main as glasses_main

        raise SystemExit(_delegate_module(glasses_main, rest))

    ap.print_help()
    raise SystemExit(2)


if __name__ == "__main__":
    main()
