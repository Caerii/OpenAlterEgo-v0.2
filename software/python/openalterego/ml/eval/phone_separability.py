"""Evaluate per-phone SPD template separability (real vs sim)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..phonology.gowda_lexicon import build_lexicon
from ..phonology.phone_templates import fit_phone_templates
from ..spd.basis import ensure_gowda_spd_basis
from ..spd.features import edge_matrix, sigma_diag_vector, spd_regularize
from ...dsp.filters import preprocess_basic
from ...sim.phonology.coarticulation import build_phone_coarticulation_envelopes
from ...sim.phonology.timeline import partition_phones_in_event
from ...sim.phonology.templates import load_phone_templates


def _vec_from_segment(seg: np.ndarray, basis_q: np.ndarray, fs_hz: float) -> np.ndarray:
    proc = preprocess_basic(seg, fs_hz=float(fs_hz), mode="gowda")
    edge = spd_regularize(edge_matrix(proc))
    diag = sigma_diag_vector(edge, basis_q).astype(np.float64)
    return np.concatenate([diag, np.zeros_like(diag)])


def _collect_phone_vectors(
    signals: np.ndarray,
    events: pd.DataFrame,
    lexicon: dict[str, Sequence[str]],
    *,
    basis_q: np.ndarray,
    fs_hz: float,
    seed: int,
    template_store=None,
    use_coarticulation: bool = False,
    coarticulation_overlap_fraction: float = 0.28,
    coarticulation_min_overlap_ms: float = 10.0,
) -> Dict[str, List[np.ndarray]]:
    out: Dict[str, List[np.ndarray]] = {}
    rng = np.random.default_rng(int(seed))
    min_ov = max(1, int(round(float(coarticulation_min_overlap_ms) * float(fs_hz) / 1000.0)))
    for row in events.itertuples(index=False):
        word = str(row.label)
        phones = tuple(str(p) for p in lexicon.get(word, ()))
        if not phones:
            continue
        s0 = int(row.start_sample)
        s1 = int(row.end_sample)
        n = s1 - s0
        weights = None
        if template_store is not None:
            weights = [template_store.duration_weight(p) for p in phones]
        seg_lens = partition_phones_in_event(n, phones, rng, duration_weights=weights)
        if use_coarticulation and len(phones) > 1:
            env = build_phone_coarticulation_envelopes(
                seg_lens,
                overlap_fraction=float(coarticulation_overlap_fraction),
                min_overlap_samples=min_ov,
            )
            block = np.asarray(signals[s0:s1, :], dtype=np.float32)
            dominant = np.argmax(env, axis=0)
            t = 0
            while t < n:
                pid = int(dominant[t])
                j = t + 1
                while j < n and int(dominant[j]) == pid:
                    j += 1
                seg = block[t:j, :]
                if seg.shape[0] >= 16:
                    key = str(phones[pid]).strip().upper()
                    out.setdefault(key, []).append(_vec_from_segment(seg, basis_q, fs_hz))
                t = j
            continue
        cur = s0
        for phone, slen in zip(phones, seg_lens):
            seg = np.asarray(signals[cur : cur + int(slen), :], dtype=np.float32)
            cur += int(slen)
            if seg.shape[0] < 16:
                continue
            key = str(phone).strip().upper()
            out.setdefault(key, []).append(_vec_from_segment(seg, basis_q, fs_hz))
    return out


def _separability_score(vectors_by_phone: Dict[str, List[np.ndarray]]) -> Dict[str, Any]:
    phones = sorted(vectors_by_phone.keys())
    if len(phones) < 2:
        return {"n_phones": len(phones), "ratio": None}
    centroids = {p: np.mean(vectors_by_phone[p], axis=0) for p in phones if vectors_by_phone[p]}
    within: list[float] = []
    between: list[float] = []
    for p, vecs in vectors_by_phone.items():
        if p not in centroids:
            continue
        c = centroids[p]
        for v in vecs:
            within.append(float(np.linalg.norm(v - c)))
    keys = list(centroids.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            between.append(float(np.linalg.norm(centroids[keys[i]] - centroids[keys[j]])))
    w = float(np.median(within)) if within else 0.0
    b = float(np.median(between)) if between else 0.0
    ratio = float(b / (w + 1e-9)) if w > 0 else None
    return {
        "n_phones": len(centroids),
        "n_vectors": int(sum(len(v) for v in vectors_by_phone.values())),
        "median_within": w,
        "median_between": b,
        "between_over_within": ratio,
    }


def run_phone_separability(
    real_dir: Path,
    sim_dir: Optional[Path] = None,
    *,
    max_events: int = 400,
    seed: int = 1337,
) -> Dict[str, Any]:
    """Compare phone-cluster geometry on real vs optional sim session."""
    real_dir = Path(real_dir)
    signals = np.load(real_dir / "signals.npy", mmap_mode="r")
    events = pd.read_csv(real_dir / "events.csv")
    if len(events) > int(max_events):
        events = events.iloc[: int(max_events)].reset_index(drop=True)
    meta = json.loads((real_dir / "meta.json").read_text(encoding="utf-8"))
    fs = float(meta.get("fs_hz", 5000))
    words = sorted(set(str(x) for x in events["label"].astype(str)))
    lex = build_lexicon(words)
    basis = ensure_gowda_spd_basis(real_dir, fs_hz=int(fs), feature_mode="diag_delta")
    real_vecs = _collect_phone_vectors(
        signals, events, lex, basis_q=basis.basis_q, fs_hz=fs, seed=seed
    )
    report: Dict[str, Any] = {
        "real_dir": str(real_dir),
        "real": _separability_score(real_vecs),
        "real_templates": fit_phone_templates(real_dir, split="train", seed=seed).to_dict(),
    }
    if sim_dir is not None:
        sim_dir = Path(sim_dir)
        sim_sig = np.load(sim_dir / "signals.npy", mmap_mode="r")
        sim_ev = pd.read_csv(sim_dir / "events.csv")
        if len(sim_ev) > int(max_events):
            sim_ev = sim_ev.iloc[: int(max_events)].reset_index(drop=True)
        tpl_path = sim_dir / "meta.json"
        store = None
        use_coart = False
        coart_frac = 0.28
        coart_min_ms = 10.0
        if tpl_path.is_file():
            sm = json.loads(tpl_path.read_text(encoding="utf-8"))
            pt = sm.get("biophysical", {}).get("phone_templates_path") or sm.get("phone_templates_path")
            if pt and Path(str(pt)).is_file():
                store = load_phone_templates(str(pt))
            bio = sm.get("biophysical", {})
            use_coart = bool(bio.get("coarticulation_enabled", False))
            coart_frac = float(bio.get("coarticulation_overlap_fraction", 0.28))
            coart_min_ms = float(bio.get("coarticulation_min_overlap_ms", 10.0))
        sim_vecs = _collect_phone_vectors(
            sim_sig,
            sim_ev,
            lex,
            basis_q=basis.basis_q,
            fs_hz=fs,
            seed=seed + 1,
            template_store=store,
            use_coarticulation=use_coart,
            coarticulation_overlap_fraction=coart_frac,
            coarticulation_min_overlap_ms=coart_min_ms,
        )
        report["sim_dir"] = str(sim_dir)
        report["sim"] = _separability_score(sim_vecs)
        report["sim_coarticulation_eval"] = use_coart
        # Template centroid cosine similarity (real fit vs sim empirical)
        real_tpl = report["real_templates"]["phones"]
        overlaps: List[float] = []
        for phone, row in real_tpl.items():
            if phone not in sim_vecs or phone not in real_vecs:
                continue
            a = np.mean(real_vecs[phone], axis=0)
            b = np.mean(sim_vecs[phone], axis=0)
            denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
            overlaps.append(float(np.dot(a, b) / denom))
        report["sim_template_cosine_mean"] = float(np.mean(overlaps)) if overlaps else None
    return report
