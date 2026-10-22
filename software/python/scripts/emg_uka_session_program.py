"""Frozen three-channel session-transfer assay on EMG-UKA speaker 008.

Selection reads sessions 001-007 only. Evaluation requires its saved selection
record and then reads session 008. The ten test prompts repeat by design;
this is a closed-prompt virtual channel-subset assay, not speech dictation.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


SOURCE_SCHEMA = "openalterego.external-emg-source.v1"
RESULT_SCHEMA = "openalterego.emg-uka-session-program.v1"
DEVELOPMENT = tuple(f"{index:03d}" for index in range(1, 8))
HOLDOUT = "008"
CHANNELS = tuple(itertools.combinations(range(6), 3))
FIXED = (0, 1, 2)
ALPHA = 10.0
BINS = 5


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_source(root: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SOURCE_SCHEMA:
        raise ValueError("unexpected corpus manifest schema")
    subset = (root / "Subsets" / "test.audible").read_bytes()
    if sha256(subset) != manifest["subset_test_audible_sha256"]:
        raise ValueError("EMG-UKA subset file hash mismatch")
    return manifest


def load_adc(path: Path, offsets: Path) -> np.ndarray:
    raw = np.frombuffer(path.read_bytes(), dtype="<i2")
    if raw.size % 7 or raw.size < 7 * BINS * 3:
        raise ValueError(f"invalid seven-channel ADC length: {path}")
    samples = raw.reshape(-1, 7)
    lines = offsets.read_text(encoding="utf-8").splitlines()
    if len(lines) != 2:
        raise ValueError(f"invalid offset format: {offsets}")
    start, end = map(int, lines[1].split())
    if not 0 <= start < end <= len(samples):
        raise ValueError(f"EMG offset outside recording: {offsets}")
    return samples[start:end, :6].astype(np.float64)


def utterance_features(samples: np.ndarray) -> np.ndarray:
    """Per-utterance centering, five relative-time bins, log RMS and WL."""
    if samples.ndim != 2 or samples.shape[1] != 6 or len(samples) < BINS * 3:
        raise ValueError("unexpected EMG crop shape")
    centered = samples - samples.mean(axis=0)
    rows = []
    for chunk in np.array_split(centered, BINS):
        rms = np.sqrt(np.mean(chunk * chunk, axis=0))
        wavelength = np.mean(abs(np.diff(chunk, axis=0)), axis=0)
        rows.extend((np.log1p(rms), np.log1p(wavelength)))
    return np.stack(rows).T.reshape(-1)


def session(root: Path, session_id: str) -> tuple[np.ndarray, tuple[str, ...], str]:
    if session_id not in (*DEVELOPMENT, HOLDOUT):
        raise ValueError("unexpected speaker-008 session")
    subset = (root / "Subsets" / "test.audible").read_text(
        encoding="utf-8")
    key = f"emg_008-{session_id}:"
    lines = [line for line in subset.splitlines() if line.startswith(key)]
    if len(lines) != 1:
        raise ValueError(f"missing or duplicate subset session: {key}")
    ids = lines[0].split(":", 1)[1].split()
    if len(ids) != 10 or len(set(ids)) != 10:
        raise ValueError("expected ten unique test utterances")
    features, prompts = [], []
    digest = hashlib.sha256()
    for identifier in ids:
        speaker, recorded_session, utterance = identifier.removeprefix(
            "emg_").split("-")
        if speaker != "008" or recorded_session != session_id:
            raise ValueError("subset identifier/session mismatch")
        stem = f"{speaker}_{session_id}_{utterance}"
        files = (
            root / "emg" / speaker / session_id / f"e07_{stem}.adc",
            root / "offset" / speaker / session_id / f"offset_{stem}.txt",
            root / "Transcripts" / speaker / session_id /
            f"transcript_{stem}.txt",
        )
        for path in files:
            content = path.read_bytes()
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(len(content).to_bytes(8, "little"))
            digest.update(content)
        features.append(utterance_features(load_adc(files[0], files[1])))
        prompts.append(files[2].read_text(encoding="utf-8").strip())
    if len(set(prompts)) != 10:
        raise ValueError("expected ten distinct prompt texts")
    return np.asarray(features), tuple(prompts), digest.hexdigest()


def classify(train_x: np.ndarray, train_y: np.ndarray,
             query_x: np.ndarray, channels: tuple[int, ...]) -> np.ndarray:
    columns = np.asarray([10 * channel + feature
                          for channel in channels for feature in range(10)])
    x = train_x[:, columns]
    query = query_x[:, columns]
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    x = (x - mean) / scale
    query = (query - mean) / scale
    targets = np.eye(10)[train_y]
    weights = np.linalg.solve(x.T @ x + ALPHA * np.eye(len(columns)),
                              x.T @ targets)
    return np.argmax(query @ weights, axis=1)


def accuracy(train_x, train_y, query_x, query_y, channels) -> float:
    return float(np.mean(classify(train_x, train_y, query_x, channels)
                         == query_y))


def development_data(root: Path):
    sessions = [session(root, value) for value in DEVELOPMENT]
    labels = tuple(sorted(sessions[0][1]))
    if any(set(prompts) != set(labels) for _, prompts, _ in sessions):
        raise ValueError("speaker-008 prompt sets differ across sessions")
    index = {prompt: offset for offset, prompt in enumerate(labels)}
    xs = [features for features, _, _ in sessions]
    ys = [np.asarray([index[prompt] for prompt in prompts])
          for _, prompts, _ in sessions]
    return xs, ys, labels, {key: item[2] for key, item in
                            zip(DEVELOPMENT, sessions)}


def select(root: Path, manifest: dict) -> dict:
    xs, ys, labels, digests = development_data(root)
    scores = np.empty((len(CHANNELS), len(DEVELOPMENT)))
    for fold in range(len(DEVELOPMENT)):
        train_x = np.vstack([x for i, x in enumerate(xs) if i != fold])
        train_y = np.concatenate([y for i, y in enumerate(ys) if i != fold])
        for program_index, program in enumerate(CHANNELS):
            scores[program_index, fold] = accuracy(
                train_x, train_y, xs[fold], ys[fold], program)
    ranking = sorted(range(len(CHANNELS)),
                     key=lambda i: (-scores[i].mean(), -scores[i].min(),
                                    CHANNELS[i]))
    chosen = CHANNELS[ranking[0]]
    random = CHANNELS[int(np.random.default_rng(42).integers(len(CHANNELS)))]
    return {
        "schema": RESULT_SCHEMA, "phase": "selection",
        "scope": "Speaker 008, audible, ten repeated test prompts per session; virtual recorded-channel subsets, no silent speech or physical intervention",
        "source_download": manifest["download"],
        "archive_sha256": manifest["downloaded_archive_sha256"],
        "subset_sha256": manifest["subset_test_audible_sha256"],
        "development_sessions": DEVELOPMENT, "holdout_session": HOLDOUT,
        "source_content_digests": digests,
        "prompt_texts": labels,
        "feature": "per-utterance centered ADC; five relative-time bins; log1p RMS and mean absolute adjacent difference per channel; no future-session normalization",
        "classifier": "train-fold standardized ridge one-hot, alpha 10",
        "selection": "mean leave-one-session-out prompt accuracy; minimum fold then lexicographic tie break",
        "program_size": 3, "candidate_count": len(CHANNELS),
        "selection_fold_accuracy": {"-".join(map(str, program)):
                                    scores[i].tolist()
                                    for i, program in enumerate(CHANNELS)},
        "selected": chosen, "fixed": FIXED, "random_seed42": random,
    }


def evaluate(root: Path, manifest: dict, selection_path: Path) -> dict:
    payload = selection_path.read_bytes()
    selected = json.loads(payload)
    if (selected.get("schema") != RESULT_SCHEMA or
            selected.get("phase") != "selection" or
            selected.get("archive_sha256") !=
            manifest["downloaded_archive_sha256"] or
            selected.get("holdout_session") != HOLDOUT):
        raise ValueError("selection record does not match corpus protocol")
    xs, ys, labels, digests = development_data(root)
    if (selected["source_content_digests"] != digests or
            tuple(selected["prompt_texts"]) != labels):
        raise ValueError("selection inputs changed before evaluation")
    query_x, prompts, holdout_digest = session(root, HOLDOUT)
    if set(prompts) != set(labels):
        raise ValueError("holdout prompt set differs")
    index = {prompt: offset for offset, prompt in enumerate(labels)}
    query_y = np.asarray([index[prompt] for prompt in prompts])
    train_x, train_y = np.vstack(xs), np.concatenate(ys)
    choices = {"selected_three": tuple(selected["selected"]),
               "fixed_three": tuple(selected["fixed"]),
               "random_three": tuple(selected["random_seed42"]),
               "all_six": tuple(range(6))}
    results = {}
    for name, program in choices.items():
        predictions = classify(train_x, train_y, query_x, program)
        results[name] = {"channels": program,
                         "correct": int(np.sum(predictions == query_y)),
                         "accuracy": float(np.mean(predictions == query_y)),
                         "correct_by_prompt": (predictions == query_y).tolist()}
    return {"schema": RESULT_SCHEMA, "phase": "evaluation",
            "selection_record_sha256": sha256(payload),
            "selection_record": str(selection_path),
            "holdout_session": HOLDOUT,
            "holdout_content_digest": holdout_digest,
            "holdout_samples": len(query_y), "results": results,
            "scope": "Retrospective single-session closed-prompt virtual subset assay; accuracy is not open-vocabulary or physical-design performance"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=("select", "evaluate"), required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or (args.phase == "evaluate" and
                                args.selection is None):
        parser.error("new output and evaluation selection record required")
    manifest = check_source(args.root, args.manifest)
    result = (select(args.root, manifest) if args.phase == "select" else
              evaluate(args.root, manifest, args.selection))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"phase": result["phase"],
                      "selected": result.get("selected"),
                      "results": result.get("results")}, indent=2))


if __name__ == "__main__":
    main()
