"""Fixed-layout raw-window CNN versus pre-existing feature logistic control."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

from openalterego.research.raw_window_cnn import RawWindowCNN, normalize_window_batch
from scripts.silentwear_window_cache import cache_file


SPREAD = (0, 2, 5, 8, 11, 13)


def load_session(root: Path, cache_dir: Path, subject: str, session: int,
                 expected: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    windows, labels = [], []
    for batch in range(1, 6):
        source = root / subject / "silent" / f"sess_{session}_batch_{batch}.h5"
        path = cache_file(source, expected[source.name], cache_dir)
        with np.load(path, allow_pickle=False) as data:
            windows.append(normalize_window_batch(data["windows"], SPREAD))
            labels.append(np.asarray(data["labels"], dtype=np.int64))
    return np.concatenate(windows), np.concatenate(labels)


def fit_and_score(train: tuple[np.ndarray, np.ndarray],
                  test_loader, *, epochs: int, seed: int,
                  batch_size: int, threads: int) -> dict:
    x, y = train
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(threads)
    model = RawWindowCNN()
    weights = len(y) / (9 * np.bincount(y, minlength=9))
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,
                                                              dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  weight_decay=1e-2)
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    history = []
    for epoch in range(epochs):
        model.train()
        weighted_loss = 0.0
        for indices in np.array_split(rng.permutation(len(y)),
                                      max(1, (len(y) + batch_size - 1) // batch_size)):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_tensor[indices])
            loss = criterion(prediction, y_tensor[indices])
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.detach()) * len(indices)
        history.append({"epoch": epoch + 1, "train_loss": weighted_loss / len(y)})
    fit_seconds = time.perf_counter() - started
    test_x, test_y = test_loader()
    model.eval()
    with torch.inference_mode():
        predicted = np.concatenate([
            model(torch.from_numpy(chunk)).argmax(dim=1).numpy()
            for chunk in np.array_split(test_x,
                                        max(1, (len(test_x) + batch_size - 1) // batch_size))])
    return {"test_balanced_accuracy": float(balanced_accuracy_score(test_y,
                                                                      predicted)),
            "fit_wall_seconds": fit_seconds,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "history": history,
            "event_counts": {"train": len(y), "test": len(test_y)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=["S01", "S02", "S03"])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (not set(args.subjects).issubset({"S01", "S02", "S03"}) or
            args.epochs < 1 or args.batch_size < 1 or args.threads < 1):
        parser.error("invalid development subjects or training budget")
    config = {"subjects": args.subjects, "epochs": args.epochs,
              "seed": args.seed, "batch_size": args.batch_size,
              "threads": args.threads, "channels": list(SPREAD),
              "normalization": "unlabeled recording median absolute amplitude"}
    if args.output.exists():
        if not args.resume:
            parser.error("output exists; use --resume")
        result = json.loads(args.output.read_text(encoding="utf-8"))
        if result.get("schema") != "openalterego.raw-window-cnn.v1" or result.get("config") != config:
            parser.error("existing run does not match configuration")
    else:
        result = {"schema": "openalterego.raw-window-cnn.v1",
                  "config": config, "subjects": {},
                  "scope": "Retrospective, offline, fixed layout and event window; not streaming or fresh holdout."}
    for subject in args.subjects:
        if subject in result["subjects"]:
            print(subject, "already recorded", flush=True)
            continue
        manifest_bytes = (args.root / f"{subject}-silent-manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        expected = {Path(row["path"]).name: row["sha256"]
                    for row in manifest["files"]}
        if len(expected) != 15:
            raise ValueError(f"{subject}: expected 15 source files")
        train1 = load_session(args.root, args.cache_dir, subject, 1, expected)
        train2 = load_session(args.root, args.cache_dir, subject, 2, expected)
        train = (np.concatenate((train1[0], train2[0])),
                 np.concatenate((train1[1], train2[1])))
        row = fit_and_score(train,
                            lambda: load_session(args.root, args.cache_dir,
                                                 subject, 3, expected),
                            epochs=args.epochs, seed=args.seed,
                            batch_size=args.batch_size, threads=args.threads)
        row["input_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
        result["subjects"][subject] = row
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f"{args.output.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                             encoding="utf-8")
        os.replace(temporary, args.output)
        print(subject, row["test_balanced_accuracy"], row["fit_wall_seconds"], flush=True)


if __name__ == "__main__":
    main()
