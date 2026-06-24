"""Tests for SE-ResNet model and external dataset importers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from openalterego.ml.datasets.gaddy import convert_gaddy_raw_dir
from openalterego.ml.datasets.gowda import (
    GOWDA_TRIAL_SAMPLES,
    GOWDA_WORD_SLICE_SAMPLES,
    import_gowda_small_vocab,
    write_dataset_catalog,
)
from openalterego.ml.infer import load_model
from openalterego.ml.model import OpenAlterEgoCNN, OpenAlterEgoSEResNet, create_model


class TestSEResNet(unittest.TestCase):
    def test_forward_shapes(self) -> None:
        for arch, cls in (("cnn", OpenAlterEgoCNN), ("se_resnet", OpenAlterEgoSEResNet)):
            m = create_model(arch, channels=8, classes=4)
            self.assertIsInstance(m, cls)
            x = torch.randn(2, 8, 600)
            logits = m(x)
            self.assertEqual(tuple(logits.shape), (2, 4))

    def test_checkpoint_roundtrip(self) -> None:
        labels = ["a", "b", "c"]
        m = create_model("se_resnet", channels=4, classes=len(labels))
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.pt"
            torch.save(
                {
                    "state_dict": m.state_dict(),
                    "labels": labels,
                    "fs": 250,
                    "channels": 4,
                    "preprocess_mode": "streaming",
                    "emg_mode": "wide",
                    "arch": "se_resnet",
                },
                p,
            )
            lm = load_model(p)
            self.assertEqual(lm.arch, "se_resnet")
            x = torch.randn(1, 4, 150)
            with torch.no_grad():
                out = lm.model(x.to(lm.device))
            self.assertEqual(out.shape[-1], 3)

    def test_legacy_cnn_checkpoint(self) -> None:
        m = OpenAlterEgoCNN(channels=2, classes=2)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "legacy.pt"
            torch.save(
                {"state_dict": m.state_dict(), "labels": ["x", "y"], "fs": 250, "channels": 2},
                p,
            )
            lm = load_model(p)
            self.assertEqual(lm.arch, "cnn")


class TestGaddyImport(unittest.TestCase):
    def test_convert_raw_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "raw"
            raw.mkdir()
            emg = np.random.randn(800, 8).astype(np.float32)
            np.save(raw / "0_emg.npy", emg)
            (raw / "0_info.json").write_text(
                json.dumps({"text": "hello world", "sentence_index": 0, "silent": True}),
                encoding="utf-8",
            )
            np.save(raw / "1_emg.npy", emg)
            (raw / "1_info.json").write_text(
                json.dumps({"text": "reference", "sentence_index": -1}),
                encoding="utf-8",
            )
            out = Path(td) / "sess"
            rep = convert_gaddy_raw_dir(raw, out, label_mode="first_word")
            self.assertEqual(rep.n_events, 1)
            self.assertTrue((out / "signals.npy").is_file())
            ev = pd.read_csv(out / "events.csv")
            self.assertEqual(str(ev.iloc[0]["label"]), "hello")


class TestGowdaImport(unittest.TestCase):
    def test_small_vocab_object_array(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            data_path = td_path / "data.npy"
            labels_path = td_path / "labels.npy"
            segs = [np.random.randn(500, 31).astype(np.float32) for _ in range(3)]
            np.save(data_path, np.array(segs, dtype=object))
            np.save(labels_path, np.array(["monday", "tuesday", "january"], dtype=object))
            out = td_path / "sess"
            rep = import_gowda_small_vocab(data_path, labels_path, out, fs_hz=5000.0)
            self.assertEqual(rep.n_events, 3)
            self.assertEqual(rep.channels, 31)

    def test_trial_cube_format(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            data = np.random.randn(4, 8, 400).astype(np.float64)
            labels = np.array(
                [["monday", "march", "sixth", "nineteen"], ["tuesday", "april", "first", "twenty"]] * 2
            )
            data_path = td_path / "data.npy"
            labels_path = td_path / "labels.npy"
            np.save(data_path, data)
            np.save(labels_path, labels)
            out = td_path / "sess"
            rep = import_gowda_small_vocab(data_path, labels_path, out, fs_hz=5000.0, max_segments=2)
            self.assertGreater(rep.n_events, 0)
            self.assertEqual(rep.channels, 8)
            meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(float(meta["fs_hz"]), 5000.0)

    def test_trial_cube_event_indices_match_vstack(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            rng = np.random.default_rng(0)
            data = rng.standard_normal((2, 31, GOWDA_TRIAL_SAMPLES)).astype(np.float32)
            labels = np.array(
                [
                    ["monday", "march", "sixth", "nineteen"],
                    ["tuesday", "april", "first", "twenty"],
                ]
            )
            data_path = td_path / "data.npy"
            labels_path = td_path / "labels.npy"
            np.save(data_path, data)
            np.save(labels_path, labels)
            out = td_path / "sess"
            import_gowda_small_vocab(data_path, labels_path, out, fs_hz=5000.0)
            sig = np.load(out / "signals.npy")
            ev = pd.read_csv(out / "events.csv")
            self.assertEqual(len(ev), 8)
            pos = 0
            expected_lens = list(GOWDA_WORD_SLICE_SAMPLES) * 2
            for i, row in ev.iterrows():
                s, e = int(row["start_sample"]), int(row["end_sample"])
                self.assertEqual(s, pos)
                self.assertEqual(e - s, expected_lens[i])
                np.testing.assert_array_equal(sig[s:e], sig[pos : pos + expected_lens[i]])
                pos = e
            self.assertEqual(pos, int(sig.shape[0]))


class TestDatasetCatalog(unittest.TestCase):
    def test_catalog_written(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = write_dataset_catalog(Path(td) / "catalog.json")
            data = json.loads(p.read_text(encoding="utf-8"))
            ids = {d["id"] for d in data["datasets"]}
            self.assertIn("gaddy_silent_speech", ids)
            self.assertIn("gowda_geometry", ids)


if __name__ == "__main__":
    unittest.main()
