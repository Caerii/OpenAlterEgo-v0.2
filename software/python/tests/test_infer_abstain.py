"""Inference abstention from softmax shape."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from openalterego.ml.infer import (
    LoadedModel,
    _softmax_entropy_norm,
    _top2_margin,
    predict_preprocessed_with_abstain,
)
from openalterego.ml.model import OpenAlterEgoCNN


class TestAbstainHelpers(unittest.TestCase):
    def test_entropy_uniform(self) -> None:
        u = np.ones(4, dtype=np.float64) / 4.0
        self.assertAlmostEqual(_softmax_entropy_norm(u), 1.0, places=5)

    def test_entropy_peak(self) -> None:
        p = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.assertLess(_softmax_entropy_norm(p), 0.05)

    def test_margin(self) -> None:
        p = np.array([0.7, 0.2, 0.1], dtype=np.float64)
        self.assertAlmostEqual(_top2_margin(p), 0.5)


class TestPredictWithAbstain(unittest.TestCase):
    def setUp(self) -> None:
        labels = ["a", "b", "c"]
        model = OpenAlterEgoCNN(channels=2, classes=3)
        self.lm = LoadedModel(
            model=model,
            labels=labels,
            fs=100,
            channels=2,
            device=torch.device("cpu"),
            preprocess_mode="streaming",
            emg_mode="standard",
        )

    def test_abstain_on_uniform_probs(self) -> None:
        x = np.zeros((16, 2), dtype=np.float32)
        with mock.patch.object(self.lm.model, "forward", return_value=torch.zeros(1, 3)):
            tok, conf, abst = predict_preprocessed_with_abstain(
                self.lm,
                x,
                abstain_entropy_norm_max=0.95,
            )
            self.assertTrue(abst)
            self.assertIn(tok, self.lm.labels)

    def test_no_abstain_when_peaked(self) -> None:
        x = np.zeros((16, 2), dtype=np.float32)
        with mock.patch.object(
            self.lm.model,
            "forward",
            return_value=torch.tensor([[100.0, 0.0, 0.0]]),
        ):
            _, _, abst = predict_preprocessed_with_abstain(
                self.lm,
                x,
                abstain_entropy_norm_max=0.99,
                abstain_min_margin=0.5,
            )
            self.assertFalse(abst)

    def test_margin_abstain(self) -> None:
        x = np.zeros((16, 2), dtype=np.float32)
        with mock.patch.object(
            self.lm.model,
            "forward",
            return_value=torch.tensor([[0.0, 0.0, 0.0]]),
        ):
            _, _, abst = predict_preprocessed_with_abstain(
                self.lm,
                x,
                abstain_entropy_norm_max=None,
                abstain_min_margin=0.4,
            )
            self.assertTrue(abst)


if __name__ == "__main__":
    unittest.main()
