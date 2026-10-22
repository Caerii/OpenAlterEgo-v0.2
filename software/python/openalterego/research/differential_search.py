"""Fast program search over explicit differential contact-pair channels.

The covariance objective is a model-based proxy. It must earn predictive
validity against held-out physical design changes before engineering claims.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from openalterego.research.differential_hardware import (
    DifferentialProgram, measurement_covariances)
from openalterego.research.electrode_circuit import (
    ElectrodeCircuit, MeasuredElectrodeCircuit,
    circuit_measurement_covariances)


@dataclass(frozen=True)
class CovarianceScenario:
    lead_field: np.ndarray
    source_covariance: np.ndarray
    contact_noise_variance: np.ndarray
    afe_noise_variance: np.ndarray
    circuit: ElectrodeCircuit | MeasuredElectrodeCircuit | None = None


@dataclass(frozen=True)
class SearchResult:
    channels: tuple[int, ...]
    objective_bits: float
    mean_bits: float
    minimum_bits: float
    evaluations: int
    contact_count: int


class DifferentialSearch:
    def __init__(self, *, contact_count: int,
                 pairs: Sequence[tuple[int, int]],
                 scenarios: Sequence[CovarianceScenario],
                 max_afe_channels: int, max_contacts: int,
                 risk_weight: float = 1.0):
        self.pairs = tuple(pairs)
        self.contact_count = contact_count
        self.max_afe_channels = max_afe_channels
        self.max_contacts = max_contacts
        self.risk_weight = float(risk_weight)
        if (not scenarios or max_afe_channels < 1 or max_contacts < 2 or
                not np.isfinite(risk_weight) or risk_weight < 0):
            raise ValueError("invalid search budget or risk weight")
        full = DifferentialProgram(contact_count, self.pairs,
                                   afe_channel_limit=len(self.pairs),
                                   contact_limit=contact_count)
        self.contact_masks = tuple((1 << left) | (1 << right)
                                   for left, right in self.pairs)
        signals, noises = [], []
        for scenario in scenarios:
            inputs = (full, scenario.lead_field, scenario.source_covariance,
                      scenario.contact_noise_variance,
                      scenario.afe_noise_variance)
            signal, noise = (measurement_covariances(*inputs)
                             if scenario.circuit is None else
                             circuit_measurement_covariances(
                                 *inputs, scenario.circuit))
            signals.append(signal)
            noises.append(noise)
        self.signals = np.asarray(signals)
        self.noises = np.asarray(noises)

    def contact_mask(self, channels: Sequence[int]) -> int:
        mask = 0
        for channel in channels:
            mask |= self.contact_masks[channel]
        return mask

    def feasible(self, channels: Sequence[int]) -> bool:
        if (len(channels) == 0 or len(channels) > self.max_afe_channels or
                len(set(channels)) != len(channels) or
                any(channel < 0 or channel >= len(self.pairs) for channel in channels)):
            return False
        return self.contact_mask(channels).bit_count() <= self.max_contacts

    def score_batch(self, designs: np.ndarray, *,
                    batch_size: int = 2048) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return risk-adjusted, mean, minimum information in bits/sample."""
        designs = np.asarray(designs, dtype=np.int16)
        if (designs.ndim != 2 or not len(designs) or batch_size < 1 or
                any(not self.feasible(row) for row in designs)):
            raise ValueError("invalid or infeasible design batch")
        objectives, means, minima = [], [], []
        for start in range(0, len(designs), batch_size):
            block = designs[start:start + batch_size]
            signal = self.signals[:, block[:, :, None], block[:, None, :]]
            noise = self.noises[:, block[:, :, None], block[:, None, :]]
            sign_n, log_n = np.linalg.slogdet(noise)
            sign_t, log_t = np.linalg.slogdet(noise + signal)
            if (np.any(np.abs(sign_n - 1) > 1e-8) or
                    np.any(np.abs(sign_t - 1) > 1e-8)):
                raise ValueError("candidate covariance is not positive definite")
            bits = (log_t - log_n) / (2 * np.log(2))
            if not np.isfinite(bits).all():
                raise ValueError("candidate information score is not finite")
            mean = bits.mean(axis=0)
            means.append(mean)
            minima.append(bits.min(axis=0))
            objectives.append(mean - self.risk_weight * bits.std(axis=0))
        return np.concatenate(objectives), np.concatenate(means), np.concatenate(minima)

    def beam_search(self, *, width: int, beam_size: int = 32) -> SearchResult:
        if width < 1 or width > min(self.max_afe_channels, len(self.pairs)) or beam_size < 1:
            raise ValueError("invalid search width or beam size")
        beam = [tuple()]
        evaluations = 0
        for _ in range(width):
            candidates = set()
            for prefix in beam:
                for channel in range(len(self.pairs)):
                    if channel not in prefix:
                        design = tuple(sorted(prefix + (channel,)))
                        if self.feasible(design):
                            candidates.add(design)
            if not candidates:
                raise ValueError("no feasible full-width program")
            ordered = sorted(candidates)
            objective, _, _ = self.score_batch(np.asarray(ordered, dtype=np.int16))
            evaluations += len(ordered)
            ranked = sorted(range(len(ordered)),
                            key=lambda index: (-objective[index], ordered[index]))
            beam = [ordered[index] for index in ranked[:beam_size]]
        chosen = beam[0]
        objective, mean, minimum = self.score_batch(np.asarray([chosen]))
        return SearchResult(chosen, float(objective[0]), float(mean[0]),
                            float(minimum[0]), evaluations,
                            self.contact_mask(chosen).bit_count())

    def random_budget_search(self, *, width: int, evaluations: int,
                             seed: int = 42) -> SearchResult:
        if width < 1 or width > min(self.max_afe_channels, len(self.pairs)) or evaluations < 1:
            raise ValueError("invalid random-search budget")
        rng = np.random.default_rng(seed)
        candidates = set()
        attempts = 0
        while len(candidates) < evaluations:
            if attempts > evaluations * 10000:
                raise ValueError("cannot sample enough feasible programs")
            attempts += 1
            design = tuple(sorted(map(int, rng.choice(
                len(self.pairs), size=width, replace=False))))
            if self.feasible(design):
                candidates.add(design)
        ordered = sorted(candidates)
        objective, mean, minimum = self.score_batch(np.asarray(ordered, dtype=np.int16))
        chosen_index = min(range(len(ordered)),
                           key=lambda index: (-objective[index], ordered[index]))
        chosen = ordered[chosen_index]
        return SearchResult(chosen, float(objective[chosen_index]),
                            float(mean[chosen_index]),
                            float(minimum[chosen_index]), evaluations,
                            self.contact_mask(chosen).bit_count())
