"""Illustrative uncertain-physics search on two explicit differential chains."""

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

from openalterego.research.differential_hardware import (
    DifferentialProgram, gaussian_information_bits, measurement_covariances)
from openalterego.research.differential_search import (
    CovarianceScenario, DifferentialSearch)


PAIRS = tuple([(index, index + 1) for index in range(31)] +
              [(32 + index, 33 + index) for index in range(7)])
FIXED = (1, 2, 3, 15, 16, 17, 31, 32)


def scenarios(seed: int, count: int) -> list[CovarianceScenario]:
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(count):
        lead = np.zeros((40, 8), dtype=np.float64)
        for source, nominal in enumerate((3, 11, 20, 28, 33, 35, 37, 39)):
            region = np.arange(32) if source < 4 else np.arange(32, 40)
            center = nominal + rng.normal(0, 1 if source < 4 else 0.35)
            width = 3.0 if source < 4 else 1.0
            lead[region, source] = np.exp(-0.5 * ((region - center) / width) ** 2)
        lead *= rng.lognormal(0, 0.08, size=(40, 1))
        strengths = rng.lognormal(0, 0.2, size=8)
        source_cov = np.diag(strengths)
        contact_noise = rng.lognormal(math.log(0.12), 0.35, size=40)
        hotspot = rng.choice(40, size=2, replace=False)
        contact_noise[hotspot] *= rng.uniform(4, 12, size=2)
        afe_noise = rng.lognormal(math.log(0.05), 0.15, size=len(PAIRS))
        result.append(CovarianceScenario(lead, source_cov, contact_noise,
                                         afe_noise))
    return result


def direct_scores(designs: np.ndarray, examples: list[CovarianceScenario]):
    scores = np.empty((len(examples), len(designs)))
    for design_index, design in enumerate(designs):
        program = DifferentialProgram(40, tuple(PAIRS[index] for index in design),
                                      afe_channel_limit=8, contact_limit=14)
        for scenario_index, scenario in enumerate(examples):
            signal, noise = measurement_covariances(
                program, scenario.lead_field, scenario.source_covariance,
                scenario.contact_noise_variance,
                scenario.afe_noise_variance[design])
            scores[scenario_index, design_index] = gaussian_information_bits(signal, noise)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-scenarios", type=int, default=16)
    parser.add_argument("--evaluation-scenarios", type=int, default=64)
    parser.add_argument("--beam-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    development = scenarios(42, args.development_scenarios)
    evaluation = scenarios(420, args.evaluation_scenarios)
    search = DifferentialSearch(contact_count=40, pairs=PAIRS,
                                scenarios=development, max_afe_channels=8,
                                max_contacts=14, risk_weight=1.0)
    started = time.perf_counter()
    beam = search.beam_search(width=8, beam_size=args.beam_size)
    beam_seconds = time.perf_counter() - started
    started = time.perf_counter()
    random = search.random_budget_search(width=8, evaluations=beam.evaluations)
    random_seconds = time.perf_counter() - started
    held_out = DifferentialSearch(contact_count=40, pairs=PAIRS,
                                  scenarios=evaluation, max_afe_channels=8,
                                  max_contacts=14, risk_weight=1.0)
    choices = {"beam": beam.channels, "random_matched": random.channels,
               "fixed": FIXED}
    measured = {}
    for name, channels in choices.items():
        objective, mean, minimum = held_out.score_batch(np.asarray([channels]))
        measured[name] = {"channels": list(map(int, channels)),
                          "active_contacts": held_out.contact_mask(channels).bit_count(),
                          "evaluation_objective_bits_per_sample": float(objective[0]),
                          "evaluation_mean_bits_per_sample": float(mean[0]),
                          "evaluation_minimum_bits_per_sample": float(minimum[0])}
    rng = np.random.default_rng(991)
    samples = set()
    while len(samples) < 64:
        design = tuple(sorted(map(int, rng.choice(len(PAIRS), size=8, replace=False))))
        if search.feasible(design):
            samples.add(design)
    benchmark_designs = np.asarray(sorted(samples), dtype=np.int16)
    started = time.perf_counter()
    batched = search.score_batch(benchmark_designs)
    batched_seconds = time.perf_counter() - started
    started = time.perf_counter()
    direct = direct_scores(benchmark_designs, development)
    direct_seconds = time.perf_counter() - started
    error = float(np.max(np.abs(batched[1] - direct.mean(axis=0))))
    if error > 1e-10:
        raise ValueError(f"precomputed/direct mismatch: {error}")
    result = {"schema": "openalterego.differential-program-synthetic.v1",
              "scope": "Illustrative two-chain forward/noise model; not a calibrated CSL montage or measured design improvement",
              "seeds": {"development": 42, "evaluation": 420,
                        "random_search": 42, "score_benchmark": 991},
              "topology": {"contact_count": 40, "candidate_channels": len(PAIRS),
                           "max_afe_channels": 8, "max_active_contacts": 14,
                           "pair_map": PAIRS, "fixed_channels": FIXED},
              "scenario_model": "smooth region-local lead fields, lognormal source/AFE/contact variance, two noisy contacts per scenario",
              "development_scenarios": args.development_scenarios,
              "evaluation_scenarios": args.evaluation_scenarios,
              "beam_size": args.beam_size,
              "beam_evaluations": beam.evaluations,
              "beam_seconds": beam_seconds,
              "random_matched_seconds": random_seconds,
              "choices": measured,
              "score_benchmark": {"designs": len(benchmark_designs),
                                  "batched_seconds": batched_seconds,
                                  "direct_seconds": direct_seconds,
                                  "speedup": direct_seconds / batched_seconds,
                                  "max_abs_mean_bits_error": error},
              "machine": platform.platform(), "numpy": np.__version__}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"beam_evaluations": beam.evaluations,
                      "beam_seconds": beam_seconds,
                      "random_seconds": random_seconds,
                      "choices": measured,
                      "benchmark_speedup": direct_seconds / batched_seconds}, indent=2))


if __name__ == "__main__":
    main()
