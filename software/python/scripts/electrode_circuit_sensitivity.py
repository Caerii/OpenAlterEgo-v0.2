"""Illustrative impedance-mismatch sensitivity in contact-limited search.

The lead fields and impedance draws are hypothetical. This checks circuit
coupling, search feasibility and rank sensitivity, not facial design validity.
"""

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

from openalterego.research.differential_search import DifferentialSearch
from openalterego.research.electrode_circuit import ElectrodeCircuit
from scripts.differential_program_pilot import PAIRS, scenarios


FIXED = tuple(range(5, 11))  # Six adjacent cheek channels, seven contacts.


def add_circuits(rows, seed, input_ohm):
    rng = np.random.default_rng(seed)
    result = []
    for row in rows:
        circuit = ElectrodeCircuit(
            series_ohm=rng.lognormal(math.log(2000), 0.3, size=40),
            parallel_ohm=np.clip(rng.lognormal(
                math.log(200000), 0.9, size=40), 20000, 2e6),
            capacitance_f=rng.lognormal(math.log(2e-8), 0.4, size=40),
            input_resistance_ohm=input_ohm,
            input_capacitance_f=0.0,
            frequency_hz=100.0,
            common_mode_variance=100.0,
            cmrr_db=100.0)
        result.append(type(row)(row.lead_field, row.source_covariance,
                                row.contact_noise_variance,
                                row.afe_noise_variance, circuit))
    return result


def search(rows):
    return DifferentialSearch(contact_count=40, pairs=PAIRS,
                              scenarios=rows, max_afe_channels=6,
                              max_contacts=8, risk_weight=1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ohm", type=float, default=1e8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not 1e6 <= args.input_ohm <= 1e15:
        parser.error("new output and input resistance from 1 MOhm to 1 POhm required")
    development = scenarios(42, 12)
    evaluation = scenarios(420, 24)
    ideal = search(development)
    loaded = search(add_circuits(development, 43, args.input_ohm))
    started = time.perf_counter()
    ideal_choice = ideal.beam_search(width=6, beam_size=8)
    ideal_time = time.perf_counter() - started
    started = time.perf_counter()
    loaded_choice = loaded.beam_search(width=6, beam_size=8)
    loaded_time = time.perf_counter() - started
    held = search(add_circuits(evaluation, 421, args.input_ohm))
    programs = {"ideal_selected": ideal_choice.channels,
                "circuit_selected": loaded_choice.channels,
                "fixed": FIXED}
    scores = {}
    for name, channels in programs.items():
        if not held.feasible(channels):
            raise ValueError("selected program violates contact budget")
        objective, mean, minimum = held.score_batch(
            np.asarray([channels], dtype=np.int16))
        scores[name] = {"channels": channels,
                        "active_contacts": held.contact_mask(channels).bit_count(),
                        "held_circuit_mean_proxy": float(mean[0]),
                        "held_circuit_risk_proxy": float(objective[0]),
                        "held_circuit_minimum_proxy": float(minimum[0])}
    result = {
        "schema": "openalterego.electrode-circuit-sensitivity.v1",
        "scope": "Synthetic narrowband Gaussian information proxy, no fitted facial impedance, anatomy, or measured design effect",
        "seeds": {"lead_dev": 42, "lead_eval": 420,
                  "circuit_dev": 43, "circuit_eval": 421},
        "contact_count": 40, "recorded_channel_count": len(PAIRS),
        "max_afe_channels": 6, "max_contacts": 8,
        "development_scenarios": len(development),
        "held_scenarios": len(evaluation),
        "model": {"frequency_hz": 100.0,
                  "input_resistance_ohm": args.input_ohm,
                  "input_capacitance_f": 0.0,
                  "common_mode_variance_arbitrary_units": 100.0,
                  "cmrr_db": 100.0,
                  "series_ohm_lognormal_median": 2000,
                  "parallel_ohm_lognormal_median": 200000,
                  "parallel_ohm_clipped_range": [20000, 2e6],
                  "capacitance_f_lognormal_median": 2e-8,
                  "all_ranges_are_unmeasured_stress_inputs": True},
        "ideal_search_evaluations": ideal_choice.evaluations,
        "circuit_search_evaluations": loaded_choice.evaluations,
        "ideal_search_seconds": ideal_time,
        "circuit_search_seconds": loaded_time,
        "scores": scores,
        "machine": platform.platform(), "numpy": np.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"input_ohm": args.input_ohm,
                      "ideal": scores["ideal_selected"],
                      "circuit": scores["circuit_selected"],
                      "fixed": scores["fixed"]}, indent=2))


if __name__ == "__main__":
    main()
