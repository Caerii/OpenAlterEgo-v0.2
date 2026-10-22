"""Replay forearm dry-electrode spectra as external circuit stress inputs.

Only impedance is measured here. Lead fields, contact assignment, common-mode
interference, and the putative 40-contact face array remain hypothetical.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from openalterego.research.differential_search import (
    CovarianceScenario, DifferentialSearch)
from openalterego.research.electrode_circuit import MeasuredElectrodeCircuit
from openalterego.research.impedance_spectra import load_impedance_text
from scripts.differential_program_pilot import PAIRS, scenarios
from scripts.electrode_circuit_sensitivity import FIXED


def load_pool(manifest_path: Path, data_dir: Path, group: str):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != "openalterego.external-impedance-source.v1" or
            manifest.get("record") != "https://zenodo.org/records/17691766" or
            manifest.get("license") != "CC BY 4.0"):
        raise ValueError("unexpected measured-impedance source manifest")
    names = sorted(manifest["files"])
    selected = ([name for name in names if name.lower().startswith(
        ("mxene ecoflex 5%", "mxene ecoflex 7%"))]
        if group == "mxene_5_7" else
        [name for name in names if name.startswith("TA -")])
    if len(selected) < 2:
        raise ValueError("material group has too few independent traces")
    pool = []
    for name in selected:
        checksum = manifest["files"][name]
        if not checksum.startswith("md5:"):
            raise ValueError("unexpected checksum type")
        spectrum = load_impedance_text(data_dir / name, checksum[4:])
        impedance = spectrum.at(100.0)
        if impedance.real < 0 or impedance.imag > 0:
            raise ValueError(f"non-passive 100 Hz spectrum in group: {name}")
        pool.append(impedance)
    return selected, np.asarray(pool)


def add_measured(rows, pool, seed, input_ohm):
    rng = np.random.default_rng(seed)
    result = []
    for row in rows:
        assigned = pool[rng.integers(len(pool), size=40)]
        circuit = MeasuredElectrodeCircuit(
            impedance_ohm=assigned,
            input_resistance_ohm=input_ohm,
            input_capacitance_f=0.0, frequency_hz=100.0,
            common_mode_variance=100.0, cmrr_db=100.0)
        result.append(CovarianceScenario(
            row.lead_field, row.source_covariance,
            row.contact_noise_variance, row.afe_noise_variance, circuit))
    return result


def search(rows):
    return DifferentialSearch(contact_count=40, pairs=PAIRS,
                              scenarios=rows, max_afe_channels=6,
                              max_contacts=8, risk_weight=1.0)


def paired_scenario_comparison(held, baseline, candidate, *,
                               seed=991, draws=5000):
    bits = held.score_scenarios(np.asarray([baseline, candidate],
                                                dtype=np.int16))
    differences = bits[:, 1] - bits[:, 0]
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(differences), size=(draws, len(differences)))
    interval = np.quantile(differences[sampled].mean(axis=1),
                           [0.025, 0.975])
    return {"mean_difference": float(differences.mean()),
            "paired_bootstrap_ci95": [float(value) for value in interval],
            "candidate_wins": int(np.sum(differences > 1e-12)),
            "baseline_wins": int(np.sum(differences < -1e-12)),
            "ties": int(np.sum(abs(differences) <= 1e-12)),
            "unit": "synthetic evaluation scenario",
            "draws": draws, "seed": seed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--group", choices=("mxene_5_7", "ta_spandex"),
                        required=True)
    parser.add_argument("--input-ohm", type=float, default=1e8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not 1e6 <= args.input_ohm <= 1e15:
        parser.error("new output and supported input resistance required")
    names, pool = load_pool(args.manifest, args.data_dir, args.group)
    development = scenarios(42, 12)
    evaluation = scenarios(420, 24)
    ideal = search(development)
    empirical = search(add_measured(development, pool, 43, args.input_ohm))
    ideal_choice = ideal.beam_search(width=6, beam_size=8)
    empirical_choice = empirical.beam_search(width=6, beam_size=8)
    refined_choice = empirical.refine_swaps(empirical_choice.channels)
    held = search(add_measured(evaluation, pool, 421, args.input_ohm))
    choices = {"ideal_selected": ideal_choice.channels,
               "measured_circuit_selected": empirical_choice.channels,
               "measured_circuit_refined": refined_choice.channels,
               "fixed": FIXED}
    scores = {}
    for name, channels in choices.items():
        objective, mean, minimum = held.score_batch(
            np.asarray([channels], dtype=np.int16))
        scores[name] = {"channels": channels,
                        "active_contacts": held.contact_mask(channels).bit_count(),
                        "held_proxy_mean": float(mean[0]),
                        "held_proxy_risk": float(objective[0]),
                        "held_proxy_minimum": float(minimum[0])}
    paired = paired_scenario_comparison(
        held, ideal_choice.channels, empirical_choice.channels)
    paired_refinement = paired_scenario_comparison(
        held, empirical_choice.channels, refined_choice.channels,
        seed=992)
    result = {
        "schema": "openalterego.measured-impedance-sensitivity.v1",
        "scope": "Publisher-hash-checked forearm dry-electrode impedances at 100 Hz; synthetic anatomy, common-mode input and independent contact assignments; no facial design validation",
        "source_record": "https://zenodo.org/records/17691766",
        "source_license": "CC BY 4.0",
        "material_group": args.group, "source_files": names,
        "source_md5": {name: json.loads(args.manifest.read_text())[
            "files"][name] for name in names},
        "measured_100hz_impedance_magnitude_ohm": [
            float(value) for value in abs(pool)],
        "measured_100hz_impedance_real_ohm": [
            float(value) for value in pool.real],
        "measured_100hz_impedance_imag_ohm": [
            float(value) for value in pool.imag],
        "input_resistance_ohm": args.input_ohm,
        "input_capacitance_f": 0.0, "frequency_hz": 100.0,
        "assumed_cmrr_db": 100.0,
        "arbitrary_common_mode_variance": 100.0,
        "development_scenarios": 12, "evaluation_scenarios": 24,
        "seeds": {"lead_dev": 42, "lead_eval": 420,
                  "contact_assignment_dev": 43,
                  "contact_assignment_eval": 421},
        "ideal_evaluations": ideal_choice.evaluations,
        "measured_circuit_evaluations": empirical_choice.evaluations,
        "swap_refinement_evaluations": refined_choice.evaluations,
        "scores": scores,
        "paired_measured_minus_ideal": paired,
        "paired_refined_minus_measured": paired_refinement,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"group": args.group, "input_ohm": args.input_ohm,
                      "impedance_range": [float(abs(pool).min()),
                                          float(abs(pool).max())],
                      "scores": scores}, indent=2))


if __name__ == "__main__":
    main()
