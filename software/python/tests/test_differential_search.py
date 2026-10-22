import numpy as np

from openalterego.research.differential_hardware import (
    DifferentialProgram, gaussian_information_bits, measurement_covariances)
from openalterego.research.differential_search import (
    CovarianceScenario, DifferentialSearch)


def test_precomputed_scenarios_match_direct_covariance_scoring():
    pairs = ((0, 1), (1, 2), (2, 3), (0, 3))
    scenarios = [
        CovarianceScenario(np.eye(4, 2), np.eye(2),
                           np.array([1.0, value, 1.0, 1.0]),
                           np.ones(4))
        for value in (1.0, 4.0)]
    search = DifferentialSearch(contact_count=4, pairs=pairs,
                                scenarios=scenarios,
                                max_afe_channels=2, max_contacts=3)
    designs = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int16)
    objective, mean, minimum = search.score_batch(designs, batch_size=1)
    batched = search.score_batch(designs, batch_size=3)
    scenario_bits = search.score_scenarios(designs, batch_size=1)
    np.testing.assert_allclose(scenario_bits.mean(axis=0), mean)
    np.testing.assert_allclose(scenario_bits.min(axis=0), minimum)
    for left, right in zip((objective, mean, minimum), batched):
        np.testing.assert_allclose(left, right)
    direct = np.empty((len(scenarios), len(designs)))
    for s, scenario in enumerate(scenarios):
        for d, design in enumerate(designs):
            program = DifferentialProgram(4, tuple(pairs[index] for index in design),
                                          2, 3)
            signal, noise = measurement_covariances(
                program, scenario.lead_field, scenario.source_covariance,
                scenario.contact_noise_variance, scenario.afe_noise_variance[design])
            direct[s, d] = gaussian_information_bits(signal, noise)
    np.testing.assert_allclose(mean, direct.mean(axis=0))
    np.testing.assert_allclose(scenario_bits, direct)
    np.testing.assert_allclose(minimum, direct.min(axis=0))
    np.testing.assert_allclose(objective, direct.mean(axis=0) - direct.std(axis=0))
    selected = search.beam_search(width=2, beam_size=3)
    assert search.feasible(selected.channels)
    assert selected.contact_count <= 3
    refined = search.refine_swaps(selected.channels)
    assert search.feasible(refined.channels)
    assert refined.objective_bits >= selected.objective_bits - 1e-12
    with np.testing.assert_raises_regex(ValueError, "invalid local-search"):
        search.refine_swaps((0, 0))
    random = search.random_budget_search(width=2, evaluations=3)
    assert search.feasible(random.channels)
    assert random.evaluations == 3


def test_single_channel_exchange_escapes_a_prefix_beam_choice():
    rng = np.random.default_rng(11)
    pairs = tuple((2 * index, 2 * index + 1) for index in range(7))
    for _ in range(8):
        lead = rng.normal(size=(14, 3))
        contact_noise = rng.lognormal(0, 1, size=14)
        afe_noise = rng.lognormal(-1, 1, size=7)
    search = DifferentialSearch(
        contact_count=14, pairs=pairs,
        scenarios=[CovarianceScenario(lead, np.eye(3), contact_noise,
                                      afe_noise)],
        max_afe_channels=3, max_contacts=6)
    beam = search.beam_search(width=3, beam_size=1)
    refined = search.refine_swaps(beam.channels)
    assert beam.channels == (0, 2, 4)
    assert refined.channels == (2, 3, 4)
    assert refined.objective_bits > beam.objective_bits + 0.01
