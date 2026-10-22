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
    np.testing.assert_allclose(minimum, direct.min(axis=0))
    np.testing.assert_allclose(objective, direct.mean(axis=0) - direct.std(axis=0))
    selected = search.beam_search(width=2, beam_size=3)
    assert search.feasible(selected.channels)
    assert selected.contact_count <= 3
    random = search.random_budget_search(width=2, evaluations=3)
    assert search.feasible(random.channels)
    assert random.evaluations == 3
