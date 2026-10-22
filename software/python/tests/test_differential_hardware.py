import numpy as np
import pytest

from openalterego.research.differential_hardware import (
    DifferentialProgram, gaussian_information_bits, measurement_covariances,
    project_sources)


def test_contact_budget_and_differential_projection():
    program = DifferentialProgram(3, ((0, 1), (1, 2)),
                                  afe_channel_limit=2, contact_limit=3)
    assert program.used_contacts == (0, 1, 2)
    np.testing.assert_array_equal(program.incidence(), [[1, -1, 0], [0, 1, -1]])
    sources = np.array([[1.0], [2.0]])
    lead = np.array([[1.0], [0.0], [0.0]])
    baseline = project_sources(program, sources, lead)
    with_common = project_sources(program, sources, lead,
                                  common_mode=np.array([100.0, -50.0]))
    np.testing.assert_allclose(with_common, baseline)
    np.testing.assert_allclose(baseline[:, 0], [1, 2])
    with_contact_noise = project_sources(
        program, sources, lead,
        contact_artifact=np.array([[0, 3, 0], [0, 4, 0]]))
    np.testing.assert_allclose(with_contact_noise - baseline,
                               [[-3, 3], [-4, 4]])
    with pytest.raises(ValueError, match="contact budget"):
        DifferentialProgram(3, ((0, 1), (1, 2)), 2, 2)


def test_shared_contact_noise_changes_channel_covariance_and_information():
    program = DifferentialProgram(3, ((0, 1), (1, 2)), 2, 3)
    signal, noise = measurement_covariances(
        program, np.array([[1.0], [0.0], [0.0]]),
        np.array([[1.0]]), np.array([0.0, 4.0, 0.0]),
        np.array([1.0, 1.0]))
    np.testing.assert_allclose(noise, [[5, -4], [-4, 5]])
    np.testing.assert_allclose(signal, [[1, 0], [0, 0]])
    assert gaussian_information_bits(signal, noise) > 0
    assert gaussian_information_bits(np.zeros_like(signal), noise) == 0
