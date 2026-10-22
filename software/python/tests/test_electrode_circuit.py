import numpy as np
import pytest

from openalterego.research.differential_hardware import (
    DifferentialProgram, measurement_covariances)
from openalterego.research.differential_search import (
    CovarianceScenario, DifferentialSearch)
from openalterego.research.electrode_circuit import (
    ElectrodeCircuit, circuit_measurement_covariances, circuit_transfer)


def circuit(parallel, *, input_ohm=1e8, frequency=100.0,
            common_variance=0.0):
    return ElectrodeCircuit(
        series_ohm=np.full(len(parallel), 1000.0),
        parallel_ohm=np.asarray(parallel, dtype=float),
        capacitance_f=np.full(len(parallel), 1e-8),
        input_resistance_ohm=input_ohm, input_capacitance_f=0.0,
        frequency_hz=frequency, common_mode_variance=common_variance)


def test_contact_mismatch_converts_common_mode_and_high_input_reduces_it():
    program = DifferentialProgram(2, ((0, 1),), 1, 2)
    matched = circuit([10000, 10000])
    _, common_matched = circuit_transfer(program, matched)
    assert abs(common_matched[0]) < 1e-14
    mismatched = circuit([10000, 500000])
    _, common_mismatched = circuit_transfer(program, mismatched)
    _, common_high_input = circuit_transfer(
        program, circuit([10000, 500000], input_ohm=1e12))
    assert abs(common_mismatched[0]) > 0
    assert abs(common_high_input[0]) < abs(common_mismatched[0]) / 1000
    assert abs(circuit([500000], frequency=1000).contact_transfer(1)[0]) > abs(
        circuit([500000], frequency=0).contact_transfer(1)[0])
    finite_cmrr = ElectrodeCircuit(
        matched.series_ohm, matched.parallel_ohm, matched.capacitance_f,
        matched.input_resistance_ohm, matched.input_capacitance_f,
        matched.frequency_hz, cmrr_db=80.0)
    assert abs(circuit_transfer(program, finite_cmrr)[1][0]) > 0


def test_circuit_covariance_is_hermitian_and_search_scores_it():
    pairs = ((0, 1), (1, 2), (2, 3))
    model = circuit([10000, 500000, 20000, 80000],
                    common_variance=100.0)
    lead = np.asarray([[1, 0], [0.8, 0.1], [0.1, 0.8], [0, 1]])
    scenario = CovarianceScenario(
        lead, np.eye(2), np.full(4, 0.1), np.full(3, 0.01),
        circuit=model)
    program = DifferentialProgram(4, pairs, 3, 4)
    signal, noise = circuit_measurement_covariances(
        program, scenario.lead_field, scenario.source_covariance,
        scenario.contact_noise_variance, scenario.afe_noise_variance,
        model)
    np.testing.assert_allclose(signal, signal.conj().T)
    np.testing.assert_allclose(noise, noise.conj().T)
    assert np.linalg.eigvalsh(noise).min() > 0
    search = DifferentialSearch(contact_count=4, pairs=pairs,
                                scenarios=[scenario], max_afe_channels=2,
                                max_contacts=3)
    objective, mean, minimum = search.score_batch(
        np.asarray([[0, 1]], dtype=np.int16))
    chosen = np.ix_([0, 1], [0, 1])
    expected = (np.linalg.slogdet((signal + noise)[chosen])[1] -
                np.linalg.slogdet(noise[chosen])[1]) / (2 * np.log(2))
    np.testing.assert_allclose([objective[0], mean[0], minimum[0]], expected)
    near_ideal = circuit([10000] * 4, input_ohm=1e15)
    loaded_signal, loaded_noise = circuit_measurement_covariances(
        program, scenario.lead_field, scenario.source_covariance,
        scenario.contact_noise_variance, scenario.afe_noise_variance,
        near_ideal)
    ideal_signal, ideal_noise = measurement_covariances(
        program, scenario.lead_field, scenario.source_covariance,
        scenario.contact_noise_variance, scenario.afe_noise_variance)
    np.testing.assert_allclose(loaded_signal, ideal_signal, atol=1e-8)
    np.testing.assert_allclose(loaded_noise, ideal_noise, atol=1e-8)


def test_circuit_rejects_negative_impedance_and_unknown_contact_count():
    bad = circuit([10000, -1])
    with pytest.raises(ValueError, match="invalid electrode"):
        bad.contact_transfer(2)
    good = circuit([10000, 10000])
    with pytest.raises(ValueError, match="invalid electrode"):
        good.contact_transfer(3)
