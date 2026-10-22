"""Frequency-domain electrode loading and common-mode leakage model.

This is a linear small-signal equivalent circuit, not an identified model of
facial skin, electrode polarization, motion, or a manufactured front end.
Impedance spectroscopy and circuit measurements are needed to set parameters.
"""

from dataclasses import dataclass

import numpy as np

from openalterego.research.differential_hardware import (
    DifferentialProgram, measurement_covariances)


@dataclass(frozen=True)
class ElectrodeCircuit:
    """One series resistor plus a parallel resistor/capacitor per contact."""

    series_ohm: np.ndarray
    parallel_ohm: np.ndarray
    capacitance_f: np.ndarray
    input_resistance_ohm: float
    input_capacitance_f: float
    frequency_hz: float
    common_mode_variance: float = 0.0
    cmrr_db: float | None = None

    def contact_transfer(self, contact_count: int) -> np.ndarray:
        """Loaded contact potential / unloaded contact potential."""
        series = np.asarray(self.series_ohm, dtype=np.float64)
        parallel = np.asarray(self.parallel_ohm, dtype=np.float64)
        capacitance = np.asarray(self.capacitance_f, dtype=np.float64)
        if (any(value.shape != (contact_count,) for value in
                (series, parallel, capacitance)) or
                any(not np.isfinite(value).all() for value in
                    (series, parallel, capacitance)) or
                np.any(series < 0) or np.any(parallel <= 0) or
                np.any(capacitance < 0) or
                not np.isfinite(self.input_resistance_ohm) or
                self.input_resistance_ohm <= 0 or
                not np.isfinite(self.input_capacitance_f) or
                self.input_capacitance_f < 0 or
                not np.isfinite(self.frequency_hz) or self.frequency_hz < 0 or
                not np.isfinite(self.common_mode_variance) or
                self.common_mode_variance < 0 or
                (self.cmrr_db is not None and (
                    not np.isfinite(self.cmrr_db) or self.cmrr_db < 0))):
            raise ValueError("invalid electrode circuit parameters")
        omega = 2 * np.pi * self.frequency_hz
        electrode = series + 1 / (1 / parallel + 1j * omega * capacitance)
        input_admittance = (1 / self.input_resistance_ohm +
                            1j * omega * self.input_capacitance_f)
        return 1 / (1 + electrode * input_admittance)


@dataclass(frozen=True)
class MeasuredElectrodeCircuit:
    """Measured complex contact impedance at one stated frequency."""

    impedance_ohm: np.ndarray
    input_resistance_ohm: float
    input_capacitance_f: float
    frequency_hz: float
    common_mode_variance: float = 0.0
    cmrr_db: float | None = None

    def contact_transfer(self, contact_count: int) -> np.ndarray:
        impedance = np.asarray(self.impedance_ohm, dtype=np.complex128)
        if (impedance.shape != (contact_count,) or
                not np.isfinite(impedance).all() or
                np.any(impedance.real < 0) or
                np.any(impedance.imag > 1e-9) or
                not np.isfinite(self.input_resistance_ohm) or
                self.input_resistance_ohm <= 0 or
                not np.isfinite(self.input_capacitance_f) or
                self.input_capacitance_f < 0 or
                not np.isfinite(self.frequency_hz) or self.frequency_hz < 0 or
                not np.isfinite(self.common_mode_variance) or
                self.common_mode_variance < 0 or
                (self.cmrr_db is not None and (
                    not np.isfinite(self.cmrr_db) or self.cmrr_db < 0))):
            raise ValueError("invalid measured electrode circuit")
        input_admittance = (1 / self.input_resistance_ohm +
                            1j * 2 * np.pi * self.frequency_hz *
                            self.input_capacitance_f)
        return 1 / (1 + impedance * input_admittance)


def circuit_transfer(program: DifferentialProgram,
                     circuit: ElectrodeCircuit | MeasuredElectrodeCircuit
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Return contact-to-channel transfer and common-mode gain."""
    loaded = circuit.contact_transfer(program.contact_count)
    transfer = program.incidence() * loaded[None, :]
    common_gain = transfer.sum(axis=1)
    if circuit.cmrr_db is not None:
        residual = 10 ** (-circuit.cmrr_db / 20)
        average_loaded = np.asarray([
            (loaded[left] + loaded[right]) / 2
            for left, right in program.pairs])
        common_gain = common_gain + residual * average_loaded
    return transfer, common_gain


def circuit_measurement_covariances(
    program: DifferentialProgram, lead_field: np.ndarray,
    source_covariance: np.ndarray, contact_noise_variance: np.ndarray,
    afe_noise_variance: np.ndarray,
    circuit: ElectrodeCircuit | MeasuredElectrodeCircuit,
) -> tuple[np.ndarray, np.ndarray]:
    """Narrowband Hermitian signal/noise covariance under stated assumptions.

    Contact artifacts are independent voltage sources upstream of loading.
    Common-mode interference is independent of the physiological sources.
    These covariances are spectral proxies at one stated frequency, not
    broadband sample variances or a calibrated EMG physics prediction.
    """
    # Reuse the existing dimensional and positive-semidefinite input checks.
    measurement_covariances(program, lead_field, source_covariance,
                            contact_noise_variance, afe_noise_variance)
    transfer, common_gain = circuit_transfer(program, circuit)
    pickup = transfer @ np.asarray(lead_field, dtype=np.float64)
    source = np.asarray(source_covariance, dtype=np.float64)
    signal = pickup @ source @ pickup.conj().T
    contact_noise = np.asarray(contact_noise_variance, dtype=np.float64)
    afe_noise = np.asarray(afe_noise_variance, dtype=np.float64)
    noise = ((transfer * contact_noise[None, :]) @ transfer.conj().T +
             np.diag(afe_noise) + circuit.common_mode_variance *
             np.outer(common_gain, common_gain.conj()))
    return signal, noise
