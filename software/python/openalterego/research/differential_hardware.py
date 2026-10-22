"""Contact-level measurement programs for differential wearable EMG.

This linear measurement layer is an abstraction. Its lead field and noise
parameters require calibration before its design ranking is predictive.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DifferentialProgram:
    contact_count: int
    pairs: tuple[tuple[int, int], ...]
    afe_channel_limit: int
    contact_limit: int

    def __post_init__(self):
        if self.contact_count < 2 or not self.pairs:
            raise ValueError("a differential program needs contacts and pairs")
        if len(self.pairs) > self.afe_channel_limit:
            raise ValueError("AFE channel budget exceeded")
        if len(set(self.pairs)) != len(self.pairs):
            raise ValueError("duplicate differential pair")
        for left, right in self.pairs:
            if (not 0 <= left < self.contact_count or
                    not 0 <= right < self.contact_count or left == right):
                raise ValueError("invalid contact pair")
        if len(self.used_contacts) > self.contact_limit:
            raise ValueError("physical contact budget exceeded")

    @property
    def used_contacts(self) -> tuple[int, ...]:
        return tuple(sorted({contact for pair in self.pairs for contact in pair}))

    def incidence(self) -> np.ndarray:
        """Return B so recorded channels equal contact potentials times B.T."""
        matrix = np.zeros((len(self.pairs), self.contact_count), dtype=np.float64)
        for channel, (left, right) in enumerate(self.pairs):
            matrix[channel, left] = 1.0
            matrix[channel, right] = -1.0
        return matrix


def project_sources(program: DifferentialProgram, sources: np.ndarray,
                    lead_field: np.ndarray, *,
                    contact_artifact: np.ndarray | None = None,
                    afe_artifact: np.ndarray | None = None,
                    common_mode: np.ndarray | None = None) -> np.ndarray:
    """Project source time series through contacts, shared noise, and AFE.

    `lead_field` has one row per contact and one column per source. Contact
    artifacts are added before differential measurement; AFE artifacts after.
    A uniform common-mode term therefore cancels in ideal differential pairs.
    """
    sources = np.asarray(sources, dtype=np.float64)
    lead_field = np.asarray(lead_field, dtype=np.float64)
    if (sources.ndim != 2 or lead_field.ndim != 2 or
            lead_field.shape != (program.contact_count, sources.shape[1]) or
            not np.isfinite(sources).all() or not np.isfinite(lead_field).all()):
        raise ValueError("source or lead-field dimensions differ")
    potentials = sources @ lead_field.T
    if contact_artifact is not None:
        artifact = np.asarray(contact_artifact, dtype=np.float64)
        if artifact.shape != potentials.shape:
            raise ValueError("contact artifact shape differs")
        potentials = potentials + artifact
    if common_mode is not None:
        common = np.asarray(common_mode, dtype=np.float64)
        if common.shape != (len(sources),):
            raise ValueError("common-mode time series shape differs")
        potentials = potentials + common[:, None]
    observed = potentials @ program.incidence().T
    if afe_artifact is not None:
        artifact = np.asarray(afe_artifact, dtype=np.float64)
        if artifact.shape != observed.shape:
            raise ValueError("AFE artifact shape differs")
        observed = observed + artifact
    return observed


def measurement_covariances(program: DifferentialProgram,
                            lead_field: np.ndarray,
                            source_covariance: np.ndarray,
                            contact_noise_variance: np.ndarray,
                            afe_noise_variance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return channel signal and noise covariance for a linear Gaussian model."""
    g = np.asarray(lead_field, dtype=np.float64)
    source = np.asarray(source_covariance, dtype=np.float64)
    contact = np.asarray(contact_noise_variance, dtype=np.float64)
    afe = np.asarray(afe_noise_variance, dtype=np.float64)
    channels = len(program.pairs)
    if (g.ndim != 2 or g.shape[0] != program.contact_count or
            source.shape != (g.shape[1], g.shape[1]) or
            contact.shape != (program.contact_count,) or
            afe.shape != (channels,) or np.any(contact < 0) or np.any(afe <= 0) or
            not all(np.isfinite(value).all() for value in (g, source, contact, afe))):
        raise ValueError("invalid covariance model dimensions or variances")
    if not np.allclose(source, source.T) or np.linalg.eigvalsh(source).min() < -1e-10:
        raise ValueError("source covariance must be positive semidefinite")
    b = program.incidence()
    pickup = b @ g
    signal = pickup @ source @ pickup.T
    noise = (b * contact[None, :]) @ b.T + np.diag(afe)
    return signal, noise


def gaussian_information_bits(signal: np.ndarray, noise: np.ndarray) -> float:
    """Gaussian source information per independent sample under stated model."""
    signal = np.asarray(signal, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    if (signal.ndim != 2 or signal.shape != noise.shape or
            signal.shape[0] != signal.shape[1] or
            not np.allclose(signal, signal.T) or
            not np.allclose(noise, noise.T)):
        raise ValueError("expected symmetric channel covariances")
    if np.linalg.eigvalsh(signal).min() < -1e-10 or np.linalg.eigvalsh(noise).min() <= 0:
        raise ValueError("signal must be PSD and noise positive definite")
    sign_noise, log_noise = np.linalg.slogdet(noise)
    sign_total, log_total = np.linalg.slogdet(noise + signal)
    if sign_noise <= 0 or sign_total <= 0:
        raise ValueError("covariances must yield positive determinants")
    return float((log_total - log_noise) / (2 * np.log(2)))
