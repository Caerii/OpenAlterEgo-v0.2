"""Hash-checked import of open electrode-skin impedance spectra.

The Zenodo TextrodeMisc TXT files use semicolons and decimal commas. They are
forearm three-electrode EIS measurements; interpolation does not transform
them into a facial contact or a complete two-electrode EMG circuit.
"""

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ImpedanceSpectrum:
    frequency_hz: np.ndarray
    impedance_ohm: np.ndarray
    source_md5: str

    def at(self, frequency_hz: float) -> complex:
        if (not np.isfinite(frequency_hz) or frequency_hz <= 0 or
                frequency_hz < self.frequency_hz.min() or
                frequency_hz > self.frequency_hz.max()):
            raise ValueError("requested frequency is outside measured sweep")
        log_f = np.log(self.frequency_hz)
        target = np.log(frequency_hz)
        return complex(np.interp(target, log_f, self.impedance_ohm.real),
                       np.interp(target, log_f, self.impedance_ohm.imag))


def load_impedance_text(path: Path, expected_md5: str) -> ImpedanceSpectrum:
    payload = path.read_bytes()
    digest = hashlib.md5(payload).hexdigest()
    if digest != expected_md5.lower():
        raise ValueError("impedance trace differs from publisher checksum")
    reader = csv.DictReader(payload.decode("utf-8-sig").splitlines(),
                            delimiter=";")
    if (not reader.fieldnames or not {"Frequency (Hz)", "Z' (Ω)",
                                    "-Z'' (Ω)", "Z (Ω)"}.issubset(reader.fieldnames)):
        raise ValueError("unrecognized impedance text columns")
    rows = list(reader)
    try:
        frequency = np.asarray([float(row["Frequency (Hz)"].replace(",", "."))
                                for row in rows])
        real = np.asarray([float(row["Z' (Ω)"].replace(",", "."))
                           for row in rows])
        negative_imag = np.asarray([
            float(row["-Z'' (Ω)"].replace(",", ".")) for row in rows])
        magnitude = np.asarray([float(row["Z (Ω)"].replace(",", "."))
                                for row in rows])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("unreadable impedance text values") from exc
    impedance = real - 1j * negative_imag
    if (len(frequency) < 3 or not np.isfinite(frequency).all() or
            not np.isfinite(impedance).all() or
            np.any(frequency <= 0) or
            not np.allclose(abs(impedance), magnitude, rtol=1e-4,
                            atol=1e-3)):
        raise ValueError("invalid frequency sweep or complex impedance")
    unique, inverse = np.unique(frequency, return_inverse=True)
    # Some publisher traces contain two measurements at 100 kHz. Retain the
    # information by averaging only those identical-frequency observations.
    summed = np.zeros(len(unique), dtype=np.complex128)
    count = np.zeros(len(unique), dtype=np.int32)
    np.add.at(summed, inverse, impedance)
    np.add.at(count, inverse, 1)
    return ImpedanceSpectrum(unique, summed / count, digest)
