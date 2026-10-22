import hashlib

import numpy as np
import pytest

from openalterego.research.differential_hardware import DifferentialProgram
from openalterego.research.electrode_circuit import (
    MeasuredElectrodeCircuit, circuit_transfer)
from openalterego.research.impedance_spectra import load_impedance_text


def test_decimal_comma_spectrum_hash_and_duplicate_frequency(tmp_path):
    payload = ("Frequency (Hz);Z (Ω);Z' (Ω);-Z'' (Ω)\n"
               "10;50;30;40\n"
               "100;50;30;40\n"
               "100;50;40;30\n"
               "1000;25;15;20\n").encode("utf-8")
    path = tmp_path / "measured.txt"
    path.write_bytes(payload)
    spectrum = load_impedance_text(path, hashlib.md5(payload).hexdigest())
    assert len(spectrum.frequency_hz) == 3
    assert spectrum.at(100) == 35 - 35j
    with pytest.raises(ValueError, match="outside measured sweep"):
        spectrum.at(10000)
    with pytest.raises(ValueError, match="publisher checksum"):
        load_impedance_text(path, "0" * 32)


def test_measured_contact_circuit_rejects_nonpassive_and_loads_contacts():
    program = DifferentialProgram(2, ((0, 1),), 1, 2)
    measured = MeasuredElectrodeCircuit(
        np.asarray([10000 - 20000j, 500000 - 600000j]),
        1e8, 0.0, 100.0, common_mode_variance=10)
    transfer, common = circuit_transfer(program, measured)
    assert transfer.shape == (1, 2)
    assert abs(common[0]) > 0
    bad = MeasuredElectrodeCircuit(np.asarray([-1 - 1j, 2 - 1j]),
                                   1e8, 0.0, 100)
    with pytest.raises(ValueError, match="invalid measured"):
        circuit_transfer(program, bad)
