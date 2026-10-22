import pytest

from scripts.silentwear_additive_screen import _freeze


def test_selection_receipt_ignores_checkout_line_endings(tmp_path):
    path = tmp_path / "selection.json"
    record = {"subject": "S01", "selection": {"channels": [0, 2, 4]}}
    digest = _freeze(path, record)
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert _freeze(path, record) == digest
    with pytest.raises(ValueError, match="record differs"):
        _freeze(path, {**record, "subject": "S02"})
