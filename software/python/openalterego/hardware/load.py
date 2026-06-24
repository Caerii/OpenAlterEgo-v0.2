"""Load ``.oae.json`` hardware specs and built-in presets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

from .presets import get_preset_dict, load_preset, preset_names
from .schema import HardwareSpec

PathLike = Union[str, Path]


def list_presets() -> list[str]:
    return preset_names()


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: root must be a JSON object")
    return data


def _merge_extends(data: Dict[str, Any], *, _stack: tuple[str, ...] = ()) -> Dict[str, Any]:
    """Shallow-merge ``extends`` preset chain (child overrides parent)."""
    extends = data.get("extends")
    if not extends:
        return dict(data)
    parent_key = str(extends).strip()
    if parent_key in _stack:
        raise ValueError(f"circular extends: {' -> '.join(_stack + (parent_key,))}")
    if parent_key in preset_names():
        base = get_preset_dict(parent_key)
    else:
        base_path = Path(parent_key)
        if not base_path.is_file():
            raise FileNotFoundError(f"extends target not found: {extends!r}")
        base = _merge_extends(_read_json(base_path), _stack=_stack + (parent_key,))
    child = {k: v for k, v in data.items() if k != "extends"}
    merged = dict(base)
    for key, val in child.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **val}
        else:
            merged[key] = val
    return merged


def load_spec(source: PathLike) -> HardwareSpec:
    """Load from preset name, ``.oae.json`` path, or path stem matching a preset."""
    raw = str(source).strip()
    path = Path(raw)

    if raw in preset_names():
        return load_preset(raw)

    if path.suffix.lower() in (".json", ".oae"):
        data = _merge_extends(_read_json(path))
        return HardwareSpec.model_validate(data)

    if path.suffix == "" and raw in preset_names():
        return load_preset(raw)

    # Try hardware/specs relative names from repo checkout
    candidates = [
        path,
        Path("hardware/specs") / f"{raw}.oae.json",
        Path("hardware/specs") / f"{raw}.json",
    ]
    for c in candidates:
        if c.is_file():
            data = _merge_extends(_read_json(c))
            return HardwareSpec.model_validate(data)

    raise FileNotFoundError(
        f"hardware spec not found: {source!r} (preset name or .oae.json / .json path)"
    )


def export_spec_json(spec: HardwareSpec, path: PathLike) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec.model_dump_public(), indent=2) + "\n", encoding="utf-8")
