"""Word-to-phone tables and I/O (synthetic EMG only, not a full G2P engine)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

DEFAULT_PHONE_LEXICON: Dict[str, Tuple[str, ...]] = {
    "yes": ("Y", "EH", "S"),
    "no": ("N", "OW"),
    "left": ("L", "EH", "F", "T"),
    "right": ("R", "AY", "T"),
    "select": ("S", "AH", "L", "EH", "K", "T"),
    "cancel": ("K", "AE", "N", "S", "AH", "L"),
    "up": ("AH", "P"),
    "down": ("D", "AW", "N"),
    "help": ("HH", "EH", "L", "P"),
    "start": ("S", "T", "AA", "R", "T"),
    "stop": ("S", "T", "AA", "P"),
    "ok": ("OW", "K", "EY"),
    "enter": ("EH", "N", "T", "ER"),
    "back": ("B", "AE", "K"),
    "menu": ("M", "EH", "N", "Y", "UW"),
}

_PHONE_TOKEN = re.compile(r"^[A-Za-z@][A-Za-z0-9@\-]*$")


def merge_lexicon(user: Optional[Mapping[str, Sequence[str]]]) -> Dict[str, Tuple[str, ...]]:
    """Merge user entries over DEFAULT_PHONE_LEXICON (keys lowercased)."""
    out: Dict[str, Tuple[str, ...]] = {k.lower(): tuple(v) for k, v in DEFAULT_PHONE_LEXICON.items()}
    if user:
        for k, v in user.items():
            key = str(k).lower().strip()
            out[key] = tuple(str(p).strip().upper() for p in v if str(p).strip())
    return out


def validate_lexicon(lexicon: Mapping[str, Tuple[str, ...]]) -> List[str]:
    """Return human-readable issues; empty means OK."""
    issues: List[str] = []
    for word, phones in lexicon.items():
        if not phones:
            issues.append(f"empty phone list for {word!r}")
            continue
        for p in phones:
            if not _PHONE_TOKEN.match(p):
                issues.append(f"phone {p!r} for word {word!r} has invalid token shape")
    return issues


def expand_word_to_phones(word: str, lexicon: Mapping[str, Tuple[str, ...]]) -> Tuple[str, ...]:
    k = str(word).lower().strip()
    if k in lexicon:
        return lexicon[k]
    return (f"@{k}",)


def phone_inventory(labels: Sequence[str], lexicon: Mapping[str, Tuple[str, ...]]) -> List[str]:
    """Stable insertion-order union of phones referenced by labels."""
    seen: List[str] = []
    for w in labels:
        for p in expand_word_to_phones(str(w), lexicon):
            if p not in seen:
                seen.append(p)
    return seen


def load_user_lexicon_overlay(path: Union[str, Path]) -> Dict[str, Tuple[str, ...]]:
    """Load user-only word→phones from JSON; merge+validate against defaults.

    Returns only keys present in the file (suitable for :attr:`ScenarioConfig.phone_lexicon`).
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("lexicon JSON root must be an object")
    user: Dict[str, Tuple[str, ...]] = {}
    for k, v in raw.items():
        if not isinstance(v, list):
            raise ValueError(f"value for {k!r} must be a JSON array of strings")
        user[str(k)] = tuple(str(x) for x in v)
    merged = merge_lexicon(user)
    issues = validate_lexicon({wk: merged[wk] for wk in user})
    if issues:
        raise ValueError("invalid lexicon: " + "; ".join(issues[:8]))
    return user


def load_lexicon_from_json(path: Union[str, Path]) -> Dict[str, Tuple[str, ...]]:
    """Load {\"word\": [\"P\", \"H\"], ...} from JSON; validates new entries."""
    user = load_user_lexicon_overlay(path)
    return merge_lexicon(user)
