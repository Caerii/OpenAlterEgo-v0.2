"""Datatypes for phoneme-tier simulation artifacts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhonemeSegment:
    """One phone interval inside a word-level command event."""

    start_sample: int
    end_sample: int
    phone: str
    word: str
