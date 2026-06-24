"""Phoneme-tier scaffolding for synthetic streams (lexicon, timelines, segment records).

This package does **not** perform general grapheme-to-phoneme conversion. It provides:

- A **small default lexicon** for command words (:data:`DEFAULT_PHONE_LEXICON`).
- **Timeline utilities** to split a word event into phone-length intervals.
- **Types** (:class:`PhonemeSegment`) for aligning labels with ``phonemes.csv``.

For motor-unit synergy classes in biophysical mode, see
:class:`~openalterego.sim.biophysical.stream.BiophysicalSimStream` with
``ScenarioConfig.drive_mode="phoneme"``.
"""

from __future__ import annotations

from .lexicon import (
    DEFAULT_PHONE_LEXICON,
    expand_word_to_phones,
    load_lexicon_from_json,
    load_user_lexicon_overlay,
    merge_lexicon,
    phone_inventory,
    validate_lexicon,
)
from .timeline import iter_phone_slices, partition_event_to_phones, partition_phones_in_event
from .coarticulation import (
    DEFAULT_COARTICULATION_OVERLAP_FRAC,
    build_phone_coarticulation_envelopes,
    coarticulation_overlap_ms,
    iter_coarticulated_phone_jobs,
)
from .durations import phone_duration_weight, phone_duration_weights
from .templates import PhoneTemplate, PhoneTemplateStore, load_phone_templates, save_phone_templates
from .types import PhonemeSegment

__all__ = [
    "DEFAULT_PHONE_LEXICON",
    "PhonemeSegment",
    "expand_word_to_phones",
    "iter_phone_slices",
    "load_lexicon_from_json",
    "load_user_lexicon_overlay",
    "merge_lexicon",
    "partition_event_to_phones",
    "partition_phones_in_event",
    "phone_duration_weight",
    "phone_duration_weights",
    "phone_inventory",
    "load_phone_templates",
    "save_phone_templates",
    "PhoneTemplate",
    "PhoneTemplateStore",
    "DEFAULT_COARTICULATION_OVERLAP_FRAC",
    "build_phone_coarticulation_envelopes",
    "coarticulation_overlap_ms",
    "iter_coarticulated_phone_jobs",
    "validate_lexicon",
]
