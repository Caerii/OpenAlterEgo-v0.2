"""Gowda-shaped biophysical simulation scenarios."""

from .gowda_small_vocab import (
    GOWDA_WORD_DURATION_S,
    build_gowda_dataset_config,
    build_gowda_phone_lexicon,
    build_gowda_scenario,
    gowda_small_vocab_labels,
    load_gowda_labels,
)

__all__ = [
    "GOWDA_WORD_DURATION_S",
    "build_gowda_dataset_config",
    "build_gowda_phone_lexicon",
    "build_gowda_scenario",
    "gowda_small_vocab_labels",
    "load_gowda_labels",
]
