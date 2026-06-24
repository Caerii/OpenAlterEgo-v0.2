"""Simulation utilities (synthetic signals, hardware-ish packetization, transport)."""

from .biophysical import (
    BiophysicalSimStream,
    BiophysicalSimStreamConfig,
    add_muap_spikes,
    bipolar_muap_template,
)
from .literature import (
    LITERATURE_MODEL_VERSION,
    VALID_PARADIGMS,
    resolve_sim_token_band,
)
from .stream import SimStream, SimStreamConfig, ScenarioConfig, SimEvent
from .dataset import DatasetConfig, generate_dataset
from .phonology import (
    PhonemeSegment,
    expand_word_to_phones,
    load_lexicon_from_json,
    load_user_lexicon_overlay,
    merge_lexicon,
    phone_inventory,
    validate_lexicon,
)
from .hardware import AfeSpec, PacketSpec, pack_oa_v1, parse_oa_v1, quantize_uV_to_counts, counts_to_uV
from .transport import LinkConfig, apply_link
