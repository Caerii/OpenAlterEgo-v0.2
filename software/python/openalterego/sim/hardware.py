"""Hardware-ish simulation helpers.

This module mostly re-exports packet/scaling helpers from :mod:`openalterego.acquisition.packet`
so the simulation code and the realtime acquisition code share the *exact* same framing logic.
"""

from __future__ import annotations

from ..acquisition.packet import (  # noqa: F401
    AfeSpec,
    PacketSpec,
    MAGIC,
    VERSION,
    HEADER_BYTES,
    counts_to_uV,
    parse_oa_v1,
    pack_oa_v1,
    quantize_uV_to_counts,
)
