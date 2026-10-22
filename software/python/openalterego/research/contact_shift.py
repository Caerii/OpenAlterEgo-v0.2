"""Grid-topological replay of shared-reference and digital-difference channels."""

import numpy as np

from openalterego.research.hyser_features import site_coordinates


SHIFTS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def channel_shift_maps(meta: dict, shifts=SHIFTS) -> np.ndarray:
    """Return channel mappings for simultaneous one-cell grid translations.

    An invalid destination is -1. Geometry is topological; no millimetre or
    independent-contact displacement claim follows from these mappings.
    """
    coordinates = site_coordinates(tuple(meta["site_names"]))
    pairs = np.asarray(meta["pairs"], dtype=np.int16)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or np.any(pairs < 0) or np.any(pairs >= 256):
        raise ValueError("invalid differential pair map")
    lookup = {tuple(map(int, point)): index
              for index, point in enumerate(coordinates)}
    pair_lookup = {frozenset(map(int, pair)): index
                   for index, pair in enumerate(pairs)}
    if len(pair_lookup) != len(pairs):
        raise ValueError("duplicate differential pair")
    result = np.full((len(shifts), 256 + len(pairs)), -1, dtype=np.int16)
    for shift_index, (dr, dc) in enumerate(shifts):
        for index, (grid, row, col) in enumerate(coordinates):
            result[shift_index, index] = lookup.get(
                (int(grid), int(row) + dr, int(col) + dc), -1)
        for index, (left, right) in enumerate(pairs):
            shifted = (int(result[shift_index, left]),
                       int(result[shift_index, right]))
            if min(shifted) >= 0:
                pair_index = pair_lookup.get(frozenset(shifted))
                if pair_index is None:
                    raise ValueError("translated pair absent from map")
                result[shift_index, 256 + index] = 256 + pair_index
    return result


def robust_options(options, maps: np.ndarray):
    """Keep grammar options that preserve eight contacts under every shift."""
    result = []
    for group in options:
        kept = []
        for option in group:
            if np.all(maps[:, option] >= 0):
                kept.append(option)
        if not kept:
            raise ValueError("no shift-compatible options in a program part")
        result.append(tuple(kept))
    return tuple(result)


def shift_programs(programs: np.ndarray, maps: np.ndarray) -> np.ndarray:
    """Map a batch of full programs to every shift; reject boundary crossings."""
    programs = np.asarray(programs, dtype=np.int16)
    if (programs.ndim != 2 or maps.ndim != 2 or
            np.any(programs < 0) or np.any(programs >= maps.shape[1])):
        raise ValueError("invalid programs or shift maps")
    shifted = maps[:, programs]
    if np.any(shifted < 0):
        raise ValueError("program crosses grid boundary under a requested shift")
    return np.sort(shifted, axis=2)


def align_shifted_features(features: np.ndarray, channel_map: np.ndarray,
                           *, features_per_channel: int) -> np.ndarray:
    """Place shifted measured channels at nominal indices for session replay.

    Invalid boundary channels are filled with zero. Only shift-compatible
    programs may be scored on the returned matrix.
    """
    features = np.asarray(features)
    channel_map = np.asarray(channel_map)
    if (features.ndim != 2 or channel_map.ndim != 1 or
            features_per_channel < 1 or
            features.shape[1] != len(channel_map) * features_per_channel or
            np.any(channel_map >= len(channel_map)) or
            np.any(channel_map < -1)):
        raise ValueError("invalid feature matrix or channel map")
    source = features.reshape(len(features), len(channel_map), features_per_channel)
    aligned = source[:, np.maximum(channel_map, 0)].copy()
    aligned[:, channel_map < 0] = 0
    return aligned.reshape(len(features), -1)
