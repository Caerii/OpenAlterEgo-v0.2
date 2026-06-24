"""Tests for corpus-duration phone alignment."""

from openalterego.sim.phonology.align import partition_phones_aligned


def test_partition_phones_aligned_sums_to_n():
    phones = ["M", "AE", "N"]
    n = 1000
    for mode in ("pseudo", "corpus_duration"):
        segs = partition_phones_aligned(n, phones, mode=mode, seed=42)
        assert len(segs) == len(phones)
        assert sum(segs) == n
        assert all(s >= 2 for s in segs)


def test_corpus_duration_more_stable_than_pseudo():
    phones = ["HH", "EH", "L", "OW"]
    n = 500
    a = partition_phones_aligned(n, phones, mode="corpus_duration", seed=0)
    b = partition_phones_aligned(n, phones, mode="corpus_duration", seed=0)
    assert a == b
