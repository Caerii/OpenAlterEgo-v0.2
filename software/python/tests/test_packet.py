import unittest
import numpy as np

from openalterego.acquisition.packet import AfeSpec, PacketSpec, pack_oa_v1, parse_oa_v1, quantize_uV_to_counts


class TestPacketV1(unittest.TestCase):
    def test_roundtrip_counts(self):
        afe = AfeSpec(adc_bits=16, vref_v=2.4, gain=24.0)
        spec = PacketSpec(channels=8, frames_per_packet=12)

        rng = np.random.default_rng(0)
        x_uV = rng.normal(scale=100.0, size=(12, 8)).astype(np.float32)
        counts = quantize_uV_to_counts(x_uV, afe=afe)

        payload = pack_oa_v1(counts, seq0=123, sample_index0=1000, spec=spec)
        out_uV, info = parse_oa_v1(payload, afe=afe)

        self.assertEqual(info["channels"], 8)
        self.assertEqual(info["frames"], 12)
        self.assertEqual(info["seq0"], 123)
        self.assertEqual(info["sample_index0"], 1000)

        # Quantization means we won't exactly match x_uV.
        # But if we convert back to counts, we should match.
        out_counts, _ = parse_oa_v1(payload, afe=None)
        np.testing.assert_array_equal(out_counts, counts)


if __name__ == "__main__":
    unittest.main()
