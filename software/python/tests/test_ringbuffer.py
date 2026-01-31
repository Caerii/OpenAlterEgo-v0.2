import unittest
import numpy as np

from openalterego.core.ringbuffer import RingBuffer


class TestRingBuffer(unittest.TestCase):
    def test_append_and_get_last(self):
        rb = RingBuffer(capacity=10, channels=2)
        rb.append(np.ones((3, 2), dtype=np.float32))
        self.assertEqual(rb.filled, 3)
        last = rb.get_last(2)
        self.assertEqual(last.shape, (2, 2))

        rb.append(np.zeros((10, 2), dtype=np.float32))  # overwrites
        self.assertEqual(rb.filled, 10)
        last = rb.get_last(5)
        self.assertEqual(last.shape, (5, 2))


if __name__ == "__main__":
    unittest.main()
