import unittest
import numpy as np

from openalterego.sim.stream import SimStream, SimStreamConfig, ScenarioConfig


class TestSimStream(unittest.TestCase):
    def test_generates_events(self):
        cfg = SimStreamConfig(
            fs_hz=250,
            channels=4,
            seed=1,
            realtime_clock=False,
            scenario=ScenarioConfig(labels=["a", "b"], p_event=0.9, event_duration_s=(0.2, 0.3), gap_duration_s=(0.05, 0.1)),
        )
        sim = SimStream(cfg)
        for _ in range(50):
            sim.next_chunk()

        self.assertTrue(len(sim.events) > 0)
        for ev in sim.events:
            self.assertLess(ev.start_sample, ev.end_sample)
            self.assertIn(ev.label, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
