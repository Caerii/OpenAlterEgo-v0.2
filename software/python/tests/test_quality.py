"""Tests for signal quality monitoring and motion artifact detection."""

import unittest
import numpy as np

from openalterego.dsp.quality import (
    SignalQualityMetrics,
    assess_signal_quality,
    compute_snr,
    detect_motion_artifacts,
    OnlineQualityMonitor,
    weak_channel_indices,
)


class TestSNRComputation(unittest.TestCase):
    def test_compute_snr_simple(self):
        """Test SNR computation on simple signal."""
        fs = 1000.0
        t = np.arange(0, 1.0, 1/fs)
        
        # Create signal with 100 Hz component (in signal band) and 2 Hz component (noise)
        signal_component = np.sin(2 * np.pi * 100 * t)
        noise_component = 0.1 * np.sin(2 * np.pi * 2 * t)
        x = signal_component + noise_component
        
        snr = compute_snr(x, fs, signal_band_hz=(20.0, 450.0), noise_band_hz=(0.5, 5.0))
        
        # SNR should be positive (signal > noise)
        self.assertGreater(snr, 0.0)
        self.assertIsInstance(snr, float)
    
    def test_compute_snr_multi_channel(self):
        """Test SNR computation on multi-channel signal."""
        fs = 1000.0
        t = np.arange(0, 1.0, 1/fs)
        
        # Create 2-channel signal
        signal = np.sin(2 * np.pi * 100 * t)
        noise = 0.1 * np.sin(2 * np.pi * 2 * t)
        x = np.column_stack([signal + noise, signal + 0.2 * noise])
        
        snr = compute_snr(x, fs)
        
        self.assertIsInstance(snr, float)
        self.assertGreater(snr, -np.inf)
    
    def test_compute_snr_empty_signal(self):
        """Test SNR computation on empty signal."""
        x = np.array([])
        snr = compute_snr(x, fs_hz=1000.0)
        self.assertEqual(snr, -np.inf)


class TestMotionArtifactDetection(unittest.TestCase):
    def test_detect_motion_artifacts_low_freq(self):
        """Test motion detection on signal with low-frequency drift."""
        fs = 1000.0
        t = np.arange(0, 2.0, 1/fs)
        
        # Create signal with strong low-frequency component (motion artifact)
        low_freq = 2.0 * np.sin(2 * np.pi * 2 * t)  # 2 Hz drift
        high_freq = 0.5 * np.sin(2 * np.pi * 100 * t)  # 100 Hz signal
        x = low_freq + high_freq
        
        motion_index, baseline_wander = detect_motion_artifacts(x, fs)
        
        # Should detect motion (low-frequency component is strong)
        self.assertGreater(motion_index, 0.0)
        self.assertLessEqual(motion_index, 1.0)
        self.assertGreater(baseline_wander, 0.0)
    
    def test_detect_motion_artifacts_clean_signal(self):
        """Test motion detection on clean signal (no low-frequency drift)."""
        fs = 1000.0
        t = np.arange(0, 1.0, 1/fs)
        
        # Create clean signal (only high-frequency component)
        x = np.sin(2 * np.pi * 100 * t)
        
        motion_index, baseline_wander = detect_motion_artifacts(x, fs)
        
        # Motion index should be low (clean signal)
        self.assertGreaterEqual(motion_index, 0.0)
        self.assertLessEqual(motion_index, 1.0)
    
    def test_detect_motion_artifacts_multi_channel(self):
        """Test motion detection on multi-channel signal."""
        fs = 1000.0
        t = np.arange(0, 1.0, 1/fs)
        
        # Create 2-channel signal with motion
        low_freq = np.sin(2 * np.pi * 2 * t)
        high_freq = 0.5 * np.sin(2 * np.pi * 100 * t)
        x = np.column_stack([low_freq + high_freq, low_freq + high_freq])
        
        motion_index, baseline_wander = detect_motion_artifacts(x, fs)
        
        self.assertGreater(motion_index, 0.0)
        self.assertLessEqual(motion_index, 1.0)


class TestSignalQualityAssessment(unittest.TestCase):
    def test_assess_signal_quality(self):
        """Test comprehensive signal quality assessment."""
        fs = 1000.0
        t = np.arange(0, 1.0, 1/fs)
        
        # Create signal with both signal and noise components
        signal = np.sin(2 * np.pi * 100 * t)
        noise = 0.1 * np.sin(2 * np.pi * 2 * t)
        x = signal + noise
        
        metrics = assess_signal_quality(x, fs)
        
        self.assertIsInstance(metrics, SignalQualityMetrics)
        self.assertIsNotNone(metrics.snr_db)
        self.assertGreater(metrics.motion_index, 0.0)
        self.assertLessEqual(metrics.motion_index, 1.0)
        self.assertGreater(metrics.signal_power, 0.0)
    
    def test_assess_signal_quality_multi_channel(self):
        """Test quality assessment on multi-channel signal."""
        fs = 1000.0
        t = np.arange(0, 1.0, 1/fs)
        
        signal = np.sin(2 * np.pi * 100 * t)
        noise = 0.1 * np.sin(2 * np.pi * 2 * t)
        x = np.column_stack([signal + noise, signal + 0.2 * noise])
        
        metrics = assess_signal_quality(x, fs)
        
        self.assertIsInstance(metrics, SignalQualityMetrics)
        self.assertIsNotNone(metrics.snr_db)

    def test_assess_per_channel_shapes(self) -> None:
        fs = 1000.0
        t = np.arange(0, 1.0, 1 / fs)
        signal = np.sin(2 * np.pi * 100 * t)
        noise = 0.1 * np.sin(2 * np.pi * 2 * t)
        x = np.column_stack([signal + noise, signal + 0.5 * noise])
        m = assess_signal_quality(x, fs, per_channel=True)
        self.assertIsNotNone(m.snr_db_per_channel)
        self.assertIsNotNone(m.motion_index_per_channel)
        assert m.snr_db_per_channel is not None
        assert m.motion_index_per_channel is not None
        self.assertEqual(m.snr_db_per_channel.shape, (2,))
        self.assertEqual(m.motion_index_per_channel.shape, (2,))

    def test_weak_channel_indices(self) -> None:
        snr = np.array([20.0, 19.0, 5.0, 18.0], dtype=np.float64)
        weak = weak_channel_indices(snr, deficit_db=6.0)
        self.assertIn(2, weak)
        self.assertEqual(weak_channel_indices(np.array([-np.inf, 10.0, 12.0])), [])


class TestOnlineQualityMonitor(unittest.TestCase):
    def test_online_monitor_initialization(self):
        """Test online monitor initialization."""
        monitor = OnlineQualityMonitor(fs_hz=1000.0, window_samples=1000)
        
        self.assertEqual(monitor.fs_hz, 1000.0)
        self.assertEqual(monitor.window_samples, 1000)
    
    def test_online_monitor_update(self):
        """Test online monitor update."""
        fs = 1000.0
        monitor = OnlineQualityMonitor(fs_hz=fs, window_samples=1000)
        
        # Create test signal
        t = np.arange(0, 0.5, 1/fs)
        x = np.sin(2 * np.pi * 100 * t)
        
        # Update monitor
        metrics = monitor.update(x)
        
        self.assertIsInstance(metrics, SignalQualityMetrics)

    def test_online_monitor_per_channel(self) -> None:
        fs = 1000.0
        monitor = OnlineQualityMonitor(fs_hz=fs, window_samples=1000, per_channel=True)
        t = np.arange(0, 0.5, 1 / fs)
        x = np.column_stack([np.sin(2 * np.pi * 100 * t)] * 3)
        m = monitor.update(x)
        self.assertIsNotNone(m.snr_db_per_channel)
        assert m.snr_db_per_channel is not None
        self.assertEqual(m.snr_db_per_channel.size, 3)
    
    def test_online_monitor_multi_channel(self):
        """Test online monitor with multi-channel signal."""
        fs = 1000.0
        monitor = OnlineQualityMonitor(fs_hz=fs, window_samples=1000)
        
        # Create 2-channel test signal
        t = np.arange(0, 0.5, 1/fs)
        signal = np.sin(2 * np.pi * 100 * t)
        x = np.column_stack([signal, signal])
        
        metrics = monitor.update(x)
        
        self.assertIsInstance(metrics, SignalQualityMetrics)
    
    def test_online_monitor_reset(self):
        """Test online monitor reset."""
        fs = 1000.0
        monitor = OnlineQualityMonitor(fs_hz=fs, window_samples=1000)
        
        # Update with some data
        t = np.arange(0, 0.5, 1/fs)
        x = np.sin(2 * np.pi * 100 * t)
        monitor.update(x)
        
        # Reset
        monitor.reset()
        
        # Buffer should be cleared
        self.assertIsNone(monitor._buffer)
        self.assertEqual(monitor._buffer_idx, 0)
    
    def test_online_monitor_sliding_window(self):
        """Test that monitor maintains sliding window."""
        fs = 1000.0
        window_samples = 500
        monitor = OnlineQualityMonitor(fs_hz=fs, window_samples=window_samples)
        
        # Add multiple chunks
        chunk_size = 100
        for i in range(10):
            t = np.arange(0, chunk_size / fs, 1/fs)
            x = np.sin(2 * np.pi * 100 * t)
            metrics = monitor.update(x)
            
            # Should always return valid metrics
            self.assertIsInstance(metrics, SignalQualityMetrics)


if __name__ == "__main__":
    unittest.main()
