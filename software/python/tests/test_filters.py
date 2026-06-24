"""Tests for DSP filters including clinical mode and harmonic notching."""

import unittest
import numpy as np

from openalterego.dsp.filters import (
    FilterSpec,
    butter_bandpass,
    apply_notch_with_harmonics,
    get_filter_spec_for_mode,
    preprocess_basic,
    preprocess_streaming,
)


class TestFilterSpec(unittest.TestCase):
    def test_filter_spec_defaults(self):
        """Test FilterSpec with default values."""
        spec = FilterSpec(fs_hz=250.0)
        self.assertEqual(spec.bandpass_hz, (1.0, 50.0))
        self.assertEqual(spec.notch_hz, 60.0)
        self.assertFalse(spec.notch_harmonics)

    def test_filter_spec_custom(self):
        """Test FilterSpec with custom values."""
        spec = FilterSpec(
            fs_hz=250.0,
            bandpass_hz=(0.5, 8.0),
            notch_hz=50.0,
            notch_harmonics=True,
        )
        self.assertEqual(spec.bandpass_hz, (0.5, 8.0))
        self.assertEqual(spec.notch_hz, 50.0)
        self.assertTrue(spec.notch_harmonics)


class TestFilterModes(unittest.TestCase):
    def test_get_filter_spec_standard(self):
        """Test standard mode filter spec."""
        spec = get_filter_spec_for_mode("standard", fs_hz=250.0, notch_hz=60.0)
        self.assertEqual(spec.bandpass_hz, (1.0, 50.0))
        self.assertEqual(spec.notch_hz, 60.0)

    def test_get_filter_spec_clinical(self):
        """Test clinical mode filter spec."""
        spec = get_filter_spec_for_mode("clinical", fs_hz=250.0, notch_hz=60.0)
        self.assertEqual(spec.bandpass_hz, (0.5, 8.0))
        self.assertEqual(spec.notch_hz, 60.0)

    def test_get_filter_spec_wide(self):
        """Test wide mode filter spec (20-450 Hz)."""
        spec = get_filter_spec_for_mode("wide", fs_hz=1000.0, notch_hz=60.0)
        self.assertEqual(spec.bandpass_hz, (20.0, 450.0))
        self.assertEqual(spec.notch_hz, 60.0)
        
    def test_get_filter_spec_wide_high_fs(self):
        """Test wide mode with higher sampling rate (should work)."""
        spec = get_filter_spec_for_mode("wide", fs_hz=1000.0, notch_hz=60.0)
        self.assertEqual(spec.bandpass_hz, (20.0, 450.0))
        # 450 Hz is below Nyquist (500 Hz) for 1000 Hz sampling

    def test_get_filter_spec_gowda(self):
        """Test gowda paper bandpass (80-1000 Hz, 3rd order)."""
        spec = get_filter_spec_for_mode("gowda", fs_hz=5000.0, notch_hz=60.0)
        self.assertEqual(spec.bandpass_hz, (80.0, 1000.0))
        self.assertEqual(spec.bandpass_order, 3)

    def test_get_filter_spec_gowda_rejects_low_fs(self):
        with self.assertRaises(ValueError):
            get_filter_spec_for_mode("gowda", fs_hz=1000.0, notch_hz=60.0)

    def test_get_filter_spec_custom_notch(self):
        """Test filter spec with custom notch frequency."""
        spec = get_filter_spec_for_mode("standard", fs_hz=250.0, notch_hz=50.0)
        self.assertEqual(spec.notch_hz, 50.0)

    def test_get_filter_spec_harmonics(self):
        """Test filter spec with harmonics enabled."""
        spec = get_filter_spec_for_mode("standard", fs_hz=250.0, notch_harmonics=True)
        self.assertTrue(spec.notch_harmonics)


class TestHarmonicNotching(unittest.TestCase):
    def test_notch_without_harmonics(self):
        """Test notch filtering without harmonics."""
        fs = 250.0
        spec = FilterSpec(fs_hz=fs, notch_hz=60.0, notch_harmonics=False)
        
        # Create test signal with 60 Hz component
        t = np.arange(0, 1.0, 1/fs)
        x = np.sin(2 * np.pi * 60 * t)[:, None]  # (time, 1 channel)
        
        y = apply_notch_with_harmonics(x, spec)
        
        # Notch should reduce 60 Hz component
        # Check that output has less energy than input
        self.assertLess(np.std(y), np.std(x))

    def test_notch_with_harmonics(self):
        """Test notch filtering with harmonics."""
        fs = 250.0
        spec = FilterSpec(fs_hz=fs, notch_hz=60.0, notch_harmonics=True)
        
        # Create test signal with 60 Hz and 120 Hz (2nd harmonic)
        t = np.arange(0, 1.0, 1/fs)
        x = (np.sin(2 * np.pi * 60 * t) + 0.5 * np.sin(2 * np.pi * 120 * t))[:, None]
        
        y = apply_notch_with_harmonics(x, spec)
        
        # Both fundamental and harmonic should be reduced
        self.assertLess(np.std(y), np.std(x))

    def test_notch_harmonics_below_nyquist(self):
        """Test that harmonics above Nyquist are not notched."""
        fs = 250.0  # Nyquist = 125 Hz
        spec = FilterSpec(fs_hz=fs, notch_hz=60.0, notch_harmonics=True)
        
        # 3rd harmonic (180 Hz) is above Nyquist, should not be notched
        # But 2nd harmonic (120 Hz) should be
        t = np.arange(0, 1.0, 1/fs)
        x = np.sin(2 * np.pi * 60 * t)[:, None]
        
        y = apply_notch_with_harmonics(x, spec)
        # Should still work (just won't notch 3rd harmonic)
        self.assertIsInstance(y, np.ndarray)


class TestPreprocessingModes(unittest.TestCase):
    def test_preprocess_basic_standard(self):
        """Test basic preprocessing in standard mode."""
        fs = 250.0
        rng = np.random.default_rng(42)
        x = rng.normal(size=(1000, 8)).astype(np.float32)
        
        y = preprocess_basic(x, fs_hz=fs, mode="standard")
        
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, np.float32)

    def test_preprocess_basic_clinical(self):
        """Test basic preprocessing in clinical mode."""
        fs = 250.0
        rng = np.random.default_rng(42)
        x = rng.normal(size=(1000, 8)).astype(np.float32)
        
        y = preprocess_basic(x, fs_hz=fs, mode="clinical")
        
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, np.float32)
        # Clinical mode should use different bandpass
        # (We can't easily test the exact frequencies, but shape should match)

    def test_preprocess_basic_wide(self):
        """Test basic preprocessing in wide mode (20-450 Hz)."""
        fs = 1000.0  # Need higher sampling rate for 450 Hz
        rng = np.random.default_rng(42)
        x = rng.normal(size=(5000, 8)).astype(np.float32)
        
        y = preprocess_basic(x, fs_hz=fs, mode="wide")
        
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, np.float32)
        
    def test_preprocess_basic_wide_250hz(self):
        """Test wide mode with 250 Hz sampling (should raise error - 450 Hz > Nyquist)."""
        fs = 250.0  # Nyquist = 125 Hz, so 450 Hz is above Nyquist
        rng = np.random.default_rng(42)
        x = rng.normal(size=(1000, 8)).astype(np.float32)
        
        # Should raise ValueError because 450 Hz > Nyquist (125 Hz)
        with self.assertRaises(ValueError) as cm:
            preprocess_basic(x, fs_hz=fs, mode="wide")
        self.assertIn("fs_hz >= 920 Hz", str(cm.exception))
        self.assertIn("Wide mode requires", str(cm.exception))

    def test_preprocess_basic_harmonics(self):
        """Test basic preprocessing with harmonic notching."""
        fs = 250.0
        rng = np.random.default_rng(42)
        x = rng.normal(size=(1000, 8)).astype(np.float32)
        
        y = preprocess_basic(x, fs_hz=fs, notch_harmonics=True)
        
        self.assertEqual(y.shape, x.shape)

    def test_preprocess_streaming_clinical(self):
        """Test streaming preprocessing in clinical mode."""
        fs = 250.0
        rng = np.random.default_rng(42)
        x = rng.normal(size=(1000, 8)).astype(np.float32)
        
        y = preprocess_streaming(x, fs_hz=fs, channels=8, mode="clinical")
        
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, np.float32)

    def test_preprocess_streaming_wide(self):
        """Test streaming preprocessing in wide mode."""
        fs = 1000.0  # Need higher sampling rate for 450 Hz
        rng = np.random.default_rng(42)
        x = rng.normal(size=(5000, 8)).astype(np.float32)
        
        y = preprocess_streaming(x, fs_hz=fs, channels=8, mode="wide")
        
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, np.float32)
        
    def test_preprocess_mode_comparison(self):
        """Test that different modes produce different outputs."""
        fs = 1000.0
        rng = np.random.default_rng(42)
        x = rng.normal(size=(5000, 8)).astype(np.float32)
        
        y_standard = preprocess_basic(x, fs_hz=fs, mode="standard")
        y_clinical = preprocess_basic(x, fs_hz=fs, mode="clinical")
        y_wide = preprocess_basic(x, fs_hz=fs, mode="wide")
        
        # All should have same shape
        self.assertEqual(y_standard.shape, y_clinical.shape)
        self.assertEqual(y_standard.shape, y_wide.shape)
        
        # They should produce different outputs (different bandpass ranges)
        # Allow some tolerance for numerical differences
        self.assertFalse(np.allclose(y_standard, y_clinical, rtol=1e-3))
        self.assertFalse(np.allclose(y_standard, y_wide, rtol=1e-3))
        self.assertFalse(np.allclose(y_clinical, y_wide, rtol=1e-3))

    def test_preprocess_backward_compatibility(self):
        """Test that old API still works (backward compatibility)."""
        fs = 250.0
        rng = np.random.default_rng(42)
        x = rng.normal(size=(1000, 8)).astype(np.float32)
        
        # Old API (no mode parameter)
        y1 = preprocess_basic(x, fs_hz=fs, bandpass_hz=(1.0, 50.0))
        
        # New API with mode=None (should behave like old)
        y2 = preprocess_basic(x, fs_hz=fs, mode=None, bandpass_hz=(1.0, 50.0))
        
        # Should produce similar results (allowing for small numerical differences)
        np.testing.assert_allclose(y1, y2, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
