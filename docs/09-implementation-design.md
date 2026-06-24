# Implementation Design: Research Validation Fixes

## Architecture Overview

### Design Principles
1. **Encapsulation**: Each feature in its own module with clear interfaces
2. **Testability**: All components unit-testable in isolation
3. **Backward Compatibility**: Existing code continues to work
4. **Configuration-Driven**: Use dataclasses for all config (following existing patterns)
5. **Progressive Enhancement**: Features can be used optionally

---

## Module Structure

```
openalterego/
├── users/                    # NEW: User management
│   ├── __init__.py
│   ├── profile.py           # UserProfile dataclass
│   ├── manager.py           # UserManager class
│   └── calibration.py       # Calibration workflow
├── dsp/
│   ├── filters.py           # MODIFY: Add clinical mode
│   └── online.py            # MODIFY: Support clinical mode
├── runtime/
│   └── streaming.py        # MODIFY: Adaptive thresholds
└── ml/
    └── train.py             # MODIFY: User-aware training
```

---

## 1. User Management System

### 1.1 UserProfile (`users/profile.py`)

```python
@dataclass(frozen=True)
class UserProfile:
    """Per-user configuration and metadata."""
    user_id: str
    created_at: float  # Unix timestamp
    model_path: Optional[Path] = None
    confidence_threshold: float = 0.70  # Per-user threshold
    preprocessing_mode: Literal["standard", "clinical"] = "standard"
    window_ms: int = 600
    stride_ms: int = 120
    # Calibration metadata
    calibration_date: Optional[float] = None
    calibration_samples: int = 0
    # Signal quality metrics (for adaptive thresholding)
    baseline_snr: Optional[float] = None
```

**Design Notes:**
- Frozen dataclass (immutable) for safety
- Stores user-specific config
- Can be serialized to JSON for persistence

### 1.2 UserManager (`users/manager.py`)

```python
class UserManager:
    """Manages user profiles and data directories."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_user_dir(self, user_id: str) -> Path:
        """Get user's data directory."""
    
    def load_profile(self, user_id: str) -> Optional[UserProfile]:
        """Load user profile from disk."""
    
    def save_profile(self, profile: UserProfile) -> None:
        """Save user profile to disk."""
    
    def list_users(self) -> List[str]:
        """List all registered users."""
```

**Design Notes:**
- Stateless class (methods only, no mutable state)
- File-based storage (simple, no DB dependency)
- Thread-safe operations (if needed later)

### 1.3 Calibration (`users/calibration.py`)

```python
@dataclass
class CalibrationConfig:
    """Configuration for user calibration."""
    min_samples_per_token: int = 50
    target_samples_per_token: int = 200
    tokens: List[str]  # Vocabulary to calibrate

def calibrate_user(
    user_id: str,
    data_dir: Path,
    config: CalibrationConfig,
    manager: UserManager,
) -> UserProfile:
    """Run calibration workflow for a user."""
    # 1. Collect calibration data
    # 2. Train user-specific model
    # 3. Compute confidence threshold from calibration data
    # 4. Create and save UserProfile
```

**Design Notes:**
- Pure function (no side effects except file I/O)
- Returns UserProfile (testable)
- Can be called from CLI or programmatically

---

## 2. Preprocessing Enhancements

### 2.1 Extended FilterSpec (`dsp/filters.py`)

```python
@dataclass(frozen=True)
class FilterSpec:
    fs_hz: float
    bandpass_hz: Tuple[float, float] = (1.0, 50.0)
    bandpass_order: int = 4
    notch_hz: Optional[float] = 60.0
    notch_q: float = 30.0
    notch_harmonics: bool = False  # NEW: Enable harmonic notching

# NEW: Preprocessing mode enum
PreprocessingMode = Literal["standard", "clinical", "offline", "streaming", "none"]

def get_filter_spec_for_mode(
    mode: PreprocessingMode,
    fs_hz: float,
    notch_hz: Optional[float] = None,
) -> FilterSpec:
    """Get FilterSpec for a preprocessing mode."""
    if mode == "clinical":
        return FilterSpec(fs_hz=fs_hz, bandpass_hz=(0.5, 8.0), notch_hz=notch_hz)
    else:  # standard
        return FilterSpec(fs_hz=fs_hz, bandpass_hz=(1.0, 50.0), notch_hz=notch_hz)
```

**Design Notes:**
- Backward compatible (defaults unchanged)
- Mode-based factory function
- Extensible for future modes

### 2.2 Harmonic Notching

```python
def apply_notch_with_harmonics(
    x: np.ndarray,
    spec: FilterSpec,
) -> np.ndarray:
    """Apply notch filter, optionally including harmonics."""
    y = x
    if spec.notch_hz is not None:
        # Notch fundamental
        b, a = notch_iir(spec)
        if b is not None:
            y = apply_notch(y, b, a)
        
        # Notch harmonics if enabled
        if spec.notch_harmonics:
            for harmonic in [2, 3]:  # 2nd and 3rd harmonics
                freq = spec.notch_hz * harmonic
                if freq < spec.fs_hz / 2:  # Below Nyquist
                    harmonic_spec = FilterSpec(
                        fs_hz=spec.fs_hz,
                        notch_hz=freq,
                        notch_q=spec.notch_q,
                    )
                    b, a = notch_iir(harmonic_spec)
                    if b is not None:
                        y = apply_notch(y, b, a)
    return y
```

**Design Notes:**
- Optional feature (notch_harmonics=False by default)
- Backward compatible
- Can be enabled per-user

---

## 3. Adaptive Confidence Thresholding

### 3.1 Extended StreamDecodeConfig (`runtime/streaming.py`)

```python
@dataclass
class StreamDecodeConfig:
    window_ms: int = 600
    stride_ms: int = 120
    min_confidence: float = 0.70  # Base threshold
    stable_n: int = 3
    cooldown_ms: int = 250
    blank_token: Optional[str] = None
    
    # NEW: Adaptive thresholding
    adaptive_threshold: bool = False
    threshold_alpha: float = 0.1  # EMA for threshold adaptation
    user_profile: Optional[UserProfile] = None  # For per-user threshold
```

### 3.2 Enhanced PredictionStabilizer

```python
class PredictionStabilizer:
    """Debounce with adaptive thresholding."""
    
    def __init__(self, cfg: StreamDecodeConfig):
        self.cfg = cfg
        self._hist: Deque[Tuple[str, float]] = deque(maxlen=int(cfg.stable_n))
        self._last_emit_token: Optional[str] = None
        self._last_emit_t: float = 0.0
        
        # NEW: Adaptive threshold state
        if cfg.adaptive_threshold:
            self._current_threshold = float(cfg.min_confidence)
            if cfg.user_profile and cfg.user_profile.confidence_threshold:
                self._current_threshold = cfg.user_profile.confidence_threshold
        else:
            self._current_threshold = float(cfg.min_confidence)
    
    def update(self, token: str, conf: float, *, t: float, seq: int, source: str) -> Optional[TokenEvent]:
        # ... existing logic ...
        
        # NEW: Adaptive threshold adjustment
        if self.cfg.adaptive_threshold:
            # Adjust threshold based on recent confidence distribution
            recent_confs = [x[1] for x in self._hist]
            if len(recent_confs) >= 3:
                mean_conf = np.mean(recent_confs)
                # If consistently high confidence, lower threshold slightly
                # If consistently low, raise threshold
                alpha = self.cfg.threshold_alpha
                if mean_conf > 0.85:
                    self._current_threshold = (1 - alpha) * self._current_threshold + alpha * (mean_conf - 0.1)
                elif mean_conf < 0.60:
                    self._current_threshold = (1 - alpha) * self._current_threshold + alpha * (mean_conf + 0.1)
                self._current_threshold = np.clip(self._current_threshold, 0.5, 0.95)
        
        # Use adaptive threshold
        if mean_conf < self._current_threshold:
            return None
```

**Design Notes:**
- Backward compatible (adaptive_threshold=False by default)
- Simple EMA-based adaptation
- Can be enhanced with more sophisticated logic later

---

## 4. User-Aware Training

### 4.1 Modified Training Script (`ml/train.py`)

```python
def main() -> None:
    ap = argparse.ArgumentParser()
    # ... existing args ...
    ap.add_argument("--user-id", type=str, default=None, help="User ID for personalization")
    ap.add_argument("--preprocessing-mode", type=str, default="offline", 
                    choices=["offline", "streaming", "clinical", "none"])
    
    args = ap.parse_args()
    
    # NEW: Load or create user profile
    if args.user_id:
        from ..users.manager import UserManager
        manager = UserManager(base_dir=Path("./users"))
        profile = manager.load_profile(args.user_id)
        if profile is None:
            # Create new profile
            profile = UserProfile(user_id=args.user_id, created_at=time.time())
    else:
        profile = None
    
    # Use profile's preprocessing mode if available
    if profile and profile.preprocessing_mode == "clinical":
        preprocess_mode = "clinical"
    else:
        preprocess_mode = args.preprocessing_mode
    
    # ... rest of training ...
    
    # Save model to user directory if user_id provided
    if args.user_id and profile:
        user_dir = manager.get_user_dir(args.user_id)
        model_path = user_dir / "model.pt"
        torch.save({...}, model_path)
        # Update profile
        profile = dataclasses.replace(profile, model_path=model_path)
        manager.save_profile(profile)
```

**Design Notes:**
- Optional user_id (backward compatible)
- Uses user's preprocessing mode if available
- Saves model to user directory

---

## 5. Testing Strategy

### 5.1 Unit Tests

```python
# tests/test_users.py
class TestUserManager(unittest.TestCase):
    def test_create_and_load_profile(self):
        """Test user profile creation and loading."""
    
    def test_user_directory_structure(self):
        """Test user directory creation."""

# tests/test_calibration.py
class TestCalibration(unittest.TestCase):
    def test_calibration_workflow(self):
        """Test end-to-end calibration."""
    
    def test_threshold_computation(self):
        """Test confidence threshold calculation from calibration data."""

# tests/test_filters.py (extend existing)
class TestFilters(unittest.TestCase):
    def test_clinical_preprocessing(self):
        """Test clinical mode preprocessing."""
    
    def test_harmonic_notching(self):
        """Test harmonic notch filtering."""

# tests/test_streaming.py (new)
class TestAdaptiveThresholding(unittest.TestCase):
    def test_adaptive_threshold_adjustment(self):
        """Test threshold adaptation logic."""
    
    def test_user_profile_integration(self):
        """Test user profile in streaming classifier."""
```

### 5.2 Integration Tests

```python
# tests/test_integration.py
class TestUserWorkflow(unittest.TestCase):
    def test_full_user_workflow(self):
        """Test: create user -> calibrate -> train -> serve."""
        # 1. Create user
        # 2. Run calibration
        # 3. Train model
        # 4. Load model and run inference
        # 5. Verify user-specific threshold works
```

---

## 6. CLI Enhancements

### 6.1 New Commands

```bash
# User management
openalterego user create --user-id alice
openalterego user list
openalterego user show --user-id alice

# Calibration
openalterego calibrate --user-id alice --data ./calibration_data

# Training (enhanced)
openalterego train --user-id alice --data ./session --preprocessing-mode clinical

# Serving (enhanced)
openalterego serve --user-id alice --source ble
```

---

## 7. Backward Compatibility

### Strategy
1. **Default values unchanged**: All new features opt-in
2. **Optional parameters**: User management is optional
3. **Graceful degradation**: Works without user_id (current behavior)
4. **Deprecation path**: None needed (all additive)

### Migration Path
- Existing code continues to work
- Users can gradually adopt personalization
- No breaking changes

---

## 8. Implementation Order

1. **Phase 1: Preprocessing Enhancements** (Low risk, high value)
   - Add clinical mode to filters
   - Add harmonic notching
   - Tests

2. **Phase 2: User Management Foundation** (Core infrastructure)
   - UserProfile dataclass
   - UserManager class
   - Basic tests

3. **Phase 3: Calibration Workflow** (Uses Phase 2)
   - Calibration module
   - CLI integration
   - Tests

4. **Phase 4: Adaptive Thresholding** (Uses Phase 2)
   - Enhanced PredictionStabilizer
   - Integration with UserProfile
   - Tests

5. **Phase 5: Training Integration** (Uses Phases 2-4)
   - User-aware training
   - Model storage per user
   - Tests

6. **Phase 6: Serving Integration** (Uses all phases)
   - User-aware serving
   - Load user profiles
   - Tests

---

## 9. File Structure

```
openalterego/
├── users/
│   ├── __init__.py
│   ├── profile.py          # UserProfile dataclass
│   ├── manager.py          # UserManager class
│   └── calibration.py      # Calibration workflow
├── dsp/
│   ├── filters.py          # MODIFIED: Clinical mode, harmonics
│   └── online.py           # MODIFIED: Support clinical mode
├── runtime/
│   └── streaming.py        # MODIFIED: Adaptive thresholds
├── ml/
│   └── train.py            # MODIFIED: User-aware
├── api/
│   └── server.py            # MODIFIED: User-aware serving
└── cli.py                   # MODIFIED: New commands

tests/
├── test_users.py           # NEW
├── test_calibration.py     # NEW
├── test_filters.py          # MODIFIED: Clinical mode tests
├── test_streaming.py        # NEW: Adaptive threshold tests
└── test_integration.py     # NEW: End-to-end tests
```

---

## 10. Success Criteria

- [ ] All existing tests pass
- [ ] New features have >80% test coverage
- [ ] Backward compatibility maintained
- [ ] Documentation updated
- [ ] CLI commands work end-to-end
- [ ] User calibration workflow functional
- [ ] Clinical preprocessing mode works
- [ ] Adaptive thresholding improves accuracy
