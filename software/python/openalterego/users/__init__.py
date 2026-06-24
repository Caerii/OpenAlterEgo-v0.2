"""User management and personalization for OpenAlterEgo."""

from .calibration import CalibrationConfig, CalibrationReport, calibrate_user
from .collect import collect_from_ble, collect_from_sim
from .defaults import default_users_dir
from .manager import UserManager
from .profile import UserProfile

__all__ = [
    "UserProfile",
    "UserManager",
    "default_users_dir",
    "CalibrationConfig",
    "CalibrationReport",
    "calibrate_user",
    "collect_from_sim",
    "collect_from_ble",
]
