"""User manager for handling user profiles and data directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .profile import UserProfile


class UserManager:
    """Manages user profiles and data directories.
    
    This class provides a simple file-based storage system for user profiles.
    Each user has a dedicated directory containing:
    - profile.json: User profile metadata
    - model.pt: Trained model (if available)
    - calibration/: Calibration data (if available)
    
    Example:
        manager = UserManager(base_dir=Path("./users"))
        profile = manager.load_profile("alice")
        if profile is None:
            profile = UserProfile(user_id="alice")
            manager.save_profile(profile)
    """
    
    def __init__(self, base_dir: Path | str):
        """Initialize user manager.
        
        Parameters
        ----------
        base_dir:
            Base directory for storing user data
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_user_dir(self, user_id: str) -> Path:
        """Get user's data directory.
        
        Parameters
        ----------
        user_id:
            User identifier
            
        Returns
        -------
        Path to user's directory
        """
        if not user_id or not isinstance(user_id, str):
            raise ValueError("user_id must be a non-empty string")
        user_dir = self.base_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir
    
    def get_profile_path(self, user_id: str) -> Path:
        """Get path to user's profile JSON file."""
        return self.get_user_dir(user_id) / "profile.json"
    
    def load_profile(self, user_id: str) -> Optional[UserProfile]:
        """Load user profile from disk.
        
        Parameters
        ----------
        user_id:
            User identifier
            
        Returns
        -------
        UserProfile if found, None otherwise
        """
        profile_path = self.get_profile_path(user_id)
        if not profile_path.exists():
            return None
        
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return UserProfile.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(f"Failed to load profile for user {user_id}: {e}") from e
    
    def save_profile(self, profile: UserProfile) -> None:
        """Save user profile to disk.
        
        Parameters
        ----------
        profile:
            UserProfile to save
        """
        profile_path = self.get_profile_path(profile.user_id)
        
        # Ensure user directory exists
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile.to_dict(), f, indent=2)
    
    def list_users(self) -> List[str]:
        """List all registered users.
        
        Returns
        -------
        List of user IDs
        """
        if not self.base_dir.exists():
            return []
        
        users = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and (item / "profile.json").exists():
                users.append(item.name)
        
        return sorted(users)
    
    def user_exists(self, user_id: str) -> bool:
        """Check if user exists.
        
        Parameters
        ----------
        user_id:
            User identifier
            
        Returns
        -------
        True if user profile exists
        """
        return self.get_profile_path(user_id).exists()
    
    def delete_user(self, user_id: str) -> None:
        """Delete user and all associated data.
        
        Parameters
        ----------
        user_id:
            User identifier to delete
            
        Warning
        -------
        This permanently deletes all user data. Use with caution.
        """
        user_dir = self.base_dir / user_id
        if user_dir.exists():
            import shutil
            shutil.rmtree(user_dir)
