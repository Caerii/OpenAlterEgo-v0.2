"""Tests for user management system."""

import json
import tempfile
import unittest
from pathlib import Path

from openalterego.users.profile import UserProfile
from openalterego.users.manager import UserManager


class TestUserProfile(unittest.TestCase):
    def test_create_profile(self):
        """Test creating a user profile."""
        profile = UserProfile(user_id="test_user")
        self.assertEqual(profile.user_id, "test_user")
        self.assertEqual(profile.confidence_threshold, 0.70)
        self.assertEqual(profile.preprocessing_mode, "standard")

    def test_profile_validation(self):
        """Test profile validation."""
        # Invalid confidence threshold
        with self.assertRaises(ValueError):
            UserProfile(user_id="test", confidence_threshold=1.5)
        
        # Invalid window_ms
        with self.assertRaises(ValueError):
            UserProfile(user_id="test", window_ms=0)
        
        # Invalid stride_ms
        with self.assertRaises(ValueError):
            UserProfile(user_id="test", stride_ms=1000, window_ms=600)

    def test_profile_serialization(self):
        """Test profile to/from dict."""
        profile = UserProfile(
            user_id="test",
            confidence_threshold=0.85,
            preprocessing_mode="clinical",
        )
        d = profile.to_dict()
        profile2 = UserProfile.from_dict(d)
        
        self.assertEqual(profile.user_id, profile2.user_id)
        self.assertEqual(profile.confidence_threshold, profile2.confidence_threshold)
        self.assertEqual(profile.preprocessing_mode, profile2.preprocessing_mode)
    
    def test_profile_wide_mode(self):
        """Test profile with wide preprocessing mode."""
        profile = UserProfile(
            user_id="test_wide",
            preprocessing_mode="wide",
        )
        self.assertEqual(profile.preprocessing_mode, "wide")
        
        # Test serialization with wide mode
        d = profile.to_dict()
        profile2 = UserProfile.from_dict(d)
        self.assertEqual(profile2.preprocessing_mode, "wide")


class TestUserManager(unittest.TestCase):
    def setUp(self):
        """Set up temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = UserManager(base_dir=Path(self.temp_dir))

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_create_and_save_profile(self):
        """Test creating and saving a profile."""
        profile = UserProfile(user_id="alice")
        self.manager.save_profile(profile)
        
        # Check file exists
        profile_path = self.manager.get_profile_path("alice")
        self.assertTrue(profile_path.exists())
        
        # Check JSON is valid
        with open(profile_path, "r") as f:
            data = json.load(f)
        self.assertEqual(data["user_id"], "alice")

    def test_load_profile(self):
        """Test loading a profile."""
        profile = UserProfile(user_id="bob", confidence_threshold=0.80)
        self.manager.save_profile(profile)
        
        loaded = self.manager.load_profile("bob")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.user_id, "bob")
        self.assertEqual(loaded.confidence_threshold, 0.80)

    def test_load_nonexistent_profile(self):
        """Test loading non-existent profile."""
        loaded = self.manager.load_profile("nonexistent")
        self.assertIsNone(loaded)

    def test_list_users(self):
        """Test listing users."""
        # No users initially
        self.assertEqual(len(self.manager.list_users()), 0)
        
        # Create some users
        self.manager.save_profile(UserProfile(user_id="user1"))
        self.manager.save_profile(UserProfile(user_id="user2"))
        
        users = self.manager.list_users()
        self.assertEqual(len(users), 2)
        self.assertIn("user1", users)
        self.assertIn("user2", users)

    def test_user_exists(self):
        """Test checking if user exists."""
        self.assertFalse(self.manager.user_exists("test"))
        
        self.manager.save_profile(UserProfile(user_id="test"))
        self.assertTrue(self.manager.user_exists("test"))

    def test_get_user_dir(self):
        """Test getting user directory."""
        user_dir = self.manager.get_user_dir("test")
        self.assertTrue(user_dir.exists())
        self.assertTrue(user_dir.is_dir())

    def test_delete_user(self):
        """Test deleting a user."""
        profile = UserProfile(user_id="delete_me")
        self.manager.save_profile(profile)
        self.assertTrue(self.manager.user_exists("delete_me"))
        
        self.manager.delete_user("delete_me")
        self.assertFalse(self.manager.user_exists("delete_me"))


if __name__ == "__main__":
    unittest.main()
