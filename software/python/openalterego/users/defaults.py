"""Shared defaults for user data locations."""

from __future__ import annotations

from pathlib import Path


def default_users_dir() -> Path:
    """Directory for :class:`UserManager` when not explicitly configured."""
    return Path.cwd() / "users"
