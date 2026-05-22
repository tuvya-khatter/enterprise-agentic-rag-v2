"""Shared pytest fixtures."""
import os
import sys
from pathlib import Path

# Ensure src imports work when running pytest from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Point to a safe test JWT secret
os.environ.setdefault("JWT_SECRET", "test-secret-key-for-pytest-only")
