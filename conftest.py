"""Ensure the repository root is importable when running the tests."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
