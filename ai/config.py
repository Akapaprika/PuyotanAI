"""
ai/config.py

Central configuration path and constants for AI search agents.
"""
from pathlib import Path

# Single source of truth for beam search config JSON across GUI, Bot, and Tests
CONFIG_PATH = str(Path(__file__).parent.parent / "native" / "resources" / "beam_config.json")
