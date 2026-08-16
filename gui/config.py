from pathlib import Path

# ---------------------------------------------------------------------------
# Config file path (single source of truth for the whole project)
# ---------------------------------------------------------------------------
CONFIG_PATH = str(Path(__file__).parent.parent / "native" / "resources" / "beam_config.json")

# Screen Dimensions
WINDOW_WIDTH = 750
WINDOW_HEIGHT = 700
FPS = 60
VIRTUAL_FRAME_INTERVAL_MS = 100
RANDOM_SEED = 1

# Grid / Board Dimensions
CELL_SIZE = 26
BOARD_WIDTH = 6
BOARD_HEIGHT = 14  # Visible is 13, but include hidden row
VISIBLE_HEIGHT = 13

# Standardized Color Palette (RGB tuples)
COLORS = {
    "Red": (255, 60, 60),
    "Green": (60, 220, 80),
    "Blue": (60, 120, 255),
    "Yellow": (255, 230, 50),
    "Ojama": (170, 170, 170),
    "Empty": (0, 0, 0),
    "Background": (22, 33, 62),
    "Grid": (60, 60, 80),
    "Text": (180, 190, 210),
    "Button": (70, 70, 90),
    "ButtonHover": (80, 80, 80),
    "ButtonText": (255, 255, 255),
    "GhostAlpha": 120
}

# Symbols for buttons
BUTTON_SYMBOLS = {
    "left": "←",
    "right": "→",
    "rot_r": "↻",
    "rot_l": "↺",
    "drop": "↓"
}
