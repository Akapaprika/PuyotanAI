from pathlib import Path
from ai.config import CONFIG_PATH

# Screen Dimensions & Timing
WINDOW_WIDTH = 750
WINDOW_HEIGHT = 700
FPS = 60
VIRTUAL_FRAME_INTERVAL_MS = 100
RANDOM_SEED = 1

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
    "GhostAlpha": 120,
}

# Mapping from puyotan_native.Cell to RGB tuples
def get_cell_rgb():
    import puyotan_native as p
    return {
        p.Cell.Red: COLORS["Red"],
        p.Cell.Green: COLORS["Green"],
        p.Cell.Blue: COLORS["Blue"],
        p.Cell.Yellow: COLORS["Yellow"],
        p.Cell.Ojama: COLORS["Ojama"],
        p.Cell.Empty: COLORS["Empty"],
    }

