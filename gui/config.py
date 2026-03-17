# Screen Dimensions
WINDOW_WIDTH = 750
WINDOW_HEIGHT = 700
FPS = 60
VIRTUAL_FPS = 1.0     # Virtual frames per second
VIRTUAL_FRAME_INTERVAL_MS = 100

# Grid / Board Dimensions
CELL_SIZE = 26
BOARD_WIDTH = 6
BOARD_HEIGHT = 14  # Visible is 13, but include hidden row
VISIBLE_HEIGHT = 13

# Colors (RGB) purely for blending logic or defaults
COLORS = {
    "Red": (255, 50, 50),
    "Green": (50, 255, 50),
    "Blue": (50, 50, 255),
    "Yellow": (255, 255, 50),
    "Ojama": (150, 150, 150),
    "Empty": (0, 0, 0),
    "Background": (30, 30, 40),
    "Grid": (60, 60, 60),
    "Text": (255, 255, 255),
    "Button": (70, 70, 90),
    "ButtonHover": (80, 80, 80),
    "ButtonText": (255, 255, 255),
    "GhostAlpha": 160
}

# Symbols for buttons
BUTTON_SYMBOLS = {
    "left": "←",
    "right": "→",
    "rot_r": "↻",
    "rot_l": "↺",
    "drop": "↓"
}
