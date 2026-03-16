import pygame

# Screen Dimensions
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 700  # Increased for buttons
FPS = 60
VIRTUAL_FPS = 1.0     # Virtual frames per second
VIRTUAL_FRAME_INTERVAL = 1000  # ms

# Grid / Board Dimensions
CELL_SIZE = 32
BOARD_WIDTH = 6
BOARD_HEIGHT = 14  # Visible is 13, but include hidden row
VISIBLE_HEIGHT = 13

# Player Board Offsets
P1_OFFSET_X = 50
P1_OFFSET_Y = 50
P2_OFFSET_X = 500
P2_OFFSET_Y = 50

# Colors (RGB)
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
    "GhostAlpha": 160  # Alpha for held pieces
}

# Symbols for buttons
BUTTON_SYMBOLS = {
    "left": "←",
    "right": "→",
    "rot_r": "↻",
    "rot_l": "↺",
    "drop": "↓"
}

# Next / Double Next offsets relative to board top
NEXT_OFFSET_X = 220
NEXT_OFFSET_Y = 20
DOUBLE_NEXT_OFFSET_X = 220
DOUBLE_NEXT_OFFSET_Y = 100

# Button Layouts
BUTTON_WIDTH = 60
BUTTON_HEIGHT = 40
BUTTON_MARGIN = 5

RESTART_BUTTON_RECT = (350, 20, 100, 40)

# 1P Control Buttons (X, Y, Label, ID)
P1_CONTROL_BUTTONS = [
    (50, 550, "Left", "left"),
    (115, 550, "Right", "right"),
    (180, 550, "Rot R", "rot_r"),
    (245, 550, "Rot L", "rot_l"),
    (310, 550, "Drop", "drop"),
]

# 2P Control Buttons
P2_CONTROL_BUTTONS = [
    (500, 550, "Left", "left"),
    (565, 550, "Right", "right"),
    (630, 550, "Rot R", "rot_r"),
    (695, 550, "Rot L", "rot_l"),
    (760, 550, "Drop", "drop"),
]

# Input mapping configurations could go here as well
import puyotan_native as p
COLOR_MAP = {
    p.Cell.Red: COLORS["Red"],
    p.Cell.Green: COLORS["Green"],
    p.Cell.Blue: COLORS["Blue"],
    p.Cell.Yellow: COLORS["Yellow"],
    p.Cell.Ojama: COLORS["Ojama"],
    p.Cell.Empty: COLORS["Empty"]
}
