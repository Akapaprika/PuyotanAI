"""
gui/views/board_widget.py

Custom QWidget that renders the full state of a single player's puyo board
using QPainter. This widget is entirely self-contained; it receives plain
BoardSnapshot dataclass objects and has no knowledge of the ViewModel or engine.
"""
from __future__ import annotations
from dataclasses import dataclass
import dataclasses
from typing import List, Tuple, Optional, Any
import puyotan_native as p

from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPainterPath
from PyQt6.QtCore import Qt, QRect, QRectF, QSize


# ---------------------------------------------------------------------------
# Colour mapping: engine Cell → Qt colour.
# Lives here so the ViewModel has zero rendering knowledge.
# ---------------------------------------------------------------------------
CELL_COLORS: dict[p.Cell, QColor] = {
    p.Cell.Red:    QColor(255,  60,  60),
    p.Cell.Green:  QColor( 60, 220,  80),
    p.Cell.Blue:   QColor( 60, 120, 255),
    p.Cell.Yellow: QColor(255, 230,  50),
    p.Cell.Ojama:  QColor(170, 170, 170),
    p.Cell.Empty:  QColor(  0,   0,   0, 0),
}

GHOST_ALPHA = 120          # alpha for the ghost (pending decision) piece

# ---------------------------------------------------------------------------
# Plain data objects — no Qt, no engine types.
# The ViewModel produces these; the widget consumes them.
# ---------------------------------------------------------------------------
@dataclass
class PieceSpec:
    """Colours for the two puyos of a single piece."""
    axis_color: QColor
    sub_color:  QColor

@dataclass
class BoardSnapshot:
    """All data the BoardWidget needs to paint one frame."""
    field:          Any          = None          # puyotan_native field object
    next1:          PieceSpec  = dataclasses.field(default_factory=lambda: PieceSpec(QColor(), QColor()))
    next2:          PieceSpec  = dataclasses.field(default_factory=lambda: PieceSpec(QColor(), QColor()))
    ghost_x:        int        = 2
    ghost_rot:      Any        = None            # p.Rotation value or None
    ghost_axis_col: QColor     = dataclasses.field(default_factory=QColor)
    ghost_sub_col:  QColor     = dataclasses.field(default_factory=QColor)
    show_ghost:     bool       = False
    score:          int        = 0
    non_active_ojama: int      = 0
    active_ojama:   int        = 0
    state:          str        = "WAITING"


def piece_spec_from_native(piece, cell_colors: dict) -> PieceSpec:
    """Convert a puyotan_native piece to a PieceSpec."""
    return PieceSpec(
        axis_color=cell_colors.get(piece.axis, QColor(255, 255, 255)),
        sub_color= cell_colors.get(piece.sub,  QColor(255, 255, 255)),
    )


# ---------------------------------------------------------------------------
# The widget itself
# ---------------------------------------------------------------------------
_BOARD_COLS   = 6
_BOARD_ROWS   = 13   # visible grid rows
_SKY_ROWS     = 3    # dedicated space above the grid
_GRID_COLOR   = QColor(60, 60, 80)
_BG_COLOR     = QColor(22, 33, 62)
_TEXT_COLOR   = QColor(180, 190, 210)


class BoardWidget(QWidget):
    """
    Renders a single player's puyo field.

    Layout (left-to-right inside the widget):
        | margin | board (6×13 cells) | margin | next panel | margin |
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._snapshot: Optional[BoardSnapshot] = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(QSize(200, 380))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_snapshot(self, snapshot: BoardSnapshot) -> None:
        self._snapshot = snapshot
        self.update()   # schedules a repaint

    # ------------------------------------------------------------------
    # Qt paint event
    # ------------------------------------------------------------------
    def paintEvent(self, _event):
        if self._snapshot is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        cell = self._cell_size(w, h)
        board_w = cell * _BOARD_COLS
        # Vertical layout: [SKY (2 rows)] + [GRID (13 rows)] + [INFO (approx 1.5 rows)]
        total_grid_h = cell * (_BOARD_ROWS + _SKY_ROWS)

        margin_x = max(10, (w - board_w - cell * 2 - 24) // 2)
        margin_y = max(10, (h - total_grid_h - 40) // 2)

        # Paint background
        painter.fillRect(0, 0, w, h, _BG_COLOR)

        snap = self._snapshot
        bx = margin_x + 20 # space for row labels
        by = margin_y + (cell * _SKY_ROWS) # Top of the 13-row grid

        self._draw_labels(painter, bx, by, cell)
        self._draw_grid(painter, bx, by, cell)
        self._draw_cells(painter, snap.field, bx, by, cell)
        if snap.show_ghost:
            self._draw_ghost(painter, snap, bx, by, cell)
        # Next panel starts at the top of the sky area
        self._draw_next_panel(painter, snap, bx + board_w + 12, by - (cell * (_SKY_ROWS - 1)), cell)
        self._draw_info(painter, snap, bx, by + (cell * _BOARD_ROWS) + 20, board_w)

        painter.end()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _cell_size(self, w: int, h: int) -> int:
        by_w = (w - 60) // (_BOARD_COLS + 3)
        by_h = (h - 80) // (_BOARD_ROWS + _SKY_ROWS)
        return max(16, min(by_w, by_h))

    def _cell_rect(self, col: int, row: int, ox: int, oy: int, cell: int) -> QRectF:
        """Returns QRectF for a board cell (row 0 = bottom)."""
        x = ox + col * cell
        y = oy + (_BOARD_ROWS - 1 - row) * cell
        return QRectF(x, y, cell, cell)

    def _draw_labels(self, painter: QPainter, ox: int, oy: int, cell: int):
        """Draw column indices (1-6) and row indices (1-13)."""
        font = QFont("Segoe UI", max(7, cell // 3))
        painter.setFont(font)
        painter.setPen(QPen(_TEXT_COLOR))

        # Column labels (bottom) - 1-indexed
        for col in range(_BOARD_COLS):
            r = self._cell_rect(col, 0, ox, oy, cell)
            painter.drawText(QRectF(r.left(), r.bottom() + 2, cell, cell // 2 + 4),
                             Qt.AlignmentFlag.AlignCenter, str(col + 1))

        # Row labels (left) - 1-indexed, 1 is bottom
        for row in range(_BOARD_ROWS):
            r = self._cell_rect(0, row, ox, oy, cell)
            painter.drawText(QRectF(ox - 18, r.top(), 16, cell),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, str(row + 1))

    def _draw_grid(self, painter: QPainter, ox: int, oy: int, cell: int):
        pen = QPen(_GRID_COLOR, 1)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for row in range(_BOARD_ROWS):
            for col in range(_BOARD_COLS):
                r = self._cell_rect(col, row, ox, oy, cell)
                painter.drawRect(r)

    def _draw_cells(self, painter: QPainter, field, ox: int, oy: int, cell: int):
        for row in range(_BOARD_ROWS):
            for col in range(_BOARD_COLS):
                c = field.get(col, row)
                if c == p.Cell.Empty:
                    continue
                color = CELL_COLORS.get(c, QColor(255, 255, 255))
                self._draw_puyo(painter, col, row, ox, oy, cell, color)

    def _draw_puyo(self, painter: QPainter, col: int, row: int,
                   ox: int, oy: int, cell: int, color: QColor, alpha: int = 255):
        r = self._cell_rect(col, row, ox, oy, cell)
        padding = max(1, cell // 10)
        inner = r.adjusted(padding, padding, -padding, -padding)
        c = QColor(color)
        c.setAlpha(alpha)
        # SIMPLE FLAT STYLE (Reduced load)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(c))
        painter.drawEllipse(inner)

    def _draw_ghost(self, painter: QPainter, snap: BoardSnapshot,
                    ox: int, oy: int, cell: int):
        """Draw the 'pending' piece ghost above the board."""
        x = snap.ghost_x
        rot = snap.ghost_rot
        # DRAW GHOST IN SKY AREA:
        # Axis at 14.51. Sub-puyo (Down) at 13.51.
        # Bottom of piece at 13.01. Grid border at 13.0.
        # Microscopic 0.01 cell gap for absolute tightness.
        ax, ay = x, _BOARD_ROWS + 1.51
        sx, sy = ax, ay

        if rot == p.Rotation.Up:
            sy += 1
        elif rot == p.Rotation.Right:
            sx += 1
        elif rot == p.Rotation.Down:
            sy -= 1
        elif rot == p.Rotation.Left:
            sx -= 1

        def draw_ghost_puyo(col, row, color: QColor):
            # Coordinates are allowed to be in the sky (-1 or -2 relative to top)
            # Row index matches our grid system where 12 = top row.
            r = self._cell_rect(col, row, ox, oy, cell)
            padding = max(1, cell // 10)
            inner = r.adjusted(padding, padding, -padding, -padding)
            c = QColor(color)
            c.setAlpha(GHOST_ALPHA)
            pen = QPen(c, 2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(inner)

        draw_ghost_puyo(ax, ay, snap.ghost_axis_col)
        draw_ghost_puyo(sx, sy, snap.ghost_sub_col)

    def _draw_next_panel(self, painter: QPainter, snap: BoardSnapshot,
                         nx: int, ny: int, cell: int):
        """Draw NEXT and DOUBLE-NEXT pieces to the right of the board."""
        font = QFont("Segoe UI", max(7, cell // 3))
        painter.setFont(font)
        painter.setPen(QPen(_TEXT_COLOR))

        def draw_label(text: str, y: int):
            painter.drawText(nx, y, text)

        def draw_static(spec: PieceSpec, px: int, py: int):
            # sub on top, axis below (standard puyo orientation)
            sub_r = QRectF(px, py - cell, cell * 0.8, cell * 0.8)
            ax_r  = QRectF(px, py,        cell * 0.8, cell * 0.8)
            padding = max(1, cell // 10)
            for r, col in [(sub_r, spec.sub_color), (ax_r, spec.axis_color)]:
                inner = r.adjusted(padding, padding, -padding, -padding)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(col))
                painter.drawEllipse(inner)

        lh = max(12, cell // 2)
        draw_label("NEXT",   ny + lh)
        draw_static(snap.next1, nx, ny + lh + cell + 4)

        draw_label("2nd",    ny + lh + cell * 2 + 16)
        draw_static(snap.next2, nx, ny + lh + cell * 3 + 20)

    def _draw_info(self, painter: QPainter, snap: BoardSnapshot,
                   bx: int, by: int, board_w: int):
        """Draw score, ojama count, and a small thinking badge below the board."""
        font = QFont("Segoe UI", max(8, 11))
        painter.setFont(font)
        painter.setPen(QPen(_TEXT_COLOR))
        painter.drawText(bx, by + 14, f"Score: {snap.score}")
        painter.drawText(bx, by + 28, f"Ojama: {snap.non_active_ojama} / {snap.active_ojama}")

        # Small status badge - Shifted further right for 4-digit score/ojama safety
        st = snap.state
        bg = "#374151"
        fg = "#9ca3af"
        if st == "THINKING": bg, fg = "#fffb0a", "#1a1a00"
        elif st == "READY": bg, fg = "#16a34a", "#ffffff"
        elif st == "LOCKED": bg, fg = "#dc2626", "#ffffff"

        badge_w = 58
        # Shifted +30px further right to clear long score/ojama numbers
        badge_rect = QRectF(bx + board_w - badge_w + 30, by + 2, badge_w, 18)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(bg)))
        painter.drawRoundedRect(badge_rect, 3, 3)
        
        painter.setPen(QPen(QColor(fg)))
        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, st)
