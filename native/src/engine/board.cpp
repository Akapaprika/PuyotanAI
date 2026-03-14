#include "engine/board.hpp"

namespace puyotan {

// ============================================================
// Cell-level read
//   Scan all 5 color boards; return the first hit.
//   Loop is short (5 iterations) and branch-free inside each
//   BitBoard::get() — the compiler typically unrolls it fully.
// ============================================================
Cell Board::get(int x, int y) const {
    for (int i = 0; i < kNumCellColors; ++i) {
        if (boards_[i].get(x, y))
            return static_cast<Cell>(i + 1); // +1 because Cell::Empty==0
    }
    return Cell::Empty;
}

// ============================================================
// Cell-level write
//   Clear the cell first (all color planes), then set only the
//   target plane.  Uses array index instead of switch.
// ============================================================
void Board::set(int x, int y, Cell color) {
    clear(x, y); // reset all planes at this cell
    if (color != Cell::Empty)
        boards_[idx(color)].set(x, y);
}

// ============================================================
// Cell-level clear
//   Clear the bit for (x, y) in every color plane.
//   No branches: every plane is unconditionally visited.
//   This is safe because at most one plane can be set per cell.
// ============================================================
void Board::clear(int x, int y) {
    for (auto& bb : boards_)
        bb.clear(x, y);
}

// ============================================================
// Place a puyo at the invisible spawn row (row 13, 0-indexed).
//   Bounds check: valid columns are 0 through kWidth-1.
//   After calling this, run Gravity::execute() to drop the piece.
// ============================================================
void Board::place_piece(int col, Cell color) {
    // column must be in [0, kWidth)
    if (static_cast<unsigned>(col) < static_cast<unsigned>(config::Board::kWidth))
        set(col, config::Board::kSpawnRow, color);
}

// ============================================================
// BitBoard-level read/write (used by Gravity, Chain, etc.)
//   Direct array indexing – no switch, no branch.
// ============================================================
const BitBoard& Board::get_bitboard(Cell color) const {
    return boards_[idx(color)];
}

void Board::set_bitboard(Cell color, const BitBoard& bb) {
    boards_[idx(color)] = bb;
}

} // namespace puyotan
