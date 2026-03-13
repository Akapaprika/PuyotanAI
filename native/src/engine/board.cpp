#include "engine/board.hpp"

namespace puyotan {

Cell Board::get(int x, int y) const {
    if (board_red_.get(x, y)) return Cell::Red;
    if (board_green_.get(x, y)) return Cell::Green;
    if (board_blue_.get(x, y)) return Cell::Blue;
    if (board_yellow_.get(x, y)) return Cell::Yellow;
    if (board_ojama_.get(x, y)) return Cell::Ojama;
    return Cell::Empty;
}

void Board::set(int x, int y, Cell color) {
    // First clear any existing color
    clear(x, y);

    switch (color) {
        case Cell::Red: board_red_.set(x, y); break;
        case Cell::Green: board_green_.set(x, y); break;
        case Cell::Blue: board_blue_.set(x, y); break;
        case Cell::Yellow: board_yellow_.set(x, y); break;
        case Cell::Ojama: board_ojama_.set(x, y); break;
        case Cell::Empty: break;
    }
}

void Board::clear(int x, int y) {
    board_red_.clear(x, y);
    board_green_.clear(x, y);
    board_blue_.clear(x, y);
    board_yellow_.clear(x, y);
    board_ojama_.clear(x, y);
}

const BitBoard& Board::get_bitboard(Cell color) const {
    switch (color) {
        case Cell::Red: return board_red_;
        case Cell::Green: return board_green_;
        case Cell::Blue: return board_blue_;
        case Cell::Yellow: return board_yellow_;
        case Cell::Ojama: return board_ojama_;
        default: {
            static BitBoard empty_board;
            return empty_board;
        }
    }
}

}
