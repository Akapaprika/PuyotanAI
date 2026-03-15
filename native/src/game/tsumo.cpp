#include <puyotan/game/tsumo.hpp>
#include <cmath>
#include <algorithm>

namespace puyotan {

Tsumo::Tsumo(uint32_t seed) : seed_(seed) {
    fillPool();
}

void Tsumo::setSeed(uint32_t seed) {
    seed_ = seed;
    fillPool();
}

PuyoPiece Tsumo::get(int index) const {
    if (pool_.empty()) {
        return PuyoPiece{Cell::Empty, Cell::Empty};
    }
    int idx = index % config::Rule::kTsumoPoolSize;
    if (idx < 0) {
        idx += config::Rule::kTsumoPoolSize;
    }
    return pool_[idx];
}

int Tsumo::nextInt(int max) {
    int32_t signed_y = static_cast<int32_t>(seed_);
    signed_y ^= (signed_y << 13);
    signed_y ^= (signed_y >> 17);
    signed_y ^= (signed_y << 15);
    seed_ = static_cast<uint32_t>(signed_y);
    
    int r = std::abs(signed_y);
    return r % max;
}

Cell Tsumo::nextKind() {
    switch (nextInt(config::Rule::kColors)) {
        case 0:  return Cell::Red;
        case 1:  return Cell::Green;
        case 2:  return Cell::Blue;
        case 3:  return Cell::Yellow;
        default: return Cell::Empty;
    }
}

void Tsumo::fillPool() {
    pool_.clear();
    pool_.reserve(config::Rule::kTsumoPoolSize);
    for (int i = 0; i < config::Rule::kTsumoPoolSize; ++i) {
        Cell axis = nextKind();
        Cell sub = nextKind();
        pool_.emplace_back(axis, sub);
    }
}

} // namespace puyotan
