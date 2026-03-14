#include "engine/tsumo.hpp"
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
        return PuyoPiece();
    }
    // Match Python/JS negative index behavior or just wrap around
    int idx = index % config::Rule::kTsumoPoolSize;
    if (idx < 0) {
        idx += config::Rule::kTsumoPoolSize;
    }
    return pool_[idx];
}

uint32_t Tsumo::next() {
    // Random.js next():
    // this.y = this.y ^ (this.y << 13)
    // this.y = this.y ^ (this.y >> 17)
    // this.y = this.y ^ (this.y << 15)
    // return this.y >>> 0
    seed_ ^= (seed_ << 13);
    seed_ ^= (seed_ >> 17);
    seed_ ^= (seed_ << 15);
    return seed_;
}

int Tsumo::nextInt(int max) {
    // Random.js nextInt(max):
    // const r = Math.abs(this.next())
    // return r % max
    
    // In C++, if we treat next() as uint32_t, its "absolute value" 
    // depends on whether JS treats it as signed.
    // Random.js: this.y is likely treated as signed 32-bit int by ^, <<, >>.
    // In JS, bitwise ops operate on 32-bit signed ints.
    
    int32_t signed_y = static_cast<int32_t>(seed_);
    // Equivalent to Random.js: this.y ^ (this.y << 13) ...
    signed_y ^= (signed_y << 13);
    signed_y ^= (signed_y >> 17);
    signed_y ^= (signed_y << 15);
    seed_ = static_cast<uint32_t>(signed_y);
    
    // nextInt uses Math.abs(signed_y) % max
    int r = std::abs(signed_y);
    return r % max;
}

Cell Tsumo::nextKind() {
    // Next._nextKind() calls nextInt(Rule::kColors)
    switch (nextInt(config::Rule::kColors)) {
        case 0: return Cell::Red;
        case 1: return Cell::Green;
        case 2: return Cell::Blue;
        case 3: return Cell::Yellow;
        default: return Cell::Empty; // Should not happen
    }
}

void Tsumo::fillPool() {
    pool_.clear();
    pool_.reserve(config::Rule::kTsumoPoolSize);
    
    // We need a temporary copy of the seed or we just update the member seed_
    // In JS, constructor takes seed and next() updates this.y.
    // Next.init() uses the random object.
    
    for (int i = 0; i < config::Rule::kTsumoPoolSize; ++i) {
        Cell axis = nextKind();
        Cell sub = nextKind();
        pool_.emplace_back(axis, sub);
    }
}

} // namespace puyotan
