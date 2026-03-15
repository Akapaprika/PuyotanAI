#pragma once

#include <vector>
#include <cstdint>
#include <puyotan/core/board.hpp>

namespace puyotan {

/**
 * Tsumo
 *   Generates and maintains a pool of puyo pairs.
 */
class Tsumo {
public:
    explicit Tsumo(uint32_t seed = 0);

    PuyoPiece get(int index) const;
    void setSeed(uint32_t seed);
    uint32_t getSeed() const { return seed_; }

private:
    uint32_t seed_;
    std::vector<PuyoPiece> pool_;

    int nextInt(int max);
    Cell nextKind();
    void fillPool();
};

} // namespace puyotan
