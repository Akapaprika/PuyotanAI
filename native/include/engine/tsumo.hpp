#pragma once

#include <vector>
#include <cstdint>
#include "engine/board.hpp"
#include "config/engine_config.hpp"

namespace puyotan {

/**
 * PuyoPiece
 *   Represents a pair of puyos (tsumo).
 */
struct PuyoPiece {
    Cell axis;
    Cell sub;

    PuyoPiece() : axis(Cell::Empty), sub(Cell::Empty) {}
    PuyoPiece(Cell a, Cell s) : axis(a), sub(s) {}
};

/**
 * Tsumo
 *   Generates and maintains a pool of puyo pairs using Xorshift32.
 *   Matches the original Puyotan logic exactly.
 */
class Tsumo {
public:
    /**
     * @param seed The random seed for Xorshift32.
     */
    explicit Tsumo(uint32_t seed = 0);

    /**
     * Returns the pair at the specified index from the pool.
     * Uses modulo kTsumoPoolSize internally.
     */
    PuyoPiece get(int index) const;

    /**
     * Resets the generator with a new seed and refills the pool.
     */
    void setSeed(uint32_t seed);

    /**
     * Returns the current seed.
     */
    uint32_t getSeed() const { return seed_; }

private:
    uint32_t seed_;
    std::vector<PuyoPiece> pool_;

    /**
     * Xorshift32 implementation matching Random.js next()
     */
    uint32_t next();

    /**
     * Returns an integer in range [0, max-1]
     */
    int nextInt(int max);

    /**
     * Returns a random puyo kind (Red, Green, Blue, or Yellow)
     */
    Cell nextKind();

    /**
     * Refills the pool with 1000 pieces.
     */
    void fillPool();
};

} // namespace puyotan
