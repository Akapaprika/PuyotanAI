#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace puyotan {

/**
 * Computes Generalized Advantage Estimation (GAE) entirely in C++ 
 * to eliminate Python loop overhead. Returns the advantage tensor.
 * 
 * Expected shapes:
 *   rewards: [num_steps, num_envs]
 *   values: [num_steps, num_envs]
 *   dones: [num_steps, num_envs]
 *   next_value: [num_envs]
 */
pybind11::array_t<float> computeGae(
    pybind11::array_t<float> rewards,
    pybind11::array_t<float> values,
    pybind11::array_t<float> dones,
    pybind11::array_t<float> next_value,
    float gamma,
    float lam);

} // namespace puyotan
