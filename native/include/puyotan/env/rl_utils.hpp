#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace puyotan {

/**
 * @brief Computes Generalized Advantage Estimation (GAE) with high performance.
 * 
 * Migrated from Python to C++ to eliminate loop overhead in RL training.
 * Uses OpenMP for parallelizing across environment dimensions.
 * 
 * @param rewards tensor of shape [num_steps, num_envs].
 * @param values tensor of shape [num_steps, num_envs].
 * @param dones tensor of shape [num_steps, num_envs].
 * @param next_value tensor of shape [num_envs].
 * @param gamma Discount factor.
 * @param lam GAE smoothing parameter.
 * @return Advantage tensor of shape [num_steps, num_envs].
 */
pybind11::array_t<float> computeGae(
    pybind11::array_t<float> rewards,
    pybind11::array_t<float> values,
    pybind11::array_t<float> dones,
    pybind11::array_t<float> next_value,
    float gamma,
    float lam);

} // namespace puyotan
