#include <puyotan/env/rl_utils.hpp>
#include <vector>

namespace puyotan {

pybind11::array_t<float> computeGae(
    pybind11::array_t<float> rewards,
    pybind11::array_t<float> values,
    pybind11::array_t<float> dones,
    pybind11::array_t<float> next_value,
    float gamma,
    float lam) {
    
    pybind11::buffer_info rt_info = rewards.request();
    int num_steps = static_cast<int>(rt_info.shape[0]);
    int num_envs = static_cast<int>(rt_info.shape[1]);

    auto r_ptr = static_cast<const float*>(rt_info.ptr);
    auto v_ptr = static_cast<const float*>(values.request().ptr);
    auto d_ptr = static_cast<const float*>(dones.request().ptr);
    auto nv_ptr = static_cast<const float*>(next_value.request().ptr);

    pybind11::array_t<float> advantages({num_steps, num_envs});
    auto adv_ptr = static_cast<float*>(advantages.mutable_data());

    std::vector<float> last_gae(num_envs, 0.0f);
    float gamma_lam = gamma * lam;

    // Time loop must be sequential because each advantage 'adv[t]' depends on 'last_gae' 
    // calculated from 'adv[t+1]'. This is the standard backward pass for GAE.
    for (int t = num_steps - 1; t >= 0; --t) {
        const float* curr_r = r_ptr + t * num_envs;
        const float* curr_v = v_ptr + t * num_envs;
        const float* curr_d = d_ptr + t * num_envs;
        const float* next_v = (t == num_steps - 1) ? nv_ptr : (v_ptr + (t + 1) * num_envs);
        float* curr_adv = adv_ptr + t * num_envs;

        // OpenMP for batch environment computations
        #pragma omp parallel for
        for (int i = 0; i < num_envs; ++i) {
            float not_done = 1.0f - curr_d[i];
            float delta = curr_r[i] + gamma * next_v[i] * not_done - curr_v[i];
            last_gae[i] = delta + gamma_lam * not_done * last_gae[i];
            curr_adv[i] = last_gae[i];
        }
    }

    return advantages;
}

} // namespace puyotan
