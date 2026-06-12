#pragma once

#include <fstream>
#include <string>
#include <filesystem>
#include <external/nlohmann/json.hpp>
#include <puyotan/search/beam_search.hpp>

namespace puyotan::search {

/**
 * @class BeamConfigLoader
 * @brief Loads and patches BeamConfig from/to a JSON file with static in-memory caching.
 */
class BeamConfigLoader {
  private:
    static inline nlohmann::json s_cached_json;
    static inline std::filesystem::file_time_type s_last_write_time;
    static inline std::string s_cached_path;
    static inline bool s_has_cache = false;

    static nlohmann::json getJson(const std::string& path) {
        try {
            auto current_time = std::filesystem::last_write_time(path);
            if (s_has_cache && path == s_cached_path && current_time == s_last_write_time) {
                return s_cached_json;
            }
            std::ifstream ifs(path);
            if (ifs.is_open()) {
                nlohmann::json j;
                ifs >> j;
                s_cached_json = j;
                s_last_write_time = current_time;
                s_cached_path = path;
                s_has_cache = true;
                return j;
            }
        } catch (...) {
            if (s_has_cache && path == s_cached_path) {
                return s_cached_json;
            }
        }
        return nlohmann::json::object();
    }

  public:
    static BeamConfig load(const std::string& path) {
        BeamConfig cfg{};
        nlohmann::json j = getJson(path);
        if (j.is_discarded() || j.empty()) return cfg;

        if (j.contains("beam_width") && j["beam_width"].is_number_integer())
            cfg.beam_width = j["beam_width"].get<int>();

        if (j.contains("look_ahead") && j["look_ahead"].is_number_integer())
            cfg.look_ahead = j["look_ahead"].get<int>();

        if (j.contains("eval_weights") && j["eval_weights"].is_object())
            applyPatch(cfg.eval_weights, j["eval_weights"]);

        return cfg;
    }

    static BeamConfig applyProfile(BeamConfig cfg,
                                   const std::string& path,
                                   const std::string& profile_name) {
        nlohmann::json j = getJson(path);
        if (j.is_discarded() || j.empty()) return cfg;

        if (!j.contains("profiles") || !j["profiles"].is_object()) return cfg;
        const auto& profiles = j["profiles"];
        if (!profiles.contains(profile_name)) return cfg;
        if (!profiles[profile_name].is_object()) return cfg;

        applyPatch(cfg.eval_weights, profiles[profile_name]);
        return cfg;
    }

    static void save(const std::string& path, const BeamConfig& cfg) {
        nlohmann::json j = getJson(path);
        if (j.empty() || j.is_discarded()) {
            j = nlohmann::json::object();
        }

        j["beam_width"]  = cfg.beam_width;
        j["look_ahead"]  = cfg.look_ahead;

        auto& ew = j["eval_weights"];
        const auto& w = cfg.eval_weights;
        ew["potential_score_scale"]   = w.potential_score_scale;
        ew["connectivity_bonus"]      = w.connectivity_bonus;
        ew["isolated_penalty"]        = w.isolated_penalty;
        ew["buried_penalty"]          = w.buried_penalty;
        ew["height_variance_penalty"] = w.height_variance_penalty;
        ew["death_col_penalty"]       = w.death_col_penalty;
        ew["fire_bias"]               = w.fire_bias;
        ew["edge_column_bonus"]       = w.edge_column_bonus;
        ew["edge_column_threshold"]   = w.edge_column_threshold;
        ew["use_fast_potential"]      = w.use_fast_potential;

        std::ofstream ofs(path);
        ofs << j.dump(2);

        try {
            s_cached_json = j;
            s_last_write_time = std::filesystem::last_write_time(path);
            s_cached_path = path;
            s_has_cache = true;
        } catch (...) {
            s_has_cache = false;
        }
    }

  private:
    static void applyPatch(BeamEvalWeights& w, const nlohmann::json& patch) {
        for (auto& [key, val] : patch.items()) {
            if (key.starts_with("_comment")) continue;
            if      (key == "potential_score_scale"   && val.is_number()) w.potential_score_scale   = val.get<float>();
            else if (key == "connectivity_bonus"       && val.is_number()) w.connectivity_bonus       = val.get<float>();
            else if (key == "isolated_penalty"         && val.is_number()) w.isolated_penalty         = val.get<float>();
            else if (key == "buried_penalty"           && val.is_number()) w.buried_penalty           = val.get<float>();
            else if (key == "height_variance_penalty"  && val.is_number()) w.height_variance_penalty  = val.get<float>();
            else if (key == "death_col_penalty"        && val.is_number()) w.death_col_penalty        = val.get<float>();
            else if (key == "fire_bias"                && val.is_number()) w.fire_bias                = val.get<float>();
            else if (key == "edge_column_bonus"        && val.is_number()) w.edge_column_bonus        = val.get<float>();
            else if (key == "edge_column_threshold"    && val.is_number()) w.edge_column_threshold    = val.get<float>();
            else if (key == "use_fast_potential"       && val.is_boolean()) w.use_fast_potential      = val.get<bool>();
        }
    }
};

} // namespace puyotan::search
