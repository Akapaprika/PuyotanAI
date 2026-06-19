#pragma once

#include <fstream>
#include <string>
#include <filesystem>
#include <external/nlohmann/json.hpp>
#include <puyotan/search/beam_search.hpp>

namespace puyotan::search {

/**
 * @class BeamConfigLoader
 * @brief Loads and saves Solo/VS BeamConfig from/to a JSON file with static in-memory caching.
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
    static BeamConfig loadSolo(const std::string& path) {
        BeamConfig cfg{};
        nlohmann::json j = getJson(path);
        if (j.is_discarded() || j.empty()) return cfg;
        if (!j.contains("solo") || !j["solo"].is_object()) return cfg;

        const auto& section = j["solo"];
        if (section.contains("beam_width") && section["beam_width"].is_number_integer())
            cfg.beam_width = section["beam_width"].get<int>();

        if (section.contains("look_ahead") && section["look_ahead"].is_number_integer())
            cfg.look_ahead = section["look_ahead"].get<int>();

        if (section.contains("eval_weights") && section["eval_weights"].is_object())
            applyPatch(cfg.eval_weights, section["eval_weights"]);

        return cfg;
    }

    static BeamConfig loadVs(const std::string& path) {
        BeamConfig cfg{};
        nlohmann::json j = getJson(path);
        if (j.is_discarded() || j.empty()) return cfg;
        if (!j.contains("vs") || !j["vs"].is_object()) return cfg;

        const auto& section = j["vs"];
        if (section.contains("beam_width") && section["beam_width"].is_number_integer())
            cfg.beam_width = section["beam_width"].get<int>();

        if (section.contains("look_ahead") && section["look_ahead"].is_number_integer())
            cfg.look_ahead = section["look_ahead"].get<int>();

        if (section.contains("eval_weights") && section["eval_weights"].is_object())
            applyPatch(cfg.eval_weights, section["eval_weights"]);

        return cfg;
    }

    static void saveSolo(const std::string& path, const BeamConfig& cfg) {
        nlohmann::json j = getJson(path);
        if (j.empty() || j.is_discarded()) {
            j = nlohmann::json::object();
        }

        auto& solo = j["solo"];
        solo["beam_width"] = cfg.beam_width;
        solo["look_ahead"] = cfg.look_ahead;

        auto& ew = solo["eval_weights"];
        const auto& w = cfg.eval_weights;
        ew["potential_score_scale"]   = w.potential_score_scale;
        ew["connectivity_bonus"]      = w.connectivity_bonus;
        ew["isolated_penalty"]        = w.isolated_penalty;
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

    static void saveVs(const std::string& path, const BeamConfig& cfg) {
        nlohmann::json j = getJson(path);
        if (j.empty() || j.is_discarded()) {
            j = nlohmann::json::object();
        }

        auto& vs = j["vs"];
        vs["beam_width"] = cfg.beam_width;
        vs["look_ahead"] = cfg.look_ahead;

        auto& ew = vs["eval_weights"];
        const auto& w = cfg.eval_weights;
        ew["potential_score_scale"]   = w.potential_score_scale;
        ew["connectivity_bonus"]      = w.connectivity_bonus;
        ew["isolated_penalty"]        = w.isolated_penalty;
        ew["buried_penalty"]          = w.buried_penalty;
        ew["fire_bias"]               = w.fire_bias;
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
            else if (key == "fire_bias"                && val.is_number()) w.fire_bias                = val.get<float>();
            else if (key == "use_fast_potential"       && val.is_boolean()) w.use_fast_potential      = val.get<bool>();
        }
    }
};

} // namespace puyotan::search
