#pragma once

#include <fstream>
#include <string>
#include <external/nlohmann/json.hpp>
#include <puyotan/search/beam_search.hpp>

namespace puyotan::search {

/**
 * @class BeamConfigLoader
 * @brief Loads, applies profiles to, and saves BeamConfig from/to a JSON file.
 *
 * The JSON schema mirrors gui/beam_config.json:
 *   - "beam_width"    : int
 *   - "look_ahead"    : int
 *   - "eval_weights"  : object matching BeamEvalWeights fields
 *   - "profiles"      : object of named patch objects (keys starting with "_comment" are ignored)
 *
 * Designed to be the single source of truth for evaluation parameters,
 * enabling future automatic tuning loops entirely within C++.
 */
class BeamConfigLoader {
  public:
    /**
     * @brief Load a BeamConfig from a JSON file.
     *
     * If the file cannot be opened or parsed, returns a default-constructed
     * BeamConfig (same as declaring `BeamConfig cfg;`).
     *
     * @param path  Absolute or relative path to the JSON config file.
     * @return      Populated BeamConfig.
     */
    static BeamConfig load(const std::string& path) {
        BeamConfig cfg{};
        std::ifstream ifs(path);
        if (!ifs.is_open()) return cfg;

        nlohmann::json j;
        try {
            ifs >> j;
        } catch (...) {
            return cfg;
        }

        if (j.contains("beam_width") && j["beam_width"].is_number_integer())
            cfg.beam_width = j["beam_width"].get<int>();

        if (j.contains("look_ahead") && j["look_ahead"].is_number_integer())
            cfg.look_ahead = j["look_ahead"].get<int>();

        if (j.contains("eval_weights") && j["eval_weights"].is_object())
            applyPatch(cfg.eval_weights, j["eval_weights"]);

        return cfg;
    }

    /**
     * @brief Apply a named profile patch to an existing BeamConfig.
     *
     * The profile must be defined under the "profiles" key in the same JSON file
     * that was used to produce `cfg`. If the profile is not found, `cfg` is
     * returned unchanged.
     *
     * @param cfg           Config to patch (value-copied).
     * @param path          Path to the JSON config file (to re-read profiles).
     * @param profile_name  Name of the profile to apply.
     * @return              Patched BeamConfig.
     */
    static BeamConfig applyProfile(BeamConfig cfg,
                                   const std::string& path,
                                   const std::string& profile_name) {
        std::ifstream ifs(path);
        if (!ifs.is_open()) return cfg;

        nlohmann::json j;
        try {
            ifs >> j;
        } catch (...) {
            return cfg;
        }

        if (!j.contains("profiles") || !j["profiles"].is_object()) return cfg;
        const auto& profiles = j["profiles"];
        if (!profiles.contains(profile_name)) return cfg;
        if (!profiles[profile_name].is_object()) return cfg;

        applyPatch(cfg.eval_weights, profiles[profile_name]);
        return cfg;
    }

    /**
     * @brief Save a BeamConfig back to a JSON file.
     *
     * Writes only the scalar fields (beam_width, look_ahead, eval_weights).
     * Existing "profiles" and "_comment" keys in the file are preserved if the
     * file already exists; otherwise a minimal JSON is written.
     *
     * Intended for use by future automatic tuning features that want to persist
     * updated weights without discarding user-authored profiles.
     *
     * @param path  Path to write.
     * @param cfg   Config to serialize.
     */
    static void save(const std::string& path, const BeamConfig& cfg) {
        nlohmann::json j;

        // Try to preserve existing content (profiles, comments, etc.)
        {
            std::ifstream ifs(path);
            if (ifs.is_open()) {
                try { ifs >> j; } catch (...) { j = nlohmann::json::object(); }
            } else {
                j = nlohmann::json::object();
            }
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
        ew["chain_bonus_per_step"]    = w.chain_bonus_per_step;
        ew["chain_power"]             = w.chain_power;
        ew["use_fast_potential"]      = w.use_fast_potential;

        std::ofstream ofs(path);
        ofs << j.dump(2);
    }

  private:
    /**
     * @brief Apply key-value pairs from a JSON object onto a BeamEvalWeights.
     *        Keys starting with "_comment" are silently ignored.
     */
    static void applyPatch(BeamEvalWeights& w, const nlohmann::json& patch) {
        for (auto& [key, val] : patch.items()) {
            if (key.starts_with("_comment")) continue;
            if      (key == "potential_score_scale"   && val.is_number()) w.potential_score_scale   = val.get<float>();
            else if (key == "connectivity_bonus"       && val.is_number()) w.connectivity_bonus       = val.get<float>();
            else if (key == "isolated_penalty"         && val.is_number()) w.isolated_penalty         = val.get<float>();
            else if (key == "buried_penalty"           && val.is_number()) w.buried_penalty           = val.get<float>();
            else if (key == "height_variance_penalty"  && val.is_number()) w.height_variance_penalty  = val.get<float>();
            else if (key == "death_col_penalty"        && val.is_number()) w.death_col_penalty        = val.get<float>();
            else if (key == "chain_bonus_per_step"     && val.is_number()) w.chain_bonus_per_step     = val.get<float>();
            else if (key == "chain_power"              && val.is_number()) w.chain_power              = val.get<float>();
            else if (key == "use_fast_potential"       && val.is_boolean()) w.use_fast_potential      = val.get<bool>();
        }
    }
};

} // namespace puyotan::search
