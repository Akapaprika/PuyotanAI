#pragma once

#include <fstream>
#include <string>
#include <filesystem>
#include <mutex>
#include <external/nlohmann/json.hpp>
#include <puyotan/search/beam_config.hpp>

namespace puyotan::search {

/**
 * @class BeamConfigLoader
 * @brief Loads and saves Solo/VS BeamConfig from/to a JSON file with static in-memory caching.
 *
 * All filesystem operations use std::error_code (no exceptions) to remain compatible
 * with /EHs-c- builds.
 */
class BeamConfigLoader {
  private:
    static inline std::mutex                        s_mutex;
    static inline nlohmann::json                    s_cached_json;
    static inline std::filesystem::file_time_type   s_last_write_time;
    static inline std::string                       s_cached_path;
    static inline bool                              s_has_cache = false;

    // ── JSON cache ────────────────────────────────────────────────────────────

    static nlohmann::json getJson(const std::string& path) {
        std::lock_guard<std::mutex> lock(s_mutex);
        std::error_code ec;
        auto mtime = std::filesystem::last_write_time(path, ec);
        if (!ec && s_has_cache && path == s_cached_path && mtime == s_last_write_time)
            return s_cached_json;
        std::ifstream ifs(path);
        if (ifs.is_open()) {
            nlohmann::json j;
            ifs >> j;
            if (!j.is_discarded()) {
                s_cached_json      = j;
                s_last_write_time  = mtime;
                s_cached_path      = path;
                s_has_cache        = true;
                return j;
            }
        }
        return s_has_cache && path == s_cached_path ? s_cached_json
                                                     : nlohmann::json::object();
    }

    static void updateCache(const std::string& path, const nlohmann::json& j) {
        std::lock_guard<std::mutex> lock(s_mutex);
        std::error_code ec;
        auto mtime = std::filesystem::last_write_time(path, ec);
        if (!ec) {
            s_cached_json     = j;
            s_last_write_time = mtime;
            s_cached_path     = path;
            s_has_cache       = true;
        } else {
            s_has_cache = false;
        }
    }

    // ── Common section loader ─────────────────────────────────────────────────

    template <typename Cfg>
    static void loadCommonSection(Cfg& cfg, const nlohmann::json& section) {
        auto getInt   = [&](const char* k, auto& dst) {
            if (section.contains(k) && section[k].is_number_integer())
                dst = section[k].template get<std::remove_reference_t<decltype(dst)>>();
        };
        auto getFloat = [&](const char* k, float& dst) {
            if (section.contains(k) && section[k].is_number())
                dst = section[k].template get<float>();
        };
        getInt  ("beam_width",          cfg.beam_width);
        getInt  ("look_ahead",          cfg.look_ahead);
        getInt  ("dbs_max_similar",     cfg.dbs_max_similar);
        getInt  ("full_beam_depth",     cfg.full_beam_depth);
        getFloat("min_beam_width_ratio", cfg.min_beam_width_ratio);
    }

    // ── Eval-weights patch (key-value iteration, handles all weight types) ────

    static void applyPatch(SoloBeamEvalWeights& w, const nlohmann::json& patch) {
        for (auto& [key, val] : patch.items()) {
            if (key.starts_with("_comment")) continue;
            if (key == "potential_score_scale" && val.is_number())
                w.potential_score_scale = static_cast<int32_t>(val.get<double>());
        }
    }

    static void applyPatch(VsBeamEvalWeights& w, const nlohmann::json& patch) {
        for (auto& [key, val] : patch.items()) {
            if (key.starts_with("_comment")) continue;
            if      (key == "potential_score_scale"       && val.is_number()) w.potential_score_scale               = static_cast<int32_t>(val.get<double>());
            else if (key == "connectivity_bonus"          && val.is_number()) w.connectivity_bonus                  = static_cast<int32_t>(val.get<double>());
            else if (key == "isolated_penalty"            && val.is_number()) w.isolated_penalty                    = static_cast<int32_t>(val.get<double>());
            else if (key == "buried_penalty"              && val.is_number()) w.buried_penalty                      = static_cast<int32_t>(val.get<double>());
            else if (key == "fire_bias"                   && val.is_number()) w.fire_bias_permille                  = static_cast<int32_t>(val.get<double>() * 1000);
            else if (key == "incoming_ojama_penalty"      && val.is_number()) w.incoming_ojama_penalty              = static_cast<int32_t>(val.get<double>());
            else if (key == "incoming_threat_bias"        && val.is_number()) w.incoming_threat_bias_permille       = static_cast<int32_t>(val.get<double>() * 1000);
            else if (key == "counter_attack_bias"         && val.is_number()) w.counter_attack_bias_permille        = static_cast<int32_t>(val.get<double>() * 1000);
            else if (key == "timing_advantage_bias"       && val.is_number()) w.timing_advantage_bias_permille      = static_cast<int32_t>(val.get<double>() * 1000);
            else if (key == "urgency_weight"              && val.is_number()) w.urgency_weight_permille             = static_cast<int32_t>(val.get<double>() * 1000);
            else if (key == "lethal_danger_scale"         && val.is_number()) w.lethal_danger_scale                 = static_cast<int32_t>(val.get<double>());
            else if (key == "effective_strike_multiplier" && val.is_number()) w.effective_strike_multiplier_permille = static_cast<int32_t>(val.get<double>() * 1000);
        }
    }

  public:
    // ── Load ─────────────────────────────────────────────────────────────────

    static SoloBeamConfig loadSolo(const std::string& path) {
        SoloBeamConfig cfg{};
        nlohmann::json j = getJson(path);
        if (j.is_discarded() || j.empty() || !j.contains("solo") || !j["solo"].is_object())
            return cfg;
        const auto& section = j["solo"];
        loadCommonSection(cfg, section);
        if (section.contains("eval_weights") && section["eval_weights"].is_object())
            applyPatch(cfg.eval_weights, section["eval_weights"]);
        cfg.recompute_beam_widths();
        return cfg;
    }

    static VsBeamConfig loadVs(const std::string& path) {
        VsBeamConfig cfg{};
        nlohmann::json j = getJson(path);
        if (j.is_discarded() || j.empty() || !j.contains("vs") || !j["vs"].is_object())
            return cfg;
        const auto& section = j["vs"];
        loadCommonSection(cfg, section);
        if (section.contains("eval_weights") && section["eval_weights"].is_object())
            applyPatch(cfg.eval_weights, section["eval_weights"]);
        cfg.recompute_beam_widths();
        return cfg;
    }

    // ── Save ─────────────────────────────────────────────────────────────────

    static void saveSolo(const std::string& path, const SoloBeamConfig& cfg) {
        nlohmann::json j = getJson(path);
        if (j.empty() || j.is_discarded()) j = nlohmann::json::object();

        auto& solo = j["solo"];
        solo["beam_width"]          = cfg.beam_width;
        solo["look_ahead"]          = cfg.look_ahead;
        solo["dbs_max_similar"]     = cfg.dbs_max_similar;
        solo["full_beam_depth"]     = cfg.full_beam_depth;
        solo["min_beam_width_ratio"]= cfg.min_beam_width_ratio;

        auto& ew = solo["eval_weights"];
        ew["potential_score_scale"] = cfg.eval_weights.potential_score_scale;

        { std::ofstream ofs(path); ofs << j.dump(2); }
        updateCache(path, j);
    }

    static void saveVs(const std::string& path, const VsBeamConfig& cfg) {
        nlohmann::json j = getJson(path);
        if (j.empty() || j.is_discarded()) j = nlohmann::json::object();

        auto& vs = j["vs"];
        vs["beam_width"]          = cfg.beam_width;
        vs["look_ahead"]          = cfg.look_ahead;
        vs["dbs_max_similar"]     = cfg.dbs_max_similar;
        vs["full_beam_depth"]     = cfg.full_beam_depth;
        vs["min_beam_width_ratio"]= cfg.min_beam_width_ratio;

        const auto& w = cfg.eval_weights;
        auto& ew = vs["eval_weights"];
        ew["potential_score_scale"]           = w.potential_score_scale;
        ew["connectivity_bonus"]              = w.connectivity_bonus;
        ew["isolated_penalty"]                = w.isolated_penalty;
        ew["buried_penalty"]                  = w.buried_penalty;
        ew["fire_bias"]                       = w.fire_bias_permille / 1000.0;
        ew["incoming_ojama_penalty"]          = w.incoming_ojama_penalty;
        ew["incoming_threat_bias"]            = w.incoming_threat_bias_permille / 1000.0;
        ew["counter_attack_bias"]             = w.counter_attack_bias_permille / 1000.0;
        ew["timing_advantage_bias"]           = w.timing_advantage_bias_permille / 1000.0;
        ew["urgency_weight"]                  = w.urgency_weight_permille / 1000.0;
        ew["lethal_danger_scale"]             = w.lethal_danger_scale;
        ew["effective_strike_multiplier"]     = w.effective_strike_multiplier_permille / 1000.0;

        { std::ofstream ofs(path); ofs << j.dump(2); }
        updateCache(path, j);
    }
};

} // namespace puyotan::search
