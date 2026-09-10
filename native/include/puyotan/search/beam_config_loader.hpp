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
        auto getBool  = [&](const char* k, bool& dst) {
            if (section.contains(k) && section[k].is_boolean())
                dst = section[k].template get<bool>();
        };
        getInt  ("beam_width",               cfg.beam_width);
        getInt  ("look_ahead",               cfg.look_ahead);
        getInt  ("dbs_max_similar",          cfg.dbs_max_similar);
        getInt  ("dbs_max_similar_end",      cfg.dbs_max_similar_end);
        getInt  ("dbs_empty_threshold_high", cfg.dbs_empty_threshold_high);
        getInt  ("dbs_empty_threshold_low",  cfg.dbs_empty_threshold_low);
        getBool ("dbs_auto_fill",            cfg.dbs_auto_fill);
        getInt  ("full_beam_depth",          cfg.full_beam_depth);
        getFloat("min_beam_width_ratio",     cfg.min_beam_width_ratio);
        getInt  ("main_chain_threshold",     cfg.main_chain_threshold);
        getInt  ("dynamic_lookahead_margin", cfg.dynamic_lookahead_margin);
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

    static void applyPatch(MatchBeamEvalWeights& w, const nlohmann::json& patch) {
        for (auto& [key, val] : patch.items()) {
            if (key.starts_with("_comment")) continue;
            if      (key == "potential_score_scale"           && val.is_number()) w.potential_score_scale           = static_cast<int32_t>(val.get<double>());
            else if (key == "reckless_fire_penalty_permille"  && val.is_number()) w.reckless_fire_penalty_permille  = static_cast<int32_t>(val.get<double>());
            else if (key == "actual_score_weight"             && val.is_number()) w.actual_score_weight             = static_cast<int32_t>(val.get<double>());
            else if (key == "connectivity_bonus"              && val.is_number()) w.connectivity_bonus              = static_cast<int32_t>(val.get<double>());
            else if (key == "isolated_penalty"                && val.is_number()) w.isolated_penalty                = static_cast<int32_t>(val.get<double>());
            else if (key == "buried_penalty"                  && val.is_number()) w.buried_penalty                  = static_cast<int32_t>(val.get<double>());
            else if (key == "active_ojama_coeff"              && val.is_number()) w.active_ojama_coeff              = static_cast<int32_t>(val.get<double>());
            else if (key == "pending_ojama_penalty"           && val.is_number()) w.pending_ojama_penalty           = static_cast<int32_t>(val.get<double>());
            else if (key == "height_danger_threshold"         && val.is_number()) w.height_danger_threshold         = static_cast<int32_t>(val.get<double>());
            else if (key == "height_danger_penalty"           && val.is_number()) w.height_danger_penalty           = static_cast<int32_t>(val.get<double>());
            else if (key == "win_score"                       && val.is_number()) w.win_score                       = static_cast<int32_t>(val.get<double>());
            else if (key == "draw_score"                      && val.is_number()) w.draw_score                      = static_cast<int32_t>(val.get<double>());
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
        if (section.contains("pv_elite_count") && section["pv_elite_count"].is_number_integer())
            cfg.pv_elite_count = section["pv_elite_count"].get<int>();
        if (section.contains("elite_keep") && section["elite_keep"].is_number_integer())
            cfg.elite_keep = section["elite_keep"].get<int>();
        if (section.contains("micro_ply") && section["micro_ply"].is_number_integer())
            cfg.micro_ply = std::max(1, section["micro_ply"].get<int>());
        if (section.contains("fire_trigger_empty_cells") && section["fire_trigger_empty_cells"].is_number_integer())
            cfg.fire_trigger_empty_cells = section["fire_trigger_empty_cells"].get<int>();
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

    static MatchBeamConfig loadMatch(const std::string& path) {
        MatchBeamConfig cfg{};
        nlohmann::json j = getJson(path);
        if (j.is_discarded() || j.empty() || !j.contains("vs_match") || !j["vs_match"].is_object())
            return cfg;
        const auto& section = j["vs_match"];
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
        solo["beam_width"]               = cfg.beam_width;
        solo["look_ahead"]               = cfg.look_ahead;
        solo["dbs_max_similar"]          = cfg.dbs_max_similar;
        solo["dbs_max_similar_end"]      = cfg.dbs_max_similar_end;
        solo["dbs_empty_threshold_high"] = cfg.dbs_empty_threshold_high;
        solo["dbs_empty_threshold_low"]  = cfg.dbs_empty_threshold_low;
        solo["dbs_auto_fill"]            = cfg.dbs_auto_fill;
        solo["pv_elite_count"]           = cfg.pv_elite_count;
        solo["elite_keep"]               = cfg.elite_keep;
        solo["micro_ply"]                = cfg.micro_ply;
        solo["full_beam_depth"]          = cfg.full_beam_depth;
        solo["min_beam_width_ratio"]     = cfg.min_beam_width_ratio;
        solo["main_chain_threshold"]     = cfg.main_chain_threshold;
        solo["dynamic_lookahead_margin"] = cfg.dynamic_lookahead_margin;
        solo["fire_trigger_empty_cells"] = cfg.fire_trigger_empty_cells;

        auto& ew = solo["eval_weights"];
        ew["potential_score_scale"] = cfg.eval_weights.potential_score_scale;

        { std::ofstream ofs(path); ofs << j.dump(2); }
        updateCache(path, j);
    }

    static void saveVs(const std::string& path, const VsBeamConfig& cfg) {
        nlohmann::json j = getJson(path);
        if (j.empty() || j.is_discarded()) j = nlohmann::json::object();

        auto& vs = j["vs"];
        vs["beam_width"]               = cfg.beam_width;
        vs["look_ahead"]               = cfg.look_ahead;
        vs["dbs_max_similar"]          = cfg.dbs_max_similar;
        vs["dbs_max_similar_end"]      = cfg.dbs_max_similar_end;
        vs["dbs_empty_threshold_high"] = cfg.dbs_empty_threshold_high;
        vs["dbs_empty_threshold_low"]  = cfg.dbs_empty_threshold_low;
        vs["dbs_auto_fill"]            = cfg.dbs_auto_fill;
        vs["full_beam_depth"]          = cfg.full_beam_depth;
        vs["min_beam_width_ratio"]     = cfg.min_beam_width_ratio;
        vs["main_chain_threshold"]     = cfg.main_chain_threshold;
        vs["dynamic_lookahead_margin"] = cfg.dynamic_lookahead_margin;
        vs["enable_attack_search"]     = cfg.enable_attack_search;

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

    static void saveMatch(const std::string& path, const MatchBeamConfig& cfg) {
        nlohmann::json j = getJson(path);
        if (j.empty() || j.is_discarded()) j = nlohmann::json::object();

        auto& match_sec = j["vs_match"];
        match_sec["beam_width"]               = cfg.beam_width;
        match_sec["look_ahead"]               = cfg.look_ahead;
        match_sec["dbs_max_similar"]          = cfg.dbs_max_similar;
        match_sec["dbs_max_similar_end"]      = cfg.dbs_max_similar_end;
        match_sec["dbs_empty_threshold_high"] = cfg.dbs_empty_threshold_high;
        match_sec["dbs_empty_threshold_low"]  = cfg.dbs_empty_threshold_low;
        match_sec["dbs_auto_fill"]            = cfg.dbs_auto_fill;
        match_sec["full_beam_depth"]          = cfg.full_beam_depth;
        match_sec["min_beam_width_ratio"]     = cfg.min_beam_width_ratio;
        match_sec["main_chain_threshold"]     = cfg.main_chain_threshold;
        match_sec["dynamic_lookahead_margin"] = cfg.dynamic_lookahead_margin;

        const auto& w = cfg.eval_weights;
        auto& ew = match_sec["eval_weights"];
        ew["potential_score_scale"]          = w.potential_score_scale;
        ew["reckless_fire_penalty_permille"] = w.reckless_fire_penalty_permille;
        ew["actual_score_weight"]            = w.actual_score_weight;
        ew["connectivity_bonus"]             = w.connectivity_bonus;
        ew["isolated_penalty"]               = w.isolated_penalty;
        ew["buried_penalty"]                 = w.buried_penalty;
        ew["active_ojama_coeff"]             = w.active_ojama_coeff;
        ew["pending_ojama_penalty"]          = w.pending_ojama_penalty;
        ew["height_danger_threshold"]        = w.height_danger_threshold;
        ew["height_danger_penalty"]          = w.height_danger_penalty;
        ew["win_score"]                      = w.win_score;
        ew["draw_score"]                     = w.draw_score;

        { std::ofstream ofs(path); ofs << j.dump(2); }
        updateCache(path, j);
    }
};

} // namespace puyotan::search
