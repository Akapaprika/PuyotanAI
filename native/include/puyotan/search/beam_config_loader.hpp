#pragma once

#include <fstream>
#include <string>
#include <filesystem>
#include <mutex>
#include <external/nlohmann/json.hpp>
#include <puyotan/search/beam_search.hpp>
#include <puyotan/search/negamax_search.hpp>
#include <puyotan/search/abs_search.hpp>

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
    static inline std::mutex s_mutex;

    static nlohmann::json getJson(const std::string& path) {
        std::lock_guard<std::mutex> lock(s_mutex);
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
    static SoloBeamConfig loadSolo(const std::string& path) {
        SoloBeamConfig cfg{};
        nlohmann::json j = getJson(path);
        if (j.is_discarded() || j.empty()) return cfg;
        if (!j.contains("solo") || !j["solo"].is_object()) return cfg;

        const auto& section = j["solo"];
        if (section.contains("beam_width") && section["beam_width"].is_number_integer())
            cfg.beam_width = section["beam_width"].get<int>();

        if (section.contains("look_ahead") && section["look_ahead"].is_number_integer())
            cfg.look_ahead = section["look_ahead"].get<int>();

        if (section.contains("dbs_max_similar") && section["dbs_max_similar"].is_number_integer())
            cfg.dbs_max_similar = section["dbs_max_similar"].get<int>();

        if (section.contains("eval_weights") && section["eval_weights"].is_object())
            applyPatch(cfg.eval_weights, section["eval_weights"]);

        return cfg;
    }

    static VsBeamConfig loadVs(const std::string& path) {
        VsBeamConfig cfg{};
        nlohmann::json j = getJson(path);
        if (j.is_discarded() || j.empty()) return cfg;
        if (!j.contains("vs") || !j["vs"].is_object()) return cfg;

        const auto& section = j["vs"];
        if (section.contains("beam_width") && section["beam_width"].is_number_integer())
            cfg.beam_width = section["beam_width"].get<int>();

        if (section.contains("look_ahead") && section["look_ahead"].is_number_integer())
            cfg.look_ahead = section["look_ahead"].get<int>();

        if (section.contains("dbs_max_similar") && section["dbs_max_similar"].is_number_integer())
            cfg.dbs_max_similar = section["dbs_max_similar"].get<int>();

        if (section.contains("enable_attack_search") && section["enable_attack_search"].is_boolean())
            cfg.enable_attack_search = section["enable_attack_search"].get<bool>();

        if (section.contains("eval_weights") && section["eval_weights"].is_object())
            applyPatch(cfg.eval_weights, section["eval_weights"]);

        return cfg;
    }

    static VsBeamConfig loadEnemy(const std::string& path) {
        VsBeamConfig cfg{};
        nlohmann::json j = getJson(path);
        if (j.is_discarded() || j.empty()) return cfg;
        if (!j.contains("enemy") || !j["enemy"].is_object()) return cfg;

        const auto& section = j["enemy"];
        if (section.contains("beam_width") && section["beam_width"].is_number_integer())
            cfg.beam_width = section["beam_width"].get<int>();

        if (section.contains("look_ahead") && section["look_ahead"].is_number_integer())
            cfg.look_ahead = section["look_ahead"].get<int>();

        if (section.contains("dbs_max_similar") && section["dbs_max_similar"].is_number_integer())
            cfg.dbs_max_similar = section["dbs_max_similar"].get<int>();

        if (section.contains("enable_attack_search") && section["enable_attack_search"].is_boolean())
            cfg.enable_attack_search = section["enable_attack_search"].get<bool>();

        if (section.contains("eval_weights") && section["eval_weights"].is_object())
            applyPatch(cfg.eval_weights, section["eval_weights"]);

        return cfg;
    }

    static NegamaxConfig loadNegamax(const std::string& path) {
        NegamaxConfig cfg{};
        cfg.vs_config = loadVs(path);
        cfg.interior_vs_config = cfg.vs_config;

        nlohmann::json j = getJson(path);
        if (j.is_discarded() || j.empty()) return cfg;
        if (!j.contains("negamax") || !j["negamax"].is_object()) return cfg;

        const auto& section = j["negamax"];
        if (section.contains("depth") && section["depth"].is_number_integer())
            cfg.depth = section["depth"].get<int>();

        if (section.contains("candidate_n") && section["candidate_n"].is_number_integer())
            cfg.candidate_n = section["candidate_n"].get<int>();

        if (section.contains("interior_candidate_n") && section["interior_candidate_n"].is_number_integer())
            cfg.interior_candidate_n = section["interior_candidate_n"].get<int>();

        if (section.contains("chain_cutoff_enabled") && section["chain_cutoff_enabled"].is_boolean())
            cfg.chain_cutoff_enabled = section["chain_cutoff_enabled"].get<bool>();

        if (section.contains("use_interior_beam_config") && section["use_interior_beam_config"].is_boolean())
            cfg.use_interior_config = section["use_interior_beam_config"].get<bool>();

        if (section.contains("interior_beam_width") && section["interior_beam_width"].is_number_integer())
            cfg.interior_vs_config.beam_width = section["interior_beam_width"].get<int>();

        if (section.contains("interior_look_ahead") && section["interior_look_ahead"].is_number_integer())
            cfg.interior_vs_config.look_ahead = section["interior_look_ahead"].get<int>();

        if (section.contains("interior_dbs_max_similar") && section["interior_dbs_max_similar"].is_number_integer())
            cfg.interior_vs_config.dbs_max_similar = section["interior_dbs_max_similar"].get<int>();

        return cfg;
    }

    static AbsConfig loadAbs(const std::string& path) {
        AbsConfig cfg{};
        // Inherit default VS eval weights
        cfg.eval_weights = loadVs(path).eval_weights;

        nlohmann::json j = getJson(path);
        if (j.is_discarded() || j.empty()) return cfg;
        if (!j.contains("abs") || !j["abs"].is_object()) return cfg;

        const auto& section = j["abs"];
        if (section.contains("depth") && section["depth"].is_number_integer())
            cfg.depth = section["depth"].get<int>();

        if (section.contains("chain_cutoff_enabled") && section["chain_cutoff_enabled"].is_boolean())
            cfg.chain_cutoff_enabled = section["chain_cutoff_enabled"].get<bool>();

        if (section.contains("my_category_budgets") && section["my_category_budgets"].is_object()) {
            const auto& mb = section["my_category_budgets"];
            if (mb.contains("build")  && mb["build"].is_number_integer())  cfg.my_budgets.build  = mb["build"].get<int>();
            if (mb.contains("crush")  && mb["crush"].is_number_integer())  cfg.my_budgets.crush  = mb["crush"].get<int>();
            if (mb.contains("strike") && mb["strike"].is_number_integer()) cfg.my_budgets.strike = mb["strike"].get<int>();
            if (mb.contains("evade")  && mb["evade"].is_number_integer())  cfg.my_budgets.evade  = mb["evade"].get<int>();
        }

        if (section.contains("opp_category_budgets") && section["opp_category_budgets"].is_object()) {
            const auto& ob = section["opp_category_budgets"];
            if (ob.contains("build")  && ob["build"].is_number_integer())  cfg.opp_budgets.build  = ob["build"].get<int>();
            if (ob.contains("crush")  && ob["crush"].is_number_integer())  cfg.opp_budgets.crush  = ob["crush"].get<int>();
            if (ob.contains("strike") && ob["strike"].is_number_integer()) cfg.opp_budgets.strike = ob["strike"].get<int>();
            if (ob.contains("evade")  && ob["evade"].is_number_integer())  cfg.opp_budgets.evade  = ob["evade"].get<int>();
        }

        if (section.contains("eval_weights") && section["eval_weights"].is_object()) {
            applyPatch(cfg.eval_weights, section["eval_weights"]);
        }

        return cfg;
    }

    static void saveSolo(const std::string& path, const SoloBeamConfig& cfg) {
        nlohmann::json j = getJson(path);
        if (j.empty() || j.is_discarded()) {
            j = nlohmann::json::object();
        }

        auto& solo = j["solo"];
        solo["beam_width"] = cfg.beam_width;
        solo["look_ahead"] = cfg.look_ahead;
        solo["dbs_max_similar"] = cfg.dbs_max_similar;

        auto& ew = solo["eval_weights"];
        const auto& w = cfg.eval_weights;
        ew["potential_score_scale"]   = w.potential_score_scale;

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

    static void saveVs(const std::string& path, const VsBeamConfig& cfg) {
        nlohmann::json j = getJson(path);
        if (j.empty() || j.is_discarded()) {
            j = nlohmann::json::object();
        }

        auto& vs = j["vs"];
        vs["beam_width"] = cfg.beam_width;
        vs["look_ahead"] = cfg.look_ahead;
        vs["dbs_max_similar"] = cfg.dbs_max_similar;
        vs["enable_attack_search"] = cfg.enable_attack_search;

        auto& ew = vs["eval_weights"];
        const auto& w = cfg.eval_weights;
        ew["potential_score_scale"]   = w.potential_score_scale;
        ew["fire_bias"]               = w.fire_bias;
        ew["incoming_ojama_penalty"]  = w.incoming_ojama_penalty;

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

    static void saveEnemy(const std::string& path, const VsBeamConfig& cfg) {
        nlohmann::json j = getJson(path);
        if (j.empty() || j.is_discarded()) {
            j = nlohmann::json::object();
        }

        auto& enemy = j["enemy"];
        enemy["beam_width"] = cfg.beam_width;
        enemy["look_ahead"] = cfg.look_ahead;
        enemy["dbs_max_similar"] = cfg.dbs_max_similar;
        enemy["enable_attack_search"] = cfg.enable_attack_search;

        auto& ew = enemy["eval_weights"];
        const auto& w = cfg.eval_weights;
        ew["potential_score_scale"]   = w.potential_score_scale;
        ew["fire_bias"]               = w.fire_bias;
        ew["incoming_ojama_penalty"]  = w.incoming_ojama_penalty;

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
    static void applyPatch(SoloBeamEvalWeights& w, const nlohmann::json& patch) {
        for (auto& [key, val] : patch.items()) {
            if (key.starts_with("_comment")) continue;
            if (key == "potential_score_scale" && val.is_number()) w.potential_score_scale = val.get<float>();
        }
    }

    static void applyPatch(VsBeamEvalWeights& w, const nlohmann::json& patch) {
        for (auto& [key, val] : patch.items()) {
            if (key.starts_with("_comment")) continue;
            if      (key == "potential_score_scale"   && val.is_number()) w.potential_score_scale   = val.get<float>();
            else if (key == "fire_bias"                && val.is_number()) w.fire_bias                = val.get<float>();
            else if (key == "incoming_ojama_penalty"   && val.is_number()) w.incoming_ojama_penalty   = val.get<float>();
            else if (key == "incoming_threat_bias"     && val.is_number()) w.incoming_threat_bias     = val.get<float>();
            else if (key == "counter_attack_bias"      && val.is_number()) w.counter_attack_bias      = val.get<float>();
            else if (key == "timing_advantage_bias"    && val.is_number()) w.timing_advantage_bias    = val.get<float>();
            else if (key == "urgency_weight"           && val.is_number()) w.urgency_weight           = val.get<float>();
            else if (key == "lethal_danger_scale"      && val.is_number()) w.lethal_danger_scale      = val.get<float>();
            else if (key == "effective_strike_multiplier" && val.is_number()) w.effective_strike_multiplier = val.get<float>();
        }
    }
};

} // namespace puyotan::search
