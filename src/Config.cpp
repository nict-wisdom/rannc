//
// Created by Masahiro Tanaka on 2019-06-26.
//



#include "Config.h"

namespace rannc {

    const std::string DEFAULT_CONF_DIR_NAME = ".pyrannc";
    const std::string RANNC_CONF_DIR_VAR = "RANNC_CONF_DIR";

    std::string toUpper(std::string s) {
        std::transform(s.cbegin(), s.cend(), s.begin(), toupper);
        return s;
    }

    namespace config {
        const std::string RANNC_CONF_FILE = "rannc_conf.toml";

        const char SHOW_CONFIG_ITEMS[] = "show_config_items";
        const char PROFILING[] = "profiling";
        const char GRAPH_PROFILING[] = "graph_profiling";
        const char PROFILE_DUMP_PATH[] = "profile_dump_path";
        const char DUMP_GRAPH[] = "dump_graph";
        const char DUMP_GRAPH_PREFIX[] = "dump_graph_prefix";
        const char PARTITION_NUM[] = "partition_num";
        const char REPLICA_NUM[] = "replica_num";
        const char PIPELINE_NUM[] = "pipeline_num";
        const char VALIDATE_COMM[] = "validate_comm";
        const char DISPLAY_COMM_VALUE[] = "display_comm_value";
        const char CONSOLIDATE_GRADS[] = "consolidate_grads";
        const char SKIP_GRAD_SCALING[] = "skip_grad_scaling";
        const char PROFILING_ITER[] = "profiling_iter";
        const char CHECKPOINTING[] = "checkpointing";
        const char CHECKPOINTING_NO_LAST[] = "checkpointing_no_last";
        const char DISABLE_BALANCING[] = "disable_balancing";
        const char OPT_PARAM_FACTOR[] = "opt_param_factor";
        const char AUTO_PARALLEL[] = "auto_parallel";
        const char P2P_COMM[] = "p2p_comm";
        const char DECOMPOSER[] = "decomposer";
        const char USE_AMP_MASTER_PARAM[] = "use_amp_master_param";
        const char MEM_LIMIT_GB[] = "mem_limit_gb";
        const char MIN_PARTITON_NUM[] = "min_partition_num";
        const char MAX_PARTITON_NUM[] = "max_partition_num";
        const char MEM_MARGIN[] = "mem_margin";
        const char MAX_MP_NUM[] = "max_mp_num";
        const char DO_UNCOARSENING[] = "do_uncoarsening";
        const char DO_COARSENING[] = "do_coarsening";
        const char MIN_PIPELINE[] = "min_pipeline";
        const char MAX_PIPELINE[] = "max_pipeline";
        const char SAVE_DEPLOYMENT[] = "save_deployment";
        const char LOAD_DEPLOYMENT[] = "load_deployment";
        const char DEPLOYMENT_FILE[] = "deployment_file";
        const char SAVE_GRAPH_PROFILE[] = "save_graph_profile";
        const char LOAD_GRAPH_PROFILE[] = "load_graph_profile";
        const char GRAPH_PROFILE_FILE[] = "graph_profile_file";
        const char TRACE_EVENTS[] = "trace_events";
        const char EVENT_TRACE_FILE[] = "event_trace_file";
        const char VERIFY_RECOMP[] = "verify_recomp";
        const char DP_SEARCH_ALL[] = "dp_search_all";
        const char MIN_PIPELINE_BS[] = "min_pipeline_bs";
        const char COARSEN_BY_TIME[] = "coarsen_by_time";
        const char LIMIT_DEV_NUM_POT[] = "limit_dev_num_pot";
        const char VERIFY_PARTITIONING[] = "verify_partitioning";
        const char ALLOC_REPL_FLAT[] = "alloc_repl_flat";
        const char SYNC_ALLREDUCE[] = "sync_allreduce";

        const char CONF_DIR[] = "conf_dir";

        ConfigValue Config::convertValue(const toml::value& val, const ConfigType& type) {
            switch (type) {
                case ConfigType::INT: return ConfigValue{toml::get<int>(val)};
                case ConfigType::FLOAT: return ConfigValue{toml::get<float>(val)};
                case ConfigType::BOOL: return ConfigValue{toml::get<bool>(val)};
                case ConfigType::STRING: return ConfigValue{toml::get<std::string>(val)};
            }
        }

        ConfigValue Config::convertValue(const std::string& val, const ConfigType& type) {
            switch (type) {
                case ConfigType::INT: return ConfigValue{std::stoi(val)};
                case ConfigType::FLOAT: return ConfigValue{std::stof(val)};
                case ConfigType::BOOL: {
                    bool b;
                    std::istringstream(val) >> std::boolalpha >> b;
                    return ConfigValue{b};
                };
                case ConfigType::STRING: return ConfigValue{val};
            }
        }

        void Config::display() {
            if (!values_.empty()) {
                std::stringstream ss;
                ss << "Config items:";
                for (const auto &item: values_) {
                    ss << std::endl << " " << item.first << "=" << toString(item.second);
                }
                std::cout << ss.str() << std::endl;
            }
        }

        Config::Config() {
            const std::vector<ConfigItem> config_items = {
                    makeConfigItem(SHOW_CONFIG_ITEMS, false),
                    makeConfigItem(PROFILING, false),
                    makeConfigItem(GRAPH_PROFILING, false),
                    makeConfigItem(PROFILE_DUMP_PATH, std::string("test_default_config")),
                    makeConfigItem(DUMP_GRAPH, false),
                    makeConfigItem(DUMP_GRAPH_PREFIX, std::string("/tmp/rannc_graph")),
                    makeConfigItem(PARTITION_NUM, 0),
                    makeConfigItem(REPLICA_NUM, 0),
                    makeConfigItem(PIPELINE_NUM, 0),
                    makeConfigItem(VALIDATE_COMM, false),
                    makeConfigItem(DISPLAY_COMM_VALUE, false),
                    makeConfigItem(CONSOLIDATE_GRADS, true),
                    makeConfigItem(SKIP_GRAD_SCALING, true),
                    makeConfigItem(PROFILING_ITER, 1),
                    makeConfigItem(CHECKPOINTING, false),
                    makeConfigItem(CHECKPOINTING_NO_LAST, false),
                    makeConfigItem(DISABLE_BALANCING, true),
                    makeConfigItem(OPT_PARAM_FACTOR, 2),
                    makeConfigItem(AUTO_PARALLEL, false),
                    makeConfigItem(P2P_COMM, true),
                    makeConfigItem(DECOMPOSER, std::string("ml_part")),
                    makeConfigItem(USE_AMP_MASTER_PARAM, false),
                    makeConfigItem(MEM_LIMIT_GB, 0),
                    makeConfigItem(MIN_PARTITON_NUM, 5),
                    makeConfigItem(MAX_PARTITON_NUM, 10),
                    makeConfigItem(MEM_MARGIN, 0.1f),
                    makeConfigItem(MAX_MP_NUM, 8),
                    makeConfigItem(DO_UNCOARSENING, false),
                    makeConfigItem(DO_COARSENING, true),
                    makeConfigItem(MIN_PIPELINE, 1),
                    makeConfigItem(MAX_PIPELINE, 32),
                    makeConfigItem(SAVE_DEPLOYMENT, false),
                    makeConfigItem(LOAD_DEPLOYMENT, false),
                    makeConfigItem(DEPLOYMENT_FILE, std::string("/tmp/rannc_deployment.bin")),
                    makeConfigItem(SAVE_GRAPH_PROFILE, false),
                    makeConfigItem(LOAD_GRAPH_PROFILE, false),
                    makeConfigItem(GRAPH_PROFILE_FILE, std::string("/tmp/rannc_graph_profiles.bin")),
                    makeConfigItem(TRACE_EVENTS, false),
                    makeConfigItem(EVENT_TRACE_FILE, std::string("/tmp/rannc_event_trace.json")),
                    makeConfigItem(VERIFY_RECOMP, false),
                    makeConfigItem(DP_SEARCH_ALL, false),
                    makeConfigItem(MIN_PIPELINE_BS, 1),
                    makeConfigItem(COARSEN_BY_TIME, true),
                    makeConfigItem(LIMIT_DEV_NUM_POT, true),
                    makeConfigItem(VERIFY_PARTITIONING, false),
                    makeConfigItem(ALLOC_REPL_FLAT, true),
                    makeConfigItem(SYNC_ALLREDUCE, true),

                    makeConfigItem(CONF_DIR, "")
            };

            for (const auto &item: config_items) {
                items_[item.name] = item;
            }

            fs::path conf_file;
            fs::path conf_dir;
            if(const char* conf_dir_var = std::getenv(RANNC_CONF_DIR_VAR.c_str())) {
                conf_dir = conf_dir_var;
                if (!fs::exists(conf_dir)) {
                    throw std::invalid_argument("Config dir not found: " + conf_dir.string());
                }

                conf_file = conf_dir / RANNC_CONF_FILE;
            } else {
                const fs::path home_dir = getHomeDir();
                conf_dir = home_dir / DEFAULT_CONF_DIR_NAME;
                conf_file = conf_dir / RANNC_CONF_FILE;
            }

            if (fs::exists(conf_file)) {
                values_[CONF_DIR] = conf_dir.string();

                const auto data  = toml::parse(conf_file.string());
                for (const auto& it: data.as_table()) {
//                for(const auto& [name, val] : data.as_table())

                    const std::string& name = it.first;
                    toml::value val = it.second;

                    if (!contains(items_, name)) {
                        std::cerr << "Ignored unknown item in " << conf_file << ": " << name << std::endl;
                        continue;
                    }

                    const auto& item = items_.at(name);
                    values_[name] = convertValue(val, item.type);
                }
            }

            // Overwrite by values in env
            for (const auto &item: config_items) {
                std::stringstream ss;
                ss << "RANNC_" << toUpper(item.name);
                std::string env_item_name = ss.str();

                if(const char* env_val = std::getenv(env_item_name.c_str())) {
                    try {
                        values_[item.name] = convertValue(std::string{env_val}, item.type);
                    } catch (std::exception& e) {
                        std::stringstream ss;
                        ss << "Failed to parse a config value " << env_item_name
                            << ": " << env_val;
                        throw std::invalid_argument(ss.str());
                    }
               }
            }
        }

        std::string toString(const ConfigValue& value) {
            std::stringstream ss;
            switch (value.type) {
                case ConfigType::INT: ss << value.int_val; break;
                case ConfigType::FLOAT: ss << value.float_val; break;
                case ConfigType::BOOL: ss << value.bool_val; break;
                case ConfigType::STRING: ss << value.str_val; break;
            }
            return ss.str();
        }
    }
}