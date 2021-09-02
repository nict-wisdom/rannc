//
// Created by Masahiro Tanaka on 2019-06-26.
//

#ifndef PYRANNC_CONFIG_H
#define PYRANNC_CONFIG_H

#include <unordered_map>
#include <toml.hpp>

#include "Common.h"

namespace rannc {

    namespace config {
        extern const char SHOW_CONFIG_ITEMS[];
        extern const char PROFILING[];
        extern const char GRAPH_PROFILING[];
        extern const char PROFILE_DUMP_PATH[];
        extern const char DUMP_GRAPH[];
        extern const char DUMP_GRAPH_PREFIX[];
        extern const char PARTITION_NUM[];
        extern const char REPLICA_NUM[];
        extern const char PIPELINE_NUM[];
        extern const char VALIDATE_COMM[];
        extern const char DISPLAY_COMM_VALUE[];
        extern const char DISPLAY_ACT_VALUE[];
        extern const char CONSOLIDATE_GRADS[];
        extern const char PROFILING_ITER[];
        extern const char CHECKPOINTING[];
        extern const char CHECKPOINTING_NO_LAST[];
        extern const char DISABLE_BALANCING[];
        extern const char OPT_PARAM_FACTOR[];
        extern const char AUTO_PARALLEL[];
        extern const char P2P_COMM[];
        extern const char DECOMPOSER[];
        extern const char USE_AMP_MASTER_PARAM[];
        extern const char MEM_LIMIT_GB[];
        extern const char MIN_PARTITON_NUM[];
        extern const char MAX_PARTITON_NUM[];
        extern const char MEM_MARGIN[];
        extern const char MAX_MP_NUM[];
        extern const char DO_UNCOARSENING[];
        extern const char DO_COARSENING[];
        extern const char MIN_PIPELINE[];
        extern const char MAX_PIPELINE[];
        extern const char SAVE_DEPLOYMENT[];
        extern const char LOAD_DEPLOYMENT[];
        extern const char DEPLOYMENT_FILE[];
        extern const char SAVE_GRAPH_PROFILE[];
        extern const char LOAD_GRAPH_PROFILE[];
        extern const char GRAPH_PROFILE_FILE[];
        extern const char TRACE_EVENTS[];
        extern const char EVENT_TRACE_FILE[];
        extern const char VERIFY_RECOMP[];
        extern const char DP_SEARCH_ALL[];
        extern const char MIN_PIPELINE_BS[];
        extern const char COARSEN_BY_TIME[];
        extern const char LIMIT_DEV_NUM_POT[];
        extern const char LIMIT_DEV_NUM_MORE_THAN_BS[];
        extern const char PROFILE_BY_ACC[];
        extern const char VERIFY_PARTITIONING[];
        extern const char ALLOC_REPL_FLAT[];
        extern const char SYNC_ALLREDUCE[];
        extern const char DUMP_DP_NODE_PROFILES[];
        extern const char DUMP_DP_CACHE[];
        extern const char PARTITIONING_DRY_RUN_NP[];
        extern const char CONF_DIR[]; // this is special because Config itself sets this item

        enum class ConfigType {
            INT,
            FLOAT,
            BOOL,
            STRING
        };

        struct ConfigValue {
            ConfigValue() {}
            ConfigValue(int v) : int_val(v), type(ConfigType::INT) {}
            ConfigValue(float v) : float_val(v), type(ConfigType::FLOAT) {}
            ConfigValue(bool v) : bool_val(v), type(ConfigType::BOOL) {}
            ConfigValue(std::string v) : str_val(std::move(v)), type(ConfigType::STRING) {}

            template <typename T>
            T get() const;

            template <typename T>
            void set(T val);

            int int_val;
            float float_val;
            bool bool_val;
            std::string str_val;

            ConfigType type;
        };

        template <>
        inline int ConfigValue::get() const { return int_val; }

        template <>
        inline float ConfigValue::get() const { return float_val; }

        template <>
        inline bool ConfigValue::get() const { return bool_val; }

        template <>
        inline std::string ConfigValue::get() const { return str_val; }

        template <>
        inline void ConfigValue::set(int val) { int_val = val; }

        template <>
        inline void ConfigValue::set(float val) { float_val = val; }

        template <>
        inline void ConfigValue::set(bool val) { bool_val = val; }

        template <>
        inline void ConfigValue::set(std::string val) { str_val = std::move(val); }

        struct ConfigItem {
            std::string name;
            ConfigType type;
            ConfigValue default_val;
        };

        template <typename T>
        struct ValueTypeTrait;

        template <>
        struct ValueTypeTrait<int> {
            static constexpr ConfigType type = ConfigType::INT;
        };

        template <>
        struct ValueTypeTrait<float> {
            static constexpr ConfigType type = ConfigType::FLOAT;
        };

        template <>
        struct ValueTypeTrait<bool> {
            static constexpr ConfigType type = ConfigType::BOOL;
        };

        template <>
        struct ValueTypeTrait<std::string> {
            static constexpr ConfigType type = ConfigType::STRING;
        };

        template <>
        struct ValueTypeTrait<const char*> {
            static constexpr ConfigType type = ConfigType::STRING;
        };

        template <typename T>
        ConfigItem makeConfigItem(const std::string& name, T val) {
            return {name, ValueTypeTrait<T>::type, val};
        }

        class Config {
        public:
            Config(const Config&) = delete;
            Config& operator=(const Config&) = delete;
            Config(Config&&) = delete;
            Config& operator=(Config&&) = delete;

            static Config& get() {
                static Config instance;
                return instance;
            }

            template <typename T>
            T getDefaultVal(const std::string& name) {
                const auto& item = items_.at(name);
                return item.default_val.get<T>();
            }

            template <typename T>
            T getVal(const std::string& name) {
                if (contains(values_, name)) {
                    return values_.at(name).get<T>();
                }
                return getDefaultVal<T>(name);
            }

            template <typename T>
            void setVal(const std::string& name, T val) {
                values_[name].set(val);
            }

            void display();

        private:
            Config();
            ~Config() = default;

            ConfigValue convertValue(const toml::value& val, const ConfigType& type);
            ConfigValue convertValue(const std::string& val, const ConfigType& type);

            std::unordered_map<std::string, ConfigItem> items_;
            std::unordered_map<std::string, ConfigValue> values_;
        };

        std::string toString(const ConfigValue& value);
    }
}

#endif //PYRANNC_CONFIG_H
