//
// Created by Masahiro Tanaka on 2019-06-26.
//

#ifndef PYRANNC_CONFIG_H
#define PYRANNC_CONFIG_H

#include <toml.hpp>
#include <unordered_map>

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
extern const char MEM_LIMIT_GB[];
extern const char MIN_PARTITION_NUM[];
extern const char MAX_PARTITION_NUM[];
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

extern const char SAVE_MLPART_RESULTS[];
extern const char LOAD_MLPART_RESULTS[];
extern const char MLPART_RESULTS_FILE[];

extern const char SAVE_ALLOC_SOLUTIONS[];
extern const char LOAD_ALLOC_SOLUTIONS[];
extern const char ALLOC_SOLUTIONS_FILE_PREFIX[];

extern const char DUMP_DP_NODE_PROFILES[];
extern const char DUMP_DP_CACHE[];
extern const char PARTITIONING_DRY_RUN_NP[];
extern const char USE_MPI_TO_GATHER_DIST_PARAMS[];
extern const char RUN_WATCHDOG[];
extern const char WATCHDOG_LOCKFILE_DIR[];
extern const char FORCE_DIST_MATMUL[];
extern const char USE_NAMED_TENSORS[];
extern const char PROFILER_CACHE_SIZE[];
extern const char ENABLE_KINETO[];

extern const char
    CONF_DIR[]; // this is special because Config itself sets this item

enum class ConfigType { INT, FLOAT, BOOL, STRING };

struct ConfigValue {
  ConfigValue() {}
  ConfigValue(int v) : int_val(v), type(ConfigType::INT) {}
  ConfigValue(float v) : float_val(v), type(ConfigType::FLOAT) {}
  ConfigValue(bool v) : bool_val(v), type(ConfigType::BOOL) {}
  ConfigValue(std::string v)
      : str_val(std::move(v)), type(ConfigType::STRING) {}

  template <typename T>
  T get() const;

  int int_val;
  float float_val;
  bool bool_val;
  std::string str_val;

  ConfigType type;
};

template <>
inline int ConfigValue::get() const {
  return int_val;
}

template <>
inline float ConfigValue::get() const {
  return float_val;
}

template <>
inline bool ConfigValue::get() const {
  return bool_val;
}

template <>
inline std::string ConfigValue::get() const {
  return str_val;
}

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
    try {
      static Config instance;
      return instance;
    } catch (const std::exception& e) {
      std::cerr << "Error: " << e.what() << std::endl;
      std::exit(1);
    }
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
    values_[name] = ConfigValue(val);
  }

  bool hasValue(const std::string& name) const {
    return contains(values_, name);
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
} // namespace config
} // namespace rannc

#endif // PYRANNC_CONFIG_H
