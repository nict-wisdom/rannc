//
// Created by Masahiro Tanaka on 2019-06-12.
//
#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>

#include <boost/filesystem.hpp>
#include <iostream>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog_setup/conf.h>

#include "Config.h"
#include "Logging.h"

namespace fs = boost::filesystem;

namespace {
bool initialized = false;
}

namespace rannc {
const std::string RANNC_LOG_FILE = "logging.toml";

void initLogger() {
  config::Config& conf = config::Config::get();

  fs::path conf_dir = conf.getVal<std::string>(config::CONF_DIR);
  const fs::path conf_file = conf_dir / RANNC_LOG_FILE;

  if (fs::exists(conf_file)) {
    spdlog_setup::from_file(conf_file.string());
  }
}

std::shared_ptr<spdlog::logger> getLogger(const std::string& name) {
  if (!initialized) {
    initLogger();
    initialized = true;
  }

  auto logger = spdlog::get(name);
  if (!logger) {
    logger = spdlog::stderr_color_mt(name);
  }
  return logger;
}
} // namespace rannc