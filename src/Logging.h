//
// Created by Masahiro Tanaka on 2019-06-12.
//

#ifndef PYRANNC_LOGGING_H
#define PYRANNC_LOGGING_H

#include "spdlog/spdlog.h"

namespace rannc {
    void initLogger();
    std::shared_ptr<spdlog::logger> getLogger(const std::string& name);
}

#endif //PYRANNC_LOGGING_H
