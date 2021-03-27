//
// Created by Masahiro Tanaka on 2018/11/28.
//

#ifndef PT_RANNC_CALLRANNC_H
#define PT_RANNC_CALLRANNC_H

#include <torch/csrc/jit/ir/ir.h>

#include <Logging.h>
#include "comp/TimeCounter.h"
#include "comp/GraphLauncher.h"
#include "graph/Decomposition.h"

namespace rannc {
    class RaNNCModule;

    class RaNNCProcess {
    public:
        RaNNCProcess() = default;

        void start();

        void registerModule(const std::string& id, RaNNCModule* module);
        void unregisterModule(const std::string& id);
        std::unordered_map<std::string, RaNNCModule*> getModules();

        void clear();

        const std::shared_ptr<ParamStorage> &getParamStorage() const {
            return param_storage_;
        }

    private:
        std::unordered_map<std::string, RaNNCModule*> modules_;
        std::shared_ptr<ParamStorage> param_storage_;
        const std::shared_ptr<spdlog::logger> logger = getLogger("RaNNCProcess");
    };
}

#endif //PT_RANNC_CALLRANNC_H
