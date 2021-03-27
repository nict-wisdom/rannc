//
// Created by Masahiro Tanaka on 2019-06-12.
//

#include "RaNNCFactory.h"

#include "bind/RaNNCProcess.h"

namespace rannc {
    std::shared_ptr<RaNNCProcess> RaNNCFactory::process_;

    std::shared_ptr<RaNNCProcess> RaNNCFactory::get() {
        if (!process_) {
            process_ = std::make_shared<RaNNCProcess>();
        }
        return process_;
    }
}
