//
// Created by Masahiro Tanaka on 2020/01/31.
//

#ifndef PYRANNC_GUESSVALUETYPES_H
#define PYRANNC_GUESSVALUETYPES_H

#include <Common.h>

#include "ir.h"

namespace rannc {
    std::shared_ptr<IRGraph> guessValueTypes(const std::shared_ptr<IRGraph> &g);
}

#endif //PYRANNC_GUESSVALUETYPES_H
