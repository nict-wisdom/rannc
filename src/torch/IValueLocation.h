//
// Created by Masahiro Tanaka on 2019-07-12.
//

#ifndef PYRANNC_IVALUELOCATION_H
#define PYRANNC_IVALUELOCATION_H

#include <msgpack.hpp>

namespace rannc {
    enum class StepTypeInIValue {
        LIST,
        TUPLE
    };
}
MSGPACK_ADD_ENUM(rannc::StepTypeInIValue);


namespace rannc {
    struct StepInIValue {
        StepInIValue() : index(-1) {}

        StepInIValue(StepTypeInIValue type, size_t index) : type(type), index(index) {}

        StepTypeInIValue type;
        size_t index;

        bool operator==(const StepInIValue &rhs) const {
            return type == rhs.type &&
                   index == rhs.index;
        }

        bool operator!=(const StepInIValue &rhs) const {
            return !(rhs == *this);
        }

        MSGPACK_DEFINE (type, index);
    };

    using PathInIValue = std::vector<StepInIValue>;

    std::string toString(const PathInIValue &path);

    struct IValueLocation {
        IValueLocation() = default;

        IValueLocation(std::string name) : value_name(std::move(name)) {
        }

        IValueLocation(std::string name, PathInIValue path) : value_name(std::move(name)), path(std::move(path)) {
        }

        std::string value_name;
        PathInIValue path;

        inline bool operator==(const IValueLocation &rhs) const {
            const IValueLocation &lhs = *this;
            return lhs.value_name == rhs.value_name && toString(lhs.path) == toString(rhs.path);
        }

        inline bool operator!=(const IValueLocation &rhs) const {
            return !(this->operator==(rhs));
        }

        MSGPACK_DEFINE (value_name, path);
    };

    std::string toString(const IValueLocation &loc);
    std::string toString(const std::vector<IValueLocation> &locs);

    struct IValueLocationHash {
        std::size_t operator()(const IValueLocation &path) const {
            return std::hash<std::string>()(path.value_name + toString(path.path));
        };
    };

    IValueLocation createListElem(const IValueLocation &loc, int index);
    IValueLocation createTupleElem(const IValueLocation &loc, int index);
}

#endif //PYRANNC_IVALUELOCATION_H
