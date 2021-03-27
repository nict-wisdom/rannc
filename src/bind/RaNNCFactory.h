//
// Created by Masahiro Tanaka on 2019-06-12.
//

#ifndef PYRANNC_RANNCFACTORY_H
#define PYRANNC_RANNCFACTORY_H

#include <memory>

namespace rannc {
    class RaNNCProcess;

    class RaNNCFactory {
    public:
        static std::shared_ptr<RaNNCProcess> get();

    private:
        RaNNCFactory() = default;
        ~RaNNCFactory() = default;

        static std::shared_ptr<RaNNCProcess> process_;
    };
};


#endif //PYRANNC_RANNCFACTORY_H
