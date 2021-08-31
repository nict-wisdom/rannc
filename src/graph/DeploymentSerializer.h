//
// Created by Masahiro Tanaka on 2020/05/20.
//

#ifndef PYRANNC_DEPLOYMENTSERIALIZER_H
#define PYRANNC_DEPLOYMENTSERIALIZER_H

#include "Decomposition.h"

namespace rannc {
    void save(const std::string& file, const Deployment& deployment,
            int world_size, long dev_mem);

    struct DeploymentState {
        Deployment deployment;
        int world_size;
        long dev_mem;

        MSGPACK_DEFINE(deployment, world_size, dev_mem);
    };

    DeploymentState loadDeploymentState(const std::string& file);
    Deployment loadDeployment(const std::string& file, int world_size, long dev_mem);
    Deployment loadDeployment(const std::string& file);
}

#endif //PYRANNC_DEPLOYMENTSERIALIZER_H
