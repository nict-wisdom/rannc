//
// Created by Masahiro Tanaka on 2019-07-22.
//

#include "AllReduceRunner.h"

#include <comm/SComm.h>

namespace rannc {
    struct AllReduceComm {
    };

    void AllReduceRunner::createCommunicator(int tag, const std::unordered_set<int>& ranks) {
    }

    void AllReduceRunner::destroy() {
    }

    void doReduce(const std::unordered_map<int, std::vector<at::Tensor>> &param_grads, bool allreduce) {
        throw std::logic_error("MPI allreduce is not supported.");
    }

    void AllReduceRunner::allreduce(const std::unordered_map<int, std::vector<at::Tensor>> &param_grads) {
        doReduce(param_grads, true);
    }

    void AllReduceRunner::reduce(const std::unordered_map<int, std::vector<at::Tensor>> &param_grads) {
        doReduce(param_grads, false);
    }

    std::string AllReduceRunner::getImplName() {
        return "MPI";
    }
}
