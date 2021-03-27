//
// Created by Masahiro Tanaka on 2018/11/28.
//

#include <iostream>

#include <torch/torch.h>

#include <comm/MPIUtil.h>
#include <Common.h>
#include <comp/NodeProfiler.h>
#include <comm/ObjectComm.h>
#include <cuda/CudaUtil.h>

#include "RaNNCProcess.h"
#include "graph/FairWeightDecomposer.h"


namespace rannc {

    void RaNNCProcess::start() {
        int provided;
        int required = MPI_THREAD_SINGLE;
        MPI_Init_thread(nullptr, nullptr, required, &provided);

        const std::string master_node = mpi::getProcessorName();
        logger->info("RaNNC started on rank {} ({})", mpi::getRank(), master_node);

        config::Config& conf = config::Config::get();
        if (mpi::getRank() == 0 && conf.getVal<bool>(config::SHOW_CONFIG_ITEMS)) {
            conf.display();
        }

        ObjectComm& ocomm = ObjectComm::get();

        std::unordered_map<std::string, std::unordered_set<int>> my_devices;
        my_devices[mpi::getProcessorName()] = getCudaDeviceIds();
        auto all_devices = ocomm.allgather(my_devices);

        std::unordered_map<std::string, std::unordered_set<int>> node_ranks;
        std::unordered_map<std::string, std::vector<int>> node_devices;

        int rank_idx = 0;
        for (const auto& d: all_devices) {
            for (const auto& it: d) {
                auto devs = setToVector(it.second);
                std::sort(devs.begin(), devs.end());
                node_devices[it.first] = devs;
                node_ranks[it.first].insert(rank_idx);
            }
            rank_idx++;
        }

        bool no_cuda = false;
        for (const auto& it: node_devices) {
            if (it.second.empty()) {
                no_cuda = true;
            }
        }
        if (no_cuda) {
            if (mpi::isMaster()) {
                logger->warn("One or more worker nodes have no CUDA devices. RaNNC may work in the CPU mode.");
            }
        } else {
            std::unordered_map<int, int> dev_alloc; // rank -> dev
            if (mpi::isMaster()) {
                logger->info("CUDA device assignments:");
            }

            for (const auto &node_name: keys(node_ranks)) {
                auto node_ranks_vec = setToVector(node_ranks.at(node_name));
                std::sort(node_ranks_vec.begin(), node_ranks_vec.end());

                const auto &node_dev_vecs = node_devices.at(node_name);

                size_t dev_count = node_dev_vecs.size();
                if (dev_count < node_ranks_vec.size()) {
                    std::stringstream ss;
                    ss << "Not enough CUDA devices are available on " << node_name << ". #devices="
                       << node_dev_vecs.size()
                       << " #workers=" << node_ranks_vec.size();
                    throw std::runtime_error(ss.str());
                }

                for (size_t i = 0; i < node_ranks_vec.size(); i++) {
                    int r = node_ranks_vec.at(i);
                    int dev = node_dev_vecs.at(i);
                    dev_alloc[r] = dev;
                    if (mpi::isMaster()) {
                        logger->info(" Worker {}: device{}@{}", r, dev, node_name);
                    }
                }
            }

            cudaSetDevice(dev_alloc.at(mpi::getRank()));
        }

        param_storage_ = std::make_shared<ParamStorage>();
   }

    void RaNNCProcess::registerModule(const std::string& id, RaNNCModule* module) {
        modules_[id] = module;
    }
    void RaNNCProcess::unregisterModule(const std::string& id) {
        modules_.erase(id);
    }
    std::unordered_map<std::string, RaNNCModule*> RaNNCProcess::getModules() {
        return modules_;
    }

    void RaNNCProcess::clear() {
        SComm::get().destroy();
        AllReduceRunner::get().destroy();

        if (param_storage_) {
            param_storage_->clear();
        }
    }
}
