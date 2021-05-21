//
// Created by Masahiro Tanaka on 2021/05/13.
//

#include "DistributedParamLocator.h"

#include "comm/ObjectComm.h"
#include "comm/MPIUtil.h"
#include "comm/NCCLWrapper.h"
#include "comm/SComm.h"


namespace rannc {

    int DistributedParamLocator::store(long pid, const at::Tensor& param) {
        int owner = doRegister(pid, param, mpi::getAllRanks());
        if (mpi::getRank() == owner) {
            spdlog::info("Placed {} on rank {}: {}", pid, owner, join_as_str(getTensorDim(param)));
            params_[pid] = param;
        }
        return owner;
    }

    at::Tensor DistributedParamLocator::load(long pid) {

        if (!contains(owners_, pid)) {
            std::stringstream ss;
            ss << "Parameter not found in DistributedParamLocator: " << pid;
            throw std::invalid_argument(ss.str());
        }

        int owner = owners_.at(pid);

        assert(contains(ir_types_, pid));

        at::TensorOptions options;
        at::Tensor buf;
        if (mpi::getRank() == owner) {
            assert(contains(params_, pid));
            buf = params_.at(pid).cuda();
        } else {
            options = options.dtype(fromIRTensorElemTypeToScalarType(ir_types_.at(pid).getTensorElemType()))
                    .device(c10::Device(c10::DeviceType::CUDA));
            buf = torch::zeros(ir_types_.at(pid).getTensorDim(), options);
        }

        TagMap& tag_map = TagMap::get();
        int tag = tag_map.getRankSetTag(mpi::getAllRanks());
        nccl_.createCommunicator(tag, mpi::getAllRanks());
        nccl_.bcast(tag, mpi::getAllRanks(), owner, {buf});

        return buf.detach().cpu();
    }

    void DistributedParamLocator::disable(long pid) {
        owners_.erase(pid);
        params_.erase(pid);
        ir_types_.erase(pid);
    }

    void DistributedParamLocator::fetchStart() {
        TagMap& tag_map = TagMap::get();
        comm_tag_ = tag_map.getRankSetTag(mpi::getAllRanks());
        nccl_.createCommunicator(comm_tag_, mpi::getAllRanks());

        if (mpi::getRank() != 0) {
            long global_pid;
            MPI_Status status;
            MPI_Recv(&global_pid, 1, MPI_LONG, 0, FETCH_TAG, MPI_COMM_WORLD, &status);

            while (global_pid != 0) {
                assert(contains(global_id_to_local_, global_pid));
                long pid = global_id_to_local_.at(global_pid);
                assert(contains(params_, pid));
                const auto param = params_.at(pid).cuda();
                nccl_.send(comm_tag_, 0, param);

                MPI_Recv(&global_pid, 1, MPI_LONG, 0, FETCH_TAG, MPI_COMM_WORLD, &status);
            }
        }
    }

    at::Tensor DistributedParamLocator::fetch(long pid) {
        assert(contains(owners_, pid));
        int owner = owners_.at(pid);

        if (mpi::getRank() == owner) {
            assert(contains(params_, pid));
            return params_.at(pid);
        }

        MPI_Send(&pid, 1, MPI_LONG, owner, FETCH_TAG, MPI_COMM_WORLD);

        at::TensorOptions options;
        options = options.dtype(fromIRTensorElemTypeToScalarType(ir_types_.at(pid).getTensorElemType()))
                .device(c10::Device(c10::DeviceType::CUDA))
                .requires_grad(true);
        auto buf = torch::zeros(ir_types_.at(pid).getTensorDim(), options);
        nccl_.recv(comm_tag_, owner, buf);
        return buf;
    }

    void DistributedParamLocator::fetchEnd() {
        if (mpi::getRank() == 0) {
            long pid = 0;
            for (int i=1; i < mpi::getSize(); i++) {
                MPI_Send(&pid, 1, MPI_LONG, i, FETCH_TAG, MPI_COMM_WORLD);
            }
        }
    }
}