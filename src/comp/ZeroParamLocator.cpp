//
// Created by Masahiro Tanaka on 2021/05/13.
//

#include "ZeroParamLocator.h"

#include "comm/ObjectComm.h"
#include "comm/MPIUtil.h"
#include "comm/NCCLWrapper.h"
#include "comm/SComm.h"


namespace rannc {
    const int ZeroParamLocator::FETCH_TAG = 10;

    int ZeroParamLocator::store(long pid, const at::Tensor& param) {

        int np = mpi::getSize();

        int64_t min_size = INT64_MAX;
        int min_idx = -1;
        for (int i=0; i<np; i++) {
            if (sizes_[i] < min_size) {
                min_size = sizes_[i];
                min_idx = i;
            }
        }

        int64_t param_size = param.numel() * param.element_size();
        if (mpi::getRank() == min_idx) {
            spdlog::info("Placed {} on rank {}: {}", pid, min_idx, join_as_str(getTensorDim(param)));
            params_[pid] = param;
        }

        ObjectComm& ocomm = ObjectComm::get();
        long global_id = pid;
        global_id = ocomm.bcast(global_id);
        global_id_to_local_[global_id] = pid;

        sizes_[min_idx] += param_size;
        owners_[pid] = min_idx;
        ir_types_[pid] = toIRType(param);

//        IRType ir_type = toIRType(param);
//        assert(ir_type.getBaseType() == IRBaseType::TENSOR);
//        elem_types_[pid] = ir_type.getTensorElemType();

        MPI_Barrier(MPI_COMM_WORLD);

        return min_idx;
    }

    at::Tensor ZeroParamLocator::load(long pid) {

        if (!contains(owners_, pid)) {
            std::stringstream ss;
            ss << "Parameter not found in ZeroParamLocator: " << pid;
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

    void ZeroParamLocator::fetchStart() {
        TagMap& tag_map = TagMap::get();
        int tag = tag_map.getRankSetTag(mpi::getAllRanks());
        nccl_.createCommunicator(tag, mpi::getAllRanks());

        if (mpi::getRank() != 0) {
            long global_pid;
            MPI_Status status;
            MPI_Recv(&global_pid, 1, MPI_LONG, 0, FETCH_TAG, MPI_COMM_WORLD, &status);

            while (global_pid != 0) {
                assert(contains(global_id_to_local_, global_pid));
                long pid = global_id_to_local_.at(global_pid);
                assert(contains(params_, pid));
                const auto param = params_.at(pid).cuda();
                nccl_.send(tag, 0, param);

                MPI_Recv(&global_pid, 1, MPI_LONG, 0, FETCH_TAG, MPI_COMM_WORLD, &status);
            }
        }
    }

    at::Tensor ZeroParamLocator::fetch(long pid) {
        assert(contains(owners_, pid));
        int owner = owners_.at(pid);

        TagMap& tag_map = TagMap::get();
        int tag = tag_map.getRankSetTag(mpi::getAllRanks());

        at::TensorOptions options;
        options = options.dtype(fromIRTensorElemTypeToScalarType(ir_types_.at(pid).getTensorElemType()))
                .device(c10::Device(c10::DeviceType::CUDA))
                .requires_grad(true);
        auto buf = torch::zeros(ir_types_.at(pid).getTensorDim(), options);
        nccl_.recv(tag, owner, buf);
        return buf;
    }

    void ZeroParamLocator::fetchEnd() {
        if (mpi::getRank() == 0) {
            long pid = 0;
            for (int i=1; i<mpi::getSize(); i++) {
                MPI_Send(&pid, 1, MPI_LONG, i, FETCH_TAG, MPI_COMM_WORLD);
            }
        }
    }
}