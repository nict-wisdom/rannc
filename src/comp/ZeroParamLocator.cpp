//
// Created by Masahiro Tanaka on 2021/05/13.
//

#include "ZeroParamLocator.h"

#include "comm/ObjectComm.h"
#include "comm/MPIUtil.h"
#include "comm/NCCLWrapper.h"
#include "comm/SComm.h"


namespace rannc {

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

        int64_t param_size = 0;
        if (mpi::getRank() == min_idx) {
            spdlog::info("Placed {} on rank {}: {}", pid, min_idx, join_as_str(getTensorDim(param)));
            param_size = param.numel() * param.element_size();
            params_[pid] = param;
        }

        ObjectComm& ocomm = ObjectComm::get();
        param_size = ocomm.bcast(param_size, min_idx);

        sizes_[min_idx] += param_size;
        owners_[pid] = min_idx;
        shapes_[pid] = getTensorDim(param);

        IRType ir_type = toIRType(param);
        assert(ir_type.getBaseType() == IRBaseType::TENSOR);
        elem_types_[pid] = ir_type.getTensorElemType();

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

        assert(contains(shapes_, pid));
        assert(contains(elem_types_, pid));

        NCCLWrapper& nccl = NCCLWrapper::get();

        at::TensorOptions options;
        at::Tensor buf;
        if (mpi::getRank() == owner) {
            assert(contains(params_, pid));
            buf = params_.at(pid);
        } else {
            options = options.dtype(fromIRTensorElemTypeToScalarType(elem_types_.at(pid)))
                    .device(c10::Device(c10::DeviceType::CUDA));
            buf = torch::zeros(shapes_.at(pid), options);
        }

        TagMap& tag_map = TagMap::get();
        int tag = tag_map.getRankSetTag(mpi::getAllRanks());
        nccl.createCommunicator(tag, mpi::getAllRanks());
        nccl.bcast(tag, mpi::getAllRanks(), owner, {buf});

        return buf.detach().cpu();
    }


}