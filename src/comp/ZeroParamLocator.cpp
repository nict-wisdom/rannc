//
// Created by Masahiro Tanaka on 2021/05/13.
//

#include "ZeroParamLocator.h"

#include "comm/ObjectComm.h"
#include "comm/MPIUtil.h"


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
            spdlog::info("Placed {} on rank {}", pid, min_idx);
            param_size = param.numel() * param.element_size();
            params_[pid] = param;
        }

        ObjectComm& ocomm = ObjectComm::get();
        param_size = ocomm.bcast(param_size, min_idx);

        sizes_[min_idx] += param_size;
        owners_[pid] = min_idx;

        MPI_Barrier(MPI_COMM_WORLD);

        return min_idx;
    }

    at::Tensor ZeroParamLocator::load(long pid) {


        return at::Tensor();
    }


}