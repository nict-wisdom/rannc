//
// Created by Masahiro Tanaka on 2021/05/19.
//

#include <comm/ObjectComm.h>
#include "DistributedParamLocatorBase.h"

namespace rannc {
    const int DistributedParamLocatorBase::FETCH_TAG = 10;

    int DistributedParamLocatorBase::doRegister(long pid, const at::Tensor& param) {

        int np = mpi::getSize();

        int64_t min_size = INT64_MAX;
        int min_idx = -1;
        for (int i=0; i<np; i++) {
            if (sizes_[i] < min_size) {
                min_size = sizes_[i];
                min_idx = i;
            }
        }

        ObjectComm& ocomm = ObjectComm::get();
        long global_id = pid;
        global_id = ocomm.bcast(global_id);
        global_id_to_local_[global_id] = pid;

        int64_t param_size = param.numel() * param.element_size();
        sizes_[min_idx] += param_size;
        owners_[pid] = min_idx;
        ir_types_[pid] = toIRType(param);

        MPI_Barrier(MPI_COMM_WORLD);

        return min_idx;
    }

    int DistributedParamLocatorBase::getOwner(long pid) {
        assert(contains(owners_, pid));
        return owners_.at(pid);
    }
}