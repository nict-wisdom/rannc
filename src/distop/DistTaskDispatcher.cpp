//
// Created by Masahiro Tanaka on 2022/02/04.
//

#include "DistTaskDispatcher.h"
#include <comm/SComm.h>

namespace rannc {

DistTaskDispatcher::DistTaskDispatcher()
    : nccl_(NCCLWrapper::get()), dpl_(DistributedParamLocator::get()) {}

void DistTaskDispatcher::start() {
  TagMap& tag_map = TagMap::get();
  comm_tag_ = tag_map.getRankSetTag(mpi::getAllRanks());
  nccl_.createCommunicator(comm_tag_, mpi::getAllRanks());

  if (mpi::getRank() != 0) {
    int task_type_buf;

    bool running = true;
    while (running) {
      MPI_Bcast(&task_type_buf, 1, MPI_INT, 0, MPI_COMM_WORLD);

      const auto task_type = static_cast<DistTaskType>(task_type_buf);
      switch (task_type) {
        case DistTaskType::GET_PARAM: {
          logger->trace("Received GET_PARAM");
          long param_id;
          MPI_Bcast(&param_id, 1, MPI_LONG, 0, MPI_COMM_WORLD);
          logger->trace("Received pid={}", param_id);
          dpl_.load(dpl_.pidToLocal(param_id));
          break;
        }
        case DistTaskType::PROFILE:
          logger->trace("Received PROFILE");
          break;
        case DistTaskType::STOP:
          logger->trace("Received STOP");
          running = false;
          break;
      }
    }
  }
}

at::Tensor DistTaskDispatcher::getParam(long param_id) {
  int task_type_buf = static_cast<int>(DistTaskType::GET_PARAM);
  MPI_Bcast(&task_type_buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&param_id, 1, MPI_LONG, 0, MPI_COMM_WORLD);

  return dpl_.load(dpl_.pidToLocal(param_id));
}

void DistTaskDispatcher::stop() {
  if (mpi::getRank() == 0) {
    int task_type_buf = static_cast<int>(DistTaskType::STOP);
    MPI_Bcast(&task_type_buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }
}

}; // namespace rannc
