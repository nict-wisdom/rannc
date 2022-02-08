//
// Created by Masahiro Tanaka on 2019-07-22.
//

#include <nccl.h>

#include "NCCLWrapper.h"
#include "ObjectComm.h"

#include <comm/SComm.h>
#include <comp/EventRecorder.h>
#include <cuda/CudaUtil.h>

namespace rannc {

int DEFAULT_TAG = 100;
constexpr size_t NCCL_MAX_COLL_OP_NUM = 2048;

void NCCLBulkJobExecutor::flush() {
  NCCLWrapper& nccl = NCCLWrapper::get();
  nccl.syncWithErrorCheck();
  for (const auto& f : pre_comm_jobs_) {
    f();
  }
  nccl.syncWithErrorCheck();
  for (const auto& f : comm_jobs_) {
    ncclGroupStart();
    f();
    ncclGroupEnd();
  }
  nccl.syncWithErrorCheck();
  for (const auto& f : post_comm_jobs_) {
    f();
  }
  nccl.syncWithErrorCheck();

  pre_comm_jobs_.clear();
  comm_jobs_.clear();
  post_comm_jobs_.clear();
}

void NCCLBulkJobExecutor::doAddJob(
    std::function<void(void)> f, std::vector<std::function<void(void)>>& jobs) {
  if (run_immediate_) {
    f();
  } else {
    jobs.push_back(f);
  }
}

void NCCLBulkJobExecutor::addPreCommJob(std::function<void(void)> f) {
  if (run_immediate_) {
    ncclGroupStart();
  }
  doAddJob(f, pre_comm_jobs_);
  if (run_immediate_) {
    ncclGroupEnd();

    NCCLWrapper& nccl = NCCLWrapper::get();
    nccl.syncWithErrorCheck();
  }
}
void NCCLBulkJobExecutor::addCommJob(std::function<void(void)> f) {
  if (run_immediate_) {
    ncclGroupStart();
  }
  doAddJob(f, comm_jobs_);
  if (run_immediate_) {
    ncclGroupEnd();
    NCCLWrapper& nccl = NCCLWrapper::get();
    nccl.syncWithErrorCheck();
  }
}
void NCCLBulkJobExecutor::addPostCommJob(std::function<void(void)> f) {
  if (run_immediate_) {
    ncclGroupStart();
  }
  doAddJob(f, post_comm_jobs_);
  if (run_immediate_) {
    ncclGroupEnd();
    NCCLWrapper& nccl = NCCLWrapper::get();
    nccl.syncWithErrorCheck();
  }
}

struct AllReduceComm {
  ncclComm_t* comm;
};

void NCCLWrapper::createCommunicator(
    int tag, const std::unordered_set<int>& ranks) {
  if (contains(comm_map_, tag)) {
    return;
  }

  std::vector<int> rank_vec = setToVector(ranks);
  std::sort(rank_vec.begin(), rank_vec.end());
  logger->trace(
      "Creating nccl comm. tag={} ranks={}", tag, join_as_str(rank_vec));

  int local_root = rank_vec.front();
  ncclUniqueId id;

  if (mpi::getRank() == local_root) {
    ncclGetUniqueId(&id);

    for (int r : rank_vec) {
      if (r != local_root) {
        MPI_Send(
            (void*)&id, sizeof(id), MPI_BYTE, r, DEFAULT_TAG, MPI_COMM_WORLD);
      }
    }
  } else {
    MPI_Status st;
    MPI_Recv(
        (void*)&id, sizeof(id), MPI_BYTE, local_root, DEFAULT_TAG,
        MPI_COMM_WORLD, &st);
  }

  int local_rank = getLocalRank(vectorToSet(rank_vec), mpi::getRank());
  ncclComm_t* ncomm = new ncclComm_t;
  ncclCommInitRank(ncomm, rank_vec.size(), id, local_rank);

  auto* comm_info = new AllReduceComm;
  comm_info->comm = ncomm;
  comm_map_[tag] = comm_info;

  ranks_to_tag_[ranks] = tag;

  logger->trace("Finished creating nccl comm. tag={}", tag);
}

void NCCLWrapper::destroy() {
  destroyAllCommunicators();
  comm_map_.clear();
  ranks_to_tag_.clear();
  buf_cache_.clear();
}

ncclDataType_t getReduceNcclDataType(const at::Tensor& t) {
  ncclDataType_t datatype;
  switch (t.scalar_type()) {
    case at::ScalarType::Half:
      datatype = ncclFloat16;
      break;
    case at::ScalarType::Float:
      datatype = ncclFloat32;
      break;
    case at::ScalarType::Double:
      datatype = ncclFloat64;
      break;
    case at::ScalarType::Int:
      datatype = ncclInt32;
      break;
    case at::ScalarType::Long:
      datatype = ncclInt64;
      break;

#if defined(__NCCL_SUPPORTS_BFLOAT16__)
    case at::ScalarType::BFloat16:
      datatype = ncclBfloat16;
      break;
#else
    case at::ScalarType::BFloat16:
      datatype = ncclFloat32;
      break;
#endif
    default:
      std::stringstream ss;
      ss << "Unsupported type given to NCCL: " << toString(t.scalar_type());
      throw std::invalid_argument(ss.str());
  }
  return datatype;
}

ncclDataType_t getRedistNcclDataType(const IRTensorElemType t) {
  ncclDataType_t datatype;
  switch (t) {
    case IRTensorElemType::HALF:
      datatype = ncclFloat16;
      break;
    case IRTensorElemType::FLOAT:
      datatype = ncclFloat32;
      break;
    case IRTensorElemType::DOUBLE:
      datatype = ncclFloat64;
      break;
    case IRTensorElemType::INT:
      datatype = ncclInt32;
      break;
    case IRTensorElemType::LONG:
      datatype = ncclInt64;
      break;

#if defined(__NCCL_SUPPORTS_BFLOAT16__)
    case IRTensorElemType::BFLOAT16:
      datatype = ncclBfloat16;
      break;
#else
    case IRTensorElemType::BFLOAT16:
      datatype = ncclFloat16;
      break;
#endif

    default:
      std::stringstream ss;
      ss << "Unsupported type given to NCCL: " << toString(t);
      throw std::invalid_argument(ss.str());
  }
  return datatype;
}

void runCollectiveCommBuf(
    std::unordered_map<int, AllReduceComm*>& comm_map, int tag,
    const std::vector<at::Tensor>& send_tensors,
    const std::vector<at::Tensor>& recv_tensors, const std::vector<int>& roots,
    const std::string& op_name,
    const std::function<ncclResult_t(
        void*, void*, size_t, int, ncclDataType_t, ncclComm_t*)>& f) {
  assert(send_tensors.size() == recv_tensors.size() || recv_tensors.empty());
  assert(send_tensors.size() == roots.size() || roots.empty());

  assert(contains(comm_map, tag));
  AllReduceComm* comm_info = comm_map.at(tag);
  ncclComm_t* ncomm = comm_info->comm;

  // NCCL's limitation
  assert(send_tensors.size() <= NCCL_MAX_COLL_OP_NUM);

  std::stringstream ss;
  size_t elem_sum = 0;
  for (const auto& grad : send_tensors) {
    elem_sum += getTensorElemCount(grad);
  }
  ss << "nccl_" << op_name << "_tag_" << tag << "_elem_" << elem_sum;
  recordStart(ss.str());

#if not defined(__NCCL_SUPPORTS_BFLOAT16__)
  for (size_t i = 0; i < send_tensors.size(); i++) {
    const auto& ten = send_tensors.at(i);
    if (ten.scalar_type() == c10::ScalarType::BFloat16) {
      throw std::runtime_error("Parameters in bfloat16 are not supported.");
    }
  }
#endif

  ncclGroupStart();
  for (size_t index = 0; index < send_tensors.size(); index++) {
    const auto& send_ten = send_tensors.at(index);

    assert(send_ten.is_contiguous());
    void* sendptr = send_ten.data_ptr();
    void* recvptr =
        recv_tensors.empty() ? sendptr : recv_tensors.at(index).data_ptr();

    ncclDataType_t datatype = getReduceNcclDataType(send_ten);

    int root = index < roots.size() ? roots.at(index) : -1;
    f(sendptr, recvptr, getTensorElemCount(send_ten), root, datatype, ncomm);
  }
  ncclGroupEnd();

  recordEnd(ss.str());
}

void runCollectiveComm(
    std::unordered_map<int, AllReduceComm*>& comm_map, int tag,
    const std::vector<at::Tensor>& send_tensors,
    const std::vector<at::Tensor>& recv_tensors, const std::vector<int>& roots,
    const std::string& op_name,
    const std::function<ncclResult_t(
        void*, void*, size_t, int, ncclDataType_t, ncclComm_t*)>& f) {
  assert(send_tensors.size() == recv_tensors.size() || recv_tensors.empty());
  assert(send_tensors.size() == roots.size() || roots.empty());

  std::vector<at::Tensor> send_tensors_buf;
  std::vector<at::Tensor> recv_tensors_buf;
  std::vector<int> roots_buf;

  send_tensors_buf.reserve(NCCL_MAX_COLL_OP_NUM);
  recv_tensors_buf.reserve(NCCL_MAX_COLL_OP_NUM);
  roots_buf.reserve(NCCL_MAX_COLL_OP_NUM);

  for (size_t i = 0; i < send_tensors.size(); i++) {
    send_tensors_buf.push_back(send_tensors.at(i));
    if (!recv_tensors.empty()) {
      recv_tensors_buf.push_back(recv_tensors.at(i));
    }
    if (!roots.empty()) {
      roots_buf.push_back(roots.at(i));
    }

    if (send_tensors_buf.size() == NCCL_MAX_COLL_OP_NUM) {
      runCollectiveCommBuf(
          comm_map, tag, send_tensors_buf, recv_tensors_buf, roots_buf, op_name,
          f);
      send_tensors_buf.clear();
      recv_tensors_buf.clear();
      roots_buf.clear();
    }
  }

  runCollectiveCommBuf(
      comm_map, tag, send_tensors_buf, recv_tensors_buf, roots_buf, op_name, f);
}

void NCCLWrapper::doAllreduce(
    int tag, const std::vector<at::Tensor>& tensors, ncclRedOp_t red_op) {
  runCollectiveComm(
      comm_map_, tag, tensors, {}, {}, "allreduce",
      [red_op](
          void* sendptr, void* recvptr, size_t count, int root,
          ncclDataType_t datatype, ncclComm_t* ncomm) {
        // in-place only
        return ncclAllReduce(
            sendptr, sendptr, count, datatype, red_op, *ncomm,
            (cudaStream_t) nullptr);
      });
}

void NCCLWrapper::allreduce(int tag, const std::vector<at::Tensor>& tensors) {
  doAllreduce(tag, tensors, ncclSum);
}

void NCCLWrapper::allreduceMin(
    int tag, const std::vector<at::Tensor>& tensors) {
  doAllreduce(tag, tensors, ncclMin);
}

void NCCLWrapper::allreduceMax(
    int tag, const std::vector<at::Tensor>& tensors) {
  doAllreduce(tag, tensors, ncclMax);
}

void NCCLWrapper::reduce(
    int tag, const std::vector<at::Tensor>& tensors,
    const std::vector<int>& roots) {
  assert(tensors.size() == roots.size());
  runCollectiveComm(
      comm_map_, tag, tensors, {}, roots, "reduce",
      [](void* sendptr, void* recvptr, size_t count, int root,
         ncclDataType_t datatype, ncclComm_t* ncomm) {
        // in-place only
        return ncclReduce(
            sendptr, sendptr, count, datatype, ncclSum, root, *ncomm,
            (cudaStream_t) nullptr);
      });
}

void NCCLWrapper::bcast(
    int tag, const std::vector<at::Tensor>& tensors,
    const std::vector<int>& roots) {
  runCollectiveComm(
      comm_map_, tag, tensors, {}, roots, "bcast",
      [](void* sendptr, void* recvptr, size_t count, int root,
         ncclDataType_t datatype, ncclComm_t* ncomm) {
        // in-place only
        return ncclBcast(
            sendptr, count, datatype, root, *ncomm, (cudaStream_t) nullptr);
      });
}

void NCCLWrapper::allgather(
    int tag, const std::vector<at::Tensor>& tensors,
    const std::vector<at::Tensor>& out_bufs) {
  return runCollectiveComm(
      comm_map_, tag, tensors, out_bufs, {}, "allgather",
      [](void* sendptr, void* recvptr, size_t count, int root,
         ncclDataType_t datatype, ncclComm_t* ncomm) {
        return ncclAllGather(
            sendptr, recvptr, count, datatype, *ncomm, (cudaStream_t) nullptr);
      });
}

std::string getBoolBufKey(const RouteDP& route, const std::string& action) {
  std::stringstream ss;
  ss << toString(route) << "_" << action;
  return ss.str();
}

void NCCLWrapper::redist(
    void* send_ptr, void* recv_ptr, const RouteDP& route,
    const IRType& global_type, const RedistArgs& redist_args) {
  if (!contains(getRanksInRoute(route), mpi::getRank())) {
    return;
  }

  const auto& global_dim = global_type.getTensorDim();

  assert(global_type.getBaseType() == IRBaseType::TENSOR);

  // Special case for bool
  bool convert_bool = global_type.getTensorElemType() == IRTensorElemType::BOOL;
  const auto& elem_type =
      convert_bool ? IRTensorElemType::INT : global_type.getTensorElemType();

  void* tmp_send_ptr = send_ptr;
  void* tmp_recv_ptr = recv_ptr;
  long send_count_sum = sum(redist_args.sendcounts);
  long recv_count_sum = sum(redist_args.recvcounts);
  if (convert_bool) {
    if (send_ptr != nullptr) {
      at::TensorOptions options;
      options = options.device(torch::Device(torch::kCUDA))
                    .dtype(c10::ScalarType::Bool);

      auto send_ten = torch::from_blob(send_ptr, {send_count_sum}, options);
      auto send_buf_ten = buf_cache_.get(
          getBoolBufKey(route, "send"),
          IRType::createTensorType(elem_type, {send_count_sum}, false));
      send_buf_ten.zero_();
      send_buf_ten.masked_fill_(send_ten, 1);
      tmp_send_ptr = send_buf_ten.data_ptr();
    }

    if (recv_ptr != nullptr) {
      auto recv_buf_ten = buf_cache_.get(
          getBoolBufKey(route, "recv"),
          IRType::createTensorType(elem_type, {recv_count_sum}, false));
      tmp_recv_ptr = recv_buf_ten.data_ptr();
    }
  }

  TagMap& tag_map = TagMap::get();
  const auto comm_ranks = getRanksInRoute(route);
  int tag = tag_map.getRankSetTag(comm_ranks);

  int my_local_rank = getLocalRank(comm_ranks, mpi::getRank());
  size_t n_ranks = comm_ranks.size();

  assert(contains(comm_map_, tag));
  AllReduceComm* comm_info = comm_map_.at(tag);
  ncclComm_t* ncomm = comm_info->comm;
  ncclDataType_t datatype = getRedistNcclDataType(elem_type);

  const auto elem_size = getTensorElemSize(elem_type);
  job_executor_.addCommJob([n_ranks, redist_args, my_local_rank, tmp_send_ptr,
                            tmp_recv_ptr, datatype, elem_size, ncomm,
                            send_count_sum]() {
    auto stream = getStream();
    for (size_t i = 0; i < n_ranks; i++) {
      int sc = redist_args.sendcounts[i];
      int rc = redist_args.recvcounts[i];

      if (i < my_local_rank) { // send first
        if (sc > 0) {
          ncclSend(
              (char*)tmp_send_ptr + redist_args.sdispls[i] * elem_size, sc,
              datatype, i, *ncomm, (cudaStream_t)stream);
        }
        if (rc > 0) {
          ncclRecv(
              (char*)tmp_recv_ptr + redist_args.rdispls[i] * elem_size, rc,
              datatype, i, *ncomm, (cudaStream_t)stream);
        }
      } else if (i > my_local_rank) {
        if (rc > 0) {
          ncclRecv(
              (char*)tmp_recv_ptr + redist_args.rdispls[i] * elem_size, rc,
              datatype, i, *ncomm, (cudaStream_t)stream);
        }
        if (sc > 0) {
          ncclSend(
              (char*)tmp_send_ptr + redist_args.sdispls[i] * elem_size, sc,
              datatype, i, *ncomm, (cudaStream_t)stream);
        }
      }
    }
  });

  int sc_local = redist_args.sendcounts[my_local_rank];
  if (sc_local > 0) {
    cudaMemcpyAsync(
        (char*)tmp_recv_ptr + redist_args.rdispls[my_local_rank] * elem_size,
        (char*)tmp_send_ptr + redist_args.sdispls[my_local_rank] * elem_size,
        sc_local * elem_size, cudaMemcpyDeviceToDevice,
        (cudaStream_t)getStream());
  }

  if (convert_bool) {
    job_executor_.addPostCommJob([recv_ptr, tmp_recv_ptr, recv_count_sum]() {
      if (recv_ptr != nullptr) {
        at::TensorOptions buf_opts;
        buf_opts = buf_opts.device(torch::Device(torch::kCUDA))
                       .dtype(c10::ScalarType::Int);
        auto recv_buf_ten =
            torch::from_blob(tmp_recv_ptr, {recv_count_sum}, buf_opts);

        at::TensorOptions options;
        options = options.device(torch::Device(torch::kCUDA))
                      .dtype(c10::ScalarType::Bool);
        auto recv_ten = torch::from_blob(recv_ptr, {recv_count_sum}, options);
        {
          at::NoGradGuard no_grad;
          recv_ten.copy_(recv_buf_ten.gt(0));
        }
      }
    });
  }
}

void NCCLWrapper::startBulk() {
  job_executor_.setRunImmediate(false);
}

void NCCLWrapper::endBulk() {
  job_executor_.flush();
  job_executor_.setRunImmediate(true);
}

void NCCLWrapper::checkCommError(int tag) {
  if (!contains(comm_map_, tag)) {
    return;
  }
  AllReduceComm* comm_info = comm_map_.at(tag);
  ncclComm_t* ncomm = comm_info->comm;

  ncclResult_t ncclAsyncErr;
  ncclResult_t ncclErr = ncclCommGetAsyncError(*ncomm, &ncclAsyncErr);
  ncclErr = ncclCommGetAsyncError(*ncomm, &ncclAsyncErr);
  if (ncclErr != ncclSuccess) {
    spdlog::warn("NCCL Error : ncclCommGetAsyncError returned {}", ncclErr);
    throw CommErrorException("NCCL Error : ncclCommGetAsyncError", ncclErr);
  }

  if (ncclAsyncErr != ncclSuccess) {
    ncclErr = ncclCommAbort(*ncomm);
    if (ncclErr != ncclSuccess) {
      spdlog::warn("NCCL Error : ncclCommAbort returned {}", ncclErr);
    }
    throw CommErrorException("NCCL Error : ncclCommAbort", ncclErr);
  }
}

void NCCLWrapper::checkAllCommErrors() {
  for (const auto& it : comm_map_) {
    checkCommError(it.first);
  }
}

void NCCLWrapper::syncWithErrorCheck() {
  cudaError_t cudaErr;
  while (true) {
    cudaErr = cudaStreamQuery(nullptr);
    if (cudaErr == cudaSuccess) {
      return;
    }

    if (cudaErr != cudaErrorNotReady) {
      spdlog::info("CUDA Error : cudaStreamQuery returned {}", cudaErr);
      throw CommErrorException("CUDA Error : cudaStreamQuery", cudaErr);
    }

    checkAllCommErrors();

    pthread_yield();
  }
}

void NCCLWrapper::destroyCommunicator(int tag) {
  if (contains(comm_map_, tag)) {
    AllReduceComm* comm_info = comm_map_.at(tag);
    ncclCommDestroy(*comm_info->comm);
    delete comm_info->comm;
    delete comm_info;
    comm_map_.erase(tag);
  }
}

void NCCLWrapper::destroyAllCommunicators() {
  std::vector<int> tags;
  for (const auto& it : comm_map_) {
    tags.push_back(it.first);
  }
  for (const auto& tag : tags) {
    destroyCommunicator(tag);
  }
}

void NCCLWrapper::abortAllCommunicators() {
  for (const auto& it : comm_map_) {
    ncclResult_t ncclErr = ncclCommAbort(*it.second->comm);
    if (ncclErr != ncclSuccess) {
      spdlog::warn("NCCL Error : ncclCommAbort returned {}", ncclErr);
    }
  }
}

void NCCLWrapper::recreateCommunicator(int tag) {
  if (contains(comm_map_, tag)) {
    AllReduceComm* comm_info = comm_map_.at(tag);

    ncclResult_t ncclErr;
    ncclErr = ncclCommAbort(*comm_info->comm);
    if (ncclErr != ncclSuccess) {
      spdlog::warn("NCCL Error : ncclCommAbort returned {}", ncclErr);
    }
    destroyCommunicator(tag);

    std::unordered_map<int, std::unordered_set<int>> tag_to_ranks;
    for (const auto& it : ranks_to_tag_) {
      tag_to_ranks[it.second] = it.first;
    }
    assert(contains(tag_to_ranks, tag));
    const auto& ranks = tag_to_ranks.at(tag);

    if (contains(ranks, mpi::getRank())) {
      createCommunicator(tag, ranks);
    }
  }
}
void NCCLWrapper::recreateAllCommunicators() {
  std::vector<int> sorted_tags;
  std::unordered_map<int, std::unordered_set<int>> tag_to_ranks;
  for (const auto& it : ranks_to_tag_) {
    sorted_tags.push_back(it.second);
    tag_to_ranks[it.second] = it.first;
  }
  std::sort(sorted_tags.begin(), sorted_tags.end());

  for (int tag : sorted_tags) {
    recreateCommunicator(tag);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

void NCCLWrapper::commWithRetry(const std::function<void()>& f) {
  while (true) {
    try {
      f();
      return;
    } catch (CommErrorException& e) {
      spdlog::warn("commWithRetry {}", e.what());
      sleep(30);
      recreateAllCommunicators();
    }
  }
}

std::string NCCLWrapper::getImplName() {
  return "NCCL";
}

void createRouteCommunicator(const std::vector<RouteDP>& routes) {
  TagMap& tag_map = TagMap::get();
  NCCLWrapper& ar = NCCLWrapper::get();

  for (const auto& r : routes) {
    const auto ranks = getRanksInRoute(r);
    int tag = tag_map.getRankSetTag(ranks);
    if (contains(ranks, mpi::getRank())) {
      ar.createCommunicator(tag, ranks);
    }
  }
}

} // namespace rannc
