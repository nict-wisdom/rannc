//
// Created by Masahiro Tanaka on 2019-07-05.
//

#ifndef PYRANNC_SCOMM_H
#define PYRANNC_SCOMM_H

#include <mpi.h>
#include <torch/cuda.h>
#include <torch/torch.h>

#include <comm/MPIUtil.h>
#include <comm/SComm.h>
#include <Common.h>
#include <graph/ir.h>

#include "comp/BatchSizeCalculator.h"
#include "torch/TorchUtil.h"

namespace rannc {

using unique_group_ptr =
    std::unique_ptr<MPI_Group, std::function<void(MPI_Group*)>>;
unique_group_ptr unique_group(MPI_Group* ptr);
using unique_comm_ptr =
    std::unique_ptr<MPI_Comm, std::function<void(MPI_Comm*)>>;
unique_comm_ptr unique_comm(MPI_Comm* ptr);

using GroupMap = std::unordered_map<int, unique_group_ptr>;
using CommMap = std::unordered_map<int, unique_comm_ptr>;

class TagMap {
 public:
  static TagMap& get() {
    static TagMap instance;
    return instance;
  }

  int getRankSetTag(const std::unordered_set<int>& ranks);
  int getRouteTag(const RouteDP& route);
  int getRouteSourceTag(const RouteDP& route);

  void sync();

  TagMap() = default;
  TagMap(const TagMap&) = delete;
  TagMap& operator=(const TagMap&) = delete;
  TagMap(TagMap&&) = delete;
  TagMap& operator=(TagMap&&) = delete;
  ~TagMap() = default;

 private:
  template <typename T>
  int generate(const T& key) {
    int tag = mpi::generateTag(key);
    while (contains(tags_, tag)) {
      tag++;
      tag %= mpi::getTagUB();
    }
    tags_.insert(tag);
    return tag;
  }

  int getStrTag(const std::string& name) {
    if (!contains(value_map_, name)) {
      value_map_[name] = generate(name);
    }

    return value_map_[name];
  }

  std::unordered_map<long, int> param_map_;
  std::unordered_map<std::string, int> value_map_;
  std::unordered_map<std::unordered_set<int>, int, IntSetHash> param_comm_map_;
  std::unordered_set<int> tags_;
};

class SComm {
 public:
  static SComm& get();

  void setPipeline(int pipeline_num, int64_t global_batch_size, bool is_bwd);
  void startSplit(int split_index);
  std::string getKey(const RouteDP& route) const;

  bool isLastLocalSplit(
      const std::unordered_set<int>& ranks, int my_rank, int split_index) const;
  int getFirstLocalSplitIndex(
      const std::unordered_set<int>& ranks, int my_rank) const;
  int getNextLocalSplitIndex(
      const std::unordered_set<int>& ranks, int my_rank, int split_index) const;
  int getPrevLocalSplitIndex(
      const std::unordered_set<int>& ranks, int my_rank, int split_index) const;

  torch::jit::IValue bcastIValue(
      const torch::jit::IValue& ivalue, const RouteDP& route);
  IValueMap bcastIValueMap(const IValueMap& ivalue_map, const RouteDP& route);

  torch::jit::IValue distribute(
      const torch::jit::IValue& tensor, const RouteDP& route, bool is_bwd,
      int split_delay = 0);

  torch::jit::IValue distribute(
      const torch::jit::IValue& tensor, const RouteDP& route, bool is_bwd,
      const IRType& global_type, int split_delay = 0);

  int64_t allReduceSumBatchSize(int64_t batch_size);
  int64_t allReduceMaxBatchSize(int64_t batch_size);

  MPI_Comm getCommunicator(int tag, const std::unordered_set<int>& ranks);

  void destroy();

  SComm(const SComm&) = delete;
  SComm& operator=(const SComm&) = delete;
  SComm(SComm&&) = delete;
  SComm& operator=(SComm&&) = delete;
  ~SComm() = default;

 private:
  SComm();

  MPI_Comm getRouteCommunicator(const RouteDP& route);
  torch::jit::IValue doDistribute(
      const torch::jit::IValue& val, const IRType& global_type,
      const RouteDP& route, bool is_fwd, int split_delay);
  torch::jit::IValue distributeBatchTensor(
      const torch::jit::IValue& val, const IRType& global_type,
      const RouteDP& route, int split_delay);
  torch::jit::IValue distributeLossTensor(
      const torch::jit::IValue& val, const IRType& global_type,
      const RouteDP& route, bool weight, int split_delay);
  at::Tensor bcastTensor(
      const torch::jit::IValue& ivalue, const IRType& ir_type,
      const RouteDP& route, MPI_Comm comm);

  size_t getSplitBatchSize(int split_index);
  size_t getCurrentSplitBatchSize(int split_delay);

  RedistArgs getRedistArgs(
      int my_rank, int64_t batch_size, const std::vector<int64_t>& dim,
      const std::unordered_set<int>& src_ranks,
      const std::unordered_set<int>& dest_ranks, int split_index);

  BufferTensorCache buf_cache_;
  BatchSizeCalculator bs_calc_;

  int pipeline_num_;
  int split_index_;
  bool is_bwd_;

  GroupMap group_map_;
  CommMap comm_map_;

  std::shared_ptr<spdlog::logger> logger = getLogger("SComm");
};
} // namespace rannc

#endif // PYRANNC_SCOMM_H
