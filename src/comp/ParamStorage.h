//
// Created by Masahiro Tanaka on 2019-06-13.
//

#ifndef PYRANNC_PARAMSTORAGE_H
#define PYRANNC_PARAMSTORAGE_H

#include <torch/torch.h>

#include <comm/NCCLWrapper.h>
#include <graph/Decomposition.h>
#include <Logging.h>
#include "DistributedGradLocator.h"
#include "SlicedParamLocator.h"

namespace rannc {

class ParamStorage;

class GradConsolidation {
 public:
  GradConsolidation(
      ParamStorage* param_storage, std::vector<long> param_ids,
      bool use_amp_master_params, bool consolidate_master_params);
  void consolidate();
  std::vector<at::Tensor> getConsolidatedGrads();
  ~GradConsolidation() = default;

 private:
  std::vector<long> param_ids_;
  std::unordered_map<long, at::Tensor> grad_tensors_;
  std::unordered_map<at::ScalarType, at::Tensor, EnumHash<at::ScalarType>>
      consolidated_grads_;
  std::unordered_map<long, at::Tensor> amp_master_grad_tensors_;
  std::unordered_map<at::ScalarType, at::Tensor, EnumHash<at::ScalarType>>
      consolidated_master_grads_;
  ParamStorage* param_storage_;

  at::Tensor consolidated_amp_fp32_next_;
  std::unordered_map<long, at::Tensor> amp_grad_tensors_next_;

  bool use_amp_master_params_;
  bool consolidate_master_params_;

  const std::shared_ptr<spdlog::logger> logger = getLogger("ParamStorage");
};

class ParamStorage {
 public:
  ParamStorage() = default;

  void registerParam(
      long param_id, const at::Tensor& param_tensor, bool buffer,
      bool distributed);
  void unregisterParam(long param_id);
  std::unordered_map<std::string, long> getParamIDs(
      const std::string& graph_id, bool include_buffer) const;
  long getParamID(const std::string& graph_id, const std::string& name) const;
  at::Tensor getParamTensor(
      const std::string& graph_id, const std::string& name) const;
  at::Tensor getParamTensor(long param_id) const;
  at::Tensor getAmpMasterParamTensor(long param_id) const;
  bool hasParam(long param_id) const;
  bool hasAmpMasterParam(long param_id) const;
  bool distributed(long param_id) const;
  bool sliced(long param_id) const;
  bool zeroEnabled(const std::string& graph_id) const;
  at::Tensor getLocalParamSegment(long param_id) const;
  std::tuple<int64_t, int64_t> getLocalParamRange(long param_id) const;
  at::Tensor gatherTensorZero(const at::Tensor& ten, long param_id);
  at::Tensor gatherTensorSliced(const at::Tensor& ten, long param_id);
  std::unordered_set<int> getRanks(long param_id) const;

  void deploy(
      const Deployment& decomp,
      const std::unordered_map<std::string, long>& graph_params,
      bool enable_zero, const ParamPartitionMap& param_partitions);
  void useParam(
      const std::string& graph_id, const std::string& name, long param_id);

  long globalToLocal(long global_param_id);
  long localToGlobal(long local_param_id);

  void allReduceParamGrads(const std::string& graph_id);
  void allReduceParamGradsZero(const std::string& graph_id);
  void clearParamGrads(const std::string& graph_id);
  void bcastParamsZero(const std::string& graph_id, bool grad);
  void prepareBackward(const std::string& graph_id);
  void scaleGrads(const std::string& graph_id, bool amp_master_grads);
  void unscaleGrads(const std::string& graph_id, bool amp_master_grads);

  void setGradToLocalParamSegment(const std::string& graph_id);
  void alignBufferZero(const std::string& graph_id);

  void registerAmpMasterParam(
      long model_param_id, long master_param_id,
      const at::Tensor& param_tensor);
  void clipGradNorm(
      const std::string& graph_id, double max_grad_norm, bool use_amp_master);
  double calcGradGlobalL2Norm(const std::string& graph_id, bool use_amp_master);

  at::Tensor syncParam(long param_id, bool amp_master_param);
  at::Tensor syncParamGrad(long param_id, bool amp_master_param);
  at::Tensor gatherParam(long param_id, bool amp_master_param);
  at::Tensor gatherParamGrad(long param_id, bool amp_master_param);
  at::Tensor gatherParamZero(long param_id, bool amp_master_param);
  at::Tensor gatherParamGradZero(long param_id, bool amp_master_param);

  IRType getParamType(long param_id);
  IRType getParamType(const std::string& graph_id, const std::string& name);

  bool isConsolidate() const {
    return consolidate_;
  }

  void setConsolidate(bool consolidate) {
    consolidate_ = consolidate;
  }

  bool getAllreduceAmpMasterParams() const {
    return allreduce_amp_master_params_;
  }

  void setAllreduceAmpMasterParams(bool allreduce_amp_master_params) {
    allreduce_amp_master_params_ = allreduce_amp_master_params;
  }

  void useAmpMasterParams(
      const std::string& graph_id, bool use_amp_master_params);

  void releaseGraphParams(const std::string& graph_id);

  void clear();

  virtual ~ParamStorage() = default;

  static void syncOnInit(bool initValue);

 protected:
  void syncParamOnInit(long param_id, const std::unordered_set<int>& ranks);
  virtual void doReleaseParam(long param_id);
  void doScaleGrads(
      const std::string& graph_id, bool unscale, bool amp_master_grads);
  at::Tensor doSyncParam(long param_id, bool grad, bool amp_master_param);
  at::Tensor doGatherParam(long param_id, bool grad, bool amp_master_param);
  at::Tensor doGatherParamZero(long param_id, bool grad, bool amp_master_param);
  void consolidateGrads(const std::string& graph_id);
  std::vector<int> sortCommTags(const std::string& graph_id);

  std::unordered_map<std::string, std::unordered_map<std::string, long>>
      graph_params_;
  std::unordered_map<std::string, std::unordered_map<std::string, long>>
      unused_params_; // used to calc global norm, the value is a global id
  std::unordered_map<long, at::Tensor> params_;
  std::unordered_set<long> buffer_ids_;
  std::unordered_set<long> dist_ids_;
  std::unordered_map<long, std::unordered_set<int>> ranks_;

  // For param sharing
  // Ranks of params in a graph deployed on this rank
  std::unordered_map<
      std::string, std::unordered_map<long, std::unordered_set<int>>>
      my_param_ranks_;

  std::unordered_map<long, int> ref_counts_;

  std::unordered_map<long, at::Tensor> amp_master_params_;
  std::unordered_map<long, long>
      amp_param_id_map_; // master param -> model param

  std::unordered_map<long, long> id_global_to_local_;
  std::unordered_map<long, long> id_local_to_global_;

  std::unordered_map<long, IRType> param_types_;

 private:
  bool consolidate_ = false;
  bool allreduce_amp_master_params_ = false;
  std::unordered_map<
      std::string, std::unordered_map<int, std::shared_ptr<GradConsolidation>>>
      grad_cons_;
  std::unordered_map<std::string, std::unordered_map<int, std::vector<long>>>
      grouped_params_;
  std::unordered_map<int, std::unordered_set<int>> tag_rank_set_;
  std::unordered_map<std::string, bool> use_amp_master_params_;
  std::unordered_map<std::string, std::shared_ptr<DistributedGradLocator>>
      zero_grad_locators_;
  std::unordered_map<std::string, std::shared_ptr<SlicedParamLocator>>
      sliced_param_locators_;

  static bool sync_on_init_;

  const std::shared_ptr<spdlog::logger> logger = getLogger("ParamStorage");
};
} // namespace rannc

#endif // PYRANNC_PARAMSTORAGE_H
