//
// Created by Masahiro Tanaka on 2019-06-13.
//

#include <comm/AllReduceRunner.h>
#include <comm/ObjectComm.h>
#include <graph/Decomposition.h>
#include <cuda/CudaUtil.h>
#include "Common.h"
#include "ConfiguredTorch.h"
#include "ParamStorage.h"
#include "comm/SComm.h"
#include "EventRecorder.h"

namespace rannc {

    std::vector<long> getSortedParamIDs(const std::unordered_map<std::string, long>& param_map) {
        std::vector<long> param_ids;
        param_ids.reserve(param_map.size());
        for (const auto& param_name: keys(param_map, true)) {
            param_ids.push_back(param_map.at(param_name));
        }
        return param_ids;
    }

    GradConsolidation::GradConsolidation(ParamStorage* param_storage, std::vector<long> param_ids,
                                         bool use_amp_master_params, bool consolidate_master_params)
            :param_ids_(std::move(param_ids)), param_storage_(param_storage),
             use_amp_master_params_(use_amp_master_params), consolidate_master_params_(consolidate_master_params) {

        std::unordered_map<at::ScalarType, int64_t, EnumHash<at::ScalarType>> grad_elem_sum;
        std::unordered_map<at::ScalarType, std::unordered_map<long, int64_t>, EnumHash<at::ScalarType>> offsets;

        // Do not sort param IDs because they are local
        for (long pid: param_ids_) {
            const auto p = param_storage_->getParamTensor(pid);
            const auto stype = p.scalar_type();
            assert(stype == at::ScalarType::Float || stype == at::ScalarType::Half);

            offsets[stype][pid] = grad_elem_sum[stype];
            grad_elem_sum[stype] += p.numel();
        }

        for (const auto& it: grad_elem_sum) {
            const auto stype = it.first;
            at::TensorOptions options;
            if (torch::cuda::is_available()) {
                options = options.device(c10::Device(c10::DeviceType::CUDA));
            } else {
                options = options.device(c10::Device(c10::DeviceType::CPU));
            }

            if (consolidate_master_params_ && stype == at::ScalarType::Half) {
                options = options.dtype(at::ScalarType::Float);
                consolidated_master_grads_[stype] = torch::zeros({it.second}, options);
            } else {
                options = options.dtype(stype);
                consolidated_grads_[stype] = torch::zeros({it.second}, options);
            }
        }

        if (use_amp_master_params_ && contains(consolidated_grads_, at::ScalarType::Float)) {
            // We need another buffer for FP32 gradients with amp.
            // amp stashes param.grad and set None to param.grad before backward
            // because scaling by amp destroys accumulated gradients.
            const auto& current = consolidated_grads_.at(at::ScalarType::Float);
            consolidated_amp_fp32_next_ = torch::zeros_like(current);
        }

        assert(!param_ids_.empty());
        for (long pid: param_ids_) {
            const auto p = param_storage_->getParamTensor(pid);
            const auto stype = p.scalar_type();
            const auto offset = offsets[stype][pid];

            if (consolidate_master_params_ && stype == at::ScalarType::Half) {
                amp_master_grad_tensors_[pid] = consolidated_master_grads_[stype].narrow(0, offset, p.numel())
                        .view(p.sizes()).detach();
                continue;
            }

            grad_tensors_[pid] = consolidated_grads_[stype].narrow(0, offset, p.numel())
                    .view(p.sizes()).detach();

            if (stype == at::ScalarType::Float
                    && use_amp_master_params_
                    && contains(consolidated_grads_, at::ScalarType::Float)) {
                assert(consolidated_amp_fp32_next_.defined());

                amp_grad_tensors_next_[pid] = consolidated_amp_fp32_next_.narrow(0, offset, p.numel())
                            .view(p.sizes()).detach();
            } else {
                amp_grad_tensors_next_[pid] = grad_tensors_[pid];
            }
        }
    }

    void GradConsolidation::consolidate() {
        if (use_amp_master_params_ && contains(consolidated_grads_, at::ScalarType::Float)) {
            // swap buffers
            // Amp stashes FP32 grad and sets None to param.grad.
            // The buffer associated with grad_tensors_ keeps alive,
            // but we need to set another buffer to param.grad
            std::unordered_map<long, at::Tensor> tmp_grad_tensors = amp_grad_tensors_next_;
            amp_grad_tensors_next_ = grad_tensors_;
            grad_tensors_ = tmp_grad_tensors;

            at::Tensor tmp_cons = consolidated_grads_.at(at::ScalarType::Float);
            consolidated_grads_[at::ScalarType::Float] = consolidated_amp_fp32_next_;
            consolidated_amp_fp32_next_ = tmp_cons;
        }

        for (long pid: param_ids_) {
            auto p = param_storage_->getParamTensor(pid);
//<<<<<<< HEAD
            const auto stype = p.scalar_type();
//=======
//            auto& grad = getMutableGradRef(p);
//            auto& con_grad = grad_tensors_.at(pid);
//>>>>>>> 460e873f383b414d65f7a0c8a6d922b2be7c19c6

            if (consolidate_master_params_ && stype == at::ScalarType::Half) {
                // set to master grad
                auto amp_master_param = param_storage_->getAmpMasterParamTensor(pid);
//                auto& grad = amp_master_param.grad();
                auto& grad = getMutableGradRef(amp_master_param);
                if (!grad.defined()) {
                    assert(contains(amp_master_grad_tensors_, pid));
                    grad = amp_master_grad_tensors_.at(pid);
                    {
                        at::NoGradGuard no_grad;
                        grad.zero_();
                    }
                }
            } else {
                // set to model grad
                auto& grad = getMutableGradRef(p);
                auto& con_grad = grad_tensors_.at(pid);

                if (grad.defined()) {
                    // Ready
                    if (grad.data_ptr() == con_grad.data_ptr()) {
                        continue;
                    } else {
                        // Replace
                        {
                            at::NoGradGuard no_grad;
                            con_grad.copy_(grad);
                            grad = con_grad;
                        }
                    }
                } else {
                    // no gradient tied to param tensor
                    grad = con_grad;
                    {
                        at::NoGradGuard no_grad;
                        grad.zero_();
                    }
                }
            }
        }
    }

    std::vector<at::Tensor> GradConsolidation::getConsolidatedGrads() {
        std::vector<at::Tensor> grads;
        for (const auto& it: consolidated_grads_) {
            grads.push_back(it.second);
        }
        for (const auto& it: consolidated_master_grads_) {
            grads.push_back(it.second);
        }

        return grads;
    }

    bool ParamStorage::sync_on_init_ = true;

    void ParamStorage::registerParam(long param_id, const at::Tensor& param_tensor, bool buffer) {
        assert(!contains(params_, param_id));

        params_[param_id] = param_tensor;

        long global_id = param_id;
        ObjectComm& ocomm = ObjectComm::get();
        global_id = ocomm.bcast(global_id);
        id_global_to_local_[global_id]= param_id;
        id_local_to_global_[param_id]= global_id;

        if (buffer) {
            buffer_ids_.insert(param_id);
        }
    }

    void ParamStorage::unregisterParam(long param_id) {
        std::vector<std::string> graph_ids = keys(graph_params_);
        for (auto& gid: graph_ids) {
            auto& graph_params = graph_params_[gid];
            std::vector<std::string> to_delete;
            for (const auto& param_it: graph_params) {
                if (param_it.second == param_id) {
                    to_delete.push_back(param_it.first);
                }
            }
            for (const auto& pname: to_delete) {
                assert(contains(graph_params, pname));
                long pid = graph_params.at(pname);
                unused_params_[gid][pname] = localToGlobal(pid);
                graph_params.erase(pname);
            }
        }

        size_t removed = params_.erase(param_id);
        assert(removed == 1);
    }

    const std::unordered_map<std::string, long> ParamStorage::getParamIDs(const std::string& graph_id, bool buffer) {
        if (!buffer) {
            return graph_params_[graph_id];
        }

        std::unordered_map<std::string, long> only_params;
        for (const auto& it: graph_params_[graph_id]) {
            if (!contains(buffer_ids_, it.second)) {
                only_params[it.first] = it.second;
            }
        }
        return only_params;
    }

    long ParamStorage::getParamID(const std::string& graph_id, const std::string& name) {
        assert(contains(graph_params_, graph_id));
        const auto& params = graph_params_.at(graph_id);
        assert(contains(params, name));
        return params.at(name);
    }

    at::Tensor ParamStorage::getParamTensor(const std::string& graph_id, const std::string& name) {
        return getParamTensor(getParamID(graph_id, name));
    }

    at::Tensor ParamStorage::getParamTensor(long param_id) const {
        if (!contains(params_, param_id)) {
            std::stringstream ss;
            ss << "Parameter not found. ID=" << param_id;
            throw std::invalid_argument(ss.str());
        }
        return params_.at(param_id);
    }

    at::Tensor ParamStorage::getAmpMasterParamTensor(long param_id) const {
        if (!hasAmpMasterParam(param_id)) {
            std::stringstream ss;
            ss << "Amp master parameter not found. ID=" << param_id;
            throw std::invalid_argument(ss.str());
        }
        return amp_master_params_.at(param_id);
    }

    bool ParamStorage::hasAmpMasterParam(long param_id) const {
        return contains(amp_master_params_, param_id);
    }

    void ParamStorage::allReduceParamGrads(const std::string& graph_id) {
        auto& graph_grouped_params = grouped_params_[graph_id];

        std::stringstream ss;
        ss << "ParamStorage::allReduceParamGrads_graph_" << graph_id;
        recordStart(ss.str());

        AllReduceRunner& ar = AllReduceRunner::get();
        bool sync_allreduce = config::Config::get().getVal<bool>(config::SYNC_ALLREDUCE);

        std::vector<int> sorted_tags = keys(graph_grouped_params, true);
        std::sort(sorted_tags.begin(), sorted_tags.end(), [this](int t1, int t2) {
            const auto& ranks1 = this->tag_rank_set_.at(t1);
            const auto& ranks2 = this->tag_rank_set_.at(t2);
           return ranks1.size() < ranks2.size();
        });

        for (int tag: sorted_tags) {
            assert(contains(tag_rank_set_, tag));
            const auto& ranks = tag_rank_set_.at(tag);
            if (contains(ranks, mpi::getRank())) {
                if (consolidate_) {
                    const auto& graph_grad_cons = grad_cons_[graph_id];
                    ar.allreduce(tag, ranks, graph_grad_cons.at(tag)->getConsolidatedGrads());
                } else {
                    const auto &param_ids = graph_grouped_params.at(tag);
                    std::vector<at::Tensor> grads;
                    grads.reserve(param_ids.size());
                    for (long pid: param_ids) {
                        auto param = allreduce_amp_master_params_ && hasAmpMasterParam(pid)
                                ? getAmpMasterParamTensor(pid) : getParamTensor(pid);
                        auto& grad = param.grad();
                        if (grad.defined()) {
                            grads.push_back(grad);
                        }
                    }
                    ar.allreduce(tag, ranks, grads);
                }
            }
            if (sync_allreduce) {
                MPI_Barrier(MPI_COMM_WORLD);
            }
        }

        recordEnd(ss.str());
    }

    void ParamStorage::clearParamGrads(const std::string& graph_id) {
        if (consolidate_) {
            const auto& graph_grad_cons = grad_cons_[graph_id];
            for (const auto& it: graph_grad_cons) {
                for (auto& g: it.second->getConsolidatedGrads()) {
                    g.zero_();
                }
            }
        } else {
            for (const auto& it: getParamIDs(graph_id, false)) {
                getParamTensor(it.second).zero_();
            }
        }
    }

    void ParamStorage::doScaleGrads(const std::string& graph_id, bool unscale, bool amp_master_grads) {
        SComm& scomm = SComm::get();
        int64_t batch_size = scomm.getBatchSize();
        for (const auto& it: my_param_ranks_[graph_id]) {
            double ratio = getDpRatio(batch_size, it.second, mpi::getRank());
            long pid = it.first;
            torch::NoGradGuard no_grad;
            auto p = amp_master_grads && hasAmpMasterParam(pid) ?
                        getAmpMasterParamTensor(pid) : getParamTensor(pid);
            if (p.grad().defined()) {
                if (unscale) {
                    p.grad().mul_(1./ratio);
                } else {
                    p.grad().mul_(ratio);
                }
            }
        }
    }

    void ParamStorage::scaleGrads(const std::string& graph_id, bool amp_master_grads) {
        doScaleGrads(graph_id, false, amp_master_grads);
    }

    void ParamStorage::unscaleGrads(const std::string& graph_id, bool amp_master_grads) {
        doScaleGrads(graph_id, true, amp_master_grads);
    }

    void ParamStorage::consolidateGrads(const std::string& graph_id) {
        if (consolidate_) {
            if (!contains(grad_cons_, graph_id)) {
                return;
            }
            const auto graph_grad_cons = grad_cons_.at(graph_id);
            for (auto &it: graph_grad_cons) {
                it.second->consolidate();
            }
        }
    }

    void ParamStorage::registerAmpMasterParam(long model_param_id, long master_param_id, const at::Tensor& param_tensor) {
        amp_master_params_[model_param_id] = param_tensor;
        amp_param_id_map_[master_param_id] = model_param_id;
    }

    void ParamStorage::clipGradNorm(const std::string& graph_id, double max_grad_norm, bool use_amp_master) {

        double global_norm = calcGradGlobalL2Norm(graph_id, use_amp_master) + 1e-6;
        if (global_norm > max_grad_norm) {
            double scale =  max_grad_norm / global_norm;
            for (const auto &it: getParamIDs(graph_id)) {
                long pid = it.second;
                at::Tensor param;
                if (use_amp_master && contains(amp_master_params_, pid)) {
                    param = amp_master_params_.at(pid);
                } else {
                    param = getParamTensor(pid);
                }

                if (param.grad().defined()) {
                    param.grad().mul_(scale);
                }
            }
        }
    }

    double ParamStorage::calcGradGlobalL2Norm(const std::string& graph_id, bool use_amp_master) {

        std::unordered_map<long, float> norms;
        for (const auto& it: getParamIDs(graph_id)) {
            long pid = it.second;
            at::Tensor ten;
            if (use_amp_master && contains(amp_master_params_, pid)) {
                ten = amp_master_params_.at(pid);
            } else {
                ten = getParamTensor(pid);
            }
            if (ten.grad().defined()) {
                norms[id_local_to_global_.at(pid)] = ten.grad().norm(2, c10::ScalarType::Float).item<float>();
            }
        }

        ObjectComm& ocomm = ObjectComm::get();
        std::vector<std::unordered_map<long, float>> gathered_norms = ocomm.allgather(norms, MPI_COMM_WORLD);

        for (const auto& gn: gathered_norms) {
            for (const auto& it: gn) {
                long global_pid = it.first;
                if (!contains(norms, global_pid)) {
                    norms[global_pid] = it.second;
                }
            }
        }

        std::unordered_map<std::string, long> params_on_rank = getParamIDs(graph_id);
        std::unordered_map<std::string, long> all_param_ids;
        all_param_ids.reserve(params_on_rank.size());
        for (const auto& it: params_on_rank) {
            all_param_ids[it.first] = localToGlobal(it.second);
        }

        if (contains(unused_params_, graph_id)) {
            std::unordered_map<std::string, long> &params_not_on_rank = unused_params_.at(graph_id);
            all_param_ids.reserve(all_param_ids.size() + params_not_on_rank.size());
            for (const auto &it: params_not_on_rank) {
                all_param_ids[it.first] = it.second; // the id is global
            }
        }

        double norm_sq_sum = 0;
        for (const auto global_pid: getSortedParamIDs(all_param_ids)) {
            if (contains(norms, global_pid)) {
                // norms may not contain global_pid because it does not include an element if the param tensor does not have its gradient.
                // this happens when the parameter is not used in the graph.
                double norm = norms.at(global_pid);
                norm_sq_sum += norm * norm;
            }
        }
        return sqrt(norm_sq_sum);
    }

    void ParamStorage::useParam(const std::string& graph_id, const std::string& name, long param_id) {
        graph_params_[graph_id][name] = param_id;
        ref_counts_[param_id]++;
    }

    void ParamStorage::deploy(const Deployment &decomp, const std::unordered_map<std::string, long>& graph_params) {
        logger->trace("ParamStorage::deployGraph starting: graph_id={}", decomp.id);

        auto& graph_id = decomp.id;
        for (const auto& it: graph_params) {
            useParam(graph_id,  it.first, it.second);
        }

        std::unordered_map<long, std::string> id_to_name;
        for (const auto& it: getParamIDs(graph_id)) {
            id_to_name[it.second] = it.first;
        }

        // check ranks of each param
        // subgraph_id -> param_name -> ranks
        std::unordered_map<std::string, std::unordered_map<std::string, std::unordered_set<int>>> param_rank_maps = getParamRanks(decomp);

        // Sort param by name, not by id
        auto& graph_grouped_params = grouped_params_[graph_id];
        std::vector<long> sorted_param_ids = getSortedParamIDs(graph_params);
        SComm& scomm = SComm::get();
        auto& tag_map = TagMap::get();
        tag_map.sync();

        std::unordered_map<PairKey<std::string, int>, std::unordered_set<std::string>, PairHash<std::string,int>> param_graph_set;
        for (const auto& param_id: sorted_param_ids) {
            for (const auto &it: param_rank_maps) {
                const auto &sg_id = it.first;
                const std::unordered_map<std::string, std::unordered_set<int>> &sg_rank_maps = it.second;

                assert(contains(id_to_name, param_id));
                const auto &param_name = id_to_name.at(param_id);
                if (!contains(sg_rank_maps, param_name)) {
                    continue;
                }
                const auto &param_ranks = sg_rank_maps.at(param_name);

                for (auto r: param_ranks) {
                    ranks_[param_id].insert(r);

                    PairKey<std::string, int> k{param_name, r};
                    param_graph_set[k].insert(sg_id);
                }

                if (contains(param_ranks, mpi::getRank())) {
                    my_param_ranks_[graph_id][param_id] = param_ranks;
                }
            }
        }

        // assert if shared param found on the same rank
        for (const auto& it: param_graph_set) {
            if (it.second.size() > 1) {
                std::stringstream ss;
                ss << "Parameter sharing among graphs on the same rank is not allowed: rank=" << it.first.second
                    << " name=" << it.first.first << " graphs=" << join_as_str(it.second);
                throw std::runtime_error(ss.str());
            }
        }

        size_t i = 1;
        for (const auto& param_id: sorted_param_ids) {
            if(!contains(ranks_, param_id)) {
                logger->debug("Param {} {} is not used.", param_id, id_to_name[param_id]);
                continue;
            }
            const auto &param_ranks = ranks_.at(param_id);

            initParam(param_id, param_ranks);
            if (mpi::getRank() == 0 && sync_on_init_) {
                logger->debug("Synchronized param {}/{}", i, sorted_param_ids.size());
            }

            assert(contains(id_to_name, param_id));
            const auto &param_name = id_to_name.at(param_id);
            logger->trace("ParamStorage::deployGraph deployed param: name={} param_id={} ranks={} ({}/{})",
                          param_name, param_id, join_as_str(param_ranks),
                          i, sorted_param_ids.size());

            AllReduceRunner& ar = AllReduceRunner::get();
            int comm_tag = tag_map.getRankSetTag(param_ranks);

            if (!contains(tag_rank_set_, comm_tag)) {
                if (contains(param_ranks, mpi::getRank())) {
                    logger->trace("Creating MPI communicator for params: tag={} ranks={}", comm_tag,
                                  join_as_str(param_ranks));

                    MPI_Comm param_comm = scomm.getCommunicator(comm_tag, param_ranks);
                    logger->trace("Finished creating MPI communicator for params: tag={} ranks={}", comm_tag,
                                  join_as_str(param_ranks));
                    MPI_Barrier(param_comm);

                    logger->trace("Creating communicator for params: tag={} ranks={}", comm_tag,
                                  join_as_str(param_ranks));
                    ar.createCommunicator(comm_tag, param_ranks);
                    logger->trace("Finished creating communicator for params: tag={} ranks={}", comm_tag,
                                  join_as_str(param_ranks));
                }
            }
            graph_grouped_params[comm_tag].push_back(param_id);
            tag_rank_set_[comm_tag] = param_ranks;
            i++;
        }

        if (consolidate_) {
            for (const auto &it: graph_grouped_params) {
                const auto &ranks = tag_rank_set_.at(it.first);
                if (contains(ranks, mpi::getRank())) {
                    assert(contains(use_amp_master_params_, graph_id));
                    grad_cons_[graph_id][it.first] = std::make_shared<GradConsolidation>(
                            this, it.second, use_amp_master_params_.at(graph_id), allreduce_amp_master_params_);
                }
            }
        }

        logger->trace("ParamStorage::deployGraph deployed all params. graph_id={}", decomp.id);
    }

    void ParamStorage::initParam(long param_id, const std::unordered_set<int>& ranks) {
        at::Tensor &param_tensor = params_.at(param_id);

        assert(param_tensor.is_contiguous());
        syncStream();

        static bool msg_shown = false;

        if (sync_on_init_) {
            if (mpi::getRank() == 0) {
                logger->trace("Synchronizing param {}", param_id);
                if (!msg_shown) {
                    logger->info("Synchronizing parameters ...");
                }
            }
            mpi::checkMPIResult(MPI_Bcast(param_tensor.data_ptr(), param_tensor.numel(),
                                          scalarTypeToMPIDatatype(param_tensor.type().scalarType()), 0,
                                          MPI_COMM_WORLD));
        } else {
            if (!msg_shown && mpi::getRank() == 0) {
                logger->info("Skipped parameter synchronization");
            }
        }
        msg_shown = true;
    }

    long ParamStorage::globalToLocal(long global_param_id) {
        assert(contains(id_global_to_local_, global_param_id));
        return id_global_to_local_.at(global_param_id);
    }

    long ParamStorage::localToGlobal(long local_param_id) {
        if (contains(amp_param_id_map_, local_param_id)) {
            local_param_id = amp_param_id_map_.at(local_param_id);
            assert(contains(id_local_to_global_, local_param_id));
        }

        if (contains(id_local_to_global_, local_param_id)) {
            return id_local_to_global_.at(local_param_id);
        }

        std::stringstream ss;
        ss << "No parameter found: local_param_id: " << local_param_id;
        throw std::invalid_argument(ss.str());
    }

    at::Tensor ParamStorage::syncParam(long param_id) {
        return doSyncParam(param_id, false);
    }

    at::Tensor ParamStorage::gatherParam(long param_id, int dest) {
        return doGatherParam(param_id, dest, false);
    }

    at::Tensor ParamStorage::syncParamGrad(long param_id) {
        return doSyncParam(param_id, true);
    }

    at::Tensor ParamStorage::gatherParamGrad(long param_id, int dest) {
        return doGatherParam(param_id, dest, true);
    }

    at::Tensor ParamStorage::doSyncParam(long param_id, bool grad) {
        // check if param exists
        if (!contains(ranks_, param_id)) {
            logger->debug("Ranks of param {} is not set.", param_id);
            return at::Tensor();
        }

        assert(contains(ranks_, param_id));
        auto param_ranks = setToVector(ranks_.at(param_id));
        std::sort(param_ranks.begin(), param_ranks.end());

        // bcast type
        int root = param_ranks.front();
        SComm& scomm = SComm::get();

        auto& tag_map = TagMap::get();
        int tag = tag_map.getRankSetTag(mpi::getAllRanks());
        int src_tag = tag_map.getRankSetTag({root});
        RouteDP route(IValueLocation("PARAM_SYNC"), {root}, setToVector(mpi::getAllRanks()),
                      tag, src_tag, RouteTypeDP::BROADCAST);
        torch::jit::IValue buf;
        if (mpi::getRank() == root) {
            if (grad) {
                const auto& param = params_.at(param_id);
                if (param.grad().defined()) {
                    buf = param.grad();
                } else {
                    buf = at::Tensor();
                }
            } else {
                buf = params_.at(param_id);
            }
        }

        const auto result = scomm.bcastIValue(buf, route);
        const auto result_cpu = toCPU(result, true);
        assert(result_cpu.isTensor());
        return result_cpu.toTensor();
    }

    at::Tensor ParamStorage::doGatherParam(long param_id, int dest, bool grad) {
        // check if param exists
        if (!contains(ranks_, param_id)) {
            logger->debug("Ranks of param {} is not set.", param_id);
            return at::Tensor();
        }

        assert(contains(ranks_, param_id));
        auto param_ranks = setToVector(ranks_.at(param_id));
        std::sort(param_ranks.begin(), param_ranks.end());
        int src = param_ranks.front();

        SComm& scomm = SComm::get();

        std::unordered_set<int> route_ranks = {src, dest};
        auto& tag_map = TagMap::get();
        int tag = tag_map.getRankSetTag(route_ranks);
        int src_tag = tag_map.getRankSetTag({src});
        RouteDP route(IValueLocation("PARAM_GATHER"), {src}, {dest},
                      tag, src_tag, RouteTypeDP::BROADCAST);

        at::Tensor ret;
        if (mpi::getRank() == src) {
            if (grad) {
                const auto& param = params_.at(param_id);
                if (param.grad().defined()) {
                    ret = param.grad();
                } else {
                    ret = at::Tensor();
                }
            } else {
                ret = params_.at(param_id);
            }
            if (mpi::getRank() != dest) {
                scomm.bcastIValue(ret, route);
            }
        } else if (mpi::getRank() == dest) {
            torch::jit::IValue buf;
            const auto result = scomm.bcastIValue(buf, route);
            assert(result.isTensor());
            ret = result.toTensor();
            ret = ret.cpu().detach();
        }

        MPI_Barrier(MPI_COMM_WORLD);
        return ret;
    }

    void ParamStorage::useAmpMasterParams(const std::string& graph_id, bool use_amp_master_params){
        use_amp_master_params_[graph_id] = use_amp_master_params;
    }

    void ParamStorage::releaseGraphParams(const std::string& graph_id) {

        for (const auto& it: graph_params_[graph_id]) {
            const auto& name = it.first;
            long param_id = graph_params_[graph_id][name];
            ref_counts_[param_id]--;

            doReleaseParam(param_id);
        }

        for (const auto& it: unused_params_[graph_id]) {
            const auto& name = it.first;
            assert(contains(id_global_to_local_, it.second));
            long param_id = id_global_to_local_.at(it.second);
            ref_counts_[param_id]--;

            doReleaseParam(param_id);
        }

        graph_params_.erase(graph_id);
        grad_cons_.erase(graph_id);
        grouped_params_.erase(graph_id);
        unused_params_.erase(graph_id);
    }

    void ParamStorage::clear() {
        graph_params_.clear();
        unused_params_.clear();
        params_.clear();
        buffer_ids_.clear();
        ranks_.clear();
        my_param_ranks_.clear();
        ref_counts_.clear();
        amp_master_params_.clear();
        amp_param_id_map_.clear();
        id_global_to_local_.clear();
        id_local_to_global_.clear();
        grad_cons_.clear();
        grouped_params_.clear();
        tag_rank_set_.clear();
        use_amp_master_params_.clear();
    }

    const std::unordered_set<int>& ParamStorage::getRanks(long param_id) {
        return ranks_[param_id];
    }

    void ParamStorage::doReleaseParam(long param_id) {
        if (ref_counts_[param_id] == 0) {
            logger->trace("Releasing parameter. id={}", param_id);

            params_.erase(param_id);
            ranks_.erase(param_id);
            ref_counts_.erase(param_id);

            long global_id = id_local_to_global_.at(param_id);
            id_local_to_global_.erase(param_id);
            id_global_to_local_.erase(global_id);
        }
    }

    void ParamStorage::syncOnInit(bool initValue) {
        sync_on_init_ = initValue;
    }
}