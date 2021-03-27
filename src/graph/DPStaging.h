//
// Created by Masahiro Tanaka on 2020/04/11.
//

#ifndef PYRANNC_DPSTAGING_H
#define PYRANNC_DPSTAGING_H

#include "MLGraph.h"
#include "ProfilerUtil.h"

namespace rannc {
    class GraphMergeHelper {
    public:
        GraphMergeHelper(MLGraph graph);
        std::shared_ptr<IRGraph> merge(size_t from, size_t to);

    private:
        MLGraph graph_;
        std::unordered_map<std::string, std::shared_ptr<MLNode>> node_map_;
        std::vector<std::string> node_ids_;
        GraphMergeCache graph_merge_cache_;
    };


    class DPStaging {
    public:
        DPStaging(std::shared_ptr<GraphProfiler> profiler, size_t batch_size, size_t dev_mem, bool use_amp_master_params)
            :prof_util_(std::move(profiler)), batch_size_(batch_size), dev_mem_(dev_mem), use_amp_master_params_(use_amp_master_params) {}
        AllocSolution runDpComm(const MLGraph& graph, size_t dev_num);

    private:
        AllocSolution doRunDpComm(const MLGraph& graph, size_t stage_num, size_t dev_num_per_group,
                int replica_num, int pipeline_num, bool checkpointing);
        long estimateTime(const AllocSolution& sol);

        GraphProfile estimateProf(const MLGraph& graph, size_t from, size_t to, size_t dev_num,
                bool checkpointing);
        std::string makeNodeEvalSummary(const MLGraph& graph, size_t dev_num, size_t pipeline_num);
        std::string makeNodeMemSummary(const MLGraph& graph, size_t dev_num, size_t pipeline_num);
        std::string doMakeNodeSummary(const MLGraph& graph, size_t dev_num, size_t pipeline_num,
                                      const std::string& label, const std::function<long(const GraphProfile& prof, const MLNode& node, size_t repl)>& f);
        ProfilerUtil prof_util_;
        size_t batch_size_;
        size_t dev_mem_;
        GraphMergeCache graph_merge_cache_;
        bool use_amp_master_params_;

        const std::shared_ptr<spdlog::logger> logger = getLogger("DPStaging");
    };
}



#endif //PYRANNC_DPSTAGING_H
