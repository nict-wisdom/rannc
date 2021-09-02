//
// Created by Masahiro Tanaka on 2020/02/24.
//

#include "MLPartDecomposer.h"
#include "Partitioner.h"
#include "DPStaging.h"

namespace rannc {

    Deployment MLPartDecomposer::decompose(const std::shared_ptr<IRGraph> &ir_graph) {

        logger->trace("MLPartDecomposer::decompose starting");

        if (dev_mem_ > 0) {
            const auto mem_limit = config::Config::get().getVal<int>(config::MEM_LIMIT_GB);
            if (mem_limit > 0) {
                dev_mem_ = std::min(dev_mem_, (size_t) (mem_limit * 1024L * 1024L * 1024L));
            }
            const auto mem_margin = config::Config::get().getVal<float>(config::MEM_MARGIN);
            dev_mem_ *= (1 - mem_margin);
            logger->info("Available device memory: {}", dev_mem_);
        } else {
            logger->warn("No CUDA device found on workers. Assuming (almost) unlimited host memory when assigning subgraphs.");
            dev_mem_ = 2 * 1024L * 1024L * 1024L * 1024L; // 2TB
        }

        logger->info("Starting model partitioning ... (this may take a very long time)");

        const int max_pipeline = config::Config::get().getVal<int>(config::MAX_PIPELINE);
        const int min_pipeline_bs = config::Config::get().getVal<int>(config::MIN_PIPELINE_BS);
        const bool coarsen_by_time = config::Config::get().getVal<bool>(config::COARSEN_BY_TIME);
        MLPartitioner partitioner(sg_prof_, worker_num_, dev_mem_, max_pipeline, min_pipeline_bs,
                                  use_amp_master_params_, coarsen_by_time, enable_zero_, worker_num_);
        MLGraph part_graph = partitioner.partition(ir_graph);

        ///////////
        logger->trace("Starting DP: id={} #nodes={}", ir_graph->getName(),
                      part_graph.nodes.size());
        DPStaging dp(sg_prof_, ir_graph, batch_size_, dev_mem_, use_amp_master_params_, enable_zero_);
        AllocSolution sol = dp.runDpComm(part_graph, worker_num_);
        logger->trace("Finished DP: id={}", ir_graph->getName());

        Partition new_part = createPartition(ir_graph, sol.graphs);

        // graph names in new_part are different with those in sol.repl_nums
        std::vector<std::string> ordered_graph_ids;
        for (const auto& g: sol.graphs) {
            ordered_graph_ids.push_back(g->getName());
        }
        assert(ordered_graph_ids.size() == new_part.order.size());
        std::unordered_map<std::string, int> repl_nums;
        for (const auto& it: new_part.subgraphs) {
            assert(contains(sol.repl_nums, it.first));
            repl_nums[it.first] = sol.repl_nums.at(it.first);
        }

        const auto repl = replicate(new_part, repl_nums, sol.pipeline_num, batch_size_);
        logger->trace("Partitioning finished: id={}", ir_graph->getName());

        std::unordered_map<std::string, std::unordered_set<int>> alloc;

        if (config::Config::get().getVal<bool>(config::ALLOC_REPL_FLAT)) {
            logger->trace("searchAllocationFlat");
            alloc = searchAllocationFlat(repl, worker_num_, dev_mem_);
        } else {
            logger->trace("searchAllocationSimple");
            alloc = searchAllocationSimple(repl, worker_num_, dev_mem_);
        }

        if (alloc.empty()) {
            throw std::runtime_error("Failed to allocate gpus to subgraphs.");
        }

        for (const auto &it: alloc) {
            logger->info(" Assigned subgraph {} to rank{}", it.first, join_as_str(it.second));
        }
        Deployment deployment = createDeployment(repl, alloc, worker_num_);
        deployment.pipeline_num = sol.pipeline_num;
        deployment.checkpointing = sol.checkpointing;
        logger->trace("MLPartDecomposer::decompose finished");

        return deployment;
    }
}