//
// Created by Masahiro Tanaka on 2020/02/24.
//

#include "MLPartDecomposer.h"
#include "DPStaging.h"
#include "Partitioner.h"

namespace rannc {

Deployment MLPartDecomposer::decompose(
    const std::shared_ptr<IRGraph>& ir_graph) {
  logger->trace("MLPartDecomposer::decompose starting");

  config::Config& conf = config::Config::get();
  if (conf_.dev_mem > 0) {
    const auto mem_limit = conf.getVal<int>(config::MEM_LIMIT_GB);
    if (mem_limit > 0) {
      conf_.dev_mem =
          std::min(conf_.dev_mem, (size_t)(mem_limit * 1024L * 1024L * 1024L));
    }
    const auto mem_margin = conf.getVal<float>(config::MEM_MARGIN);
    conf_.dev_mem *= (1 - mem_margin);
    logger->info("Available device memory: {}", conf_.dev_mem);
  } else {
    logger->warn(
        "No CUDA device found on workers. Assuming (almost) unlimited host memory when assigning subgraphs.");
    conf_.dev_mem = 2 * 1024L * 1024L * 1024L * 1024L; // 2TB
  }

  logger->info(
      "Starting model partitioning ... (this may take a very long time)");

  const bool coarsen_by_time = conf.getVal<bool>(config::COARSEN_BY_TIME);
  MLPartitioner partitioner(sg_prof_, conf_, coarsen_by_time);
  MLGraph part_graph = partitioner.partition(ir_graph);

  ///////////
  logger->trace(
      "Starting DP: id={} #nodes={}", ir_graph->getName(),
      part_graph.nodes.size());
  DPStaging dp(sg_prof_, ir_graph, conf_);
  AllocSolution sol = dp.runDpComm(part_graph);
  logger->trace("Finished DP: id={}", ir_graph->getName());

  Partition new_part = createPartition(ir_graph, sol.graphs);

  // graph names in new_part are different with those in sol.repl_nums
  std::vector<std::string> ordered_graph_ids;
  for (const auto& g : sol.graphs) {
    ordered_graph_ids.push_back(g->getName());
  }
  assert(ordered_graph_ids.size() == new_part.order.size());
  std::unordered_map<std::string, int> repl_nums;
  for (const auto& it : new_part.subgraphs) {
    assert(contains(sol.repl_nums, it.first));
    repl_nums[it.first] = sol.repl_nums.at(it.first);
  }

  const auto repl =
      replicate(new_part, repl_nums, sol.pipeline_num, conf_.batch_size);
  logger->trace("Partitioning finished: id={}", ir_graph->getName());

  std::unordered_map<std::string, std::unordered_set<int>> alloc;

  if (config::Config::get().getVal<bool>(config::ALLOC_REPL_FLAT)) {
    logger->trace("searchAllocationFlat");
    alloc = searchAllocationFlat(repl, conf_.dev_num, conf_.dev_mem);
  } else {
    logger->trace("searchAllocationSimple");
    alloc = searchAllocationSimple(repl, conf_.dev_num, conf_.dev_mem);
  }

  if (alloc.empty()) {
    throw std::runtime_error("Failed to allocate gpus to subgraphs.");
  }

  for (const auto& it : alloc) {
    logger->info(
        " Assigned subgraph {} to rank{}", it.first, join_as_str(it.second));
  }
  Deployment deployment = createDeployment(repl, alloc, conf_.dev_num);
  deployment.pipeline_num = sol.pipeline_num;
  deployment.checkpointing = sol.checkpointing;
  deployment.offload_params = conf_.offload_params;
  logger->trace("MLPartDecomposer::decompose finished");

  return deployment;
}
} // namespace rannc