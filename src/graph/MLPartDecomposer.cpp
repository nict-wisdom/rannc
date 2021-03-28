//
// Created by Masahiro Tanaka on 2020/02/24.
//

#include "MLPartDecomposer.h"
#include "Partitioner.h"
#include "DPStaging.h"

namespace rannc {

    Partition createPartition(const std::shared_ptr<IRGraph>& ir_graph,
                              const std::vector<std::shared_ptr<IRGraph>>& subgraphs) {

        // value name -> graph id
        // a value can be an input of one or more graphs
        std::unordered_map<std::string, std::unordered_set<std::string>> in_vals;
        // Only one graph produces a value
        std::unordered_map<std::string, std::string> out_vals;
        std::unordered_map<std::string, std::shared_ptr<IRGraph>> sg_map;
        std::vector<std::string> sg_order;

        // graph id -> input value names
        std::unordered_map<std::string, std::unordered_set<std::string>> created_cons;

        for (const auto& sg: subgraphs) {
            for (const auto& in: sg->getInputNames()) {
                if (!sg->getValue(in).isParam()) {
                    in_vals[in].insert(sg->getName());
                }
            }
            for (const auto& out: sg->getOutputNames()) {
                if (!contains(ir_graph->getInputNames(), out)) {
                    out_vals[out] = sg->getName();
                }
            }

            sg_map[sg->getName()] = sg;
            sg_order.push_back(sg->getName());
        }

        std::vector<GraphConnection> connections;
        for (const auto& sg: subgraphs) {
            for (const auto& in: sg->getInputNames()) {
                if (!sg->getValue(in).isParam()) {
                    if (contains(out_vals, in)) {
                        const auto &src_id = out_vals.at(in);
                        const auto &tgt_id = sg->getName();
                        if (src_id != tgt_id) {
                            GraphConnection con{in, src_id, tgt_id, src_id + "_" + in,
                                                tgt_id + "_" + in};
                            connections.push_back(con);

                            created_cons[tgt_id].insert(in);
                        }
                    }
                }
            }
        }

        for (const auto& in: ir_graph->getInputNames()) {
            if (!ir_graph->getValue(in).isParam()) {
                if(!contains(in_vals, in)) {
                    // unused input
                    continue;
                }
                for (const auto &tgt_id: in_vals.at(in)) {
                    if (!contains(created_cons[tgt_id], in)) { ;
                        GraphConnection con{in, "MASTER", tgt_id, "MASTER_" + in,
                                            tgt_id + "_" + in};
                        connections.push_back(con);
                        created_cons[tgt_id].insert(in);
                    }
                }
            }
        }

        for (const auto& out: ir_graph->getOutputNames()) {
            assert(contains(out_vals, out));
            const auto& src_id = out_vals.at(out);
            GraphConnection con{out, src_id, "MASTER", src_id + "_" + out,
                                "MASTER_" + out};
            connections.push_back(con);
        }

        return Partition{ir_graph->getName(), ir_graph, sg_map, connections, sg_order};
    }

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
        MLPartitioner partitioner(sg_prof_, mpi::getSize(), dev_mem_, max_pipeline, min_pipeline_bs,
                                  use_amp_master_params_, coarsen_by_time);
        MLGraph part_graph = partitioner.partition(ir_graph);

        ///////////
        logger->trace("Starting DP: id={} #nodes={}", ir_graph->getName(),
                      part_graph.nodes.size());
        DPStaging dp(sg_prof_, batch_size_, dev_mem_, use_amp_master_params_);
        AllocSolution sol = dp.runDpComm(part_graph, mpi::getSize());
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

        const auto repl = replicate(new_part, repl_nums, batch_size_);
        logger->trace("Partitioning finished: id={}", ir_graph->getName());

        std::unordered_map<std::string, std::unordered_set<int>> alloc;

        if (config::Config::get().getVal<bool>(config::ALLOC_REPL_FLAT)) {
            logger->trace("searchAllocationFlat");
            alloc = searchAllocationFlat(repl, mpi::getSize(), dev_mem_);
        } else {
            logger->trace("searchAllocationSimple");
            alloc = searchAllocationSimple(repl, mpi::getSize(), dev_mem_);
        }

        if (alloc.empty()) {
            throw std::runtime_error("Failed to allocate gpus to subgraphs.");
        }

        for (const auto &it: alloc) {
            logger->info(" Assigned subgraph {} to rank{}", it.first, join_as_str(it.second));
        }
        Deployment deployment = createDeployment(repl, alloc);
        deployment.pipeline_num = sol.pipeline_num;
        deployment.checkpointing = sol.checkpointing;
        logger->trace("MLPartDecomposer::decompose finished");

        return deployment;
    }
}