//
// Created by Masahiro Tanaka on 2020/02/04.
//

#include "SchedulingDecomposer.h"

namespace rannc {

    bool anyHasMultipleTargets(const std::vector<Vertex> nodes, const BGraph& bg) {
        for (const auto& v: nodes) {
            if (target_nodes(v, bg).size() > 1) {
                return true;
            }
        }
        return false;
    }

    void splitSerAndPar(BGraph& bg) {
        for (const auto& v: all_nodes<Vertex, BGraph>(bg)) {
            bg[v].ranks.clear();
        }
        int rank = 0;
        for (const auto& v: all_nodes_topo<Vertex, BGraph>(bg)) {
            const auto srcs = source_nodes(v, bg);
            if (srcs.size() > 1 || srcs.empty() || anyHasMultipleTargets(srcs, bg)) {
                rank++;
            }
            bg[v].ranks.insert(rank);
        }
    }

    void setConstRank(BGraph& bg, int rank) {
        for (const auto& v: all_nodes_topo<Vertex, BGraph>(bg)) {
            bg[v].ranks.insert(rank);
        }
    }

    void copyRanks(const BGraph& src, BGraph& tgt) {
        std::unordered_map<std::string, Vertex> ver_map;
        for (const auto& v: all_nodes<Vertex, BGraph>(tgt)) {
            ver_map[tgt[v].id] = v;
        }
        for (const auto& v: all_nodes<Vertex, BGraph>(src)) {
            tgt[ver_map[src[v].id]].ranks = src[v].ranks;
        }
    }

   std::unordered_map<std::string, std::unordered_set<int>> searchSubgraphAllocation(
            const PartitionDP& partition,
            size_t dev_count, size_t dev_mem) {

       std::unordered_map<std::string, std::unordered_set<int>> ret;
       int i=0;
       for (const auto& it: partition.replica_nums) {
            int repl_num = it.second;
            for (int r_idx=0; r_idx<repl_num; r_idx++) {
                ret[it.first].insert(i % mpi::getSize());
                i++;
            }
       }
       return ret;
    }

    void setProfileTime(BGraph& bg, const std::unordered_map<std::string, GraphProfile>& node_profiles) {
        for (const auto& v: all_nodes<Vertex, BGraph>(bg)) {
            if (bg[v].type == NODE) {
                if (contains(node_profiles, bg[v].id)) {
                    const GraphProfile& prof = node_profiles.at(bg[v].id);
                    bg[v].fwd_time = prof.fwd_time;
                    bg[v].bwd_time = prof.bwd_time;
                }
            }
        }
    }

    Deployment SchedulingDecomposer::decompose(const std::shared_ptr<IRGraph> &irGraph) {

        int conf_n_partition = config::Config::get().getVal<int>(config::PARTITION_NUM);

        logger->trace("SchedulingDecomposer::decompose starting");

        if (dev_mem_ > 0) {
            dev_mem_ -= 1024L * 1024L * 1024L;
        } else {
            logger->warn(
                    "No CUDA device found on workers. Assuming (almost) unlimited host memory when assigning subgraphs.");
            dev_mem_ = 2 * 1024L * 1024L * 1024L * 1024L; // 2TB
        }

        int n_partition = conf_n_partition;
        if (n_partition <= 0) {
            size_t graph_size = irGraph->getSizeInByte();
            n_partition = (int) (graph_size / dev_mem_) + 1;
            logger->info("The partition num was automatically set to {}. To manually set this value, "
                         "add 'partition_num' to ~/.pyrannc/rannc_conf.toml", n_partition);
        }
        assert(n_partition != 0);

        BGraph bg = toBGL(irGraph);
        setProfileTime(bg, node_profiles_);

        std::ofstream file_org("decomp_org.dot");
        boost::write_graphviz(file_org, bg, vertex_rank_label_writer<BGraph>(bg));

        BGraph batch_bg = copyGraphWithBatch(bg);
        std::ofstream file("decomp_batch.dot");
        boost::write_graphviz(file, batch_bg, vertex_rank_label_writer<BGraph>(batch_bg));

        setConstRank(batch_bg, 1);
        copyRanks(batch_bg, bg);
        std::ofstream file_batch_marked("decomp_batch_marked.dot");
        boost::write_graphviz(file_batch_marked, bg, vertex_rank_label_writer<BGraph>(bg));

        splitSerAndPar(batch_bg);
        std::ofstream file2("decomp_batch_sp.dot");
        boost::write_graphviz(file2, batch_bg, vertex_rank_label_writer<BGraph>(batch_bg));

        copyRanks(batch_bg, bg);
        std::ofstream file2_1("decomp_all.dot");
        boost::write_graphviz(file2_1, bg, vertex_rank_label_writer<BGraph>(bg));

        fixNonBatchRanks(bg);
        std::ofstream file3("decomp_all_fix.dot");
        boost::write_graphviz(file3, bg, vertex_rank_label_writer<BGraph>(bg));

        int replica_num = config::Config::get().getVal<int>(config::REPLICA_NUM);

        Partition partition = createPartition(bg);

        for (const auto &it: partition.subgraphs) {
            BGraph sg = toBGL(it.second);
            setProfileTime(sg, node_profiles_);
            std::ofstream sg_file("test_graph_" + it.first + ".dot");
            boost::write_graphviz(sg_file, sg, vertex_rank_label_writer<BGraph>(sg));
        }

        int profiling_iter = config::Config::get().getVal<int>(config::PROFILING_ITER);
        const auto prof = sg_prof_->profile(partition.subgraphs, profiling_iter);
        std::unordered_map<std::string, long> node_fwd_time;
        std::unordered_map<std::string, long> node_bwd_time;
        for (const auto& v: all_nodes<Vertex, BGraph>(bg)) {
            node_fwd_time[bg[v].id] = bg[v].fwd_time;
            node_bwd_time[bg[v].id] = bg[v].bwd_time;
        }

        for (const auto& it: prof.node_profiles) {
            const auto& sg_name = it.first;
            const auto& sg = partition.subgraphs.at(sg_name);
            long acc_fwd_time = 0;
            long acc_bwd_time = 0;
            for (const auto& n: sg->getNodes()) {
                acc_fwd_time += node_fwd_time.at(n.getId());
                acc_bwd_time += node_bwd_time.at(n.getId());
            }
            const GraphProfile& p = it.second;
            spdlog::info("{} fwd={} bwd={} mem={} acc_fwd={} acc_bwd={}",
                    it.first, p.fwd_time, p.bwd_time, p.max_allocated_mem,
                    acc_fwd_time, acc_bwd_time);
        }

        // Use IRGraph in the argument because createPartition loses the order of inputs/outputs
        partition.graph = irGraph;
        logger->trace("ProfiledWeightDecomposer::decompose created partition. id={}", partition.id);

        logger->trace("ProfiledWeightDecomposer::decompose creating replications. id={} repl_num={}", partition.id,
                      replica_num);

        std::unordered_map<std::string, int> repl_nums;
        for (const auto &it: partition.subgraphs) {
            repl_nums[it.first] = replica_num;
        }
        PartitionDP partitionDp = replicate(partition, repl_nums, batch_size_);
        logger->trace("SchedulingDecomposer::decompose created PartitionDP. id={}", partitionDp.id);

        logger->info("Assigning {} subgraphs to {} device(s) ... (mem: {} per device)",
                     partitionDp.subgraphs.size(), mpi::getSize(), dev_mem_);

        std::unordered_map<std::string, std::unordered_set<int>> alloc = searchSubgraphAllocation(partitionDp, mpi::getSize(), dev_mem_ * 2);
        if (alloc.empty()) {
            throw std::runtime_error("Failed to allocate gpus to subgraphs.");
        }

        for (const auto &it: alloc) {
            logger->info(" Assigned subgraph {} to rank{}", it.first, join_as_str(it.second));
        }
        Deployment deployment = createDeployment(partitionDp, alloc);
        deployment.pipeline_num = 1;
        logger->trace("SchedulingDecomposer::decompose finished");

        return deployment;
    }
}