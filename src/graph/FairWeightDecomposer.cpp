//
// Created by Masahiro Tanaka on 2019-03-15.
//
#include <cassert>

#include <boost/range/algorithm/for_each.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>

#include <graph/ir.h>
#include <Config.h>
#include "FairWeightDecomposer.h"
#include "Decomposition.h"

namespace rannc {


    size_t calcValueSize(const IRValue& value, int replica_num) {
        size_t val_size = value.getSizeInByte();
        if (value.isParam()) {
            // under testing, param + gradient + amp master param + buffer*2 (1 + 1 + 2 + 2*2)
//            return val_size * 9;
            return val_size;
        } else if (value.isBatch()) {
            return val_size / replica_num;
        }
        return val_size;
    }

    void chooseRanksOnNonCriticalPaths(BGraph &g) {
        std::vector<Vertex> rev_topo_vertices;
        boost::topological_sort(g, std::back_inserter(rev_topo_vertices));

        // check nodes and values in a reversed topological sorted order
        for (auto &v: rev_topo_vertices) {
            if (g[v].ranks.empty()) {
                const auto& val = g[v].value;
                std::vector<Vertex> targets = target_nodes<Vertex, BGraph>(v, g);
                assert(!targets.empty());
                if (val.isBatch()) {
                    assert(!g[targets.front()].ranks.empty());
                    g[v].ranks = g[targets.front()].ranks;
                } else {
                    for (const auto& tgt: targets) {
                        assert(!g[tgt].ranks.empty());
                        for (int r: g[tgt].ranks) {
                            g[v].ranks.insert(r);
                        }
                    }
                }
//                spdlog::info("{} has no rank. set to {}", g[v].name, join_as_str(g[v].ranks));
            }
        }
    }

    void splitByCriticalPath(BGraph& g, int n_partition) {
        auto input_nodes = graph_regular_input_nodes<Vertex, BGraph>(g);
        auto output_nodes = graph_output_nodes<Vertex, BGraph>(g);

        std::vector<boost::graph_traits<BGraph>::vertices_size_type> distances(boost::num_vertices(g));
        auto dist_pmap = boost::make_iterator_property_map(distances.begin(), get(boost::vertex_index, g));

        std::vector<Vertex> parents(boost::num_vertices(g));
        auto parents_pmap = boost::make_iterator_property_map(parents.begin(), get(boost::vertex_index, g));

        int longest_path_len = 0;
        std::vector<Vertex> longest_path_rev;

        for (auto& in_v: input_nodes) {
            auto v1 = boost::record_distances(dist_pmap, boost::on_tree_edge());
            auto v2 = boost::record_predecessors(parents_pmap, boost::on_tree_edge());
            auto visitor = boost::visitor(boost::make_bfs_visitor(std::make_pair(v1, v2)));
            boost::breadth_first_search(g, in_v, visitor);

            for (const auto& out_v: output_nodes) {
                spdlog::info("path length from {} to {}: {}", g[in_v].name, g[out_v].name, dist_pmap[out_v]);

                if (dist_pmap[out_v] > longest_path_len) {
                    longest_path_len = dist_pmap[out_v];

                    longest_path_rev.clear();
                    longest_path_rev.reserve(dist_pmap[out_v] + 1);
                    longest_path_rev.push_back(out_v);
                    Vertex v = out_v;
                    for (int i=0; i<longest_path_len; i++) {
                        Vertex p = parents_pmap[v];
                        longest_path_rev.push_back(p);
                        v = p;
                    }
                }
            }
        }

        int pos = 0;
        int rank = 1;
        std::vector<Vertex> longest_path = reverse(longest_path_rev);
        for (const auto& v: longest_path) {
            g[v].ranks.insert(rank);
            spdlog::info(" path {} {} rank={}", pos, g[v].name, rank);
            if (pos > longest_path_len*rank/n_partition) {
                rank++;
            }
            pos++;
        }

        chooseRanksOnNonCriticalPaths(g);
    }

    Deployment FairWeightDecomposer::decompose(const std::shared_ptr<IRGraph>& irGraph) {

        int conf_n_partition = config::Config::get().getVal<int>(config::PARTITION_NUM);

        logger->trace("FairWeightDecomposer::decompose starting");

        if (dev_mem_ > 0) {
            dev_mem_ -= 1024L * 1024L * 1024L;
        } else {
            logger->warn("No CUDA device found on workers. Assuming (almost) unlimited host memory when assigning subgraphs.");
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

        int replica_num = config::Config::get().getVal<int>(config::REPLICA_NUM);

        BGraph g = toBGL(irGraph);
        std::vector<Vertex> topo_vertices, rev_topo_vertices;
        boost::topological_sort(g, std::back_inserter(rev_topo_vertices));

        boost::for_each(rev_topo_vertices | boost::adaptors::reversed, [&topo_vertices](Vertex v) {
            topo_vertices.push_back(v);
        });

        // strategy 1
        const auto split = splitByValueSizes(g, n_partition);
        setRanksOnGraph(g, split);
        // strategy 2
//        splitByCriticalPath(g, n_partition);

        // fix decomposition
        fixNonBatchRanks(g);

        Partition partition = createPartition(g);
        for (const auto& it: partition.subgraphs) {
            const auto& name = it.first;
            const auto& sg = it.second;
            logger->trace(" AFTER createPartition: Subgraph {} size={} param_size={}", name, sg->getSizeInByte(), sg->getParamSizeInByte());
        }

        // Use IRGraph in the argument because createPartition loses the order of inputs/outputs
        partition.graph = irGraph;
        logger->trace("FairWeightDecomposer::decompose created partition. id={}", partition.id);

        logger->trace("FairWeightDecomposer::decompose creating replications. id={} repl_num={}", partition.id, replica_num);

        std::unordered_map<std::string, int> repl_nums;
        for (const auto& it: partition.subgraphs) {
            repl_nums[it.first] = replica_num;
        }
        int conf_n_pipeline = config::Config::get().getVal<int>(config::PIPELINE_NUM);
        PartitionDP partitionDp = replicate(partition, repl_nums, conf_n_pipeline, batch_size_);
        logger->trace("FairWeightDecomposer::decompose created PartitionDP. id={}", partitionDp.id);

//        logger->info("Created subgraphs: partitions={} replications={}", n_partition, replica_num);
//        for (const auto& it: partitionDp.subgraphs) {
//            const auto& name = it.first;
//            const auto& sg = it.second;
////            logger->info(" AFTER replicate: Subgraph {} size={} opt_param_size={} sum={}", name, sg->getSizeInByte(), sg->getParamSizeInByte()*6,
////                         sg->getSizeInByte() + sg->getParamSizeInByte()*6);
//        }

//        const auto alloc_mem = sg_prof_->profile(partitionDp.subgraphs);
//        for (const auto& it: alloc_mem) {
//            const auto& prof = it.second;
//            logger->info("{} fwd_time={} bwd_time={} mem={}", it.first, prof.fwd_time, prof.bwd_time, prof.max_allocated_mem);
//        }


        logger->info("Assigning {} subgraphs to {} device(s) ... (mem: {} per device) pipeline={}",
                     partitionDp.subgraphs.size(), mpi::getSize(), dev_mem_, conf_n_pipeline);

        std::unordered_map<std::string, std::unordered_set<int>> alloc = searchAllocation(partitionDp, mpi::getSize(), dev_mem_*2);
        if (alloc.empty()) {
            throw std::runtime_error("Failed to allocate gpus to subgraphs.");
        }

        for (const auto& it: alloc) {
            logger->info(" Assigned subgraph {} to rank{}", it.first, join_as_str(it.second));
        }
        Deployment deployment = createDeployment(partitionDp, alloc, mpi::getSize());
        logger->trace("FairWeightDecomposer::decompose finished");

        deployment.checkpointing = config::Config::get().getVal<bool>(config::CHECKPOINTING);
        deployment.pipeline_num = conf_n_pipeline;
        return deployment;
    }
}
