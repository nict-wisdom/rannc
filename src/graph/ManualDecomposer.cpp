//
// Created by Masahiro Tanaka on 2019-07-01.
//
#include <regex>
#include <boost/range/algorithm/for_each.hpp>
#include <boost/range/adaptor/reversed.hpp>

#include <Config.h>

#include "Decomposition.h"
#include "ManualDecomposer.h"

namespace rannc {

    int getRank(const std::string& val_name) {
        std::regex pattern("_r([0-9]+)_.*");
        std::cmatch match;

        if (std::regex_match(val_name.c_str(), match, pattern)) {
            assert(match.size() == 2);
            return std::atoi(match.str(1).c_str());
        } else {
            return 0;
        }
    }

    Deployment ManualDecomposer::decompose(const std::shared_ptr<IRGraph>& irGraph, int n_partition, int64_t batch_size) {
        logger->info("ManualDecomposer::decompose starting");

        BGraph g = toBGL(irGraph);

        std::vector<Vertex> topo_vertices, rev_topo_vertices;
        boost::topological_sort(g, std::back_inserter(rev_topo_vertices));

        boost::for_each(rev_topo_vertices | boost::adaptors::reversed, [&topo_vertices](Vertex v) {
            topo_vertices.push_back(v);
        });
//
//        for (const auto& v: topo_vertices) {
//            int rank = getRank(g[v].name);
//            g[v].rank = rank;
//        }
//
//        // Set node's rank to the rank of its output
//        for (const auto& v: topo_vertices) {
//            if (g[v].type == NODE) {
//                for (const auto& tgt: target_nodes(v, g)) {
//                    if (g[tgt].rank != 0) {
//                        g[v].rank = g[tgt].rank;
//                        break;
//                    }
//                }
//            }
//        }
//        fixConstantRank(g);
//        mergeNodesFromInput(g);

        bool dump_graph = config::Config::get().getVal<bool>(config::DUMP_GRAPH);
        std::string dump_graph_prefix = config::Config::get().getVal<std::string>(config::DUMP_GRAPH_PREFIX);
        if (dump_graph) {
            std::ofstream file(dump_graph_prefix + "_manual_r0.dot");
            boost::write_graphviz(file, g, vertex_rank_label_writer<BGraph>(g));
        }

        Partition partition = createPartition(g);
        std::unordered_map<std::string, int> repl_nums;
        for (const auto& it: partition.subgraphs) {
            repl_nums[it.first] = 1;
        }
        PartitionDP partitionDp = replicate(partition, repl_nums, batch_size);

        std::unordered_map<std::string, int> alloc = searchAllocation(partitionDp,
                                                                      mpi::getSize()-1, 15 * (1024L * 1024L * 1024L));

        for (const auto& a: alloc) {
            std::cout << "subgraph " << a.first << " ranks=" << a.second << std::endl;
        }

        logger->info("ManualDecomposer::decompose finished");

        return createDeployment(partitionDp, alloc);
    }

}
