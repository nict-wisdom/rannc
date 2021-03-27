//
// Created by Masahiro Tanaka on 2019-05-30.
//

#include "spdlog/sinks/stdout_color_sinks.h"

#include "NodeProfiler.h"
#include "graph/ConvertGraph.h"

namespace rannc {

    ProfilingResult NodeProfiler::profile(const std::shared_ptr<IRGraph>& ir_graph, int iteration) {

        std::unordered_map<std::string, std::shared_ptr<IRGraph>> graphs;
        std::unordered_map<std::string, IRNode> node_map;

        const auto& nodes = ir_graph->getNodes();
        for (const auto& n: nodes) {
            std::vector<IRNode> pf_nodes = {n};
            std::unordered_map<std::string, IRValue> pf_values;
            for (const auto& in: n.getInputNames()) {
                pf_values[in] = ir_graph->getValue(in);
            }
            for (const auto& out: n.getOutputNames()) {
                pf_values[out] = ir_graph->getValue(out);
            }

            std::vector<std::string> input_names;
            for (const auto& in: n.getInputNames()) {
                const auto& v = ir_graph->getValue(in);
                if (!v.isParam()) {
                    input_names.push_back(v.getName());
                }
            }
            for (const auto& in: n.getInputNames()) {
                const auto& v = ir_graph->getValue(in);
                if (v.isParam()) {
                    input_names.push_back(v.getName());
                }
            }

            const auto& id = n.getId();
            graphs[id] = std::make_shared<IRGraph>(id, pf_nodes, pf_values, input_names, n.getOutputNames());
            node_map[id] = n;
        }

        return sg_prof_->profile(graphs, iteration);
    }
}
