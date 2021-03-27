//
// Created by Masahiro Tanaka on 2019-06-13.
//

#include <comm/SComm.h>
#include "GraphLauncher.h"

namespace rannc {

    bool isBatch(const std::shared_ptr<IRGraph> &graph, const std::string &value_name) {
        return graph->getValue(value_name).isBatch();
    }

    IValueMap GraphLauncher::alignBatch(const IValueMap &input, int batch_size, const std::shared_ptr<IRGraph> &graph,
            bool zero_pad) {
        IValueMap pad_input;
        for (const auto &it: input) {
            const auto& loc = it.first;
            const auto& ival = it.second;
            if (isBatch(graph, loc.value_name)) {
                pad_input[loc] = alignTensorsInIValue(ival, batch_size, zero_pad);
            } else {
                pad_input[loc] = ival;
            }
        }
        return pad_input;
    }

    void createRouteCommunicator(const std::vector<RouteDP>& routes) {
        TagMap& tag_map = TagMap::get();
        AllReduceRunner& ar = AllReduceRunner::get();

        for (const auto& r: routes) {
            const auto ranks = getRanksInRoute(r);
            int tag = tag_map.getRankSetTag(ranks);
            if (contains(ranks, mpi::getRank())) {
                ar.createCommunicator(tag, ranks);
            }
        }
    }

    void GraphLauncher::deployGraph(const Deployment &deployment) {

        logger->trace("GraphLauncher::deployGraph starting");

        deployment_ = deployment;

        driver_[deployment.id] = std::make_shared<GraphConnector>(
                deployment.id, param_storage_, value_storage_,
                this->function_storage_
        );
        driver_[deployment.id]->deployGraph(deployment);

//        std::atomic_bool bwd_run_flag = {false};
//        bwd_running_.emplace(deployment.id, std::move(bwd_run_flag));
//        bwd_running_[deployment.id].

        TagMap& tag_map = TagMap::get();
        tag_map.sync();
        createRouteCommunicator(deployment.fwd_in_routes);
        createRouteCommunicator(deployment.fwd_routes);
        createRouteCommunicator(deployment.fwd_out_routes);
        createRouteCommunicator(deployment.bwd_in_routes);
        createRouteCommunicator(deployment.bwd_routes);
        createRouteCommunicator(deployment.bwd_out_routes);

        logger->trace("GraphLauncher::deployGraph finished");
    }

    void GraphLauncher::undeployGraph(const std::string& id) {
        if (enable_profiling_) {
            logger->info("Rank {} launcher profiling summary\n{}", mpi::getRank(),
                         time_counter_.summary<std::chrono::milliseconds>());
        }

        driver_.erase(id);
    }

    IValueMap GraphLauncher::compute(const std::string& id, bool is_bwd,
            int64_t batch_size, const IValueMap &inputs, const std::function<void(int64_t)> &set_bs,
            std::vector<RouteDP> &in_routes, std::vector<RouteDP> &out_routes) {

        // Assume we already padded the global batch size according to the world size
        assert(batch_size % mpi::getSize() == 0);

        logger->trace("GraphLauncher::compute starting");

        SComm& scomm = SComm::get();

        int actual_pipeline_num = pipeline_num_ > batch_size ? batch_size : pipeline_num_;

        // Routes from/to subgraphs
        std::unordered_map<std::string, std::unordered_map<IValueLocation, RouteDP, IValueLocationHash>> in_route_map;
        for (const auto& r: in_routes) { in_route_map[r.dest_graph][r.location] = r; }
        std::unordered_map<std::string, std::unordered_map<IValueLocation, RouteDP, IValueLocationHash>> out_route_map;
        for (const auto& r: out_routes) { out_route_map[r.source_graph][r.location] = r; }

        // *global* batch size in the pipeline
        std::vector<int64_t> split_batch_sizes = getSplitBatchSizes(batch_size, actual_pipeline_num);
        // *local* batch size of *this split* in the pipeline
        std::vector<int64_t> local_split_batch_sizes = getLocalSplitBatchSizes(split_batch_sizes, mpi::getSize(), mpi::getRank());

        /////////////////////////////////////////////////////
        // Step 1: distribute (inputs)
        /////////////////////////////////////////////////////
        std::vector<std::unordered_map<std::string, IValueMap>> graph_inputs;
        graph_inputs.reserve(actual_pipeline_num);
        AllReduceRunner& ar = AllReduceRunner::get();
//        ar.startBulk();
        for (int i=0; i<actual_pipeline_num; i++) {
            std::unordered_map<std::string, IValueMap> split_inputs;

            // *global* batch sizes of this split in the pipeline
            int64_t split_bs = split_batch_sizes.at(i);
            set_bs(split_bs);
            scomm.startSplit(i);

            for (const auto &r: in_routes) {
                assert(contains(inputs, r.location));
                const auto &val = inputs.at(r.location);

                torch::jit::IValue send_val;
                if (r.ir_value.isLoss() || r.ir_value.isBatch()) {
                    send_val = sliceOrWeightTensorsInIValue(val, local_split_batch_sizes, i);
                } else {
                    throw std::runtime_error(
                            "Unexpected type of graph input. route=" + toString(r));
                }

                auto ir_type = r.ir_value.getType();
                ir_type.setBatchSize(split_bs);

                logger->trace("Sending input via route {} split={} {}", toString(r), i, toString(toIRType(send_val)));
                const auto in = scomm.distribute(send_val, r, is_bwd, ir_type);
                logger->trace("Received input via route {} split={} {}", toString(r), i, toString(toIRType(in)));

                if (!in.isNone()) {
                    split_inputs[r.dest_graph][r.location] = in;
                }
            }

            graph_inputs.push_back(split_inputs);
        }
//        ar.endBulk();

        /////////////////////////////////////////////////////
        // Step 2: compute
        /////////////////////////////////////////////////////
        std::vector<std::unordered_map<std::string, IValueMap>> graph_driver_out; // graph_id -> IValueMap
        for (int i=0; i<actual_pipeline_num; i++) {
            assert(graph_inputs.size() > i);
            assert(local_split_batch_sizes.size() > i);

            std::unordered_map <std::string, IValueMap> split_driver_out;

            int64_t split_bs = split_batch_sizes.at(i);
            set_bs(split_bs);
            scomm.startSplit(i);
            if (is_bwd) {
                split_driver_out = driver_[id]->backward(id, graph_inputs.at(i), i);
            } else {
                split_driver_out = driver_[id]->forward(id, graph_inputs.at(i), i,
                                                        torch::autograd::GradMode::is_enabled());
            }
            graph_driver_out.push_back(split_driver_out);
        }

        /////////////////////////////////////////////////////
        // Step 3: distribute (outputs)
        /////////////////////////////////////////////////////
        std::vector<std::unordered_map<std::string, IValueMap>> graph_outputs;
        for (int i=0; i<actual_pipeline_num; i++) {
            assert(graph_driver_out.size() > i);

            std::unordered_map <std::string, IValueMap> &split_driver_out = graph_driver_out.at(i);

            // *global* batch sizes of this split in the pipeline
            int64_t split_bs = split_batch_sizes.at(i);
            set_bs(split_bs);
            scomm.startSplit(i);

            std::unordered_map <std::string, IValueMap> split_out;
            for (const auto &g_it: out_route_map) {
                const auto &sg_name = g_it.first;
                const auto &sg_out_routes = g_it.second;

                for (const auto &r_it: sg_out_routes) {
                    const auto &loc = r_it.first;
                    const auto &route = r_it.second;
                    logger->trace("Processing graph output route: {} split={}", toString(route), i);

                    torch::jit::IValue send_val;
                    if (contains(split_driver_out[sg_name], loc)) {
                        send_val = split_driver_out[sg_name].at(loc);
                    }
                    logger->trace("Sending output {} via route {} split={}", toString(toIRType(send_val)),
                                  toString(route), i);
                    auto ir_type = route.ir_value.getType();
                    ir_type.setBatchSize(split_bs);

                    const auto out = scomm.distribute(send_val, route, is_bwd, ir_type);
                    logger->trace("Received output {} via route: {} split={}", toString(toIRType(out)),
                                  toString(route), i);

                    if (!out.isNone()) {
                        split_out[sg_name][loc] = out;
                    }
                }
            }

            graph_outputs.push_back(std::move(split_out));
        }

        IValueMap ret;

        // Merge split results
        for (const auto& g_it: out_route_map) {
            const auto& sg_name = g_it.first;
            const auto& sg_out_routes = g_it.second;

            for (const auto& r_it: sg_out_routes) {
                const auto& loc = r_it.first;
                const auto& route = r_it.second;

                std::vector<torch::jit::IValue> loc_values;
                for (int i=0; i<actual_pipeline_num; i++) {
                    const auto& split_recv_map = graph_outputs.at(i);
                    if (!contains(split_recv_map, sg_name)) break;

                    const auto& sg_split_recv_map = split_recv_map.at(sg_name);
                    if (!contains(sg_split_recv_map, loc)) break;

                    loc_values.push_back(sg_split_recv_map.at(loc));
                }

                const auto &ir_val = route.ir_value;
                if (!loc_values.empty()) {
                    if (ir_val.isLoss() || ir_val.isBatch()) {
                        ret[loc] = concatOrSumTensorsInIValues(loc_values, batch_size);
                    } else {
                        throw std::runtime_error("Unexpected type of graph input. route=" + toString(route));
                    }
                }
            }
        }

        logger->trace("GraphLauncher::compute finished");

        return ret;
    }

    torch::jit::IValue GraphLauncher::forward(const std::string &id, const IValueMap &inputs) {

//        spdlog::info("GraphLauncher::forward starting id={} rng_state={}", id,
//                     toString(getRngState()));

        logger->trace("GraphLauncher::forward starting");
        time_counter_.start("GraphLauncher::forward");

        int64_t input_batch_size = (int64_t) guessBatchSize(inputs);
        int64_t max_local_batch_size = mpi::allReduceMaxBatchSize(input_batch_size);

        last_batch_size_ = max_local_batch_size;
        const auto pad_inputs = alignBatch(inputs, max_local_batch_size, deployment_.graph, false);

        int64_t global_batch_size = max_local_batch_size * mpi::getSize();
        SComm& scomm = SComm::get();
        scomm.startFwd(global_batch_size);

        std::function<void(int64_t)> set_bs = [&scomm](size_t batch_size) {
                    return scomm.startFwd(batch_size);
                };

        auto outputs = compute(id, false, global_batch_size, pad_inputs, set_bs, deployment_.fwd_in_routes, deployment_.fwd_out_routes);

        outputs = alignBatch(outputs, input_batch_size, deployment_.graph, false);

        const auto &output_names = deployment_.graph->getOutputNames();
        assert(output_names.size() == 1);
        assert(contains(outputs, output_names.front()));

        auto output_value = outputs.at(output_names.front());

        time_counter_.stop("GraphLauncher::forward");
        logger->trace("GraphLauncher::forward finished");

        return output_value;
    }

    IValueMap GraphLauncher::backward(const std::string &id, const IValueMap &inputs) {

        std::atomic_bool& graph_bwd_running = bwd_running_[id];
        bool expected_running = false;
        bool result = graph_bwd_running.compare_exchange_strong(expected_running, true);
        if (!result) {
            throw std::runtime_error("Concurrent calls of backward are not allowed. "
                                     "This happens when backward is called on a tensor produced from multiple output tensors "
                                     "of a RaNNC module.");
        }

        logger->trace("GraphLauncher::backward starting");
        time_counter_.start("GraphLauncher::backward");

        int64_t input_batch_size = (int64_t) guessBatchSize(inputs);
        if (input_batch_size < 0) { // maybe loss
            input_batch_size = last_batch_size_;
        }
        int64_t max_local_batch_size = mpi::allReduceMaxBatchSize(input_batch_size);
        const auto pad_inputs = alignBatch(inputs, max_local_batch_size, deployment_.graph, true);

        int64_t global_batch_size = max_local_batch_size * mpi::getSize();

        // Scale if a batch
        // loss value is scaled by scommtensor
        IValueMap scaled_inputs;
        for (const auto& route: deployment_.bwd_in_routes) {
            double scale = getDpRatio(global_batch_size, mpi::getAllRanks(), mpi::getRank());
            scaled_inputs[route.location] = transformTensorsInIValue(pad_inputs.at(route.location),
                    [scale](const at::Tensor& t){
                const auto& dim = getTensorDim(t);
                if (dim.empty()) {
                    return t;
                }
                torch::NoGradGuard no_grad;
                return t.mul(scale).detach();
            });
        }

        SComm& scomm = SComm::get();
        std::function<void(int64_t)> set_bs = [&scomm](size_t batch_size) {
            return scomm.startBwd(batch_size);
        };
        auto outputs = compute(id, true, global_batch_size, scaled_inputs, set_bs, deployment_.bwd_in_routes, deployment_.bwd_out_routes);

        outputs = alignBatch(outputs, input_batch_size, deployment_.graph, false);

        time_counter_.stop("GraphLauncher::backward");
        logger->trace("GraphLauncher::backward finished");

        graph_bwd_running.store(false);

        return outputs;
    }
}
