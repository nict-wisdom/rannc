//
// Created by Masahiro Tanaka on 2019-06-16.
//

#include <unistd.h>
#include <future>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <comm/SComm.h>
#include <Config.h>

#include "EventRecorder.h"
#include "GraphConnector.h"

namespace {

int getDelay(
    const rannc::RouteDP& r,
    const std::unordered_map<std::string, int>& graph_order) {
  return graph_order.at(r.dest_graph) - graph_order.at(r.source_graph) - 1;
}

int getMaxDelay(
    const std::vector<rannc::RouteDP>& routes,
    const std::unordered_map<std::string, int>& graph_order) {
  int max_delay = 0;
  for (const auto& r : routes) {
    assert(rannc::contains(graph_order, r.source_graph));
    assert(rannc::contains(graph_order, r.dest_graph));

    int send_delay = getDelay(r, graph_order);
    max_delay = std::max(max_delay, send_delay);
  }
  return max_delay;
}
} // namespace

namespace rannc {

bool containsUndefinedTensor(const IValueMap& values) {
  for (const auto& it : values) {
    if (it.second.isTensor()) {
      const auto t = it.second.toTensor();
      if (!t.defined()) {
        return true;
      }
    }
  }
  return false;
}

torch::jit::IValue GraphConnector::distributeOutput(
    bool is_bwd, const RouteDP& r, int split_index, int flush_offset,
    const std::unordered_map<std::string, int>& graph_order) {
  assert(contains(graph_order, r.source_graph));
  assert(contains(graph_order, r.dest_graph));
  int send_delay =
      graph_order.at(r.dest_graph) - graph_order.at(r.source_graph) - 1;

  int tgt_split = split_index + flush_offset - send_delay;
  if (tgt_split >= 0) {
    assert(contains(split_values_, tgt_split));
    assert(contains(split_values_.at(tgt_split), r.source_graph));
    auto& sg_values = split_values_.at(tgt_split).at(r.source_graph);

    torch::jit::IValue send_value;
    if (contains(sg_values, r.location)) {
      const auto event_key = getFuncKey(
          "GraphConnector", "driver_output_to_cuda", deployment_.id, tgt_split,
          false);
      recordStart(event_key);
      send_value = toCUDAIfAvailable(sg_values.at(r.location), true, false);
      recordEnd(event_key);
    }

    const auto event_key = getCommKey(
        "GraphConnector", "send", r, split_index, toIRType(send_value));
    recordStart(event_key);
    int split_delay = send_delay - flush_offset;
    logger->trace(
        "Sending output via route {} split={} split_delay={} tgt_split={}",
        toString(r), split_index, split_delay, tgt_split);
    SComm& scomm = SComm::get();
    auto recv_val = scomm.distribute(send_value, r, is_bwd, split_delay);
    recordEnd(event_key);
    logger->trace(
        "Finished sending output via route {} split={} split_delay={}, tgt_split={}",
        toString(r), split_index, split_delay, tgt_split);

    return recv_val;
  } else {
    logger->trace("Delaying send via route: {}", toString(r));
  }
  return torch::jit::IValue();
}

std::unordered_map<std::string, std::shared_ptr<IRGraph>> getGraphsOnRank(
    const Deployment& deployment, int rank) {
  std::unordered_map<std::string, std::shared_ptr<IRGraph>> graphs;

  for (const auto& it : deployment.subgraphs) {
    if (contains(deployment.allocation.at(it.first), rank)) {
      graphs[it.first] = it.second;
    }
  }
  return graphs;
}

std::vector<RouteDP> sortRoutes(
    const std::vector<RouteDP>& routes,
    const std::unordered_map<std::string, int>& graph_order) {
  std::vector<RouteDP> ret = routes;
  std::stable_sort(
      ret.begin(), ret.end(),
      [&graph_order](const RouteDP& r1, const RouteDP& r2) {
        int d1 = getDelay(r1, graph_order);
        int d2 = getDelay(r2, graph_order);
        assert(d1 >= 0);
        assert(d2 >= 0);

        if (d1 == d2) {
          return r1.location.value_name < r2.location.value_name;
        }
        return d1 < d2;
      });
  return ret;
}

std::vector<RouteDP> filterRoutes(
    const std::vector<RouteDP>& routes, const std::string& sg_name,
    const std::function<std::string(const RouteDP&)>& f) {
  std::vector<RouteDP> filtered_routes;
  for (const auto& r : routes) {
    if (f(r) == sg_name) {
      filtered_routes.push_back(r);
    }
  }
  return filtered_routes;
}

std::vector<RouteDP> sortRecvRoutes(
    const std::vector<RouteDP>& routes, const std::string& sg_name,
    const std::unordered_map<std::string, int>& graph_order) {
  std::vector<RouteDP> recv_routes = filterRoutes(
      routes, sg_name, [](const RouteDP& r) { return r.dest_graph; });
  return sortRoutes(recv_routes, graph_order);
}

std::vector<RouteDP> sortSendRoutes(
    const std::vector<RouteDP>& routes, const std::string& sg_name,
    const std::unordered_map<std::string, int>& graph_order) {
  std::vector<RouteDP> recv_routes = filterRoutes(
      routes, sg_name, [](const RouteDP& r) { return r.source_graph; });
  return sortRoutes(recv_routes, graph_order);
}

void GraphConnector::deployGraph() {
  assert(deployment_.pipeline_num <= 1 || deployment_.checkpointing);

  graphs_ = getGraphsOnRank(deployment_, mpi::getRank());

  for (const auto& sg_name : deployment_.fwd_graph_order) {
    if (contains(keys(graphs_), sg_name)) {
      fwd_sorted_graph_ids_.push_back(sg_name);
    }
  }
  for (const auto& sg_name : deployment_.bwd_graph_order) {
    if (contains(keys(graphs_), sg_name)) {
      bwd_sorted_graph_ids_.push_back(sg_name);
    }
  }

  assert(
      deployment_.fwd_graph_order.size() == deployment_.bwd_graph_order.size());
  for (size_t i = 0; i < deployment_.fwd_graph_order.size(); i++) {
    fwd_graph_order_[deployment_.fwd_graph_order.at(i)] = i;
    bwd_graph_order_[deployment_.bwd_graph_order.at(i)] = i;
  }

  for (const auto& sg_name : deployment_.fwd_graph_order) {
    if (!contains(graphs_, sg_name)) {
      continue;
    }
    fwd_recv_routes_[sg_name] =
        sortRecvRoutes(deployment_.fwd_routes, sg_name, fwd_graph_order_);
    fwd_send_routes_[sg_name] =
        sortSendRoutes(deployment_.fwd_routes, sg_name, fwd_graph_order_);

    for (const auto& r : fwd_recv_routes_[sg_name]) {
      fwd_inc_edges_count_[r.dest_graph][r.location]++;
    }
  }

  for (const auto& sg_name : deployment_.bwd_graph_order) {
    if (!contains(graphs_, sg_name)) {
      continue;
    }
    bwd_recv_routes_[sg_name] =
        sortRecvRoutes(deployment_.bwd_routes, sg_name, bwd_graph_order_);
    bwd_send_routes_[sg_name] =
        sortSendRoutes(deployment_.bwd_routes, sg_name, bwd_graph_order_);

    for (const auto& r : bwd_recv_routes_[sg_name]) {
      bwd_inc_edges_count_[r.dest_graph][r.location]++;
    }
  }

  for (const auto& it : graphs_) {
    const auto& subgraph = it.second;
    logger->trace("GraphConnector::deployGraph receiving constants and params");

    auto ir_constants = graphConstantValues(subgraph);
    IValueMap constants;
    for (const auto& irc : ir_constants) {
      constants[irc.getName()] = value_storage_->getValue(irc.getName());
    }
    logger->trace("GraphConnector::deployGraph received constants");

    auto ir_params = graphParamValues(subgraph);
    std::unordered_map<std::string, at::Tensor> param_tensors;
    for (const auto& irp : ir_params) {
      param_tensors[irp.getName()] =
          param_storage_->getParamTensor(deployment_.id, irp.getName());
    }
    logger->trace("GraphConnector::deployGraph received params");

    const std::string& sg_name = subgraph->getName();
    driver_.createModule(
        sg_name, deployment_.id, subgraph, constants, this->functions_,
        param_tensors, toDriverExecConf(deployment_));

    checkpointing_[sg_name] = deployment_.checkpointing;
    assert(contains(deployment_.allocation, sg_name));
    allocation_[sg_name] = deployment_.allocation.at(sg_name);
  }

  pipeline_num_ = deployment_.pipeline_num;
  max_fwd_delay_ = getMaxDelay(deployment_.fwd_routes, fwd_graph_order_);
  max_bwd_delay_ = getMaxDelay(deployment_.bwd_routes, bwd_graph_order_);
}

void GraphConnector::runDriver(
    std::unordered_set<std::string>& graphs_done,
    std::unordered_map<std::string, IValueMap>& values, int split_index,
    const std::function<IValueMap(const std::string&, const IValueMap&, int)>&
        f,
    const std::function<std::vector<std::string>(
        const std::shared_ptr<IRGraph>&)>& input_names_getter,
    const std::function<bool(const IValueMap&, int)>& skip) {
  for (const auto& it : graphs_) {
    const auto& sg_name = it.first;
    const auto& subgraph = it.second;

    if (contains(graphs_done, sg_name))
      continue;

    auto& sg_values = values[sg_name];
    const auto input_names = input_names_getter(subgraph);
    bool graph_ready = isGraphReady(input_names, sg_values);

    if (graph_ready) {
      IValueMap graph_inputs;
      for (const auto& in_name : input_names) {
        IValueLocation loc(in_name);
        assert(contains(sg_values, loc));
        graph_inputs[in_name] = sg_values.at(loc);
      }

      IValueMap driver_out;
      logger->trace("Starting driver split_index={}", split_index);
      if (!skip(graph_inputs, split_index)) {
        driver_out = f(sg_name, graph_inputs, split_index);
      }
      logger->trace("Finished driver split_index={}", split_index);
      for (const auto& it_out : driver_out) {
        if (!contains(sg_values, it_out.first)) {
          const auto event_key = getFuncKey(
              "GraphConnector", "driver_output_to_cpu", deployment_.id,
              split_index, false);
          recordStart(event_key);
          sg_values[it_out.first] = toCPU(it_out.second, true, true);
          recordEnd(event_key);
        }
      }
      graphs_done.insert(sg_name);
    }
  }
}

std::unordered_map<std::string, IValueMap> GraphConnector::compute(
    const std::string& id, bool is_bwd,
    const std::unordered_map<std::string, IValueMap>& inputs, int split_index,
    const std::unordered_map<std::string, std::vector<RouteDP>>& recv_routes,
    const std::unordered_map<std::string, std::vector<RouteDP>>& send_routes,
    const std::unordered_map<std::string, int>& graph_order,
    const std::vector<std::string>& sorted_graph_ids, int max_delay,
    const std::function<IValueMap(const std::string&, const IValueMap&, int)>&
        func,
    const std::function<void(
        std::unordered_map<std::string, IValueMap>&,
        std::unordered_map<
            std::string,
            std::unordered_map<
                IValueLocation, std::vector<torch::jit::IValue>,
                IValueLocationHash>>&)>& aggr,
    const std::function<std::vector<std::string>(
        const std::shared_ptr<IRGraph>&)>& input_names_getter,
    const std::function<bool(const IValueMap&, int)>& skip) {
  logger->trace(
      "GraphConnector::compute starting. id={} is_bwd={} split={}", id, is_bwd,
      split_index);

  if (split_index == 0) {
    split_values_.clear();
  }
  split_values_[split_index] = inputs; // copy
  std::unordered_set<std::string> graphs_done;

  SComm& scomm = SComm::get();
  std::unordered_map<
      std::string,
      std::unordered_map<
          IValueLocation, std::vector<torch::jit::IValue>, IValueLocationHash>>
      recv_values;

  for (const auto& sg_name : sorted_graph_ids) {
    // recv
    assert(contains(recv_routes, sg_name));
    for (const auto& r : recv_routes.at(sg_name)) {
      assert(contains(r.dests, mpi::getRank()));
      const auto event_key =
          getCommKey("GraphConnector", "recv", r, split_index);

      recordStart(event_key);
      logger->trace("Receiving input via route {}", toString(r));
      const auto recv_val = scomm.distribute(torch::jit::IValue(), r, is_bwd);
      if (!recv_val.isNone()) {
        recv_values[r.dest_graph][r.location].push_back(recv_val);
      }
      logger->trace("Received input via route {}", toString(r));
      recordEnd(event_key);
    }
    aggr(split_values_[split_index], recv_values);

    // compute
    runDriver(
        graphs_done, split_values_[split_index], split_index, func,
        input_names_getter, skip);

    // send
    assert(contains(send_routes, sg_name));
    for (const auto& r : send_routes.at(sg_name)) {
      assert(contains(split_values_[split_index], r.source_graph));
      const auto recv_val =
          distributeOutput(is_bwd, r, split_index, 0, graph_order);

      if (!recv_val.isNone()) {
        recv_values[r.dest_graph][r.location].push_back(recv_val);
      }
    }

    // flush
    if (split_index + 1 == pipeline_num_) { // last split
      logger->trace("Starting to flush. max_delay={}", max_delay);
      for (int i = 1; i <= max_delay; i++) {
        for (const auto& r : send_routes.at(sg_name)) {
          if (getDelay(r, graph_order) >= i) {
            logger->trace("Flushing step={} route={}", i, toString(r));
            distributeOutput(is_bwd, r, split_index, i, graph_order);
          }
        }
      }
    }
  }

  logger->trace(
      "GraphConnector::compute finished. id={} split={}", id, split_index);

  return split_values_[split_index];
}

std::unordered_map<std::string, IValueMap> GraphConnector::forward(
    const std::string& id,
    const std::unordered_map<std::string, IValueMap>& inputs, int split_index,
    bool grad_mode) {
  const auto event_key =
      getFuncKey("GraphConnector", "forward", id, split_index, grad_mode);
  recordStart(event_key);

  logger->trace(
      "GraphConnector::forward starting. id={} split={}", id, split_index);
  time_counter_.start("GraphConnector::forward");

  const auto func = [this](
                        const std::string& id, const IValueMap& inputs,
                        int split_index) {
    auto& driver = this->driver_;
    auto& cp = this->checkpointing_;
    assert(contains(cp, id));

    this->rng_states_[id][split_index] = getRngState();

    if (cp.at(id)) {
      if (split_index == 0) {
        for (int s = 0; s < pipeline_num_; s++) {
          inputs_[id][s].clear();
        }
      }

      c10::cuda::CUDAStream stream = c10::cuda::getStreamFromPool();
      {
        TraceEvent evt(getFuncKey(
            "GraphConnector", "input_to_cpu_async", id, split_index, false));
        c10::cuda::CUDAStreamGuard guard(stream);

        inputs_[id][split_index] = toCPU(inputs, true, true);
        copy_to_cpu_events_[id][split_index].record(stream);
      }

      SComm& scomm = SComm::get();
      assert(contains(allocation_, id));
      const std::unordered_set<int>& sg_ranks = allocation_.at(id);
      // Before computing the *last* split in pipeline, start moving inputs for
      // the *first* split to gpu
      if (scomm.isLastLocalSplit(
              allocation_.at(id), mpi::getRank(), split_index)) { //
        TraceEvent evt(
            getFuncKey("GraphConnector", "input_to_gpu", id, 0, false));
        c10::cuda::CUDAStreamGuard guard(stream);
        int first_split =
            scomm.getFirstLocalSplitIndex(sg_ranks, mpi::getRank());

        // Make sure that copy to cpu has finished
        copy_to_cpu_events_[id][first_split].block(stream);

        inputs_[id][first_split] =
            toCUDAIfAvailable(inputs_[id][first_split], true, true);
        copy_to_gpu_events_[id][first_split].record(stream);
      }

      torch::NoGradGuard no_grad;
      auto outputs = driver.forward(id, inputs, split_index);

      // for debugging
      if (verify_recomp_) {
        for (const auto& it : outputs) {
          last_cp_outputs_[id][split_index][it.first] =
              cloneTensorsInIValue(it.second);
        }
      }

      return outputs;
    } else {
      return driver.forward(id, inputs, split_index);
    }
  };

  const auto aggr = [this](
                        std::unordered_map<std::string, IValueMap>& values,
                        std::unordered_map<
                            std::string,
                            std::unordered_map<
                                IValueLocation, std::vector<torch::jit::IValue>,
                                IValueLocationHash>>& recv_values) {
    for (const auto& g_it : this->fwd_inc_edges_count_) {
      const auto& sg_name = g_it.first;
      for (const auto& count_it : g_it.second) {
        const auto& loc = count_it.first;
        const auto& edge_counts = count_it.second;

        assert(edge_counts == 1);

        if (contains(values, sg_name) && contains(values.at(sg_name), loc)) {
          continue;
        }
        if (recv_values[sg_name][loc].size() == edge_counts) {
          values[sg_name][loc] = recv_values.at(sg_name).at(loc).front();
        }
      }
    }
  };

  // Records whether a split is skipped
  const auto skip = [this](const IValueMap& values, int split) {
    for (const auto& it : values) {
      if (it.second.isTensor()) {
        const auto t = it.second.toTensor();
        if (!t.defined()) {
          this->skip_fwd_split_[split] = true;
          return true;
        }
      }
    }
    this->skip_fwd_split_[split] = false;
    return false;
  };

  auto values = compute(
      id, false, inputs, split_index, fwd_recv_routes_, fwd_send_routes_,
      fwd_graph_order_, fwd_sorted_graph_ids_, max_fwd_delay_, func, aggr,
      getNonParamInputNames, skip);

  recordEnd(event_key);

  return values;
}

std::unordered_map<std::string, IValueMap> GraphConnector::backward(
    const std::string& id,
    const std::unordered_map<std::string, IValueMap>& inputs, int split_index) {
  logger->trace(
      "GraphConnector::backward starting. id={} split={}", id, split_index);
  time_counter_.start("GraphConnector::backward");

  recordStart(getFuncKey("GraphConnector", "backward", id, split_index, false));

  const auto func = [this](
                        const std::string& id, const IValueMap& inputs,
                        int split_index) {
    auto& driver = this->driver_;
    auto& cp = this->checkpointing_;
    assert(contains(cp, id));

    if (cp.at(id)) {
      const auto stashed_rng_state = getRngState();

      assert(contains(this->rng_states_, id));
      assert(contains(this->rng_states_.at(id), split_index));
      {
        torch::autograd::AutoGradMode gm(true);
        setRngState(this->rng_states_.at(id).at(split_index));

        SComm& scomm = SComm::get();
        assert(contains(allocation_, id));
        const std::unordered_set<int>& sg_ranks = allocation_.at(id);

        // We don't need inputs for old splits
        int prev_split =
            scomm.getPrevLocalSplitIndex(sg_ranks, mpi::getRank(), split_index);
        if (prev_split >= 0) {
          inputs_[id][prev_split].clear();
        }

        // Move inputs to cuda for the *next* split. This intends to overlap the
        // copy and backward. Note that inputs for the first split has already
        // been moved when finishing forward for the last split
        if (!scomm.isLastLocalSplit(sg_ranks, mpi::getRank(), split_index)) {
          TraceEvent evt(getFuncKey(
              "GraphConnector", "input_to_gpu_async", id, split_index, false));

          c10::cuda::CUDAStream stream = c10::cuda::getStreamFromPool();

          int next_split = scomm.getNextLocalSplitIndex(
              sg_ranks, mpi::getRank(), split_index);
          c10::cuda::CUDAStreamGuard guard(stream);

          // Make sure that copy to cpu has finished
          at::cuda::CUDAEvent& prev_evt = copy_to_cpu_events_[id][next_split];
          prev_evt.block(stream);

          inputs_[id][next_split] =
              toCUDAIfAvailable(inputs_[id][next_split], true, true);

          at::cuda::CUDAEvent& cuda_evt =
              copy_to_gpu_events_[id][split_index + 1];
          cuda_evt.record(stream);
        }

        // We have to wait for inputs back to gpu
        copy_to_gpu_events_[id][split_index].block(
            c10::cuda::getCurrentCUDAStream());
        const auto outputs =
            driver_.forward(id, inputs_[id][split_index], split_index);

        // for debugging
        if (verify_recomp_) {
          for (const auto& it : outputs) {
            assert(contains(last_cp_outputs_[id][split_index], it.first));

            if (it.second.isTensor()) {
              const auto& tgt = last_cp_outputs_[id][split_index].at(it.first);
              assert(tgt.isTensor());
              if (it.second.toTensor().equal(tgt.toTensor())) {
                spdlog::info("recomp matched on {}", toString(it.first));
              } else {
                spdlog::info(
                    "recomp mismatched on {}: A={} B={}", toString(it.first),
                    tensorToString(it.second.toTensor()),
                    tensorToString(tgt.toTensor()));
              }
            }
          }
        }
      }
      setRngState(stashed_rng_state);
    }
    return driver.backward(id, inputs, split_index);
  };

  const auto aggr = [this](
                        std::unordered_map<std::string, IValueMap>& values,
                        std::unordered_map<
                            std::string,
                            std::unordered_map<
                                IValueLocation, std::vector<torch::jit::IValue>,
                                IValueLocationHash>>& recv_values) {
    for (const auto& g_it : this->bwd_inc_edges_count_) {
      const auto& sg_name = g_it.first;
      for (const auto& count_it : g_it.second) {
        const auto& loc = count_it.first;
        const auto& edge_counts = count_it.second;

        if (contains(values, sg_name) && contains(values.at(sg_name), loc)) {
          continue;
        }
        if (recv_values[sg_name][loc].size() == edge_counts) {
          values[sg_name][loc] =
              sumTensorsInIValues(recv_values.at(sg_name).at(loc));
        }
      }
    }
  };

  // Check if a split is skipped at a forward pass
  const auto skip = [this](const IValueMap& values, int split) {
    assert(contains(this->skip_fwd_split_, split));
    return this->skip_fwd_split_.at(split);
  };

  auto values = compute(
      id, true, inputs, split_index, bwd_recv_routes_, bwd_send_routes_,
      bwd_graph_order_, bwd_sorted_graph_ids_, max_bwd_delay_, func, aggr,
      getGradOutputNames, skip);
  logger->trace(
      "GraphConnector::backward finished. id={} split={}", id, split_index);

  recordEnd(getFuncKey("GraphConnector", "backward", id, split_index, false));

  return values;
}

void GraphConnector::enableDropout(const std::string& id, bool enable) {
  driver_.enableDropout(id, enable);
}
} // namespace rannc
