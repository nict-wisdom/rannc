//
// Created by Masahiro Tanaka on 2019/12/20.
//

#include <regex>

#include <graph/GuessValueTypes.h>
#include <torch/TorchEngine.h>
#include <torch/TorchUtil.h>
#include "comp/ParamStorage.h"
#include "ConfiguredTorch.h"
#include "graph/ConvertGraph.h"
#include "GraphProfiler.h"

namespace rannc {

std::string profItemToStr(const ProfileItemKey& prof_key) {
  std::vector<std::string> graph_str_vec;
  graph_str_vec.reserve(prof_key.ir_graphs.size());
  for (const auto& it : prof_key.ir_graphs) {
    IRGraph renamed("PROF_GRAPH", *it.second);
    graph_str_vec.push_back(toString(renamed));
  }
  std::sort(graph_str_vec.begin(), graph_str_vec.end());

  std::stringstream ss;
  ss << "graphs=" << join_as_str(graph_str_vec) << "_bs=" << prof_key.batch_size
     << "_repl_num=" << prof_key.repl_num << "_iteration=" << prof_key.iteration
     << "_checkpointing=" << prof_key.checkpointing;

  return ss.str();
}

size_t getKey(const ProfileItemKey& prof_key) {
  return std::hash<std::string>()(profItemToStr(prof_key));
}

bool ProfileDB::hasRecord(const ProfileItemKey& prof_key) {
  return contains(items_, getKey(prof_key));
}

void ProfileDB::add(const ProfileItem& prof_item) {
  const auto& key = prof_item.key;
  if (hasRecord(key)) {
    return;
  }

  items_[getKey(key)] = prof_item;
}

ProfilingResult ProfileDB::get(const ProfileItemKey& prof_key) {
  assert(hasRecord(prof_key));
  return items_.at(getKey(prof_key)).profile;
}

const std::unordered_map<size_t, ProfileItem>& ProfileDB::getItems() const {
  return items_;
}

std::string getFwdTimeKey(const std::string& id) {
  std::stringstream time_key;
  time_key << "fwd_time_" << id;
  return time_key.str();
}

std::string getBwdTimeKey(const std::string& id) {
  std::stringstream time_key;
  time_key << "bwd_time_" << id;
  return time_key.str();
}

std::shared_ptr<IRGraph> setInputTypes(
    const std::shared_ptr<IRGraph>& g,
    const std::unordered_map<std::string, IRType>& value_types) {
  std::unordered_map<std::string, IRValue> values = g->getValues();
  for (auto& in_name : g->getInputNames()) {
    IRValue v = g->getValue(in_name);
    if (v.isParam()) {
      continue;
    }
    if (v.isFunction()) {
      continue;
    }
    if (!contains(value_types, v.getName())) {
      throw std::invalid_argument(
          "No type information in profiling results: " + v.getName());
    }
    v.setType(value_types.at(v.getName()));
    values[in_name] = v;
  }

  return std::make_shared<IRGraph>(
      g->getName(), g->getNodes(), values, g->getInputNames(),
      g->getOutputNames());
}

std::unordered_map<std::string, at::Tensor> paramsToCuda(
    const std::unordered_map<std::string, at::Tensor>& params) {
  std::unordered_map<std::string, at::Tensor> cuda_params;
  for (const auto& it : params) {
    if (it.second.is_cuda()) {
      cuda_params[it.first] = it.second;
    } else {
      cuda_params[it.first] = toCUDAIfAvailable(it.second, true);
    }
  }
  return cuda_params;
}

std::vector<IRValue> getParamValues(const std::shared_ptr<IRGraph>& graph) {
  std::vector<IRValue> results;
  for (const auto& in : graph->getInputNames()) {
    const auto& v = graph->getValue(in);
    if (v.isParam()) {
      results.push_back(v);
    }
  }
  return results;
}

void GraphValueCache::put(
    const std::string& name, const torch::jit::IValue& value) {}

torch::jit::IValue GraphValueCache::get(
    const std::string& name, size_t batch_size) {}

static at::Dimname dimnameFromString(const std::string& str) {
  return at::Dimname::fromSymbol(at::Symbol::dimname(str));
}

static std::vector<at::Dimname> dimnamesFromStringVector(
    const std::vector<std::string>& names) {
  std::vector<at::Dimname> dim_names;
  for (const auto& n : names) {
    dim_names.push_back(dimnameFromString(n));
  }
  return dim_names;
}

std::vector<at::Dimname> createDimnames(
    const std::string& val_name, const at::Tensor& ten, bool batch) {
  std::vector<std::string> dim_names;
  size_t idx = 0;
  for (const auto& n : ten.names()) {
    if (batch && idx == 0) {
      dim_names.emplace_back("N");
    } else {
      std::string name =
          std::regex_replace(val_name, std::regex(R"([_:\[\]\.]+)"), "_");
      name = std::regex_replace(name, std::regex("_+"), "_");

      std::stringstream ss;
      ss << name << "_D" << idx;
      dim_names.push_back(ss.str());
    }
    idx++;
  }

  return dimnamesFromStringVector(dim_names);
}

at::Tensor setDimNamesOnTensor(
    const at::Tensor& t, const IValueLocation& loc, bool batch,
    std::unordered_map<
        IValueLocation, std::vector<at::Dimname>, IValueLocationHash>&
        dim_names) {
  const auto new_names = createDimnames(toString(loc), t, batch);
  std::vector<at::Dimname> merged_names;
  if (contains(dim_names, loc)) {
    const auto& current_names = dim_names.at(loc);
    for (size_t idx = 0; idx < new_names.size(); idx++) {
      if (idx < current_names.size() && !current_names.at(idx).isWildcard()) {
        merged_names.push_back(current_names.at(idx));
      } else {
        merged_names.push_back(new_names.at(idx));
      }
    }
  } else {
    merged_names = new_names;
  }
  dim_names[loc] = merged_names;

  bool requires_grad = t.requires_grad();
  at::Tensor ret = t.rename(merged_names).detach();
  ret.set_requires_grad(requires_grad);

  return ret;
}

torch::jit::IValue setDimNamesOnIValue(
    const torch::jit::IValue& ivalue, const std::string& name, bool batch,
    std::unordered_map<
        IValueLocation, std::vector<at::Dimname>, IValueLocationHash>&
        dim_names) {
  return transformTensorsInIValueWithPath(
      ivalue, name,
      [&dim_names, batch](const at::Tensor& t, const IValueLocation& loc) {
        return setDimNamesOnTensor(t, loc, batch, dim_names);
      });
}

torch::jit::IValue resetDimNamesOnIValue(const torch::jit::IValue& ivalue) {
  return transformTensorsInIValue(ivalue, [](const at::Tensor& t) {
    bool requires_grad = t.requires_grad();
    return t.rename({}).detach().set_requires_grad(requires_grad);
  });
}

void recordDimNamesFromIValue(
    const torch::jit::IValue& ivalue, const std::string& name,
    std::unordered_map<
        IValueLocation, std::vector<at::Dimname>, IValueLocationHash>&
        dim_names) {
  transformTensorsInIValueWithPath(
      ivalue, name,
      [&dim_names](const at::Tensor& t, const IValueLocation& loc) {
        std::vector<at::Dimname> dims;
        for (const auto& d : t.names()) {
          dims.push_back(d);
        }
        dim_names[loc] = dims;
        return t;
      });
  return;
}

void recordDimNamesFromIValue(
    const IValueMap& values,
    std::unordered_map<
        IValueLocation, std::vector<at::Dimname>, IValueLocationHash>&
        dim_names) {
  for (const auto& it : values) {
    recordDimNamesFromIValue(it.second, it.first.value_name, dim_names);
  }
}

std::pair<IValueMap, GraphProfile> GraphProfiler::computeGraph(
    const std::shared_ptr<IRGraph>& subgraph,
    const IValueMap& graph_inputs_with_names,
    const std::unordered_map<std::string, at::Tensor>& graph_params,
    int iteration, IValueMap& values, int split_index, bool checkpointing,
    bool trace_dim_names) {
  emptyCache();

  ProfilingResult ret_profiles;

  const std::string& id = subgraph->getName();
  std::string fwd_time_key = getFwdTimeKey(id);
  std::string bwd_time_key = getBwdTimeKey(id);

  driver_.createModule(id, id, subgraph, constants_, functions_, graph_params);

  // clear grads
  long param_size = 0;
  for (auto& param_it : getGraphParams(subgraph)) {
    auto& param = param_it.second;
    param_size += param.numel() * param.element_size();
    getMutableGradRef(param) = at::Tensor();
  }

  long alloc_mem_start = getAllocatedMemory();
  resetMaxAllocatedMemory();
  IValueMap graph_inputs = graph_inputs_with_names;
  IValueMap driver_out;
  TimeCounter time_counter(true);
  for (size_t i = 0; i < iteration; i++) {
    driver_out.clear();

    auto& driver = driver_;
    if (checkpointing) {
      // Run forward enabling grad to judge whether bwd is required or not
      driver_out = driver.forward(id, graph_inputs, split_index);
      bool run_bwd = setRequiresGrad(subgraph, driver_out) > 0;

      // Clear to release memory
      driver_out.clear();
      {
        torch::NoGradGuard no_grad;
        driver_out = measureTime<IValueMap>(
            [&driver, &id, &graph_inputs, split_index]() {
              return driver.forward(id, graph_inputs, split_index);
            },
            i, iteration / 2, time_counter, fwd_time_key);
      }

      if (run_bwd) {
        driver_out = measureTime<IValueMap>(
            [&driver, &id, &graph_inputs, this, &subgraph, split_index]() {
              const auto out = driver.forward(id, graph_inputs, split_index);
              setRequiresGrad(subgraph, out);
              this->backward(subgraph, out, split_index);
              return out;
            },
            i, iteration / 2, time_counter, bwd_time_key);
      }
    } else {
      driver_out = measureTime<IValueMap>(
          [&driver, &id, &graph_inputs, split_index]() {
            return driver.forward(id, graph_inputs, split_index);
          },
          i, iteration / 2, time_counter, fwd_time_key);
      if (trace_dim_names) {
        recordDimNamesFromIValue(driver_out, dim_names_);
        for (auto& it : graph_inputs) {
          graph_inputs[it.first] = resetDimNamesOnIValue(it.second);
        }
        for (auto& it : driver_out) {
          driver_out[it.first] = resetDimNamesOnIValue(it.second);
        }
      }

      if (setRequiresGrad(subgraph, driver_out) > 0) {
        measureTime(
            [this, &subgraph, &driver_out, split_index]() {
              this->backward(subgraph, driver_out, split_index);
            },
            i, iteration / 2, time_counter, bwd_time_key);
      }
    }
  }

  driver_.destroyModule(id);

  long bwd_time = time_counter.hasRecord(bwd_time_key)
      ? time_counter.get<std::chrono::microseconds>(bwd_time_key)
      : 0;

  // Add param size because params remain on a cuda device
  long total_mem = getMaxAllocatedMemory() - alloc_mem_start + param_size;

  GraphProfile prof{
      id, time_counter.get<std::chrono::microseconds>(fwd_time_key), bwd_time,
      total_mem, checkpointing};

  return {driver_out, prof};
}

ProfilingResult GraphProfiler::compute(
    const ProfilingInput& input, IValueMap& values, int split_index,
    bool trace_dim_names) {
  TimeCounter time_counter(true);
  ProfilingResult ret_profiles;
  std::unordered_set<std::string> graphs_done;

  std::unordered_set<IValueLocation, IValueLocationHash> avail_locs;
  for (const auto& it : values) {
    avail_locs.insert(it.first);
  }

  size_t prev_done = 0;
  size_t prev_locs = 0;
  size_t infinite = 0;

  const auto& ir_graphs = input.ir_graphs;
  while (graphs_done.size() < ir_graphs.size()) {
    for (const auto& it : ir_graphs) {
      const auto& id = it.first;
      const auto& untyped_subgraph = it.second;

      if (contains(graphs_done, id))
        continue;

      const auto input_names_org = getNonParamInputNames(untyped_subgraph);

      //  Remove Function(s) from input_names
      std::vector<std::string> input_names;
      input_names.reserve(input_names_org.size());
      for (const std::string& in : input_names_org) {
        if (untyped_subgraph->getValue(in).isFunction()) {
          continue;
        }
        input_names.push_back(in);
      }

      bool graph_ready = isGraphReady(input_names, avail_locs);

      //      spdlog::info("avail_locs num={}", avail_locs.size());
      //      for (const auto& it: avail_locs) {
      //        spdlog::info("loc={}", toString(it));
      //      }

      if (graph_ready) {
        logger->debug(
            "GraphProfiler::compute starting graph {} ({}/{})", id,
            graphs_done.size() + 1, ir_graphs.size());

        emptyCache();

        IValueMap graph_inputs;
        size_t input_size = 0;
        for (const auto& in_name : input_names) {
          assert(contains(values, in_name));

          graph_inputs[in_name] =
              contiguous(toCUDAIfAvailable(values.at(in_name), true));

          if (trace_dim_names) {
            graph_inputs[in_name] = setDimNamesOnIValue(
                graph_inputs[in_name], in_name, false, dim_names_);
          }

          ret_profiles.value_types[in_name] = toIRType(graph_inputs[in_name]);
          input_size += ret_profiles.value_types[in_name].getSizeInByte();
        }
        for (const auto& it : getGraphParams(untyped_subgraph)) {
          ret_profiles.value_types[it.first] = toIRType(it.second);
        }

        const auto& subgraph =
            setInputTypes(untyped_subgraph, ret_profiles.value_types);
        auto graph_params = paramsToCuda(getGraphParams(subgraph));

        if (trace_dim_names) {
          for (const auto& it : graph_params) {
            graph_params[it.first] =
                setDimNamesOnTensor(it.second, it.first, false, dim_names_);
          }
        }

        bool dimnames_set = false;
        std::pair<IValueMap, GraphProfile> prof_results;
        try {
          prof_results = computeGraph(
              subgraph, graph_inputs, graph_params, input.iteration, values,
              split_index, input.checkpointing, trace_dim_names);
          dimnames_set = true;
        } catch (std::runtime_error& e) {
          driver_.destroyModule(subgraph->getName());

          std::string msg = e.what();
          std::string::size_type pos1 =
              msg.find("not yet supported with named tensors");
          std::string::size_type pos2 =
              msg.find("Error when attempting to broadcast dims");

          if (pos1 == std::string::npos && pos2 == std::string::npos) {
            throw e;
          }

          if (pos2 != std::string::npos) {
            dimnames_set = true;
          }

          // Remove dim names if unsupported.
          size_t input_idx = 0;
          for (const auto& in_name : getNonParamInputNames(subgraph)) {
            if (pos2 == std::string::npos || input_idx != 0) {
              graph_inputs[in_name] =
                  resetDimNamesOnIValue(graph_inputs[in_name]);
            }
            input_idx++;
          }
          for (const auto& it : graph_params) {
            graph_params[it.first] =
                resetDimNamesOnIValue(it.second).toTensor();
          }

          // Try again without dim names
          prof_results = computeGraph(
              subgraph, graph_inputs, graph_params, input.iteration, values,
              split_index, input.checkpointing, trace_dim_names);

          // Set dim names again on params
          for (const auto& it : graph_params) {
            graph_params[it.first] =
                setDimNamesOnTensor(it.second, it.first, false, dim_names_);
          }
        }
        IValueMap& driver_out = prof_results.first;

        size_t output_size = 0;
        for (const auto& it_out : driver_out) {
          avail_locs.insert(it_out.first);

          torch::jit::IValue out_val;
          if (it_out.second.isList()) {
            out_val = toTensorListIfElemsAreTensors(it_out.second);
          } else {
            out_val = it_out.second;
          }
          if (trace_dim_names) {
            out_val = resetDimNamesOnIValue(out_val);
          }
          values[it_out.first] = toCPU(out_val, true);
          ret_profiles.value_types[it_out.first.value_name] = toIRType(out_val);

          output_size +=
              ret_profiles.value_types[it_out.first.value_name].getSizeInByte();
        }

        ret_profiles.node_profiles[id] = prof_results.second;

        driver_out.clear();

        // clear grad again
        for (auto& param_it : getGraphParams(subgraph)) {
          auto& param = param_it.second;
          getMutableGradRef(param) = at::Tensor();
        }

        emptyCache();

        graphs_done.insert(id);

        logger->trace("GraphProfiler::compute finished graph {}", id);
      } //  End if (graph_ready)
    } //  End for (it_graphs)

    //  Check the sizes of graphs_done and avail_locs.
    //  Because if the above for block did not change the sizes,
    //  it may be bug and causes infinite-loops.
    if ((graphs_done.size() == prev_done) && (avail_locs.size() == prev_locs)) {
      if (infinite > ir_graphs.size() + 1) {
        throw std::runtime_error(
            "The variable(s) was not updated in while loop. "
            "It (may) causes infinite-loops. "
            "This may be BUG, please report to developer.");
      }
      logger->debug(
          "The variable(s) was not updated in while loop. "
          "It (may) causes infinite-loops. "
          "This may be BUG, please report to developer.");
      logger->debug(
          "graphs_done = {} (prev = {}), "
          "avail_locs  = {} (prev = {}), checks={}/{}",
          graphs_done.size(), prev_done, avail_locs.size(), prev_locs, infinite,
          ir_graphs.size() + 1);
      ++infinite;
    }
    prev_done = graphs_done.size();
    prev_locs = avail_locs.size();
  } //  End while

  return ret_profiles;
}

size_t GraphProfiler::setRequiresGrad(
    const std::shared_ptr<IRGraph>& ir_graph, const IValueMap& outputs) {
  IValueMap ret;
  size_t out_idx = 0;
  size_t req_grad_count = 0;
  for (auto& it : outputs) {
    auto& out = it.second;

    const auto& paths = findPathsToTensorInIValue(out);
    for (const auto& path : paths) {
      auto elem = getElemInIValue(out, path);

      assert(elem.isTensor());
      auto out_t = elem.toTensor();

      bool requires_grad1 = false;
      if (torch::autograd::impl::get_autograd_meta(out_t)) {
        requires_grad1 =
            (bool)torch::autograd::impl::get_autograd_meta(out_t)->grad_fn_;
      }
      bool requires_grad2 = out_t.requires_grad();
      bool requires_grad = requires_grad1 || requires_grad2;

      out_t.set_requires_grad(requires_grad);

      if (requires_grad) {
        req_grad_count++;
      }
    }
    out_idx++;
  }
  return req_grad_count;
}

void GraphProfiler::backward(
    const std::shared_ptr<IRGraph>& ir_graph, const IValueMap& outputs,
    int split_idx) {
  IValueMap bwd_input;
  for (const auto& out_name : ir_graph->getOutputNames()) {
    auto& out = outputs.at(out_name);

    const auto& paths = findPathsToTensorInIValue(out);
    for (const auto& path : paths) {
      auto elem = getElemInIValue(out, path);
      assert(elem.isTensor());
      auto out_t = elem.toTensor();

      if (out_t.scalar_type() == c10::ScalarType::Float ||
          out_t.scalar_type() == c10::ScalarType::Half ||
          out_t.scalar_type() == c10::ScalarType::BFloat16 ||
          out_t.scalar_type() == c10::ScalarType::Double) {
        IValueLocation loc(out_name, path);
        auto target = toCUDAIfAvailable(torch::randn_like(out_t), true);
        bwd_input[loc] = target;
      }
    }
  }

  driver_.backward(ir_graph->getName(), bwd_input, split_idx);
}

std::unordered_map<std::string, at::Tensor> GraphProfiler::getGraphParams(
    const std::shared_ptr<IRGraph>& graph) {
  std::unordered_map<std::string, at::Tensor> graph_param_tensors;
  auto ir_params = graphParamValues(graph);
  for (const auto& irp : ir_params) {
    assert(contains(graph_params_, irp.getName()));

    if (cache_param_values_) {
      if (contains(param_cache_, irp.getName())) {
        graph_param_tensors[irp.getName()] = param_cache_.at(irp.getName());
      } else {
        logger->debug("Fetching param: {}", irp.getName());
        const auto param_tensor =
            param_storage_->getParamTensor(graph_params_.at(irp.getName()));
        graph_param_tensors[irp.getName()] = param_cache_[irp.getName()] =
            param_tensor;
      }
    } else {
      graph_param_tensors[irp.getName()] =
          param_storage_->getParamTensor(graph_params_.at(irp.getName()));
    }
  }
  return graph_param_tensors;
}

ProfilingResult GraphProfiler::doProfile(
    const ProfilingInput& input, IValueMap& values, bool trace_dim_names) {
  const auto& ir_graphs = input.ir_graphs;
  for (const auto& it : ir_graphs) {
    for (const auto& in_name : it.second->getInputNames()) {
      IValueLocation loc{in_name};
      if (contains(values_, loc)) {
        const auto pad_in = splitBatchInIValue(
            values_.at(loc), 1, batch_size_, input.replica_num, false);

        const auto& paths = findPathsToTensorInIValue(pad_in);
        for (const auto& path : paths) {
          auto elem = getElemInIValue(pad_in, path);
          assert(elem.isTensor());
          if (elem.toTensor().numel() > 1e9) {
            std::stringstream ss;
            ss << "Too many elements in tensor: " << in_name << " "
               << toString(path) << " " << toString(toIRType(pad_in))
               << " #elements=" << elem.toTensor().numel();

            throw std::runtime_error(ss.str());
          }
        }

        values[in_name] = toCPU(pad_in, true);
      }
    }
  }

  logger->trace(
      "GraphProfiler::profile starting doCompute replica_num={}",
      input.replica_num);
  ProfilingResult ret_profiles = compute(input, values, 0, trace_dim_names);
  logger->trace("GraphProfiler::profile finished doCompute");

  driver_.destroy();
  emptyCache();
  resetMaxAllocatedMemory();

  logger->debug("GraphProfiler::profile finished");

  return ret_profiles;
}

ProfilingResult GraphProfiler::profile(const ProfilingInput& input) {
  const auto& ir_graphs = input.ir_graphs;

  ProfileItemKey key{
      ir_graphs, batch_size_, input.replica_num, input.iteration,
      input.checkpointing};
  if (profile_db_.hasRecord(key)) {
    logger->trace(
        "Cache hit {} bs={} repl={} iter={} cp={}",
        join_as_str(keys(ir_graphs)), batch_size_, input.replica_num,
        input.iteration, input.checkpointing);
    return profile_db_.get(key);
  } else {
    logger->trace(
        "Cache NOT hit {} bs={} repl={} iter={} cp={}",
        join_as_str(keys(ir_graphs)), batch_size_, input.replica_num,
        input.iteration, input.checkpointing);
  }

  IValueMap values; // temporal value
  auto ret_profiles = doProfile(input, values, false);

  ProfileItem prof_item{key, ret_profiles};
  profile_db_.add(prof_item);

  return ret_profiles;
}

ProfilingResult GraphProfiler::profile(
    const ProfilingInput& input, IValueMap values) {
  auto ret_profiles = doProfile(input, values, false);
  return ret_profiles;
}

ProfilingResult GraphProfiler::init(bool trace_dim_names) {
  for (const auto& it : non_param_inputs_) {
    IValueLocation loc{it.first};
    if (!contains(values_, loc)) {
      values_[it.first] =
          alignTensorsInIValue(toCPU(it.second, true), 1, false);
    }
  }

  const auto graph_params = getGraphParams(base_graph_);
  if (trace_dim_names) {
    for (const auto& p_it : graph_params) {
      const auto dim_names = createDimnames(p_it.first, p_it.second, false);
      dim_names_[p_it.first] = dim_names;
    }
  }

  std::unordered_map<std::string, std::shared_ptr<IRGraph>> graphs;
  std::unordered_map<std::string, IRNode> node_map;
  const auto& nodes = base_graph_->getNodes();
  for (const auto& n : nodes) {
    std::vector<IRNode> pf_nodes = {n};
    std::unordered_map<std::string, IRValue> pf_values;
    for (const auto& in : n.getInputNames()) {
      pf_values[in] = base_graph_->getValue(in);
    }
    for (const auto& out : n.getOutputNames()) {
      pf_values[out] = base_graph_->getValue(out);
    }

    std::vector<std::string> input_names;
    for (const auto& in : n.getInputNames()) {
      const auto& v = base_graph_->getValue(in);
      if (!v.isParam()) {
        input_names.push_back(v.getName());
      }
    }
    for (const auto& in : n.getInputNames()) {
      const auto& v = base_graph_->getValue(in);
      if (v.isParam()) {
        input_names.push_back(v.getName());
      }
    }

    const auto& id = n.getId();
    graphs[id] = std::make_shared<IRGraph>(
        id, pf_nodes, pf_values, input_names, n.getOutputNames());
    node_map[id] = n;
  }

  IValueMap values; // temporal value
  size_t replica_num = dev_num_ * min_pipeline_num_;
  bool checkpointing = false;

  ProfilingInput input{graphs, 1, replica_num, checkpointing};
  auto ret = doProfile(input, values, trace_dim_names);

  // set types of unused values
  for (const auto& np : non_param_inputs_) {
    if (!contains(ret.value_types, np.first)) {
      ret.value_types[np.first] = toIRType(np.second);
    }
  }

  for (const auto& v : getParamValues(base_graph_)) {
    if (!contains(ret.value_types, v.getName())) {
      assert(contains(graph_params_, v.getName()));
      long pid = graph_params_.at(v.getName());
      const auto param_tensor = param_storage_->getParamTensor(pid);
      ret.value_types[v.getName()] = toIRType(param_tensor);
    }
  }

  base_graph_ = setValueTypes(base_graph_, ret.value_types);
  base_graph_ = guessValueTypes(base_graph_);
  base_graph_->setBatchSize(batch_size_);

  std::unordered_map<std::string, std::vector<std::string>> str_dim_names;
  for (const auto& it : dim_names_) {
    for (const auto& d : it.second) {
      str_dim_names[it.first.value_name].push_back(d.symbol().toQualString());
    }
  }
  base_graph_->setDimNames(str_dim_names);

  logger->trace("Scaling {} intermediate values ...", values.size());
  for (auto& it : values) {
    const auto& ir_val = base_graph_->getValue(it.first.value_name);
    if (ir_val.isBatch()) {
      values_[it.first] = scaleBatchInIValue(
          it.second, ceil(batch_size_ / (double)replica_num), 1, false);
      logger->trace(
          "Scaled {}: {} to {}", toString(it.first),
          toString(toIRType(it.second)), toString(toIRType(values_[it.first])));
    } else {
      values_[it.first] = it.second;
    }
  }

  // We run profiling to get values for further profiling of subgraphs,
  // but return cached profiling records.
  // Since the following decomposition steps use the records (e.g. sort by the
  // profiled values), value changes lead to different decompositions and
  // reduces cache hits.
  ProfileItemKey key{graphs, batch_size_, replica_num, 1, checkpointing};
  if (profile_db_.hasRecord(key)) {
    logger->trace(
        "Cache hit on init {} bs={} repl={} iter={} cp={}",
        join_as_str(keys(graphs)), batch_size_, replica_num, 1, checkpointing);
    return profile_db_.get(key);
  }

  ProfileItem item{key, ret};
  profile_db_.add(item);

  return ret;
}

void GraphProfiler::clear() {
  driver_.destroy();

  for (const auto& v : getParamValues(base_graph_)) {
    assert(contains(graph_params_, v.getName()));
    long pid = graph_params_.at(v.getName());

    if (!param_storage_->distributed(pid)) {
      getMutableGradRef(param_storage_->getParamTensor(pid)) = at::Tensor();
    }
  }
}

void GraphProfiler::load(const std::string& file) {
  std::ifstream input(file, std::ios::in | std::ios::binary);
  if (!input) {
    throw std::invalid_argument("Failed to open file: " + file);
  }

  std::vector<char> buffer(std::istreambuf_iterator<char>(input), {});
  std::unordered_map<size_t, ProfileItem> obj =
      deserialize<std::unordered_map<size_t, ProfileItem>>(buffer);

  for (const auto& it : obj) {
    profile_db_.add(it.second);
  }

  logger->debug("Loaded {} profiling item(s)", obj.size());
}

void GraphProfiler::save(const std::string& file) {
  const auto prof_data = serialize(profile_db_.getItems());

  std::ofstream out(file, std::ios::out | std::ios::binary);
  if (!out) {
    throw std::invalid_argument("Failed to open file: " + file);
  }
  out.write(reinterpret_cast<const char*>(&prof_data[0]), prof_data.size());
  out.close();
}

bool GraphProfiler::hasConstant(const IValueLocation& loc) const {
  return contains(constants_, loc);
}

void GraphProfiler::updateConstants(const IValueMap& constants) {
  for (const auto& it : constants) {
    constants_[it.first] = it.second;
  }
}

void GraphProfiler::removeConstant(const IValueLocation& loc) {
  constants_.erase(loc);
}

} // namespace rannc
