//
// Created by Masahiro Tanaka on 2020/04/11.
//

#include "ProfilerUtil.h"
#include <cuda/CudaSync.h>
#include <cuda/CudaUtil.h>
#include <distop/DistTaskDispatcher.h>
#include <distop/PartitionTensor.h>

namespace rannc {

size_t calcSize(
    const std::shared_ptr<IRGraph>& g,
    const std::function<
        std::vector<std::string>(const std::shared_ptr<IRGraph>&)>& f) {
  size_t sum = 0;
  for (const auto& in_name : f(g)) {
    const IRValue& val = g->getValue(in_name);
    if (!val.isParam()) {
      sum += val.getSizeInByte();
    }
  }
  return sum;
}

size_t calcInputSize(const std::shared_ptr<IRGraph>& g) {
  return calcSize(
      g, [](const std::shared_ptr<IRGraph>& g) { return g->getInputNames(); });
}

size_t calcOutputSize(const std::shared_ptr<IRGraph>& g) {
  return calcSize(
      g, [](const std::shared_ptr<IRGraph>& g) { return g->getOutputNames(); });
}

long calcCommTime(long cut_size) {
  // profiling results are in micro sec
  return cut_size * 1e6 / (double)(20 * 1024L * 1024L * 1024L);
}

long calcInputCommTime(const std::shared_ptr<IRGraph>& g, int repl) {
  return calcCommTime(calcInputSize(g) / repl);
}

long calcOutputCommTime(const std::shared_ptr<IRGraph>& g, int repl) {
  return calcCommTime(calcOutputSize(g) / repl);
}

long calcAllReduceTime(long size) {
  // profiling results are in micro sec
  return size * 1e6 / (double)(10 * 1024L * 1024L * 1024L);
}

size_t getOptMemSize(
    const std::shared_ptr<IRGraph>& ir_graph, const ProfilingInput& prof_in) {
  assert(contains(prof_in.replica_nums, ir_graph->getName()));
  int replica_num = prof_in.replica_nums.at(ir_graph->getName());
  int zero_dist_num = prof_in.enable_zero ? replica_num : 1;

  // This does not need to consider batch size
  size_t sum = 0;
  assert(contains(prof_in.part_info, ir_graph->getName()));
  const auto sliced_params =
      key_set(prof_in.part_info.at(ir_graph->getName()).param_partitions);
  for (const auto& v : ir_graph->getValues()) {
    const auto& val = v.second;
    if (val.isParam()) {
      int slice_num = contains(sliced_params, val.getName()) ? replica_num : 1;

      if (prof_in.use_amp_master_params) {
        if (val.getType().getTensorElemType() == IRTensorElemType::HALF) {
          sum += val.getSizeInByte() // amp holds params
              * 2 // FP32
              / zero_dist_num // Each rank holds only fragments of FP32 master
                              // params
              / slice_num;
          sum += val.getSizeInByte() // amp holds grads
              * 2 // FP32
              / zero_dist_num / slice_num;
          sum += val.getSizeInByte() // optimizer state
              * 2 // FP32
              * prof_in.opt_param_factor / zero_dist_num / slice_num;
        } else if (
            val.getType().getTensorElemType() == IRTensorElemType::FLOAT ||
            val.getType().getTensorElemType() == IRTensorElemType::BFLOAT16) {
          // we have to keep memory for stashed gradients
          sum += val.getSizeInByte() * prof_in.opt_param_factor /
                  zero_dist_num // optimizer state
                  / slice_num +
              val.getSizeInByte() / slice_num; // stashed gradients
        } else {
          spdlog::trace(
              "Unexpected param type. skipping counting size. val={} type={}",
              val.getName(), toString(val.getType().getTensorElemType()));
        }
      } else {
        sum += val.getSizeInByte() * prof_in.opt_param_factor /
            zero_dist_num // optimizer state
            / slice_num;
      }
    }
  }
  return sum;
}

size_t getAmpMasterParamSize(const std::shared_ptr<IRGraph>& ir_graph) {
  size_t sum = 0;
  for (const auto& v : ir_graph->getValues()) {
    const auto& val = v.second;
    if (val.isParam()) {
      if (val.getType().getTensorElemType() == IRTensorElemType::HALF) {
        sum += val.getSizeInByte() * 2; // FP32
      }
    }
  }
  return sum;
}

size_t calcGraphMem(
    const std::shared_ptr<IRGraph>& g, const GraphProfile& prof,
    const ProfilingInput& prof_in) {
  size_t opt_mem = getOptMemSize(g, prof_in);

  return prof.max_allocated_mem + opt_mem;
}

size_t calcGraphMem(
    const std::shared_ptr<IRGraph>& g, const GraphProfile& prof,
    size_t batch_size, ProfilingInput in) {
  assert(contains(in.replica_nums, g->getName()));
  int replica_num = in.replica_nums.at(g->getName());

  size_t bs = ceil(batch_size / (double)(replica_num * in.pipeline_num));
  auto scaled = std::make_shared<IRGraph>("scaled", *g);
  scaled->setBatchSize(bs);

  return calcGraphMem(g, prof, in) + calcCommBufSize(scaled);
}

GraphProfile makeErrorProfile() {
  GraphProfile p;
  p.fwd_time = ProfilerUtil::ERROR_VAL;
  p.bwd_time = ProfilerUtil::ERROR_VAL;
  p.max_allocated_mem = ProfilerUtil::ERROR_VAL;

  return p;
}

GraphProfile ProfilerUtil::profile(const ProfilingInput& in) {
  if (in.force_dist_matmul) {
    bool dist = true;
    for (const auto& part_info : in.part_info) {
      if (!part_info.second.valid()) {
        dist = false;
        break;
      }
    }
    if (dist) {
      return profileDist(in);
    }
  }
  return doProfile(in, [this](const ProfilingInput& input) {
    return this->profiler_->profile(input);
  });
}

GraphProfile ProfilerUtil::profileDist(const ProfilingInput& in) {
  return doProfile(in, [this](const ProfilingInput& input) {
    IValueMap in_values;
    const IValueMap& avail_vals = this->profiler_->getValues();

    for (const auto& it : input.ir_graphs) {
      for (const auto& in_name : getNonParamInputNames(it.second)) {
        assert(contains(avail_vals, in_name));
        in_values[in_name] = toCUDAIfAvailable(avail_vals.at(in_name), true);
      }
    }

    DistTaskDispatcher& dtd = DistTaskDispatcher::get();
    return dtd.profile(input, in_values);
  });
}

GraphProfile ProfilerUtil::doProfile(
    const ProfilingInput& in,
    const std::function<ProfilingResult(const ProfilingInput& input)>& f) {
  assert(in.pipeline_num > 0);
  assert(in.ir_graphs.size() == 1);

  const std::shared_ptr<IRGraph>& g = (*in.ir_graphs.begin()).second;
  assert(g);

  assert(contains(in.replica_nums, g->getName()));
  int replica_num = in.replica_nums.at(g->getName());
  assert(replica_num > 0);

  size_t bs = ceil(in.batch_size / (double)(replica_num * in.pipeline_num));

  const MLProfileKey k{g->getName(), bs, in.checkpointing};
  if (contains(profile_cache_, k)) {
    return profile_cache_.at(k);
  }

  if (!contains(max_batch_size_cache_[in.checkpointing], g->getName())) {
    max_batch_size_cache_[in.checkpointing][g->getName()] = SIZE_MAX;
  }

  size_t max_bs = max_batch_size_cache_[in.checkpointing][g->getName()];
  if (max_bs < bs) {
    profile_cache_[k] = makeErrorProfile();
    return profile_cache_.at(k);
  }

  std::unordered_map<std::string, std::shared_ptr<IRGraph>> prof_in_v;
  prof_in_v[g->getName()] = g;

  try {
    ProfilingResult prof_v = f(in);
    assert(prof_v.node_profiles.size() == 1);
    profile_cache_[k] = prof_v.node_profiles.begin()->second;
  } catch (std::exception& e) {
    std::string msg = e.what();
    std::string::size_type pos1 = msg.find("CUDA out of memory");
    std::string::size_type pos2 = msg.find("Too many elements");
    if (pos1 == std::string::npos && pos2 == std::string::npos) {
      spdlog::error(
          "Failed to profile graph: {} batch_size={} replica_num={} pipeline_num={} {}",
          g->getName(), in.batch_size, replica_num, in.pipeline_num, e.what());
      throw std::runtime_error("Failed to profile graph: " + toString(*g));
    } else {
      profile_cache_[k] = makeErrorProfile();

      if (max_batch_size_cache_[in.checkpointing][g->getName()] >= bs) {
        max_batch_size_cache_[in.checkpointing][g->getName()] = bs - 1;
      }
      profiler_->clear();
      emptyCache();
      syncWithErrorCheck();
    }
  }
  return profile_cache_.at(k);
}

void ProfilerUtil::clearCache() {
  profile_cache_.clear();
}

GraphProfile accProfileValues(
    ProfilerUtil& prof_util, const ProfilingInput& prof_in) {
  GraphProfile prof_sum{"MERGED", 0, 0, 0};

  int idx = 0;
  for (const auto& it : prof_in.ir_graphs) {
    auto graph = it.second;
    assert(contains(prof_in.part_info, graph->getName()));

    const auto prof = prof_util.profile(
        {{{graph->getName(), graph}},
         prof_in.batch_size,
         prof_in.iteration,
         prof_in.replica_nums,
         prof_in.pipeline_num,
         prof_in.checkpointing,
         prof_in.opt_param_factor,
         prof_in.use_amp_master_params,
         prof_in.enable_zero,
         prof_in.offload_params,
         prof_in.force_dist_matmul,
         {{graph->getName(), prof_in.part_info.at(graph->getName())}}});

    prof_sum.fwd_time += prof.fwd_time;
    prof_sum.bwd_time += prof.bwd_time;
    prof_sum.param_size += prof.param_size;
    prof_sum.input_size += 0;
    //    prof_sum.output_size += 0;
    prof_sum.output_size += prof.output_size;
    prof_sum.activation_size += prof.activation_size;
    prof_sum.working_mem = std::max(prof.working_mem, prof_sum.working_mem);

    //    spdlog::info("accProfileValues idx={} alloc={} param+grad={} in={}
    //    out={} act={} work={}",
    //                 idx++,
    //                 prof.max_allocated_mem,
    //                 prof.param_size*2,
    //                 prof.input_size,
    //                 prof.output_size,
    //                 prof.activation_size,
    //                 prof.working_mem);
  }

  prof_sum.max_allocated_mem =
      prof_sum.param_size * 2 + prof_sum.activation_size + prof_sum.working_mem;

  return prof_sum;
}

std::string displayGraphProfiles(
    const ProfilingInput& prof_inputs,
    const std::unordered_map<std::string, GraphProfile>& profiles) {
  std::stringstream ss;

  int dev_num = 0;
  for (const auto& it : prof_inputs.replica_nums) {
    dev_num += it.second;
  }

  ss << "Estimated profiles of subgraphs: batch_size=" << prof_inputs.batch_size
     << " np=" << dev_num << " pipeline=" << prof_inputs.pipeline_num
     << " use_amp=" << prof_inputs.use_amp_master_params
     << " zero=" << prof_inputs.enable_zero << std::endl;

  size_t idx = 0;
  for (const auto& it : prof_inputs.ir_graphs) {
    const auto& name = it.first;
    const auto& g = it.second;
    assert(contains(prof_inputs.replica_nums, name));
    assert(contains(profiles, name));

    int repl_num = prof_inputs.replica_nums.at(name);
    const auto& prof = profiles.at(name);

    size_t opt_mem = getOptMemSize(g, prof_inputs);

    size_t bs = ceil(
        prof_inputs.batch_size / (double)(repl_num * prof_inputs.pipeline_num));
    auto scaled = std::make_shared<IRGraph>("scaled", *g);
    scaled->setBatchSize(bs);
    size_t comm_buf = calcCommBufSize(scaled);

    long ar_time = calcAllReduceTime(g->getParamSizeInByte());

    size_t total = prof.max_allocated_mem + opt_mem + comm_buf;

    size_t fp32params = 0;
    size_t fp16params = 0;
    for (const auto& in_name : g->getInputNames()) {
      const auto& in_v = g->getValue(in_name);
      if (in_v.isParam()) {
        assert(in_v.getType().getBaseType() == IRBaseType::TENSOR);
        if (in_v.getType().getTensorElemType() == IRTensorElemType::FLOAT) {
          fp32params += in_v.getSizeInByte();
        } else if (
            in_v.getType().getTensorElemType() == IRTensorElemType::HALF ||
            in_v.getType().getTensorElemType() == IRTensorElemType::BFLOAT16) {
          fp16params += in_v.getSizeInByte();
        } else {
          spdlog::debug(
              "Unknown elem type of parameter {}: {}", in_name,
              toString(in_v.getType().getTensorElemType()));
        }
      }
    }

    ss << "  graph=" << g->getName() << " repl=" << repl_num
       << " fwd_time=" << prof.fwd_time << " bwd_time=" << prof.bwd_time
       << " ar_time=" << ar_time << " in_size=" << calcInputSize(scaled)
       << " out_size=" << calcOutputSize(scaled)
       << " fp32param_size=" << fp32params << " fp16param_size=" << fp16params
       << " total_mem=" << total << " (fwd+bwd=" << prof.max_allocated_mem
       << " opt=" << opt_mem << " comm=" << comm_buf << ")";

    if (idx < prof_inputs.ir_graphs.size() - 1) {
      ss << std::endl;
    }
    idx++;
  }

  return ss.str();
}
} // namespace rannc
