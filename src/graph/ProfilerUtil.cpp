//
// Created by Masahiro Tanaka on 2020/04/11.
//

#include "ProfilerUtil.h"
#include <cuda/CudaUtil.h>


namespace rannc {

    size_t calcSize(const std::shared_ptr<IRGraph>& g, const std::function<std::vector<std::string>(const std::shared_ptr<IRGraph>&)>& f) {
        size_t sum = 0;
        for (const auto& in_name: f(g)) {
            const IRValue& val = g->getValue(in_name);
            if (!val.isParam()) {
                sum += val.getSizeInByte();
            }
        }
        return sum;
    }

    size_t calcInputSize(const std::shared_ptr<IRGraph>& g) {
        return calcSize(g, [](const std::shared_ptr<IRGraph>& g) {
            return g->getInputNames();
        });
    }

    size_t calcOutputSize(const std::shared_ptr<IRGraph>& g) {
        return calcSize(g, [](const std::shared_ptr<IRGraph>& g) {
            return g->getOutputNames();
        });
    }

    long calcCommTime(long cut_size) {
        // profiling results are in micro sec
        return cut_size * 1e6 / (double) (20 * 1024L  * 1024L * 1024L);
    }

    long calcInputCommTime(const std::shared_ptr<IRGraph>& g, int repl) {
        return calcCommTime(calcInputSize(g) / repl);
    }

    long calcOutputCommTime(const std::shared_ptr<IRGraph>& g, int repl) {
        return calcCommTime(calcOutputSize(g) / repl);
    }

    long calcAllReduceTime(long size) {
        // profiling results are in micro sec
        return size * 1e6 / (double) (10 * 1024L  * 1024L * 1024L);
    }

    size_t calcGraphMem(const std::shared_ptr<IRGraph>& g, const GraphProfile& prof, bool use_amp_master_params,
                        bool enable_zero, int zero_dist_num) {
        static int opt_param_factor = config::Config::get().getVal<int>(config::OPT_PARAM_FACTOR);
        size_t opt_mem = getOptMemSize(g, opt_param_factor, use_amp_master_params, enable_zero, zero_dist_num);

        return prof.max_allocated_mem + opt_mem;
    }

    size_t calcGraphMem(const std::shared_ptr<IRGraph>& g, const GraphProfile& prof, size_t batch_size, int replica_num,
                        int pipeline_num, bool use_amp_master_params, bool enable_zero) {
        size_t bs = ceil(batch_size / (double) (replica_num*pipeline_num));
        auto scaled = std::make_shared<IRGraph>("scaled", *g);
        scaled->setBatchSize(bs);
        return calcGraphMem(g, prof, use_amp_master_params, enable_zero, replica_num) + calcCommBufSize(scaled, pipeline_num);
    }

    bool fitToMem(const std::shared_ptr<IRGraph>& g, const GraphProfile& prof, long capacity, bool use_amp_master_params) {
        return calcGraphMem(g, prof, use_amp_master_params, false, 1) < (size_t) capacity;
    }

    GraphProfile makeErrorProfile() {
        GraphProfile p;
        p.fwd_time = ProfilerUtil::ERROR_VAL;
        p.bwd_time = ProfilerUtil::ERROR_VAL;
        p.max_allocated_mem = ProfilerUtil::ERROR_VAL;

        return p;
    }

    GraphProfile ProfilerUtil::profile(const std::shared_ptr<IRGraph>& g, size_t batch_size, size_t replica_num,
                                       bool checkpointing) {
        assert(replica_num > 0);
        assert(g);

        size_t bs = ceil(batch_size / (double) replica_num);

        const MLProfileKey k{g->getName(), bs, checkpointing};
        if (contains(profile_cache_, k)) {
            return profile_cache_.at(k);
        }

        if (!contains(max_batch_size_cache_[checkpointing], g->getName())) {
            max_batch_size_cache_[checkpointing][g->getName()] = SIZE_MAX;
        }

        size_t max_bs = max_batch_size_cache_[checkpointing][g->getName()];
        if (max_bs < bs) {
            profile_cache_[k] = makeErrorProfile();
            return profile_cache_.at(k);
        }

        std::unordered_map<std::string, std::shared_ptr<IRGraph>> prof_in_v;
        prof_in_v[g->getName()] = g;

        try {
            ProfilingResult prof_v = profiler_->profile(prof_in_v, 3, replica_num, checkpointing);
            assert(prof_v.node_profiles.size() == 1);
            profile_cache_[k] = prof_v.node_profiles.begin()->second;
        } catch (std::exception& e) {
            std::string msg= e.what();
            std::string::size_type pos1 = msg.find("CUDA out of memory");
            std::string::size_type pos2 = msg.find("Too many elements");
            if (pos1 == std::string::npos && pos2 == std::string::npos) {
                spdlog::error("Failed to profile graph: {} batch_size={} replica_num={} {}", g->getName(),
                        batch_size, replica_num, e.what());
                throw std::runtime_error("Failed to profile graph: " + toString(*g));
            } else {
                profile_cache_[k] = makeErrorProfile();

                if (max_batch_size_cache_[checkpointing][g->getName()] >= bs) {
                    max_batch_size_cache_[checkpointing][g->getName()] = bs - 1;
                }
                profiler_->clear();
                emptyCache();
                syncStream();
            }
        }
        return profile_cache_.at(k);
    }
}