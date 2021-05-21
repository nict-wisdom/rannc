//
// Created by Masahiro Tanaka on 2019-07-22.
//

#ifndef PYRANNC_MPIALLREDUCERUNNER_H
#define PYRANNC_MPIALLREDUCERUNNER_H

#include <unordered_set>
#include "torch/TorchUtil.h"

namespace rannc {
    typedef struct AllReduceComm AllReduceComm;

    class NCCLBulkJobExecutor {
    public:
        void flush();
        void addPreCommJob(std::function<void(void)> f);
        void addCommJob(std::function<void(void)> f);
        void addPostCommJob(std::function<void(void)> f);

        bool runImmediate() const {
            return run_immediate_;
        }

        void setRunImmediate(bool run_immediate) {
            run_immediate_ = run_immediate;
        }

    private:
        void doAddJob(std::function<void(void)> f, std::vector<std::function<void(void)>>& jobs);

        bool run_immediate_ = true;
        std::vector<std::function<void(void)>> pre_comm_jobs_;
        std::vector<std::function<void(void)>> comm_jobs_;
        std::vector<std::function<void(void)>> post_comm_jobs_;
    };

    class NCCLWrapper {
    public:
        static NCCLWrapper& get() {
            static NCCLWrapper instance;
            return instance;
        }

        void createCommunicator(int tag, const std::unordered_set<int> &ranks);
        void destroy();

        void allreduce(int tag, const std::vector<at::Tensor> &tensors);
        void allreduceMin(int tag, const std::vector<at::Tensor> &tensors);
        void reduce(int tag, const std::vector<at::Tensor> &tensors, const std::vector<at::Tensor>& out_bufs, const std::vector<int>& roots);
        void redist(void* send_ptr, void* recv_ptr, const RouteDP& route,
                    int64_t batch_size, const IRType& global_type, int split_index);
        void bcast(int tag, const std::unordered_set<int>& ranks, int root,
                                const std::vector<at::Tensor>& tensors);
        void send(int tag, int dest, const at::Tensor& tensor);
        void recv(int tag, int dest, const at::Tensor& tensor);
        void startBulk();
        void endBulk();

        int getTag(const std::unordered_set<int> &ranks) {
            if(!contains(ranks_to_tag_, ranks)) return -1;
            return ranks_to_tag_.at(ranks);
        }

        std::string getImplName();

        NCCLWrapper(const NCCLWrapper&) = delete;
        NCCLWrapper& operator=(const NCCLWrapper&) = delete;
        NCCLWrapper(NCCLWrapper&&) = delete;
        NCCLWrapper& operator=(NCCLWrapper&&) = delete;

    private:
        NCCLWrapper() = default;

        void doAllreduce(int tag, const std::vector<at::Tensor> &tensors, ncclRedOp_t red_op);

        bool initialized = false;
        std::unordered_map<int, AllReduceComm*> comm_map_;
        std::unordered_map<std::unordered_set<int>, int, IntSetHash> ranks_to_tag_;
        BufferTensorCache buf_cache_;
        NCCLBulkJobExecutor job_executor_;

        std::shared_ptr<spdlog::logger> logger = getLogger("NCCLWrapper");
    };
}

#endif //PYRANNC_MPIALLREDUCERUNNER_H
