//
// Created by Masahiro Tanaka on 2021/09/03.
//

#ifndef PYRANNC_BATCHSIZECALCULATOR_H
#define PYRANNC_BATCHSIZECALCULATOR_H

#include "Common.h"

namespace rannc {

    class BatchSizeCalculator {
    public:
        BatchSizeCalculator() : pipeline_num_(0), global_batch_size_(0) {}
        BatchSizeCalculator(int pipeline_num, int64_t global_batch_size);

        void setPipeline(int pipeline_num, int64_t global_batch_size);

        int64_t getGlobalSplitBatchSize(int split_index) const;
        std::vector<int64_t> getAllGlobalSplitBatchSizes() const;
        int64_t getLocalSplitBatchSize(const std::unordered_set<int>& ranks, int my_rank, int split_index) const;
        std::vector<int64_t> getAllLocalSplitBatchSizes(const std::unordered_set<int>& ranks, int my_rank) const;

        std::unordered_map<int, std::vector<int64_t>> calcDistBatchDims(const std::vector<int64_t> &global_dim,
                                                                        const std::unordered_set<int> &ranks,
                                                                        int split_index) const;
        double getDpRatio(const std::unordered_set<int>& ranks, int my_rank, int split_index) const;

        bool isLastLocalSplit(const std::unordered_set<int>& ranks, int my_rank, int split_index) const;
        int getFirstLocalSplitIndex(const std::unordered_set<int>& ranks, int my_rank) const;
        int getNextLocalSplitIndex(const std::unordered_set<int>& ranks, int my_rank, int split_index) const;

    private:
        int pipeline_num_;
        int64_t global_batch_size_;
        std::vector<int64_t> split_batch_sizes_;
    };

}


#endif //PYRANNC_BATCHSIZECALCULATOR_H
