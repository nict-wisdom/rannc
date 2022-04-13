//
// Created by Masahiro Tanaka on 2021/09/03.
//

#include "BatchSizeCalculator.h"

#include "Common.h"

namespace rannc {

/**
 * Calculates the batch size for the given split index.
 * batch_size is the one for split_index. See comments for calcDistBatchDims.
 *
 * @param split_batch_size Batch size for split_index.
 * @param ranks Ranks that process batch_size in total.
 * @param split_index Index of the target split.
 * @return Batch sizes for given ranks.
 */
std::unordered_map<int, int64_t> getLocalSplitBatchSizes(
    const size_t split_batch_size, const std::unordered_set<int>& ranks,
    int split_index) {
  std::vector<int> vec_ranks = rannc::setToVector(ranks);
  std::sort(vec_ranks.begin(), vec_ranks.end());

  size_t rank_num = ranks.size();
  int64_t base_size = split_batch_size / rank_num;
  size_t mod = split_batch_size % rank_num;

  std::unordered_map<int, int64_t> batch_sizes;
  for (size_t i = 0; i < rank_num; i++) {
    int64_t ex = i < mod ? 1 : 0;
    int64_t split_size = base_size + ex;

    int rank = vec_ranks.at(i);
    batch_sizes[rank] = split_size;
  }

  if (split_index == 0) {
    return batch_sizes;
  }

  int offset = 0;
  for (int i = 0; i < split_index; i++) {
    offset = (offset + split_batch_size) % ranks.size();
  }
  std::unordered_map<int, int64_t> shifted_batch_sizes;
  shifted_batch_sizes.reserve(batch_sizes.size());

  for (int i = 0; i < vec_ranks.size(); i++) {
    int shift_i = (i + offset) % vec_ranks.size();
    shifted_batch_sizes[vec_ranks.at(shift_i)] =
        batch_sizes.at(vec_ranks.at(i));
  }

  return shifted_batch_sizes;
}

/**
 * Calculates batch sizes for splits.
 *
 * @param batch_size The global batch size (the total batch size processed by
 * all ranks and all splits).
 * @param pipeline_num The number of splits.
 * @return Batch sizes for splits.
 */
std::vector<int64_t> getSplitBatchSizes(int pipeline_num, int64_t batch_size) {
  std::vector<int> dummy_ranks;
  for (int i = 0; i < pipeline_num; i++) {
    dummy_ranks.push_back(i);
  }

  const auto batch_sizes =
      getLocalSplitBatchSizes(batch_size, vectorToSet(dummy_ranks), 0);
  std::vector<int64_t> result;
  for (int r : dummy_ranks) {
    result.push_back(batch_sizes.at(r));
  }
  return result;
}

BatchSizeCalculator::BatchSizeCalculator(
    int pipeline_num, int64_t global_batch_size)
    : pipeline_num_(pipeline_num), global_batch_size_(global_batch_size) {
  split_batch_sizes_ = getSplitBatchSizes(pipeline_num_, global_batch_size_);
}

int64_t BatchSizeCalculator::getGlobalSplitBatchSize(int split_index) const {
  assert(global_batch_size_ > 0);
  assert(pipeline_num_ > 0);

  assert(split_index < split_batch_sizes_.size());
  return split_batch_sizes_.at(split_index);
}

std::vector<int64_t> BatchSizeCalculator::getAllGlobalSplitBatchSizes() const {
  return split_batch_sizes_;
}

int64_t BatchSizeCalculator::getLocalSplitBatchSize(
    const std::unordered_set<int>& ranks, int my_rank, int split_index) const {
  assert(contains(ranks, my_rank));
  int64_t split_bs = getGlobalSplitBatchSize(split_index);

  std::unordered_map<int, int64_t> split_bs_all_ranks =
      getLocalSplitBatchSizes(split_bs, ranks, split_index);
  assert(contains(split_bs_all_ranks, my_rank));

  return split_bs_all_ranks.at(my_rank);
}

std::vector<int64_t> BatchSizeCalculator::getAllLocalSplitBatchSizes(
    const std::unordered_set<int>& ranks, int my_rank) const {
  assert(contains(ranks, my_rank));
  std::vector<int64_t> split_sizes;

  for (int split_index = 0; split_index < split_batch_sizes_.size();
       split_index++) {
    int64_t split_bs = getGlobalSplitBatchSize(split_index);
    std::unordered_map<int, int64_t> split_bs_all_ranks =
        getLocalSplitBatchSizes(split_bs, ranks, split_index);
    assert(contains(split_bs_all_ranks, my_rank));
    split_sizes.push_back(split_bs_all_ranks.at(my_rank));
  }

  return split_sizes;
}

double BatchSizeCalculator::getDpRatio(
    const std::unordered_set<int>& ranks, int my_rank, int split_index) const {
  return getLocalSplitBatchSize(ranks, my_rank, split_index) /
      (double)getGlobalSplitBatchSize(split_index);
}

/**
 * Calculates the batch size for the given split index and scales the size of
 * the head dimension. Notice that:
 * - batch_size is the one that should be processed by all the ranks, but only
 * for the given split index. This is not an effective batch size for a training
 * iteration and can be calculated by std::vector<int64_t>
 * getSplitBatchSizes(int pipeline_num, int64_t batch_size) .
 * - The head dimension may not equal the batch size.
 * Even for the same *batch_size*, the local batch size that should be processed
 * on a certain rank can differ if the split_index is different.
 *
 * @param batch_size Sum of batch sizes on all ranks for *split_index*.
 * @param global_dim Dimension of a tensor.
 * @param ranks Ranks that process *batch_size* in total.
 * @param split_index Index of the target split.
 * @return
 */
std::unordered_map<int, std::vector<int64_t>> BatchSizeCalculator::
    calcDistBatchDims(
        const std::vector<int64_t>& global_dim,
        const std::unordered_set<int>& ranks, int split_index) const {
  assert(!global_dim.empty());

  int64_t split_bs = getGlobalSplitBatchSize(split_index);
  std::unordered_map<int, int64_t> batch_sizes =
      getLocalSplitBatchSizes(split_bs, ranks, split_index);

  std::unordered_map<int, std::vector<int64_t>> results;
  for (const auto& it : batch_sizes) {
    int rank = it.first;
    auto rank_bs = it.second;

    std::vector<int64_t> rank_dim = global_dim;
    // The head dimension may not equal the batch size.
    // We assume that the head dimension is divisible by the batch size.
    assert(global_dim[0] % split_bs == 0);
    rank_dim[0] = global_dim[0] * rank_bs / split_bs;
    results[rank] = rank_dim;
  }

  return results;
}

void BatchSizeCalculator::setPipeline(
    int pipeline_num, int64_t global_batch_size) {
  pipeline_num_ = pipeline_num;
  global_batch_size_ = global_batch_size;
  split_batch_sizes_ = getSplitBatchSizes(pipeline_num_, global_batch_size_);
}
} // namespace rannc
