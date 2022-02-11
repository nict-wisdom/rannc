//
// Created by Masahiro Tanaka on 2020/02/24.
//

#ifndef PYRANNC_MLPARTDECOMPOSER_H
#define PYRANNC_MLPARTDECOMPOSER_H

#include <comp/GraphProfiler.h>
#include "Decomposition.h"

namespace rannc {

class MLPartDecomposer {
 public:
  MLPartDecomposer(
      std::shared_ptr<GraphProfiler> sg_prof, PartitioningConf conf)
      : sg_prof_(std::move(sg_prof)), conf_(std::move(conf)) {}

  Deployment decompose(const std::shared_ptr<IRGraph>& ir_graph);

 private:
  std::shared_ptr<GraphProfiler> sg_prof_;
  PartitioningConf conf_;

  const std::shared_ptr<spdlog::logger> logger = getLogger("Decomposer");
};

} // namespace rannc

#endif // PYRANNC_MLPARTDECOMPOSER_H
