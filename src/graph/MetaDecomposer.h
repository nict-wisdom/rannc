//
// Created by Masahiro Tanaka on 2020/02/06.
//

#ifndef PYRANNC_METADECOMPOSER_H
#define PYRANNC_METADECOMPOSER_H

#include <comp/GraphProfiler.h>
#include "Decomposition.h"

namespace rannc {

class GraphProfiler;
class MetaDecomposer {
 public:
  MetaDecomposer(std::shared_ptr<GraphProfiler> sg_prof, PartitioningConf conf)
      : sg_prof_(std::move(sg_prof)), conf_(std::move(conf)) {}

  Deployment decompose(
      const std::string& name, const std::shared_ptr<IRGraph>& ir_graph);

 private:
  std::shared_ptr<GraphProfiler> sg_prof_;
  PartitioningConf conf_;

  const std::shared_ptr<spdlog::logger> logger = getLogger("Decomposer");
};
} // namespace rannc

#endif // PYRANNC_METADECOMPOSER_H
