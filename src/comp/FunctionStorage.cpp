
#include "FunctionStorage.h"

namespace rannc {

//
//    Construct function table from graph.
//
void FunctionStorage::deploy(const std::shared_ptr<torch::jit::Graph>& graph) {
  for (const auto& node : graph->nodes()) {
    if (node->kind() != c10::prim::Constant) {
      continue;
    }

    const auto tp = node->output()->type();
    if (tp->kind() != c10::TypeKind::FunctionType) {
      continue;
    }

    const auto func = tp->expectRef<c10::FunctionType>().function();
    const std::string name =
        node->output()->debugName() + '@' + func->qualname().name();
    this->functions_[name] = func;

    //  Save the attr name of the function.
    this->func_attr_[name] = node->s(c10::attr::name);
  }
}

//
//    Get attr::name of the function.
//
const std::string& FunctionStorage::getAttrName(const std::string& name) const {
  return func_attr_.at(name);
}

} //  End of namespace rannc.
