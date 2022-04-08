
#ifndef PYRANNC_FUNCTIONSTRAGE_H
#define PYRANNC_FUNCTIONSTRAGE_H

#include <string>
#include <unordered_map>

#include <torch/csrc/jit/ir/ir.h>

namespace rannc {

class FunctionStorage {
 public:
  typedef std::unordered_map<std::string, torch::jit::Function*> FunctionTable;

  typedef std::unordered_map<std::string, std::string> FunctionAttr;

 public:
  const FunctionTable& getFunctions() const {
    return functions_;
  }

  /**
  **    Construct function table from graph.
  **
  **  @param [in] graph
  **  @return     void.
  **/
  void deploy(const std::shared_ptr<torch::jit::Graph>& graph);

  /**
  **    Get attr::name of the function.
  **
  **  @param [in] name    Function name.
  **  @return     string : the attr::name value.
  **/
  const std::string& getAttrName(const std::string& name) const;

 private:
  FunctionTable functions_;
  FunctionAttr func_attr_;
};

} //  End of namespace rannc.

#endif
