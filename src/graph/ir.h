//
// Created by Masahiro Tanaka on 2018-12-17.
//

#ifndef PT_RANNC_IR_H
#define PT_RANNC_IR_H

#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <msgpack.hpp>

#include "Common.h"
#include "torch/IValueLocation.h"

#include <ATen/core/jit_type.h>

namespace rannc {

enum class IRTensorElemType {
  UNDEF,
  INT,
  DOUBLE,
  FLOAT,
  HALF,
  BFLOAT16,
  LONG,
  BOOL
};
enum class IRListType { INT, FLOAT, BOOL, TENSOR, GENERIC };
enum class IRScalarType { NONE, INT, FLOAT, NUMBER, BOOL, DEVICE };
enum class IRBaseType {
  SCALAR,
  TENSOR,
  STRING,
  LIST,
  TUPLE,
  OPTIONAL,
  NONE,
  FUNCTION
};

typedef c10::FunctionTypePtr IRFunctionType;
} // namespace rannc

// define in the global namespace
// https://github.com/msgpack/msgpack-c/wiki/v1_1_cpp_adaptor#enums
MSGPACK_ADD_ENUM(rannc::IRTensorElemType);
MSGPACK_ADD_ENUM(rannc::IRListType);
MSGPACK_ADD_ENUM(rannc::IRScalarType);
MSGPACK_ADD_ENUM(rannc::IRBaseType);

namespace rannc {
class IRType {
 public:
  // for serialization
  IRType() = default;

  static IRType createScalarType(IRScalarType scalar_type) {
    return IRType(scalar_type);
  }
  static IRType createTensorType(
      IRTensorElemType tensor_type, std::vector<int64_t> dim,
      bool requires_grad) {
    return IRType(
        IRBaseType::TENSOR, IRScalarType::NONE, std::move(dim), tensor_type,
        std::vector<IRType>(), requires_grad);
  }
  static IRType createUnknownShapeTensorType(IRTensorElemType tensor_type) {
    return IRType(tensor_type);
  }
  static IRType createUnknownTensorType() {
    return IRType(
        IRBaseType::TENSOR, IRScalarType::NONE, std::vector<int64_t>(),
        IRTensorElemType::UNDEF, std::vector<IRType>(), false);
  }
  static IRType createStringType() {
    return IRType(
        IRBaseType::STRING, IRScalarType::NONE, std::vector<int64_t>(),
        IRTensorElemType::UNDEF, std::vector<IRType>(), false);
  }
  static IRType createNoneType() {
    return IRType(
        IRBaseType::NONE, IRScalarType::NONE, std::vector<int64_t>(),
        IRTensorElemType::UNDEF, std::vector<IRType>(), false);
  }

  static IRType createListType(IRListType list_type, size_t size = -1) {
    IRType type;
    type.base_type_ = IRBaseType::LIST;
    type.list_type_ = list_type;
    type.list_size = size;
    type.compound_types_ = {};
    return type;
  }

  static IRType createListType(std::vector<IRType> elem_types) {
    IRType type(
        IRBaseType::LIST, IRScalarType::NONE, std::vector<int64_t>(),
        IRTensorElemType::UNDEF, elem_types, false);
    type.list_type_ = IRListType::GENERIC;
    type.list_size = elem_types.size();
    return type;
  }

  static IRType createTensorListType(std::vector<IRType> elem_types) {
    IRType type(
        IRBaseType::LIST, IRScalarType::NONE, std::vector<int64_t>(),
        IRTensorElemType::UNDEF, elem_types, false);
    type.list_type_ = IRListType::TENSOR;
    type.list_size = elem_types.size();
    return type;
  }
  static IRType createTupleType(std::vector<IRType> compound_types) {
    return IRType(
        IRBaseType::TUPLE, IRScalarType::NONE, std::vector<int64_t>(),
        IRTensorElemType::UNDEF, std::move(compound_types), false);
  }

  /**
   *    Create function type.
   *
   *  @param [in] func_type
   **/
  static IRType createFunctionType(const IRFunctionType& func_type) {
    return IRType(func_type);
  }

  static IRType createOptionalType(IRType elem_type) {
    std::vector<IRType> compound_type;
    compound_type.push_back(std::move(elem_type));
    return IRType(
        IRBaseType::OPTIONAL, IRScalarType::NONE, std::vector<int64_t>(),
        IRTensorElemType::UNDEF, compound_type, false);
  }

  IRBaseType getBaseType() const {
    return base_type_;
  }

  IRScalarType getScalarType() const {
    return scalar_type_;
  }

  IRListType getListType() const {
    return list_type_;
  }

  const std::vector<int64_t>& getTensorDim() const {
    return tensor_dim_;
  }
  //
  //
  //        void setTensorDim(const std::vector<int64_t> &tensorDim) {
  //            tensor_dim_ = tensorDim;
  //        }

  IRTensorElemType getTensorElemType() const {
    return tensor_elem_type_;
  }

  const std::vector<IRType>& getCompoundTypes() const {
    return compound_types_;
  }
  //
  //        void setCompoundTypes(const std::vector<IRType> &compoundTypes) {
  //            compound_types_ = compoundTypes;
  //        }

  bool requiresGrad() const {
    return requires_grad_;
  }

  void setRequiresGrad(bool requiresGrad) {
    requires_grad_ = requiresGrad;
  }

  size_t getSizeInByte() const;

  size_t getListSize() const {
    return list_size;
  }

  void setBatchSize(int64_t batch_size);

  const std::string& getFunctionName() const {
    return this->func_name_;
  }

  bool operator==(const IRType& rhs) const {
    if (base_type_ != rhs.base_type_)
      return false;

    switch (base_type_) {
      case IRBaseType::SCALAR:
        return scalar_type_ == rhs.scalar_type_;
      case IRBaseType::TENSOR:
        return tensor_dim_ == rhs.tensor_dim_ &&
            tensor_elem_type_ == rhs.tensor_elem_type_;
      case IRBaseType::LIST:
        return compound_types_ == rhs.compound_types_;
      case IRBaseType::TUPLE:
        return compound_types_ == rhs.compound_types_;
      case IRBaseType::STRING:
        return compound_types_ == rhs.compound_types_;
        ;
      case IRBaseType::FUNCTION:
        return func_name_ == rhs.func_name_;
      case IRBaseType::OPTIONAL:
        return compound_types_ == rhs.compound_types_;
      case IRBaseType::NONE:
        return compound_types_ == rhs.compound_types_;
    }
  }

  bool operator!=(const IRType& rhs) const {
    return !(rhs == *this);
  }

  MSGPACK_DEFINE(
      base_type_, scalar_type_, tensor_dim_, tensor_elem_type_, requires_grad_,
      list_type_, compound_types_, list_size, func_name_);

 private:
  IRType(
      IRBaseType baseType, IRScalarType scalarType,
      std::vector<int64_t> tensorDim, IRTensorElemType tensorElemType,
      std::vector<IRType> compoundTypes, bool requires_grad)
      : base_type_(baseType),
        scalar_type_(scalarType),
        tensor_dim_(std::move(tensorDim)),
        tensor_elem_type_(tensorElemType),
        requires_grad_(requires_grad),
        compound_types_(std::move(compoundTypes)) {}
  /*
   * Creates scalar type.
   */
  IRType(IRScalarType scalar_type)
      : base_type_(IRBaseType::SCALAR),
        scalar_type_(scalar_type),
        tensor_elem_type_(IRTensorElemType::UNDEF) {}
  /*
   * Creates tensor type.
   */
  IRType(IRTensorElemType tensor_type, std::vector<int64_t> dim)
      : base_type_(IRBaseType::TENSOR),
        scalar_type_(IRScalarType::NONE),
        tensor_dim_(std::move(dim)),
        tensor_elem_type_(tensor_type),
        requires_grad_(false) {}
  /*
   * Creates tensor type, dim is unknown.
   */
  IRType(IRTensorElemType tensor_type)
      : base_type_(IRBaseType::TENSOR),
        scalar_type_(IRScalarType::NONE),
        tensor_elem_type_(tensor_type),
        requires_grad_(false) {}

  /**
   *    Initialize instance as function type.
   *
   *  @param [in] func_type
   **/
  IRType(const IRFunctionType& func_type);

  IRBaseType base_type_;

  // valid if SCALAR
  IRScalarType scalar_type_;

  // valid if TENSOR
  std::vector<int64_t> tensor_dim_;
  IRTensorElemType tensor_elem_type_;
  bool requires_grad_;

  // valid if LIST
  IRListType list_type_;

  // valid if LIST, Tuple, or OPTIONAL. Note that the number of elements is
  // always one for LIST and OPTIONAL.
  std::vector<IRType> compound_types_;

  // valid if LIST
  size_t list_size = -1;

  //  valid if FUNCTION
  std::string func_name_;
};

class IRValue {
 public:
  // for serialization
  IRValue() = default;

  IRValue(std::string name, IRValue val) {
    *this = val;
    name_ = name;
  }

  IRValue(std::string name, IRType type)
      : name_(std::move(name)),
        type_(std::move(type)),
        is_param_(false),
        is_batch_(false),
        is_loss_(false) {}

  const std::string& getName() const {
    return name_;
  }

  const IRType& getType() const {
    return type_;
  }

  void setType(const IRType& type) {
    type_ = type;
  }

  bool isParam() const {
    return is_param_;
  }

  void setParam(bool is_param) {
    is_param_ = is_param;
  }

  size_t getSizeInByte() const {
    return type_.getSizeInByte();
  }

  bool isBatch() const {
    return is_batch_;
  }

  void setBatch(bool isBatch) {
    is_batch_ = isBatch;
  }

  bool isLoss() const {
    return is_loss_;
  }

  void setLoss(bool isLoss) {
    is_loss_ = isLoss;
  }

  void setBatchSize(int64_t batch_size) {
    if (isBatch()) {
      type_.setBatchSize(batch_size);
    }
  }

  /**
  **    Check if this value has FunctionType or not.
  **/
  bool isFunction() const;

  /**
  **    Get the function name (if this value is FunctionType).
  **
  **  @return     The function name.
  **/
  const std::string getFunctionName() const;

  friend std::ostream& operator<<(std::ostream& os, const IRValue& value);

  MSGPACK_DEFINE(name_, type_, is_param_, is_batch_, is_loss_);

 private:
  std::string name_;
  IRType type_;
  bool is_param_ = false;
  bool is_batch_ = false;
  bool is_loss_ = false;
};

class IRNode {
 public:
  IRNode() = default;

  IRNode(
      std::string name, std::vector<std::string> inputNames,
      std::vector<std::string> outputNames)
      : name_(std::move(name)),
        input_names_(std::move(inputNames)),
        output_names_(std::move(outputNames)),
        id_(generateName("node_")),
        is_criterion_(false),
        is_batch_(false) {}

  const std::string& getName() const {
    return name_;
  }

  const std::vector<std::string>& getInputNames() const {
    return input_names_;
  }

  const std::vector<std::string>& getOutputNames() const {
    return output_names_;
  }

  const std::string& getId() const {
    return id_;
  }

  bool isCriterion() const {
    return is_criterion_;
  }

  void setCriterion(bool isCriterion) {
    is_criterion_ = isCriterion;
  }

  bool isBatch() const {
    return is_batch_;
  }

  void setBatch(bool isBatch) {
    is_batch_ = isBatch;
  }

  /**
  **    Check if this node is prim::Function or not.
  **
  **  @param [in] values
  **  @retval     true   : This node is prim::Function
  **  @retval     false  : Otherwise.
  **/
  bool isFunctionNode(
      const std::unordered_map<std::string, IRValue>& values) const;

  friend std::ostream& operator<<(std::ostream& os, const IRNode& node);

  MSGPACK_DEFINE(
      name_, input_names_, output_names_, id_, is_criterion_, is_batch_);

 private:
  std::string name_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::string id_;
  bool is_criterion_;
  bool is_batch_;
};

class IRGraph {
 public:
  IRGraph() = default;

  IRGraph(
      std::string name, std::vector<IRNode> nodes,
      std::unordered_map<std::string, IRValue> values,
      std::vector<std::string> inputNames, std::vector<std::string> outputNames)
      : name_(std::move(name)),
        nodes_(std::move(nodes)),
        values_(std::move(values)),
        input_names_(std::move(inputNames)),
        output_names_(std::move(outputNames)),
        is_replicable_(false) {
    checkReplicable();
  }

  IRGraph(std::string name, const IRGraph& g) : name_(std::move(name)) {
    nodes_ = g.nodes_;
    values_ = g.values_;
    input_names_ = g.input_names_;
    output_names_ = g.output_names_;
    is_replicable_ = g.is_replicable_;
  }

  const std::string& getName() const {
    return name_;
  }

  void setName(const std::string& name) {
    name_ = name;
  }

  const std::vector<IRNode>& getNodes() const {
    return nodes_;
  }

  const std::unordered_map<std::string, IRValue>& getValues() const {
    return values_;
  }

  const std::vector<std::string>& getInputNames() const {
    return input_names_;
  }

  const std::vector<std::string>& getOutputNames() const {
    return output_names_;
  }

  const IRValue& getValue(const std::string& name) const {
    if (!contains(values_, name)) {
      throw std::invalid_argument("IRGraph does not contain value: " + name);
    }

    return values_.at(name);
  }

  size_t getSizeInByte() const {
    size_t sum = 0;
    for (const auto& v : getValues()) {
      sum += v.second.getSizeInByte();
    }
    return sum;
  }

  size_t getParamSizeInByte() const {
    size_t sum = 0;
    for (const auto& v : getValues()) {
      if (v.second.isParam()) {
        sum += v.second.getSizeInByte();
      }
    }
    return sum;
  }

  std::vector<std::string> getParamNames() const {
    std::vector<std::string> results;
    for (const auto& input : getInputNames()) {
      const auto& value = getValue(input);
      if (value.isParam()) {
        results.push_back(input);
      }
    }
    return results;
  }

  bool hasValue(const std::string& name) const {
    return contains(values_, name);
  }

  /**
   *    Check if the 'node' is prim::Function or not.
   *
   *  @param [in] node
   *  @return
   **/
  bool isFunctionNode(const IRNode& node) const {
    return node.isFunctionNode(this->values_);
  }

  bool isReplicable() const {
    return is_replicable_;
  }

  void setReplicable(bool isReplicable) {
    is_replicable_ = isReplicable;
  }

  void checkReplicable();

  void setBatchSize(int64_t batch_size);

  friend std::ostream& operator<<(std::ostream& os, const IRGraph& graph);

  MSGPACK_DEFINE(
      name_, nodes_, values_, input_names_, output_names_, is_replicable_);

 private:
  std::string name_;
  std::vector<IRNode> nodes_;
  std::unordered_map<std::string, IRValue> values_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  bool is_replicable_;
};

std::vector<IRValue> graphNonParamInputValues(
    const std::shared_ptr<IRGraph>& irGraph);
std::vector<IRValue> graphParamInputValues(
    const std::shared_ptr<IRGraph>& irGraph);

/**
 * Get constant values in an IRGraph. The result is sorted by names of the
 * values.
 *
 * @param irGraph Graph to search.
 * @return Constant values.
 */
std::vector<IRValue> graphConstantValues(
    const std::shared_ptr<IRGraph>& irGraph);

/**
 * Get parameter values in an IRGraph. The result is sorted by names of the
 * values.
 *
 * @param irGraph Graph to search.
 * @return Parameter values.
 */
std::vector<IRValue> graphParamValues(const std::shared_ptr<IRGraph>& irGraph);

std::string toString(IRBaseType type);
std::string toString(IRTensorElemType type);
std::string toString(IRListType type);
size_t getTensorElemSize(IRTensorElemType type);
size_t getTensorSizeInByte(
    IRTensorElemType type, const std::vector<int64_t>& dim);

std::string toString(IRScalarType type);
size_t getScalarSize(IRScalarType type);
std::string toString(const IRType& type);

IRType getElemInIRType(const IRType& type, const PathInIValue& path);
IRType setDimToIRType(const IRType& type, const std::vector<int64_t>& dim);

std::vector<std::string> getNonParamInputNames(
    const std::shared_ptr<IRGraph>& graph);
std::vector<std::string> getParamInputNames(
    const std::shared_ptr<IRGraph>& graph);

std::vector<std::string> getGradNames(
    const std::shared_ptr<IRGraph>& graph,
    const std::vector<std::string>& names);
std::vector<std::string> getGradOutputNames(
    const std::shared_ptr<IRGraph>& graph);
std::vector<std::string> getGradInputNames(
    const std::shared_ptr<IRGraph>& graph);
bool isGraphReady(
    const std::vector<std::string>& graph_input_names,
    const std::unordered_set<IValueLocation, IValueLocationHash>& avail_locs);

std::pair<
    std::shared_ptr<IRGraph>,
    std::unordered_map<std::string, std::vector<std::string>>>
cloneSharedInputs(const std::shared_ptr<IRGraph>& ir_graph);
int64_t guessGraphBatchSize(const std::shared_ptr<IRGraph>& ir_graph);
size_t getOptMemSize(
    const std::shared_ptr<IRGraph>& ir_graph, int opt_param_factor,
    bool use_amp_master_param, bool enable_zero, int zero_dist_num);
size_t getAmpMasterParamSize(const std::shared_ptr<IRGraph>& ir_graph);
bool verifyNoDuplicatedOutputs(const std::shared_ptr<IRGraph>& g);
bool verifyNodeInputs(const std::shared_ptr<IRGraph>& g, bool show_msg = false);
std::vector<IRNode> detectUnusedNodes(const std::shared_ptr<IRGraph>& g);
std::unordered_set<std::string> findUnusedValue(
    const std::shared_ptr<IRGraph>& g);
bool noUnusedValue(const std::shared_ptr<IRGraph>& g, bool show_msg = false);
std::shared_ptr<IRGraph> removeUnusedNodes(const std::shared_ptr<IRGraph>& g);
size_t calcCommBufSize(const std::shared_ptr<IRGraph>& g);
} // namespace rannc

#endif // PT_RANNC_IR_H
