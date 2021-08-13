//
// Created by Masahiro Tanaka on 2018-12-17.
//
#include <assert.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include "ir.h"
#include "torch/IValueLocation.h"

#include <ATen/core/function.h>


namespace rannc {

    std::string toString(IRBaseType type) {
        switch (type) {
            case IRBaseType::SCALAR: return "SCALAR";
            case IRBaseType::TENSOR: return "TENSOR";
            case IRBaseType::STRING: return "STRING";
            case IRBaseType::LIST: return "LIST";
            case IRBaseType::TUPLE: return "TUPLE";
            case IRBaseType::FUNCTION:  return "FUNCTION";
            case IRBaseType::OPTIONAL: return "OPTIONAL";
            case IRBaseType::NONE: return "NONE";
        }
        throw std::invalid_argument("Failed to convert IRBaseType to string.");
    }

    std::string toString(IRTensorElemType type) {
        switch (type) {
            case IRTensorElemType::UNDEF: return "UNDEF";
            case IRTensorElemType::INT: return "INT";
            case IRTensorElemType::DOUBLE: return "DOUBLE";
            case IRTensorElemType::FLOAT: return "FLOAT";
            case IRTensorElemType::HALF: return "HALF";
            case IRTensorElemType::BFLOAT16: return "BFLOAT16";
            case IRTensorElemType::LONG: return "LONG";
            case IRTensorElemType::BOOL: return "BOOL";
        }
    }

    std::string toString(IRListType type) {
        switch (type) {
            case IRListType::INT: return "INT";
            case IRListType::FLOAT: return "FLOAT";
            case IRListType::BOOL: return "BOOL";
            case IRListType::TENSOR: return "TENSOR";
            case IRListType::GENERIC: return "GENERIC";
        }
    }

    size_t getTensorElemSize(IRTensorElemType type) {
        switch (type) {
            case IRTensorElemType::INT: return sizeof(int);
            case IRTensorElemType::DOUBLE: return sizeof(double);
            case IRTensorElemType::FLOAT: return sizeof(float);
            case IRTensorElemType::HALF: return sizeof(float) / 2;
            case IRTensorElemType::BFLOAT16: return sizeof(float) / 2;
            case IRTensorElemType::LONG: return sizeof(long);
            case IRTensorElemType::BOOL: return sizeof(bool);
            case IRTensorElemType::UNDEF: break;
        }
        throw std::invalid_argument("Unexpected tensor elem type: " + toString(type));
    }

    size_t getTensorSizeInByte(IRTensorElemType type, const std::vector<int64_t>& dim) {
        return productDim(dim) * getTensorElemSize(type);
    }

    std::string toString(IRScalarType type) {
        switch (type) {
            case IRScalarType::INT: return "INT";
            case IRScalarType::FLOAT: return "FLOAT";
            case IRScalarType::NUMBER: return "NUMBER";
            case IRScalarType::BOOL: return "BOOL";
            case IRScalarType::DEVICE: return "DEVICE";
            case IRScalarType::NONE: return "NONE";
        }
        throw std::invalid_argument("Failed to convert IRScalarType to string.");
    }
    size_t getScalarSize(IRScalarType type) {
        switch (type) {
            case IRScalarType::INT: return sizeof(long);
            case IRScalarType::FLOAT: return sizeof(double);
            case IRScalarType::BOOL: return sizeof(bool);
            case IRScalarType::NUMBER: return 0;
            case IRScalarType::DEVICE: return 0;
            case IRScalarType::NONE: return 0;
        }
        throw std::invalid_argument("Unexpected scalar type: " + toString(type));
    }

    std::vector<std::string> toString(const std::vector<IRType>& types) {
        std::vector<std::string> ret;
        ret.reserve(types.size());
        for (const auto& t: types) {
            ret.push_back(toString(t));
        }
        return ret;
    }

    std::string toString(const IRType& type) {
        std::stringstream ss;

        switch (type.getBaseType()) {
            case IRBaseType::SCALAR:
                return "SCALAR_" + toString(type.getScalarType());
            case IRBaseType::TENSOR: {
                std::string req_grad;
                if (type.requiresGrad()) {
                    req_grad = "(G)";
                }

                auto elem_type = type.getTensorElemType();
                ss << toString(elem_type) << join_as_str(type.getTensorDim()) << req_grad;
                return ss.str();
            }
            case IRBaseType::LIST: {
                ss << toString(type.getListType()) << "LIST" << join_as_str(toString(type.getCompoundTypes()));
                return ss.str();
            }
            case IRBaseType::TUPLE:{
                ss << "TUPLE" << join_as_str(toString(type.getCompoundTypes()));
                return ss.str();
            }
            case IRBaseType::STRING:
                return "STRING";
            case IRBaseType::FUNCTION:
                return "FUNCTION";
            case IRBaseType::OPTIONAL: {
                assert(type.getCompoundTypes().size() == 1);
                const auto& ct = type.getCompoundTypes().front();
                ss << "OPTIONAL[" << toString(ct) << "]";
                return ss.str();
            }
            case IRBaseType::NONE:
                return "NONE";
            default:
                break;
        }
        return "UNK";
    }

    size_t IRType::getSizeInByte() const {
        switch (base_type_) {
            case IRBaseType::SCALAR: return getScalarSize(scalar_type_);
            case IRBaseType::TENSOR: return getTensorSizeInByte(tensor_elem_type_, tensor_dim_);
            case IRBaseType::LIST:
                // list length is unknown.
                return 0;
            case IRBaseType::TUPLE: {
                size_t sum = 0;
                for (const auto &t: compound_types_) {
                    sum += t.getSizeInByte();
                }
                return sum;
            }
            case IRBaseType::STRING:
            case IRBaseType::OPTIONAL:
            case IRBaseType::NONE:
                return 0;
            case IRBaseType::FUNCTION:
                return 0;
        }
        throw std::invalid_argument("Unexpected base type: " + toString(base_type_));
    }

    void IRType::setBatchSize(int64_t batch_size) {
        switch (getBaseType()) {
            case IRBaseType::SCALAR: break;
            case IRBaseType::TENSOR: {
                if (!tensor_dim_.empty()) {
                    tensor_dim_[0] = batch_size;
                }
                return;
            }
            case IRBaseType::TUPLE:
            case IRBaseType::LIST: {
                for (auto& t: compound_types_) {
                    t.setBatchSize(batch_size);
                }
                break;
            }
            case IRBaseType::STRING:
            case IRBaseType::NONE:
            case IRBaseType::OPTIONAL:
                break;
        }
    }

    //
    //    Initialize instance as function type.
    //
    IRType::IRType(const IRFunctionType &func_type)
        : base_type_(IRBaseType::FUNCTION),
          scalar_type_(IRScalarType::NONE),
          tensor_elem_type_(IRTensorElemType::UNDEF),
          requires_grad_(false),
          func_name_(func_type->function()->qualname().name())
    { }

    //
    //    Check if this value has FunctionType or not.
    //
    bool IRValue::isFunction() const
    {
        if ( this->type_.getBaseType() != IRBaseType::FUNCTION ) {
            return  false;
        }
        return  true;
    }

    //
    //    Get the function name (if this value is FunctionType).
    //
    const std::string   IRValue::getFunctionName() const
    {
        assert( isFunction() );
        return  ( this->name_ + '@' + this->type_.getFunctionName() );
    }

    std::vector<IRValue> graphInputValues(const std::shared_ptr<IRGraph> &irGraph, bool is_param) {
        std::vector<IRValue> results;

        for (const auto &name: irGraph->getInputNames()) {
            const auto& val = irGraph->getValue(name);
            if (val.isParam() == is_param) {
                results.push_back(val);
            }
        }
        return results;
    }

    std::vector<IRValue> graphNonParamInputValues(const std::shared_ptr<IRGraph> &irGraph) {
        return graphInputValues(irGraph, false);
    }

    std::vector<IRValue> graphParamInputValues(const std::shared_ptr<IRGraph> &irGraph) {
        return graphInputValues(irGraph, true);
    }

    std::vector<IRValue> graphConstantValues(const std::shared_ptr<IRGraph> &irGraph) {

        std::unordered_map<std::string, IRValue> values;

        for (const auto &node: irGraph->getNodes()) {
            if (node.getName() != "prim::Constant") {
                continue;
            }

            for (const auto &out_name: node.getOutputNames()) {
                const auto &val = irGraph->getValue(out_name);
                if (val.isFunction()) {
                    //  Exception handling - Function in constants.
                    continue;
                }
                values[val.getName()] = val;
            }
        }
        std::vector<IRValue> results;
        for (const auto& it: values) {
            results.push_back(it.second);
        }

        return results;
    }

    std::vector<IRValue> graphParamValues(const std::shared_ptr<IRGraph> &irGraph) {
        std::vector<IRValue> results;

        for (const auto &name: keys(irGraph->getValues())) {
            const auto& val = irGraph->getValue(name);
            if (val.isParam()) {
                results.push_back(val);
            }
        }
        return results;
    }

    std::ostream &operator<<(std::ostream &os, const IRValue &value) {
        std::string val_type;
        if (value.isBatch()) {
            val_type = "(B)";
        } else if (value.isParam()) {
            val_type = "(P)";
        } else if (value.isLoss()) {
            val_type = "(L)";
        }
        os << value.getName() << val_type << "[" << toString(value.getType()) << "]";
        return os;
    }

    //
    //    Check if this node is prim::Function or not.
    //
    bool IRNode::isFunctionNode(
            const std::unordered_map<std::string, IRValue> &values) const
    {
        const IRValue &ov = values.at(this->output_names_.at(0));
        return ( ov.isFunction() );
    }

    std::ostream &operator<<(std::ostream &os, const IRNode &node) {
        os << node.name_ << "(" << join_as_str(node.input_names_) << ") => "
           << join_as_str(node.output_names_);
        return os;
    }

    std::ostream &operator<<(std::ostream &os, const IRGraph &graph) {

        os << "Graph " << graph.getName() << std::endl;
        for (const std::string &input_name: graph.getInputNames()) {
            const auto& value = graph.getValue(input_name);
            os << "Graph input: " << value << std::endl;
        }

        for (const auto &in_node: graph.getNodes()) {
            std::vector<std::string> outputs;
            for (const auto& node_out_name: in_node.getOutputNames()) {
                const auto& out_v = graph.getValue(node_out_name);
                std::stringstream ss;
                ss << out_v;
                outputs.push_back(ss.str());
            }

            std::string op_type;
            if (in_node.isBatch()) {
                op_type = "[B]";
            }

            os << "  " << in_node.getName() << op_type << "(" << join_as_str(in_node.getInputNames()) << ") => "
               << join_as_str(outputs) << std::endl;
        }

        for (const std::string &output_name: graph.getOutputNames()) {
            const auto& value = graph.getValue(output_name);
            os << "Graph output: " << value << std::endl;
        }

        return os;
    }

    void IRGraph::checkReplicable() {
        for (const auto& in_name: getInputNames()) {
            const auto& in_v = getValue(in_name);

            if (in_v.isBatch()) {
                setReplicable(true);
                return;
            }
        }
    }

    void IRGraph::setBatchSize(int64_t batch_size) {
        for (auto& it: values_) {
            auto& v = it.second;
            if (v.isBatch()) {
                v.setBatchSize(batch_size);
            }
        }
    }

    IRType doGetElemInIRType(const IRType &type, std::vector<StepInIValue>::const_iterator iter,
                      const std::vector<StepInIValue>::const_iterator &end_iter) {
        if (iter == end_iter) return type;

        assert(type.getBaseType() == IRBaseType::LIST || type.getBaseType() == IRBaseType::TUPLE);
        const auto& elem_types = type.getCompoundTypes();
        assert(elem_types.size() > iter->index);
        return elem_types.at(iter->index);
    }

    IRType getElemInIRType(const IRType &type, const PathInIValue &path) {
        if (path.begin() == path.end()) return type;
        return doGetElemInIRType(type, path.cbegin(), path.cend());
    }

    IRType setDimToIRType(const IRType &type, const std::vector<int64_t>& dim) {
        return IRType::createTensorType(type.getTensorElemType(), dim, type.requiresGrad());
    }

    std::vector<std::string> doGetInputNames(const std::shared_ptr<IRGraph> &graph,
            const std::function<bool(const std::string&)>& f) {
        const auto &input_names = graph->getInputNames();
        std::vector<std::string> ret_input_names;

        std::copy_if(input_names.begin(), input_names.end(), std::back_inserter(ret_input_names), f);
        return ret_input_names;
    }

    std::vector<std::string> getNonParamInputNames(const std::shared_ptr<IRGraph> &graph) {
        return doGetInputNames(graph, [&graph](const std::string &n) {
            return !graph->getValue(n).isParam();
        });
    }

    std::vector<std::string> getParamInputNames(const std::shared_ptr<IRGraph> &graph) {
        return doGetInputNames(graph, [&graph](const std::string &n) {
            return graph->getValue(n).isParam();
        });
    }

    std::vector<std::string> getGradNames(const std::shared_ptr<IRGraph> &graph, const std::vector<std::string>& names) {
        std::vector<std::string> grad_names;

        std::copy_if(names.begin(), names.end(), std::back_inserter(grad_names),
                     [&graph](const std::string &n) {
                         return passedForBackward(graph->getValue(n).getType());
                     });
        return grad_names;
    }

    std::vector<std::string> getGradOutputNames(const std::shared_ptr<IRGraph> &graph) {
        const auto &output_names = graph->getOutputNames();
        return getGradNames(graph, output_names);
    }

    std::vector<std::string> getGradInputNames(const std::shared_ptr<IRGraph> &graph) {
        const auto &input_names = graph->getInputNames();
        return getGradNames(graph, input_names);
    }

    bool isGraphReady(const std::vector<std::string> &graph_input_names,
                      const std::unordered_set<IValueLocation, IValueLocationHash> &avail_locs) {

        for (const auto &in_name: graph_input_names) {
            IValueLocation loc{in_name};
            if (!contains(avail_locs, loc)) return false;
        }
        return true;
    }

    int countValueRefCount(const std::shared_ptr<IRGraph> &ir_graph, const std::string& val_name) {
        int count = 0;
        for (const auto& node: ir_graph->getNodes()) {
            if (contains(node.getInputNames(), val_name)) {
                count++;
            }
        }
        return count;
    }

    std::unordered_map<std::string, int> countSharedRefs(const std::shared_ptr<IRGraph> &ir_graph,
                                                         const std::vector<IRValue>& input_vals) {
        std::unordered_map<std::string, int> shared_inputs;

        for (const auto& val: input_vals) {
            int ref_count = countValueRefCount(ir_graph, val.getName());
            if (ref_count > 1) {
                shared_inputs[val.getName()] = ref_count;
            }
        }
        return shared_inputs;
    }

    std::pair<std::shared_ptr<IRGraph>, std::unordered_map<std::string, std::vector<std::string>>> cloneSharedInputs(const std::shared_ptr<IRGraph> &ir_graph) {
        std::vector<IRValue> no_param_inputs = graphNonParamInputValues(ir_graph);
        const std::unordered_map<std::string, int> shared_no_param_inputs = countSharedRefs(ir_graph, no_param_inputs);
        std::vector<IRValue> param_inputs = graphParamInputValues(ir_graph);
        const std::unordered_map<std::string, int> shared_param_inputs = countSharedRefs(ir_graph, param_inputs);

        const std::unordered_map<std::string, int> shared_input_ref_count = addAll(shared_no_param_inputs, shared_param_inputs);

        std::unordered_map<std::string, int> ref_idx;
        std::unordered_map<std::string, std::vector<std::string>> shared_val_cl_names;

        std::vector<std::string> graph_input_names;
        for (const auto& graph_in_name: ir_graph->getInputNames()) {
            if (contains(shared_input_ref_count, graph_in_name)) {
                int ref_count = shared_input_ref_count.at(graph_in_name);
                for (int i=0; i<ref_count; i++) {
                    std::stringstream ss;
                    ss << graph_in_name << "_cl" << i;
                    shared_val_cl_names[graph_in_name].push_back(ss.str());
                    graph_input_names.push_back(ss.str());
                }
            } else {
                graph_input_names.push_back(graph_in_name);
            }
        }

        std::unordered_map<std::string, IRValue> values;
        for (const auto& it: ir_graph->getValues()) {
            const IRValue& v = it.second;
            if (contains(shared_val_cl_names, v.getName())) {
                for (const auto& cl_name: shared_val_cl_names.at(v.getName())) {
                    IRValue cl_input(cl_name, v);
                    values[cl_name] = cl_input;
                }
            } else {
                values[it.first] = v;
            }
        }

        std::vector<IRNode> nodes;
        for (const auto& n: ir_graph->getNodes()) {
            std::vector<std::string> input_names;

            for (const auto& in_name: n.getInputNames()) {
                if (contains(shared_val_cl_names, in_name)) {
                    const auto& cl_names = shared_val_cl_names.at(in_name);
                    const std::string& cl_name = cl_names.at(ref_idx[in_name]);
                    input_names.push_back(cl_name);
                    ref_idx[in_name]++;
                } else {
                    input_names.push_back(in_name);
                }
            }
            IRNode new_node(n.getName(), input_names, n.getOutputNames());
            nodes.push_back(new_node);
        }

        // cloned input can also be an output
        std::vector<std::string> output_names;
        for (const auto& o: ir_graph->getOutputNames()) {
            if (contains(shared_val_cl_names, o)) {
                const auto& clone_names = shared_val_cl_names.at(o);
                assert(!clone_names.empty());
                output_names.push_back(clone_names.front());
            } else {
                output_names.push_back(o);
            }
        }

        return {std::make_shared<IRGraph>(ir_graph->getName(),
                nodes, values, graph_input_names, output_names),
                shared_val_cl_names};
    }

    std::vector<int64_t> getBatchDim(const IRType& type) {
        switch (type.getBaseType()) {
            case IRBaseType::SCALAR: return {};
            case IRBaseType::TENSOR:
                return type.getTensorDim();
            case IRBaseType::LIST: {
                IRListType list_type = type.getListType();
                if (list_type == IRListType::TENSOR || list_type == IRListType::GENERIC) {
                    const auto& elem_types = type.getCompoundTypes();
                    for (const auto& et: elem_types) {
                        std::vector<int64_t> ret = getBatchDim(et);
                        if (!ret.empty()) {
                            return ret;
                        }
                    }
                }
                return {};
            }
            case IRBaseType::TUPLE: {
                const auto& elem_types = type.getCompoundTypes();
                for (const auto& et: elem_types) {
                    std::vector<int64_t> ret = getBatchDim(et);
                    if (!ret.empty()) {
                        return ret;
                    }
                }
                return {};
            }
            case IRBaseType::STRING: return {};
            case IRBaseType::OPTIONAL: return {};
            case IRBaseType::NONE: return {};
        }
    }

    int64_t guessGraphBatchSize(const std::shared_ptr<IRGraph>& ir_graph) {
        std::stringstream ss;
        ss << *ir_graph;

        for (const auto& in_name: ir_graph->getInputNames()) {
            const auto& val = ir_graph->getValue(in_name);
            if (val.isBatch()) {
                const auto& dim = getBatchDim(val.getType());
                assert(!dim.empty());
                return dim.front();
            }
        }
        return 0;
    }

    size_t getOptMemSize(const std::shared_ptr<IRGraph> &ir_graph, int opt_param_factor, bool use_amp_master_param,
                         bool enable_zero, int zero_dist_num) {

        if (!enable_zero) {
            zero_dist_num = 1;
        }

        // This does not need to consider batch size
        size_t sum = 0;
        for (const auto& v: ir_graph->getValues()) {
            const auto& val = v.second;
            if (val.isParam()) {
                if (use_amp_master_param) {
                    if (val.getType().getTensorElemType() == IRTensorElemType::HALF) {
                        sum += val.getSizeInByte() // amp holds params
                                * 2 // FP32
                                / zero_dist_num; // Each rank holds only fragments of FP32 master params
                        sum += val.getSizeInByte()  // amp holds grads
                               * 2; // FP32
                               // We don't divide the size of gradients by zero_dist num
                               // because allreduce in FP32 needs buffer for the whole parameters
                        sum += val.getSizeInByte() // optimizer state
                               * 2 // FP32
                               * opt_param_factor
                               / zero_dist_num;
                    } else if (val.getType().getTensorElemType() == IRTensorElemType::FLOAT
                                || val.getType().getTensorElemType() == IRTensorElemType::BFLOAT16) {
                            // we have to keep memory for stashed gradients
                        sum += val.getSizeInByte() * opt_param_factor / zero_dist_num // optimizer state
                                + val.getSizeInByte(); // stashed gradients
                    } else {
                        throw std::runtime_error("Unexpected param type: " + toString(val.getType().getTensorElemType()));
                    }
                } else {
                    sum += val.getSizeInByte() * opt_param_factor / zero_dist_num;  // optimizer state
                }
            }
        }
        return sum;
   }

    size_t getAmpMasterParamSize(const std::shared_ptr<IRGraph>& ir_graph) {
        size_t sum = 0;
        for (const auto& v: ir_graph->getValues()) {
            const auto& val = v.second;
            if (val.isParam()) {
                if (val.getType().getTensorElemType() == IRTensorElemType::HALF) {
                    sum += val.getSizeInByte()
                           * 2; // FP32
                }
            }
        }
        return sum;
    }

    bool verifyNoDuplicatedOutputs(const std::shared_ptr<IRGraph>& g) {
        std::unordered_set<std::string> out_names;

        for (const auto& g_in: g->getInputNames()) {
            if (contains(out_names, g_in)) {
                spdlog::info("Duplicated graph input detected: {}", g_in);
                return false;
            }
            out_names.insert(g_in);
        }

        for (const auto& n: g->getNodes()) {
            for (const auto& out: n.getOutputNames()) {
                if (contains(out_names, out)) {
                    spdlog::info("Duplicated output value detected: {}", out);
                    return false;
                }
                out_names.insert(out);
            }
        }
        return true;
    }

    bool verifyNodeInputs(const std::shared_ptr<IRGraph>& g, bool show_msg) {
        std::unordered_set<std::string> val_names;

        for (const auto& g_in: g->getInputNames()) {
            val_names.insert(g_in);
        }

        for (const auto& n: g->getNodes()) {
            for (const auto& in: n.getInputNames()) {
                if (!contains(val_names, in)) {
                    if (show_msg) {
                        spdlog::info("Required input of {} not found: {}", n.getName(), in);
                    }
                    return false;
                }
            }
            for (const auto& out: n.getOutputNames()) {
                val_names.insert(out);
            }
        }
        return true;
    }


    std::unordered_set<std::string> findUnusedValue(const std::shared_ptr<IRGraph>& g) {
        std::unordered_set<std::string> ref_val_names;
        std::unordered_set<std::string> unused_vals;

        // graph outputs and node inputs are required
        for (const auto& g_out: g->getOutputNames()) {
            ref_val_names.insert(g_out);
        }
        for (const auto& n: g->getNodes()) {
            for (const auto &in: n.getInputNames()) {
                ref_val_names.insert(in);
            }
        }

        for (const auto& g_in: g->getInputNames()) {
            if (!contains(ref_val_names, g_in)) {
                unused_vals.insert(g_in);
            }
        }

        for (const auto& n: g->getNodes()) {
            for (const auto &out: n.getOutputNames()) {
                if (!contains(ref_val_names, out)) {
                    unused_vals.insert(out);
                }
            }
        }
        return unused_vals;
    }

    bool noUnusedValue(const std::shared_ptr<IRGraph>& g, bool show_msg) {
        const auto unused = findUnusedValue(g);

        if (show_msg) {
            for (const auto& v: unused) {
                spdlog::info("Value {} is not used", v);
            }
        }

        return unused.empty();
    }


    std::vector<IRNode> detectUnusedNodes(const std::shared_ptr<IRGraph>& g) {
        std::unordered_set<std::string> ref_val_names;
        std::vector<IRNode> unused_nodes;

        // graph outputs and node inputs are required
        for (const auto& g_out: g->getOutputNames()) {
            ref_val_names.insert(g_out);
        }
        for (const auto& n: g->getNodes()) {
            for (const auto &in: n.getInputNames()) {
                ref_val_names.insert(in);
            }
        }

        for (const auto& n: g->getNodes()) {
            bool unused = false;
            for (const auto &out: n.getOutputNames()) {
                if (!contains(ref_val_names, out)) {
                    unused = true;
                }
            }
            if (unused) {
                unused_nodes.push_back(n);
            }
        }

        return unused_nodes;
    }

    std::unordered_set<std::string> getRequiredValues(const std::shared_ptr<IRGraph>& g) {
        std::unordered_set<std::string> ref_val_names;

        // graph outputs and node inputs are required
        for (const auto &g_out: g->getOutputNames()) {
            ref_val_names.insert(g_out);
        }
        for (const auto &n: g->getNodes()) {
            for (const auto &in: n.getInputNames()) {
                ref_val_names.insert(in);
            }
        }
        return ref_val_names;
    }

    std::shared_ptr<IRGraph> removeUnusedNodes(const std::shared_ptr<IRGraph>& g) {

        std::shared_ptr<IRGraph> new_g = g;
        auto unused_nodes = detectUnusedNodes(new_g);
        while (!unused_nodes.empty()) {
            std::vector<IRNode> new_nodes;
            std::unordered_set<std::string> unused_ids;
            for (const auto& n: unused_nodes) {
                unused_ids.insert(n.getId());
            }

            for (const auto& n: new_g->getNodes()) {
                if (!contains(unused_ids, n.getId())) {
                    new_nodes.push_back(n);
                }
            }
            new_g = std::make_shared<IRGraph>(g->getName(), new_nodes, g->getValues(),
                    g->getInputNames(), g->getOutputNames());
            unused_nodes = detectUnusedNodes(new_g);
        }

        const auto required_vals = getRequiredValues(new_g);

        std::unordered_map<std::string, IRValue> new_vals;
        for (const auto& it: new_g->getValues()) {
            if (contains(required_vals, it.first)) {
                new_vals[it.first] = it.second;
            }
        }

        std::vector<std::string> new_inputs, new_outputs;
        for (const auto& in_name: new_g->getInputNames()) {
            if (contains(required_vals, in_name)) {
                new_inputs.push_back(in_name);
            }
        }
        for (const auto& out_name: new_g->getOutputNames()) {
            if (contains(required_vals, out_name)) {
                new_outputs.push_back(out_name);
            }
        }

        return std::make_shared<IRGraph>(g->getName(), new_g->getNodes(), new_vals, new_inputs, new_outputs);
    }

    size_t calcCommBufSize(const std::shared_ptr<IRGraph>& g, int pipeline_num) {
        size_t input_size = 0;
        for (const auto& in_name: g->getInputNames()) {
            const IRValue& in_val = g->getValue(in_name);
            if (!in_val.isParam()) {
                input_size += in_val.getSizeInByte();
            }
        }

        // We do not need to keep clone inputs for each split
        // When performing gradient checkpointing, we run a backward pass soon after the forward pass finishes.
        size_t clone_input_size = 0;
        const auto cl = cloneSharedInputs(g);
        const auto& shared_val_cl_names = cl.second;
        for (const auto& it: shared_val_cl_names) {
            const IRValue& in_val = g->getValue(it.first);
            clone_input_size += in_val.getSizeInByte() * it.second.size();
        }
        // Clone inputs have their gradients.
        // The gradients are removed after they are accumulated on another tensor,
        // but we still have to consider the sizes of the gradients of clone inputs.
        clone_input_size *= 2;

        // The memory for outputs is allocated by PyTorch, and the size is included in the memory requirement measured by GraphProfiler.
        // However, we still need the output size to receiving gradients of outputs.
        size_t output_size = 0;
        for (const auto& out_name: g->getOutputNames()) {
            const IRValue& out_val = g->getValue(out_name);
            output_size += out_val.getSizeInByte();
        }

        size_t ingrad_size = 0;
        for (const auto& in_name: g->getInputNames()) {
            const IRValue& in_val = g->getValue(in_name);
            if (!in_val.isParam()) {
                const auto& type = in_val.getType();
                if (passedForBackward(type)) {
                    ingrad_size += in_val.getSizeInByte();
                }
            }
        }

        return (input_size + output_size + ingrad_size) * pipeline_num + clone_input_size;
    }



}
