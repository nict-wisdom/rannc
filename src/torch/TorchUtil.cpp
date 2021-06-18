//
// Created by Masahiro Tanaka on 2019-07-05.
//

#include <cuda_runtime_api.h>

#include <torch/csrc/jit/runtime/profiling_record.h>

#include <mpi.h>
#include <comm/SCommCommon.h>
#include <cuda/CudaUtil.h>

#include "Common.h"
#include "ConfiguredTorch.h"
#include "TorchUtil.h"
#include "graph/ir.h"

#include <c10/cuda/CUDACachingAllocator.h>

#include <torch/torch.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/CUDAGeneratorImpl.h>

namespace py = pybind11;

using c10::cuda::CUDACachingAllocator::DeviceStats;

namespace rannc {

    std::string tensorToString(const at::Tensor &t, size_t max_elem, size_t max_str_len) {
        auto t_cpu = t.flatten().slice(0, 0, std::min((int64_t) max_elem, t.numel()))
                .to(c10::Device(c10::kCPU), false, true);

        size_t size = std::min(max_elem, productDim(t.sizes()));

        if (t.scalar_type() == c10::ScalarType::Half || t.scalar_type() == c10::ScalarType::BFloat16) {
            auto float_ten = t_cpu.to(c10::ScalarType::Float, false, true).contiguous();
            return tensorPtrToString((float*)float_ten.data_ptr(), size, max_str_len);
        } else if (t.scalar_type() == c10::ScalarType::Float) {
            return tensorPtrToString((float*)t_cpu.data_ptr(), size, max_str_len);
        } else if (t.scalar_type() == c10::ScalarType::Double) {
            return tensorPtrToString((double*)t_cpu.data_ptr(), size, max_str_len);
        } else if (t.scalar_type() == c10::ScalarType::Int) {
            int* ptr = static_cast<int *>(t_cpu.data_ptr());
            return tensorPtrToString(ptr, size, max_str_len);
        } else if (t.scalar_type() == c10::ScalarType::Long) {
            long* ptr = static_cast<long *>(t_cpu.data_ptr());
            return tensorPtrToString(ptr, size, max_str_len);
        } else if (t.scalar_type() == c10::ScalarType::Byte) {
            unsigned char* ptr = static_cast<unsigned char *>(t_cpu.data_ptr());
            std::vector<unsigned short> vec;
            vec.reserve(size);
            for (size_t i=0; i<size; i++) {
                vec.push_back(*ptr);
                ptr++;
            }
            return tensorPtrToString(&vec[0], size, max_str_len);
        } else if (t.scalar_type() == c10::ScalarType::Bool) {
            bool* ptr = static_cast<bool *>(t_cpu.data_ptr());
            std::vector<int> vec;
            vec.reserve(size);
            for (size_t i=0; i<size; i++) {
                vec.push_back(*ptr);
                ptr++;
            }
            return tensorPtrToString(&vec[0], size, max_str_len);
        }
        std::stringstream ss;
        ss << "Failed to convert tensor to string. Invalid type of tensor: " << toString(t.scalar_type());
        throw std::invalid_argument(ss.str());
    }

    std::string tensorPtrToString(void* ptr, size_t size, c10::ScalarType datatype, size_t max_elem, size_t max_str_len) {

        int64_t elem_size = std::min((size_t) max_elem, size);

        if (datatype == c10::ScalarType::Long) {
            return tensorPtrToString(static_cast<long*>(ptr), elem_size, max_str_len);
        } else if (datatype == c10::ScalarType::Int) {
            return tensorPtrToString(static_cast<int*>(ptr), elem_size, max_str_len);
        } else if (datatype == c10::ScalarType::Double) {
            return tensorPtrToString(static_cast<double*>(ptr), elem_size, max_str_len);
        } else if (datatype == c10::ScalarType::Float) {
            return tensorPtrToString(static_cast<float*>(ptr), elem_size, max_str_len);
        } else if (datatype == c10::ScalarType::Half || datatype == c10::ScalarType::BFloat16) {
            const auto ten = torch::from_blob(ptr, {(int64_t)elem_size}, datatype);
            auto float_ten = ten.to(c10::ScalarType::Float, false, true).contiguous();
            return tensorPtrToString((float*)float_ten.data_ptr(), elem_size, max_str_len);
        }
        std::stringstream ss;
        ss << "Failed to convert tensor ptr to string. Invalid type of tensor: " << toString(datatype);
        throw std::invalid_argument(ss.str());
    }

    std::vector<int64_t> getTensorDim(const at::Tensor &t) {
        std::vector<int64_t> dim;
        for (auto d: t.sizes()) {
            dim.push_back(d);
        }
        return dim;
    }

    size_t getTensorElemCount(const at::Tensor &t) {
        return productDim(getTensorDim(t));
    }

    std::vector<int64_t> intArrayRefToVec(c10::IntArrayRef list) {
        std::vector<int64_t> vec;
        for (auto v: list) {
            vec.push_back(v);
        }
        return vec;
    }

    std::string tensorToDimString(const at::Tensor &t) {
        return join_as_str(intArrayRefToVec(t.sizes()));
    }

    torch::jit::IValue processTensorInIValue(const torch::jit::IValue &iv,
                                             const std::function<at::Tensor(at::Tensor)> &f) {

        if (iv.isTensor()) {
            return f(iv.toTensor());
        } else if (iv.isTensorList()) {
            auto l = iv.toTensorVector();

            std::vector<at::Tensor> dev_l;
            dev_l.reserve(l.size());
            for (const auto &t: l) {
                dev_l.push_back(f(t));
            }

            return c10::List<at::Tensor>(dev_l);
        } else if (iv.isList()) {
            auto l = iv.toListRef();
            auto source = std::move(iv).toList();
            c10::impl::GenericList ret_list(source.elementType());
            ret_list.reserve(l.size());
            for (const auto &elem: l) {
                ret_list.push_back(processTensorInIValue(elem, f));
            }
            return ret_list;
        } else if (iv.isTuple()) {
            auto l = iv.toTuple()->elements();

            std::vector<torch::jit::IValue> dev_l;
            dev_l.reserve(l.size());
            for (const auto &elem: l) {
                dev_l.push_back(processTensorInIValue(elem, f));
            }
            return c10::ivalue::Tuple::create(dev_l);
        }
        return iv;
    }

    torch::jit::IValue to(const torch::jit::IValue &iv, const torch::Device &device,
                          bool detach) {
        return processTensorInIValue(iv, [&device, detach](at::Tensor t) {
            auto ret = t.to(device, false, true);
            if (detach) {
                auto dret = ret.detach();
                dret.set_requires_grad(t.requires_grad());
                return dret;
            }
            return ret;
        });
    }

    torch::jit::IValue detach(const torch::jit::IValue &iv) {
        return processTensorInIValue(iv, [](at::Tensor t) {
            return t.detach();
        });
    }

    torch::jit::IValue toCPU(const torch::jit::IValue& iv, bool detach) {
        return to(iv, torch::Device(torch::kCPU), detach);
    }

    torch::jit::IValue toCUDAIfAvailable(const torch::jit::IValue& iv, bool detach) {
        if (torch::cuda::is_available()) {
            return to(iv, torch::Device(torch::kCUDA), detach);
        }
        return iv;
    }

    at::Tensor toCUDAIfAvailable(const at::Tensor& t, bool detach) {
        auto ret = t;
        if (torch::cuda::is_available()) {
            ret = t.to(torch::Device(torch::kCUDA), false, true);
        }
        if (detach) {
            auto dret = ret.detach();
            dret.set_requires_grad(t.requires_grad());
            return dret;
        }
        return ret;
    }

    void toCUDAInPlace(at::Tensor& t) {
        if (t.is_cuda()) {
            return;
        }

        torch::NoGradGuard no_grad;
        t.set_data(t.to(torch::Device(torch::kCUDA), false, true));
    }

    void toCPUInPlace(at::Tensor& t) {
        if (!t.is_cuda()) {
            return;
        }

        torch::NoGradGuard no_grad;
        t.set_data(t.to(torch::Device(torch::kCPU), false, true));
    }

    torch::jit::IValue contiguous(const torch::jit::IValue &iv) {
        return processTensorInIValue(iv, [](at::Tensor t) {
            return t.contiguous();
        });
    }

    torch::jit::IValue setRequiresGrad(const torch::jit::IValue &iv, bool requires_grad) {
        return processTensorInIValue(iv, [requires_grad](at::Tensor t) {
            t.set_requires_grad(requires_grad);
            return t;
        });
    }

    void doFindPathsToTensorInIValue(const torch::jit::IValue &ivalue, PathInIValue &path,
                                     std::vector<PathInIValue> &found) {
        if (ivalue.isTensor()) {
            found.push_back(path);
        } else if (ivalue.isTensorList()) {
            int index = 0;
            for (const auto& t: ivalue.toTensorVector()) {
                auto path_elem = path;
                path_elem.emplace_back(StepTypeInIValue::LIST, index);
                doFindPathsToTensorInIValue(t, path_elem, found);
                index++;
            }
        } else if (ivalue.isList()) {
            int index = 0;
            for (const auto& e: ivalue.toListRef()) {
                auto path_elem = path;
                path_elem.emplace_back(StepTypeInIValue::LIST, index);
                doFindPathsToTensorInIValue(e, path_elem, found);
                index++;
            }
        } else if (ivalue.isTuple()) {
            const auto &elem_tuple = ivalue.toTuple()->elements();
            for (size_t i = 0; i < elem_tuple.size(); i++) {
                auto path_elem = path;
                path_elem.emplace_back(StepTypeInIValue::TUPLE, i);
                doFindPathsToTensorInIValue(elem_tuple.at(i), path_elem, found);
            }
        }
    }

    std::vector<PathInIValue> findPathsToTensorInIValue(const torch::jit::IValue &ivalue) {
        std::vector<PathInIValue> found;
        PathInIValue path;
        doFindPathsToTensorInIValue(ivalue, path, found);
        return found;
    }

    torch::jit::IValue
    doGetElemInIValue(const torch::jit::IValue &ivalue, std::vector<StepInIValue>::const_iterator iter,
                      const std::vector<StepInIValue>::const_iterator &end_iter) {
        if (iter == end_iter) return ivalue;
        switch (iter->type) {
            case StepTypeInIValue::LIST:
                if (ivalue.isTensorList()) {
                    const auto &tensor_elems = ivalue.toTensorVector();
//                    const auto &tensor_elems = ivalue.toTensorList()->elements();
                    return tensor_elems.at(iter->index);
                } else if (ivalue.isList()) {
                    const auto &ivalue_elems = ivalue.toListRef();
                    return doGetElemInIValue(ivalue_elems.at(iter->index), iter + 1, end_iter);
                }
            case StepTypeInIValue::TUPLE: {
                const auto &ivalue_elems = ivalue.toTuple()->elements();
                return doGetElemInIValue(ivalue_elems.at(iter->index), iter + 1, end_iter);
            }
        }
    }

    torch::jit::IValue getElemInIValue(const torch::jit::IValue &ivalue, const PathInIValue &path) {
        if (path.begin() == path.end()) return ivalue;
        return doGetElemInIValue(ivalue, path.cbegin(), path.cend());
    }

    std::vector<IValueLocation> getKeys(const IValueMap &map) {
        std::vector<IValueLocation> locs;
        locs.reserve(map.size());
        for (const auto& it: map) {
            locs.push_back(it.first);
        }
        return locs;
    }

    IRTensorElemType toTensorElemType(const c10::ScalarType &scalarType) {
        switch (scalarType) {
            case c10::ScalarType::Int:
                return IRTensorElemType::INT;
            case c10::ScalarType::Long:
                return IRTensorElemType::LONG;
            case c10::ScalarType::Bool:
                return IRTensorElemType::BOOL;
            case c10::ScalarType::Float:
                return IRTensorElemType::FLOAT;
            case c10::ScalarType::Half:
                return IRTensorElemType::HALF;
            case c10::ScalarType::BFloat16:
                return IRTensorElemType::BFLOAT16;
            case c10::ScalarType::Double:
                return IRTensorElemType::DOUBLE;
            case c10::ScalarType::Byte:
                break;
            case c10::ScalarType::Char:
                break;
            case c10::ScalarType::Short:
                break;
            case c10::ScalarType::ComplexHalf:
                break;
            case c10::ScalarType::ComplexFloat:
                break;
            case c10::ScalarType::ComplexDouble:
                break;
            case c10::ScalarType::QInt8:
                break;
            case c10::ScalarType::Undefined:
                break;
            case c10::ScalarType::NumOptions:
                break;
        }
        std::stringstream ss;
        ss << "Failed to convert c10::ScalarType to IRTensorElemType: " << c10::toString(scalarType);
        throw std::invalid_argument(ss.str());
    }

    IRType toIRType(const torch::jit::TypePtr &type) {
        switch (type->kind()) {
            case c10::TypeKind::IntType:
                return IRType::createScalarType(IRScalarType::INT);
            case c10::TypeKind::FloatType:
                return IRType::createScalarType(IRScalarType::FLOAT);
            case c10::TypeKind::NumberType:
                return IRType::createScalarType(IRScalarType::NUMBER);
            case c10::TypeKind::BoolType:
                return IRType::createScalarType(IRScalarType::BOOL);
            case c10::TypeKind::TensorType: {
                const auto tt = type->expect<torch::jit::TensorType>();
                const auto st = tt->scalarType();
                if (st) {
                    const auto elem_type = toTensorElemType(st.value());
                    return IRType::createUnknownShapeTensorType(elem_type);
                }
                return IRType::createUnknownTensorType();
            }

            case c10::TypeKind::ListType: {
                const auto lt = type->expect<torch::jit::ListType>();
                const auto elem_type_ptr = lt->getElementType();
                const auto ir_elem_type = toIRType(lt->getElementType());

                if (ir_elem_type.getBaseType() == IRBaseType::SCALAR) {
                    if (ir_elem_type.getScalarType() == IRScalarType::INT) {
                        return IRType::createListType(IRListType::INT);
                    } else if (ir_elem_type.getScalarType() == IRScalarType::FLOAT) {
                        return IRType::createListType(IRListType::FLOAT);
                    } else if (ir_elem_type.getScalarType() == IRScalarType::BOOL) {
                        return IRType::createListType(IRListType::BOOL);
                    }
                } else if (ir_elem_type.getBaseType() == IRBaseType::TENSOR) {
                    std::vector<IRType> contained_ir_types;
                    contained_ir_types.push_back(ir_elem_type);
                    return IRType::createTensorListType(contained_ir_types);
                } else {
                    std::vector<IRType> contained_ir_types;
                    contained_ir_types.push_back(ir_elem_type);
                    return IRType::createListType(contained_ir_types);
                }
            }

            case c10::TypeKind::TupleType: {
                const auto tt = type->expect<torch::jit::TupleType>();
                const auto &contained_types = tt->containedTypes();
                std::vector<IRType> contained_ir_types;
                for (const auto &ct: contained_types) {
                    contained_ir_types.push_back(toIRType(ct));
                }
                return IRType::createTupleType(contained_ir_types);
            }

            case c10::TypeKind::OptionalType: {
                const auto ot = type->expect<torch::jit::OptionalType>();
                return IRType::createOptionalType(toIRType(ot->getElementType()));
            }

            case c10::TypeKind::DeviceObjType: {
                return IRType::createScalarType(IRScalarType::DEVICE);
            }

            case c10::TypeKind::FunctionType: {
                c10::FunctionTypePtr  func_type = type->cast<c10::FunctionType>();
                return IRType::createFunctionType(func_type);
            }

            case c10::TypeKind::StringType: {
                return IRType::createStringType();
            }

            case c10::TypeKind::NoneType: {
                return IRType::createNoneType();
            }

            case c10::TypeKind::DictType:
                break;
            case c10::TypeKind::FutureType:
                break;
            case c10::TypeKind::GeneratorType:
                break;
            case c10::TypeKind::VarType:
                break;
            case c10::TypeKind::ClassType:
                break;
        }
        std::stringstream ss;
        ss << "Failed to convert torch type to IRType. Unsupported c10::TypeKind: "
           << c10::typeKindToString(type->kind());
        throw std::invalid_argument(ss.str());
    }

    IRType toIRType(const torch::jit::IValue &ivalue) {

        if (ivalue.isInt()) {
            return IRType::createScalarType(IRScalarType::INT);
        } else if (ivalue.isBool()) {
            return IRType::createScalarType(IRScalarType::BOOL);
        } else if (ivalue.isDouble()) {
            return IRType::createScalarType(IRScalarType::FLOAT);
        } else if (ivalue.isIntList()) {
            return IRType::createListType(IRListType::INT, ivalue.toIntList().size());
        } else if (ivalue.isDoubleList()) {
            return IRType::createListType(IRListType::FLOAT, ivalue.toDoubleList().size());
        } else if (ivalue.isBoolList()) {
            return IRType::createListType(IRListType::BOOL, ivalue.toBoolList().size());
        } else if (ivalue.isTensor()) {
            auto t = ivalue.toTensor();
            if (t.defined()) {
                return IRType::createTensorType(toTensorElemType(t.scalar_type()), getTensorDim(t), t.requires_grad());
            } else {
                return IRType::createTensorType(IRTensorElemType::UNDEF, {}, false);
            }
        } else if (ivalue.isTensorList()) {
            std::vector<IRType> elem_types;
            for (const auto &t: ivalue.toTensorVector()) {
                elem_types.push_back(toIRType(t));
            }
            return IRType::createTensorListType(elem_types);
        } else if (ivalue.isList()) {
            std::vector<IRType> elem_types;
            for (const auto &t: ivalue.toListRef()) {
                elem_types.push_back(toIRType(t));
            }

            if (elem_types.empty()) {
                return  IRType::createListType(elem_types);
            }

            if (elem_types.front().getBaseType() == IRBaseType::TENSOR) {
                return IRType::createTensorListType(elem_types);
            }

            return IRType::createListType(elem_types);
        } else if (ivalue.isTuple()) {
            std::vector<IRType> elem_types;
            for (const auto &t: ivalue.toTuple()->elements()) {
                elem_types.push_back(toIRType(t));
            }
            return IRType::createTupleType(elem_types);
        } else if (ivalue.isDevice()) {
            return IRType::createScalarType(IRScalarType::DEVICE);
        } else if (ivalue.isString()) {
            return IRType::createStringType();
        } else if (ivalue.isNone()) {
            return IRType::createNoneType();
        }
        throw std::invalid_argument("Unknown ivalue type.");
    }

    IRValue toIRValue(torch::jit::Value *value) {
        const torch::jit::TypePtr &type = value->type();
        return IRValue(value->debugName(), toIRType(type));
    }

    torch::jit::TypePtr fromIRScalarType(const IRScalarType &ir_scalar_type) {
        switch (ir_scalar_type) {
            case IRScalarType::NONE:
                return torch::jit::NoneType::get();
            case IRScalarType::INT:
                return torch::jit::IntType::get();
            case IRScalarType::FLOAT:
                return torch::jit::FloatType::get();
            case IRScalarType::NUMBER:
                return torch::jit::NumberType::get();
            case IRScalarType::BOOL:
                return torch::jit::BoolType::get();
            case IRScalarType::DEVICE:
                return torch::jit::DeviceObjType::get();
        }
    }

    at::ScalarType fromIRTensorElemTypeToScalarType(IRTensorElemType ir_tensor_elem) {
        switch (ir_tensor_elem) {
            case IRTensorElemType::INT:
                return at::ScalarType::Int;
            case IRTensorElemType::DOUBLE:
                return at::ScalarType::Double;
            case IRTensorElemType::FLOAT:
                return at::ScalarType::Float;
            case IRTensorElemType::HALF:
                return at::ScalarType::Half;
            case IRTensorElemType::BFLOAT16:
                return at::ScalarType::BFloat16;
            case IRTensorElemType::LONG:
                return at::ScalarType::Long;
            case IRTensorElemType::BOOL:
                return at::ScalarType::Bool;
            case IRTensorElemType::UNDEF:
                break;
        }
        throw std::invalid_argument("The given tensor element type was None.");
    }

    at::ScalarType fromIRListTypeToScalarType(IRListType list_type) {
        switch (list_type) {
            case IRListType::INT:
                return at::ScalarType::Int;
            case IRListType::FLOAT:
                return at::ScalarType::Int;
            case IRListType::BOOL:
                return at::ScalarType::Int;
            default:
                break;
        }
        throw std::invalid_argument("The list type is not scalar: " + toString(list_type));
    }

    IRScalarType fromIRListTypeToIRScalarType(IRListType list_type) {
        switch (list_type) {
            case IRListType::INT:
                return IRScalarType::INT;
            case IRListType::FLOAT:
                return IRScalarType::FLOAT;
            case IRListType::BOOL:
                return IRScalarType::BOOL;
            default:
                break;
        }
        throw std::invalid_argument("The list type is not scalar: " + toString(list_type));
    }

    torch::jit::IValue toTensorListIfElemsAreTensors(const torch::jit::IValue &ivalue) {
        assert(ivalue.isList());

        bool tensor_elem = false;
        std::vector<torch::jit::IValue> elems;
        for (const auto &t: ivalue.toListRef()) {
            if (t.isTensor()) {
                tensor_elem = true;
            }
            elems.push_back(t);
        }
        if (tensor_elem) {
            c10::List<at::Tensor> l;
            for (const auto& t: elems) {
                l.push_back(t.toTensor());
            }
            return l;
        }
        return ivalue;
    }

    torch::jit::IValue transformTensorsInIValueWithPath(const torch::jit::IValue &ivalue,
                                                        const IValueLocation &loc, const std::function<at::Tensor(const at::Tensor &, const IValueLocation &loc)> &f) {

        if (ivalue.isTensor()) {
            return f(ivalue.toTensor(), loc);
        } else if (ivalue.isTensorList()) {
            std::vector<at::Tensor> tensors;
            int idx=0;
            for (const auto &t: ivalue.toTensorVector()) {
                tensors.push_back(f(t, createListElem(loc, idx++)));
            }
            return tensors;
        } else if (ivalue.isList()) {
            auto source = std::move(ivalue).toList();
            c10::impl::GenericList list(source.elementType());
            int idx=0;
            for (const auto &e: ivalue.toListRef()) {
                list.push_back(transformTensorsInIValueWithPath(e, createListElem(loc, idx++), f));
            }
            return list;
        } else if (ivalue.isTuple()) {
            std::vector<torch::jit::IValue> elems;
            int idx=0;
            for (const auto &e: ivalue.toTuple()->elements()) {
                elems.push_back(transformTensorsInIValueWithPath(e, createTupleElem(loc, idx++), f));
            }
            return c10::ivalue::Tuple::create(elems);
        }
        return ivalue;
    }


    torch::jit::IValue cloneTensorsInIValueWithBuffer(const torch::jit::IValue &ivalue,
                                                      const std::string& key,
                                                      BufferTensorCache& buffer) {
        return transformTensorsInIValueWithPath(ivalue, key, [&buffer](const at::Tensor &t, const IValueLocation &loc) {
            auto buf = buffer.get(toString(loc), toIRType(t));
            {
                torch::NoGradGuard no_grad;
                buf.copy_(t, false);
            }
            buf.set_requires_grad(t.requires_grad());
            return buf;
        });
    }

    torch::jit::IValue transformTensorsInIValue(const torch::jit::IValue &ivalue,
                                                const std::function<at::Tensor(const at::Tensor &)> &f) {
        IValueLocation loc("NA");
        return transformTensorsInIValueWithPath(ivalue, loc, [&f](const at::Tensor &t, const IValueLocation &loc) {
            return f(t);
        });
    }

    torch::jit::IValue transformTensorsInIValueWithPath(const torch::jit::IValue &ivalue, const std::string& name,
                                                const std::function<at::Tensor(const at::Tensor &t, const IValueLocation &loc)> &f) {
        IValueLocation loc(name);
        return transformTensorsInIValueWithPath(ivalue, loc, f);
    }

    at::Tensor padTensor(const at::Tensor &tensor, int batch_size, bool zero) {
        const auto &dim = getTensorDim(tensor);
        assert(!dim.empty());

        const int64_t original_size = dim.front();
        int64_t current_size = 0;
        std::vector<at::Tensor> targets;

        while (current_size < batch_size) {
            int64_t pad_size = std::min(original_size, batch_size - current_size);
            at::Tensor pad;
            if (original_size == pad_size) {
                pad = tensor;
            } else {
                pad = tensor.slice(0, 0, pad_size);
            }
            if (zero) {
                at::NoGradGuard no_grad;
                pad.zero_();
            }
            targets.push_back(pad);
            current_size += pad_size;
        }
        return torch::cat(targets).contiguous().detach();
    }

    at::Tensor unpadTensor(const at::Tensor &tensor, int batch_size) {
        const auto &dim = getTensorDim(tensor);
        assert(!dim.empty());

        const int64_t original_size = dim.front();

        at::Tensor dummy_result = tensor;
        if (original_size > batch_size) {
            dummy_result = tensor.slice(0, 0, batch_size);
        }
        return dummy_result.detach();
    }

    // This aligns the first dimension of the arg to *batch_size*
    // and does not care if it is actually a batch or the first dimension represents the actual batch size.
    at::Tensor alignTensor(const at::Tensor &tensor, int batch_size, bool zero_pad) {
        const auto &dim = getTensorDim(tensor);
        assert(!dim.empty());

        const int64_t original_size = dim.front();
        if (original_size > batch_size) {
            return unpadTensor(tensor, batch_size);
        } else if (original_size < batch_size) {
            return padTensor(tensor, batch_size, zero_pad);
        }
        return tensor;
    }

    torch::jit::IValue padTensorsInIValue(const torch::jit::IValue &ivalue, int batch_size, bool zero) {
        const std::function<at::Tensor(const at::Tensor &)> f = [batch_size, zero](const at::Tensor &t) {
            return padTensor(t, batch_size, zero);
        };
        return transformTensorsInIValue(ivalue, f);
    }

    torch::jit::IValue unpadTensorsInIValue(const torch::jit::IValue &ivalue, int batch_size) {
        const std::function<at::Tensor(const at::Tensor &)> f = [batch_size](const at::Tensor &t) {
            return unpadTensor(t, batch_size);
        };
        return transformTensorsInIValue(ivalue, f);
    }

    torch::jit::IValue splitBatchInIValue(const torch::jit::IValue &ivalue, size_t current_batch_size, size_t total_batch_size,
                                          size_t replica_num, bool zero_pad) {
        const std::function<at::Tensor(const at::Tensor &)> f =
                [current_batch_size, total_batch_size, replica_num, zero_pad](const at::Tensor &t) {
            std::unordered_set<int> dummy_ranks;
            for (int i=0; i < replica_num; i++) {
                dummy_ranks.insert(i);
            }
            std::unordered_map<int, std::vector<int64_t>> dist_dim = calcDistBatchDims(
                    total_batch_size, {(int64_t) total_batch_size}, dummy_ranks);
            assert(contains(dist_dim, 0));
            const auto& dim = dist_dim.at(0);
            assert(!dim.empty());

            // Actually the first dim is not always a batch size
            const auto& orig_dim = getTensorDim(t);
            if (orig_dim.empty()) {
                return t;
            }
            int64_t size_per_example = orig_dim.front() / current_batch_size;
            return alignTensor(t, size_per_example * dim.front(), zero_pad);
        };
        return transformTensorsInIValue(ivalue, f);
    }

    torch::jit::IValue scaleBatchInIValue(const torch::jit::IValue &ivalue, size_t batch_size,
                                          size_t tgt_batch_size, bool zero_pad) {
        const std::function<at::Tensor(const at::Tensor &)> f =
                [batch_size, tgt_batch_size, zero_pad](const at::Tensor &t) {
                    // Actually the first dim is not always a batch size
                    const auto& orig_dim = getTensorDim(t);
                    if (orig_dim.empty()) {
                        return t;
                    }
                    int64_t size_per_example = orig_dim.front() / batch_size;
                    return alignTensor(t, size_per_example * tgt_batch_size, zero_pad);
                };
        return transformTensorsInIValue(ivalue, f);
    }

    torch::jit::IValue replicateTensorsInIValue(const torch::jit::IValue &ivalue, size_t replica_num) {
        const std::function<at::Tensor(const at::Tensor &)> f =
                [replica_num](const at::Tensor &t) {
                    const auto& orig_dim = getTensorDim(t);
                    if (orig_dim.empty()) {
                        return t;
                    }

                    // Actually the first dim is not always a batch size
                    return alignTensor(t, orig_dim.front() * replica_num, false);
                };
        return transformTensorsInIValue(ivalue, f);
    }

    torch::jit::IValue alignTensorsInIValue(const torch::jit::IValue &ivalue,
                                            size_t target_batch_size, bool zero_pad) {
        const std::function<at::Tensor(const at::Tensor &)> f =
                [target_batch_size, zero_pad](const at::Tensor &t) {
                    if (getTensorDim(t).empty()) {
                        return t;
                    }
                    return alignTensor(t, target_batch_size, zero_pad);
                };
        return transformTensorsInIValue(ivalue, f);
    }

    torch::jit::IValue scaleTensorsInIValue(const torch::jit::IValue &ivalue, double scale) {
        const std::function<at::Tensor(const at::Tensor &)> f = [scale](const at::Tensor &t) {
            return t.mul(scale).detach();
        };
        return transformTensorsInIValue(ivalue, f);
    }

    at::Tensor toFloatIfHalf(const at::Tensor &tensor) {
        if (tensor.scalar_type() == at::ScalarType::Half) {
            return tensor.to(torch::kFloat32);
        }
        return tensor;
    }

    torch::jit::IValue toFloatTensorsInIValue(const torch::jit::IValue &ivalue) {
        return transformTensorsInIValue(ivalue, toFloatIfHalf);
    }

    at::Tensor sliceDistBatchTensor(const at::Tensor& tensor, int index, int64_t batch_size, int num) {
        const auto& dim = getTensorDim(tensor);
        std::vector<int> dummy_ranks;
        dummy_ranks.reserve(num);
        for (int i=0; i<num; i++) {
            dummy_ranks.push_back(i);
        }
        const auto dist_dims = calcDistBatchDims(batch_size, dim, vectorToSet(dummy_ranks));

        int offset = 0;
        int i;
        for (i=0; i<index; i++) {
            assert(!dist_dims.empty());
            int bs = dist_dims.at(i).front();
            offset += bs;
        }
        int size = dist_dims.at(i).front();
        return tensor.slice(0, offset, offset+size);
    }

    torch::jit::IValue sliceDistBatchTensorsInIValue(const torch::jit::IValue &ivalue, int index, int64_t batch_size,
            int num) {
        std::function<at::Tensor(const at::Tensor&)> f =
                [index, num, batch_size](const at::Tensor& t) {
                    return sliceDistBatchTensor(t, index, batch_size, num);
                };
        return transformTensorsInIValue(ivalue, f);
    }

    torch::jit::IValue weightDistLossTensorsInIValue(const torch::jit::IValue &ivalue, int index, int64_t batch_size,
                                                     int num) {
        std::function<at::Tensor(const at::Tensor&)> f =
                [index, num, batch_size](const at::Tensor& t) {
                    std::vector<int> dummy_ranks;
                    dummy_ranks.reserve(num);
                    for (int i=0; i<num; i++) {
                        dummy_ranks.push_back(i);
                    }

                    auto loss = getDpRatio(batch_size, dummy_ranks, index) * t;
//                    spdlog::info("weightDistLossTensorsInIValue v={} v'={} ratio={} (rank={} #ranks={})", tensorToString(t),
//                            tensorToString(loss), getDpRatio(batch_size, dummy_ranks, index),
//                            index, num);
                    return loss;
                };
        return transformTensorsInIValue(ivalue, f);
    }

    torch::jit::IValue sliceOrWeightTensorsInIValue(const torch::jit::IValue &ivalue,
                                                    const std::vector<int64_t>& batch_sizes, int index) {
        std::function<at::Tensor(const at::Tensor&)> f =
                [&batch_sizes, index](const at::Tensor& t) {
                    assert(index < batch_sizes.size());

                    const auto& dim = getTensorDim(t);
                    if (dim.empty()) {
                        int64_t sum = 0;
                        for (const auto& bs: batch_sizes) {
                            sum += bs;
                        }
//                        spdlog::info("@sliceOrWeightTensorsInIValue {} split={} scale={} loss={}",
//                                     toString(toIRType(t)),
//                                     index, (batch_sizes.at(index) / (double) sum),
//                                     tensorToString((batch_sizes.at(index) / (double) sum) * t));
                        return (batch_sizes.at(index) / (double) sum) * t;
                    }

                    int64_t offset = 0;
                    for (size_t i=0; i<index; i++) {
                        offset += batch_sizes.at(i);
                    }
                    return t.slice(0, offset, offset+batch_sizes.at(index));
                };
        return transformTensorsInIValue(ivalue, f);
    }

    torch::jit::IValue cloneTensorsInIValue(const torch::jit::IValue &ivalue, const c10::Device& dev) {
        std::function<at::Tensor(const at::Tensor&)> f =
                [dev](const at::Tensor& t) {

                    at::TensorOptions options;
                    options = options.device(dev)
                        .dtype(t.scalar_type());

                    auto t_copy = torch::empty(t.sizes(), options);
                    t_copy.copy_(t, false);
                    return t_copy;
                };
        return transformTensorsInIValue(ivalue, f);
    }

    torch::jit::IValue cloneTensorsInIValue(const torch::jit::IValue &ivalue) {
        std::function<at::Tensor(const at::Tensor&)> f =
                [](const at::Tensor& t) {
                    at::Tensor t_copy = torch::empty_like(t);
                    {
                        torch::NoGradGuard no_grad;
                        t_copy.copy_(t, false);
                    }
                    t_copy.set_requires_grad(t.requires_grad());
                    return t_copy;
                };
        return transformTensorsInIValue(ivalue, f);
    }

    torch::jit::IValue aggregateTensorsInIValues(const std::vector<torch::jit::IValue> &ivalues,
                                                 const std::function<at::Tensor(const std::vector<at::Tensor>&)>& f) {

        const auto& ivalue = ivalues.front();

        if (ivalue.isTensor()) {
            std::vector<at::Tensor> tensors;
            tensors.reserve(ivalues.size());
            for (const auto& iv: ivalues) {
                tensors.push_back(iv.toTensor());
            }
            return f(tensors);
        } else if (ivalue.isTensorList()) {
            size_t list_size = ivalue.toTensorVector().size();
            std::vector<at::Tensor> concat_tensors;
            for (size_t i=0; i<list_size; i++) {
                std::vector<at::Tensor> tensors;
                tensors.reserve(ivalues.size());
                for (const auto& iv: ivalues) {
                    tensors.push_back(iv.toTensorVector().at(i));
                }
                concat_tensors.push_back(f(tensors));
            }
            return concat_tensors;
        } else if (ivalue.isList()) {
            size_t list_size = ivalue.toListRef().size();
            auto source = std::move(ivalue).toList();
            c10::impl::GenericList list(source.elementType());
            for (size_t i=0; i<list_size; i++) {
                std::vector<torch::jit::IValue> elem_ivalues;
                elem_ivalues.reserve(ivalues.size());
                for (const auto& iv: ivalues) {
                    elem_ivalues.push_back(iv.toListRef().at(i));
                }
                list.push_back(aggregateTensorsInIValues(elem_ivalues, f));
            }
            return list;
        } else if (ivalue.isTuple()) {
            size_t list_size = ivalue.toTuple()->elements().size();
            std::vector<torch::jit::IValue> concat_ivalues;
            for (size_t i=0; i<list_size; i++) {
                std::vector<torch::jit::IValue> elem_ivalues;
                elem_ivalues.reserve(ivalues.size());
                for (const auto& iv: ivalues) {
                    elem_ivalues.push_back(iv.toTuple()->elements().at(i));
                }
                concat_ivalues.push_back(aggregateTensorsInIValues(elem_ivalues, f));
            }
            return c10::ivalue::Tuple::create(concat_ivalues);
        }
        return ivalue;
    }

    torch::jit::IValue concatTensorsInIValues(const std::vector<torch::jit::IValue> &ivalues) {
        return aggregateTensorsInIValues(ivalues, [](const std::vector<at::Tensor>& tensors) {
            torch::NoGradGuard no_grad;
            return torch::cat(tensors);
        });
    }

    at::Tensor sumDistLossTensors(const std::vector<at::Tensor>& tensors, size_t batch_size){
        torch::NoGradGuard no_grad;

        assert(!tensors.empty());
        const auto dist_bs = getSplitBatchSizes(batch_size, tensors.size());

        double ratio = (double) dist_bs.front() / batch_size;
        at::Tensor ret = ratio * tensors.front();
        for (size_t i=1; i<tensors.size(); i++) {
            ratio = (double) dist_bs.at(i) / batch_size;
            ret = ret + ratio * tensors.at(i);
        }
        return ret;
    }

    torch::jit::IValue sumDistLossTensorsInIValues(const std::vector<torch::jit::IValue> &ivalues, int batch_size) {
        return aggregateTensorsInIValues(ivalues, [batch_size](const std::vector<at::Tensor>& tensors) {
            return sumDistLossTensors(tensors, batch_size);
        });
    }

    torch::jit::IValue concatOrSumTensorsInIValues(const std::vector<torch::jit::IValue> &ivalues, size_t batch_size) {
        return aggregateTensorsInIValues(ivalues, [batch_size](const std::vector<at::Tensor>& tensors) {
            torch::NoGradGuard no_grad;

            assert(!tensors.empty());

            std::vector<at::Tensor> valid_tensors;
            for (const auto& t: tensors) {
                if (t.defined()) {
                    valid_tensors.push_back(t);
                }
            }

            if (getTensorDim(valid_tensors.front()).empty()) { // loss
                return sumDistLossTensors(valid_tensors, batch_size);
            }
            return torch::cat(valid_tensors);
        });
    }


    torch::jit::IValue sumTensorsInIValues(const std::vector<torch::jit::IValue> &ivalues) {
        return aggregateTensorsInIValues(ivalues, [](const std::vector<at::Tensor>& tensors) {
            assert(!tensors.empty());

            at::Tensor ret = tensors.front();
            {
                torch::NoGradGuard no_grad;
                for (size_t i=1; i<tensors.size(); i++) {
                    if (tensors.at(i).defined()) {
                        ret.add_(tensors.at(i));
                    }
                }
            }
            return ret;
        });
    }

    torch::jit::IValue concatOrSumTensorsInIValues(const std::vector<torch::jit::IValue> &ivalues) {
        return aggregateTensorsInIValues(ivalues, [](const std::vector<at::Tensor>& tensors) {
            assert(!tensors.empty());

            at::Tensor ret = tensors.front();
            if (ret.dim() == 0) {
                for (size_t i=1; i<tensors.size(); i++) {
                    ret = ret + tensors.at(i);
                }
            } else {
                return torch::cat(tensors);
            }
            return ret;
        });
    }

    int64_t guessBatchSize(const at::Tensor &t) {
        const auto dim = getTensorDim(t);
        if (dim.empty()) {
            return -1;
        }
        return dim.front();
    }

    int64_t doGuessBatchSize(const torch::jit::IValue &ivalue) {
        if (ivalue.isTensor()) {
            return guessBatchSize(ivalue.toTensor());
        } else if (ivalue.isTensorList()) {
            for (const auto &t: ivalue.toTensorVector()) {
                size_t bs = guessBatchSize(t);
                if (bs > 0) {
                    return bs;
                }
            }
            return -1;
        } else if (ivalue.isList()) {
            for (const auto &iv: ivalue.toListRef()) {
                int64_t bs = doGuessBatchSize(iv);
                if (bs > 0) {
                    return bs;
                }
            }
            return -1;
        } else if (ivalue.isTuple()) {
            for (const auto &iv: ivalue.toTuple()->elements()) {
                int64_t bs = doGuessBatchSize(iv);
                if (bs > 0) {
                    return bs;
                }
            }
            return -1;
        }
        return -1;
    }

    int64_t guessBatchSize(const std::vector<torch::jit::IValue> &ivalues) {
        for (const auto &iv: ivalues) {
            int64_t bs = doGuessBatchSize(iv);
            if (bs > 0) {
                return bs;
            }
        }
        return -1;
    }

    int64_t guessBatchSize(const IValueMap &ivalues) {
        return guessBatchSize(values(ivalues));
    }

    void initTensorGrad(at::Tensor& t) {
        if (t.grad().defined()) {
            t.grad().zero_();
        } else {
            getMutableGradRef(t) = torch::zeros_like(t);
        }
    }

    template <typename T>
    bool equalsList(const torch::jit::IValue &iv1, const torch::jit::IValue &iv2,
            const std::function<c10::List<T>(const torch::jit::IValue&)>& to_list,
            const std::function<bool(const T&, const T&)>& compare) {
        const auto l1 = to_list(iv1);
        const auto l2 = to_list(iv2);
        assert(l1.size() == l2.size());
        bool eq = true;
        for (size_t i=0; i<l1.size(); i++) {
            eq &= compare(l1.get(i), l2.get(i));
        }
        return eq;
    }

    bool equalsTuple(const torch::jit::IValue &iv1, const torch::jit::IValue &iv2,
                    const std::function<std::vector<torch::jit::IValue>(const torch::jit::IValue&)>& to_elems,
                    const std::function<bool(const torch::jit::IValue&, const torch::jit::IValue&)>& compare) {
        const auto l1 = to_elems(iv1);
        const auto l2 = to_elems(iv2);
        assert(l1.size() == l2.size());
        bool eq = true;
        for (size_t i=0; i<l1.size(); i++) {
            eq &= compare(l1.at(i), l2.at(i));
        }
        return eq;
    }

    bool equals(const torch::jit::IValue &iv1, const torch::jit::IValue &iv2) {

        if (iv1.isInt() && iv2.isInt()) {
            return iv1.toInt() == iv2.toInt();
        } else if (iv1.isBool() && iv2.isBool()) {
            return iv1.toBool() == iv2.toBool();
        } else if (iv1.isDouble() && iv2.isDouble()) {
            return iv1.toDouble() == iv2.toDouble();
        } else if (iv1.isIntList() && iv2.isIntList()) {
            return equalsList<int64_t>(iv1, iv2, [](const torch::jit::IValue& iv) {return iv.toIntList();},
                    [](const int64_t& v1, const int64_t& v2) {return v1 == v2;});
        } else if (iv1.isDoubleList() && iv2.isDoubleList()) {
            return equalsList<double>(iv1, iv2, [](const torch::jit::IValue& iv) {return iv.toDoubleList();},
                                       [](const double& v1, const double& v2) {return v1 == v2;});
        } else if (iv1.isBoolList() && iv2.isBoolList()) {
            return equalsList<bool>(iv1, iv2, [](const torch::jit::IValue& iv) {return iv.toBoolList();},
                                      [](const bool& v1, const bool& v2) {return v1 == v2;});
        } else if (iv1.isTensor() && iv2.isTensor()) {
            return torch::equal(iv1.toTensor(), iv2.toTensor());
        } else if (iv1.isTensorList() && iv2.isTensorList()) {
            return equalsList<at::Tensor>(iv1, iv2, [](const torch::jit::IValue& iv) {return iv.toTensorList();},
                                    [](const at::Tensor& v1, const at::Tensor& v2) {return torch::equal(v1, v2);});
        } else if (iv1.isList() && iv2.isList()) {
            return equalsList<torch::jit::IValue>(iv1, iv2, [](const torch::jit::IValue& iv) {return iv.toList();},
                                          [](const torch::jit::IValue& v1, const torch::jit::IValue& v2) {return equals(v1, v2);});
        } else if (iv1.isTuple() && iv2.isTuple()) {
            return equalsTuple(iv1, iv2, [](const torch::jit::IValue& iv) {return iv.toTuple()->elements();},
                                                  [](const torch::jit::IValue& v1, const torch::jit::IValue& v2) {return equals(v1, v2);});
        } else if (iv1.isDevice() && iv2.isDevice()) {
            return true;
        } else if (iv1.isNone() && iv2.isNone()) {
            return true;
        }
        return false;
    }

    void halfToFloat(void* half_buf, float* float_buf, int count) {
        const auto half_t = torch::from_blob(half_buf, {count}, c10::ScalarType::Half);
        auto float_t = torch::from_blob(float_buf, {count}, c10::ScalarType::Float);
        float_t.copy_(half_t);
    }

    void floatToHalf(float* float_buf, void* half_buf, int count) {
        const auto float_t = torch::from_blob(float_buf, {count}, c10::ScalarType::Float);
        auto half_t = torch::from_blob(half_buf, {count}, c10::ScalarType::Half);
        half_t.copy_(float_t);
    }

    std::unordered_map<std::string, torch::jit::Value*> getGraphConstantValues(
            const std::shared_ptr<torch::jit::Graph>& graph) {

        std::unordered_map<std::string, torch::jit::Value*> results;
        for (const auto& node: graph->nodes()) {
            if (node->kind() != torch::jit::prim::Constant) continue;

            for (torch::jit::Value* val: node->outputs()) {
                if (auto ivalue = toIValue(val)) {
                    results[val->debugName()] = val;
                }
            }
        }
        return results;
    }

    bool isGraphReady(const std::vector<std::string> &input_names,
                      const IValueMap& inputs) {
        std::unordered_set<IValueLocation, IValueLocationHash> locs;
        for (const auto& it: inputs) {
            locs.insert(it.first);
        }
        return isGraphReady(input_names, locs);
    }

    void emptyCache() {
        if (torch::cuda::is_available()) {
            c10::cuda::CUDACachingAllocator::emptyCache();
        }
    }

    void showMem(const std::string& prefix, int rank) {
        if (rank >=0 && mpi::getRank() != rank) {
            return;
        }

        if (torch::cuda::is_available()) {
            int dev_id = getCurrentCudaDeviceId();
            const DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(dev_id);
            spdlog::info("{}: dev={} alloc={} max_alloc={} cache={}", prefix, dev_id,
                         stats.allocated_bytes[0].current,
                         stats.allocated_bytes[0].peak,
                         stats.reserved_bytes[0].current);
        }
    }

    long getAllocatedMemory() {
        if (torch::cuda::is_available()) {
            int dev_id = getCurrentCudaDeviceId();
            const DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(dev_id);
            return stats.allocated_bytes[0].current;
        }
        return 0;
    }

    long getMaxAllocatedMemory() {
        if (torch::cuda::is_available()) {
            int dev_id = getCurrentCudaDeviceId();
            const DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(dev_id);
            return stats.allocated_bytes[0].peak;
        }
        return 0;
    }

    long getMaxCachedMemory() {
        if (torch::cuda::is_available()) {
            int dev_id = getCurrentCudaDeviceId();
            const DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(dev_id);
            return stats.reserved_bytes[0].current;
        }
        return 0;
    }

    void resetMaxAllocatedMemory() {
        if (torch::cuda::is_available()) {
            int dev_id = getCurrentCudaDeviceId();
            c10::cuda::CUDACachingAllocator::resetPeakStats(dev_id);
        }
    }

    std::string toString(const IValueMap& vals) {
        std::stringstream ss;
        for (const auto& it: vals) {
            ss << toString(it.first) << ": " << toString(toIRType(it.second)) << std::endl;
        }
        return ss.str();
    }

    std::string toString(const std::unordered_map<std::string, IValueMap>& vals) {
        std::stringstream ss;
        for (const auto& it: vals) {
            if (!it.second.empty()) {
                ss << "graph=" << it.first << std::endl << toString(it.second);
            }
        }
        return ss.str();
    }

    at::Tensor BufferTensorCache::get(const std::string& key, const IRType& type) {
        if (!contains(types_, key)) {
            types_[key] = type;
        }
        // buffer is not present or type changed

        assert(contains(types_, key));
        if (!contains(buf_tensors_, key) || toString(types_.at(key)) != toString(type)) {
            types_[key] = type;
            buf_tensors_[key] = createBufTensor(type);

//            spdlog::info("@createBufTensor {} {} numel={} nbytes={}", key, toString(type),
//                         buf_tensors_[key].numel(), buf_tensors_[key].nbytes());
        }

        assert(contains(buf_tensors_, key));
        auto ret = buf_tensors_.at(key);
        ret.set_requires_grad(type.requiresGrad());
        return ret;
    }

    void BufferTensorCache::clear() {
        buf_tensors_.clear();
        types_.clear();
    }

    at::Tensor createBufTensor(const IRType& type) {

        const auto scalar_type = fromIRTensorElemTypeToScalarType(type.getTensorElemType());
        at::TensorOptions options;
        if (torch::cuda::is_available()) {
            options = options.device(c10::Device(c10::DeviceType::CUDA));
        } else {
            options = options.device(c10::Device(c10::DeviceType::CPU));
        }
        options = options.dtype(scalar_type)
                .requires_grad(type.requiresGrad());

        at::Tensor ret = torch::zeros(type.getTensorDim(), options);

        return ret;
    }

    RngState getRngState() {
        const auto& default_cpu_gen = at::detail::getDefaultCPUGenerator();
        at::Generator cpu_gen;
        {
            std::lock_guard<std::mutex> lock(default_cpu_gen.unsafeGetGeneratorImpl()->mutex_);
            cpu_gen = default_cpu_gen.clone();
        }

        const auto& default_cuda_gen = at::cuda::detail::getDefaultCUDAGenerator();
        at::Generator cuda_gen;
        {
            std::lock_guard<std::mutex> lock(default_cuda_gen.unsafeGetGeneratorImpl()->mutex_);
            cuda_gen = default_cuda_gen.clone();
        }

        return {std::move(cpu_gen), std::move(cuda_gen)};
    }

    void setRngState(const RngState& state) {
        auto default_cpu_gen = at::detail::getDefaultCPUGenerator();
        {
            const auto def_gen = at::check_generator<at::CPUGeneratorImpl>(default_cpu_gen);
            const auto saved_gen = at::check_generator<at::CPUGeneratorImpl>(state.cpu_gen);

            std::lock_guard<std::mutex> lock(def_gen->mutex_);
            def_gen->set_engine(saved_gen->engine());
            def_gen->set_next_float_normal_sample(saved_gen->next_float_normal_sample());
            def_gen->set_next_double_normal_sample(saved_gen->next_double_normal_sample());
        }

        auto default_cuda_gen = at::cuda::detail::getDefaultCUDAGenerator();
        {
            const auto def_gen = at::check_generator<at::CUDAGeneratorImpl>(default_cuda_gen);
            const auto saved_gen = at::check_generator<at::CUDAGeneratorImpl>(state.cuda_gen);

            std::lock_guard<std::mutex> lock(def_gen->mutex_);

            def_gen->set_current_seed(saved_gen->current_seed());
            def_gen->set_philox_offset_per_thread(saved_gen->philox_offset_per_thread());
        }
    }

    at::Tensor intToBool(const at::Tensor& ten) {
        return ten.ge(0);
    }

    at::Tensor boolToInt(const at::Tensor& ten) {
        assert(ten.scalar_type() == at::ScalarType::Bool);

        at::TensorOptions opts;
        opts = opts.dtype(c10::ScalarType::Int).device(ten.device());
        auto out = torch::ones(ten.sizes(), opts);
        {
            at::NoGradGuard no_grad;
            out.copy_(ten);
        }
        return out;
    }

    bool inPlaceOpName(const std::string& name) {
        if (ends_with(name, "_")) {
            return true;
        }

        const std::vector<std::string> special_ops = {"aten::index_put"};
        if (contains(special_ops, name)) {
            return true;
        }

        return false;
    }

    at::Tensor createTensorFromIRType(const IRType& ir_type, const c10::Device& device) {
        at::TensorOptions options;
        options = options.dtype(fromIRTensorElemTypeToScalarType(ir_type.getTensorElemType()))
                .device(c10::Device(c10::DeviceType::CUDA));
        return torch::zeros(ir_type.getTensorDim(), options);
    }
}
