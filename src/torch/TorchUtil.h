//
// Created by Masahiro Tanaka on 2019-07-05.
//

#ifndef PYRANNC_TORCHUTIL_H
#define PYRANNC_TORCHUTIL_H

#include <nccl.h>
#include <torch/torch.h>

#include <ATen/Generator.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <ostream>

#include "Common.h"
#include "graph/ir.h"
#include "torch/IValueLocation.h"
#include "comm/SCommCommon.h"

namespace rannc {
    std::vector<int64_t> getTensorDim(const at::Tensor& t);
    size_t getTensorElemCount(const at::Tensor &t);
    std::vector<int64_t> intArrayRefToVec(c10::IntArrayRef list);
    std::string tensorToString(const at::Tensor& t, size_t max_elem=20, size_t max_str_len=100);
    std::string tensorPtrToString(void* ptr, size_t size, c10::ScalarType datatype, size_t max_elem=20, size_t max_str_len=100);

    template <typename T>
    std::vector<T> getTensorElems(const at::Tensor& t) {
        std::vector<T> elems;
        size_t elem_count = getTensorElemCount(t);
        elems.reserve(elem_count);
        for (size_t i=0; i<elem_count; i++) {
            T* ptr = static_cast<T*>(t.data_ptr());
            elems.push_back(ptr[i]);
        }
        return elems;
    }

    class BufferTensorCache {
    public:
        at::Tensor get(const std::string& key, const IRType& type);
        void clear();

    private:
        std::unordered_map<std::string, at::Tensor> buf_tensors_;
        std::unordered_map<std::string, IRType> types_;
    };

    torch::jit::IValue to(const torch::jit::IValue &iv, const torch::Device &device, bool detach, bool non_blocking=false);
    torch::jit::IValue detach(const torch::jit::IValue &iv);
    torch::jit::IValue toCPU(const torch::jit::IValue& iv, bool detach, bool non_blocking=false);
    torch::jit::IValue toCUDAIfAvailable(const torch::jit::IValue& iv, bool detach, bool non_blocking=false);
    at::Tensor toCUDAIfAvailable(const at::Tensor& t, bool detach, bool non_blocking=false);
    void toCUDAInPlace(at::Tensor& t);
    void toCPUInPlace(at::Tensor& t);
    torch::jit::IValue contiguous(const torch::jit::IValue &iv);
    torch::jit::IValue setRequiresGrad(const torch::jit::IValue &iv, bool requires_grad);

    std::vector<PathInIValue> findPathsToTensorInIValue(const torch::jit::IValue &ivalue);
    torch::jit::IValue getElemInIValue(const torch::jit::IValue &ivalue, const PathInIValue &path);
    using IValueMap = std::unordered_map<IValueLocation, torch::jit::IValue, IValueLocationHash>;
    std::vector<IValueLocation> getKeys(const IValueMap &map);

    IValueMap toCPU(const IValueMap& iv_map, bool detach, bool non_blocking=false);
    IValueMap toCUDAIfAvailable(const IValueMap& iv_map, bool detach, bool non_blocking=false);

    enum class IRTensorElemType;
    IRTensorElemType toTensorElemType(const c10::ScalarType& scalarType);
    class IRType;
    IRType toIRType(const torch::jit::TypePtr& type);
    IRType toIRType(const torch::jit::IValue& ivalue);
    class IRValue;
    IRValue toIRValue(torch::jit::Value* value);
    torch::jit::TypePtr fromIRScalarType(const IRScalarType& ir_scalar_type);
    at::ScalarType fromIRTensorElemTypeToScalarType(IRTensorElemType ir_tensor_elem);
    at::ScalarType fromIRListTypeToScalarType(IRListType list_type);
    IRScalarType fromIRListTypeToIRScalarType(IRListType list_type);
    torch::jit::IValue toTensorListIfElemsAreTensors(const torch::jit::IValue &ivalue);

    torch::jit::IValue transformTensorsInIValue(const torch::jit::IValue &ivalue,
                                                const std::function<at::Tensor(const at::Tensor&)>& f);
    torch::jit::IValue transformTensorsInIValueWithPath(const torch::jit::IValue &ivalue, const std::string& name,
                                                        const std::function<at::Tensor(const at::Tensor &t, const IValueLocation &loc)> &f);

    at::Tensor padTensor(const at::Tensor& tensor, int batch_size, bool zero);
    at::Tensor unpadTensor(const at::Tensor& tensor, int batch_size);
    torch::jit::IValue padTensorsInIValue(const torch::jit::IValue &ivalue, int batch_size, bool zero);
    torch::jit::IValue unpadTensorsInIValue(const torch::jit::IValue &ivalue, int batch_size);
    torch::jit::IValue alignTensorsInIValue(const torch::jit::IValue &ivalue, size_t target_batch_size, bool zero_pad);
    torch::jit::IValue splitBatchInIValue(const torch::jit::IValue &ivalue, size_t current_batch_size, size_t total__batch_size,
                                          size_t replica_num, bool zero_pad);
    torch::jit::IValue scaleBatchInIValue(const torch::jit::IValue &ivalue, size_t batch_size,
                                          size_t tgt_batch_size, bool zero_pad);
    torch::jit::IValue replicateTensorsInIValue(const torch::jit::IValue &ivalue, size_t replica_num);

    torch::jit::IValue scaleTensorsInIValue(const torch::jit::IValue &ivalue, double scale);
    at::Tensor toFloatIfHalf(const at::Tensor &tensor);
    torch::jit::IValue toFloatTensorsInIValue(const torch::jit::IValue &ivalue);
    at::Tensor sliceDistBatchTensor(const at::Tensor& tensor, int index, int64_t batch_size, int num);
    torch::jit::IValue sliceDistBatchTensorsInIValue(const torch::jit::IValue &ivalue, int index, int64_t batch_size, int num);
    torch::jit::IValue weightDistLossTensorsInIValue(const torch::jit::IValue &ivalue, int index, int64_t batch_size, int num);
    torch::jit::IValue sliceOrWeightTensorsInIValue(const torch::jit::IValue &ivalue,
                                                    const std::vector<int64_t>& batch_sizes, int index);
    torch::jit::IValue cloneTensorsInIValue(const torch::jit::IValue &ivalue);
    torch::jit::IValue cloneTensorsInIValueWithBuffer(const torch::jit::IValue &ivalue,
                                                      const std::string& key,
                                                      BufferTensorCache& buffer);

    torch::jit::IValue aggregateTensorsInIValues(const std::vector<torch::jit::IValue> &ivalues,
                                                 const std::function<at::Tensor(const std::vector<at::Tensor>&)>& f);    torch::jit::IValue concatTensorsInIValues(const std::vector<torch::jit::IValue> &ivalues);
    torch::jit::IValue sumDistLossTensorsInIValues(const std::vector<torch::jit::IValue> &ivalues, int batch_size);
    torch::jit::IValue sumTensorsInIValues(const std::vector<torch::jit::IValue> &ivalues);
    torch::jit::IValue concatOrSumTensorsInIValues(const std::vector<torch::jit::IValue> &ivalues, size_t batch_size);
    torch::jit::IValue concatOrSumTensorsInIValues(const std::vector<torch::jit::IValue> &ivalues);

    int64_t guessBatchSize(const std::vector<torch::jit::IValue> &ivalues);
    int64_t guessBatchSize(const IValueMap &ivalues);
    void initTensorGrad(at::Tensor& t);

    bool equals(const torch::jit::IValue &iv1, const torch::jit::IValue &iv2);

    void halfToFloat(void* half_buf, float* float_buf, int count);
    void floatToHalf(float* float_buf, void* half_buf, int count);

    std::unordered_map<std::string, torch::jit::Value*> getGraphConstantValues(
            const std::shared_ptr<torch::jit::Graph>& graph);
    bool isGraphReady(const std::vector<std::string> &input_names,
                      const IValueMap& inputs);

    template <typename T>
    std::vector<T> listToVector(c10::List<T> list) {
        std::vector<T> vec;
        vec.reserve(list.size());
        for (const T& e: list) {
            vec.push_back(e);
        }
        return vec;
    }

    template <typename T>
    c10::List<T> vectorToList(std::vector<T> vector) {
        c10::List<T> list;
        list.reserve(vector.size());
        for (const T& e: vector) {
            list.push_back(e);
        }
        return list;
    }

    template <typename T>
    std::vector<T> arrayRefToVector(c10::ArrayRef<T> array) {
        std::vector<T> vec;
        vec.reserve(array.size());
        for (const T& e: array) {
            vec.push_back(e);
        }
        return vec;
    }

    void emptyCache();
    void showMem(const std::string& prefix, int rank=-1);
    long getAllocatedMemory();
    long getMaxAllocatedMemory();
    void resetMaxAllocatedMemory();
    long getMaxCachedMemory();

    std::string toString(const IValueMap& vals);
    std::string toString(const std::unordered_map<std::string, IValueMap>& vals);

    at::Tensor createBufTensor(const IRType& type);


    struct RngState {
        at::Generator cpu_gen;
        at::Generator cuda_gen;
    };
    RngState getRngState();
    void setRngState(const RngState& state);

    at::Tensor intToBool(const at::Tensor& ten);
    at::Tensor boolToInt(const at::Tensor& ten);

    bool inPlaceOpName(const std::string& name);
    at::Tensor createTensorFromIRType(const IRType& ir_type, const c10::Device& device);

    template<typename T>
    bool almostEqualTensorsWithTolerance(const at::Tensor& t1, const at::Tensor& t2, T tolerance, T tolerance_ratio) {
        const auto flat_t1 = t1.flatten().cpu();
        const auto flat_t2 = t2.flatten().cpu();
        if (flat_t1.size(0) != flat_t2.size(0)) {
            return false;
        }

        auto d1 = flat_t1.accessor<T, 1>();
        auto d2 = flat_t1.accessor<T, 1>();

        for (size_t i=0; i<flat_t1.size(0); i++) {
            T v1 = d1[i];
            T v2 = d2[i];
            if (std::abs(v1 - v2) > tolerance
                || std::abs((v1 - v2) / std::max(v1, v2)) > tolerance_ratio) {
                return false;
            }
        }
        return true;
    }
}

#endif //PYRANNC_TORCHUTIL_H
