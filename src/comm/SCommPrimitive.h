//
// Created by Masahiro Tanaka on 2019/11/19.
//

#ifndef PYRANNC_SCOMMPRIMITIVE_H
#define PYRANNC_SCOMMPRIMITIVE_H

#include <comm/MPIUtil.h>
#include <comm/SCommCommon.h>
#include <torch/TorchUtil.h>

namespace rannc {

    template <typename T>
    T bcastPrimitive(const torch::jit::IValue& ivalue, int root, MPI_Comm comm,
                     const std::function<bool(const torch::jit::IValue&)>& verify,
                     const std::function<T(const torch::jit::IValue&)>& convert) {
        T v;
        if (mpi::getRank(comm) == root) {
            assert(verify(ivalue));
            v = convert(ivalue);
        }
        mpi::checkMPIResult(MPI_Bcast(&v, 1, getMPIDataType<T>(), root, comm));
        return v;
    };

    template <typename T>
    std::vector<T> toPrimitiveVector(const torch::jit::IValue& ivalue) {
        throw std::invalid_argument("Unsupported type was given to toListRef()");
    }

    template <>
    inline std::vector<int64_t> toPrimitiveVector(const torch::jit::IValue& ivalue) {
        return listToVector(ivalue.toIntList());
    }

    template <>
    inline std::vector<double> toPrimitiveVector(const torch::jit::IValue& ivalue) {
        return listToVector(ivalue.toDoubleList());
    }

    template <>
    inline std::vector<bool> toPrimitiveVector(const torch::jit::IValue& ivalue) {
        std::vector<bool> vec;
        for (const auto& v: ivalue.toBoolList()) {
            vec.push_back(v);
        }
        return vec;
    }

    template <typename T>
    std::vector<T> doBcastPrimitiveArray(const torch::jit::IValue& ivalue, const IRType& ir_type, int root, MPI_Comm comm) {
        size_t list_size = ir_type.getListSize();
        auto scalar_type = fromIRListTypeToScalarType(ir_type.getListType());
        size_t buf_size = list_size * elementSize(scalar_type);
        MPI_Datatype datatype = scalarTypeToMPIDatatype(scalar_type);

        std::unique_ptr<T> buf = std::unique_ptr<T>((T*) malloc(buf_size));
        if (mpi::getRank(comm) == root) {
            assert(ivalue.isIntList());
            std::vector<T> vec = toPrimitiveVector<T>(ivalue);
            copyFromVector(buf.get(), vec);
        }
        mpi::checkMPIResult(MPI_Bcast(buf.get(), list_size, datatype, root, comm));
        std::vector<T> result_vec;
        copyToVector(result_vec, buf.get(), list_size);
        return result_vec;
    }

    torch::jit::IValue bcastPrimitive(const torch::jit::IValue& ivalue, const IRType& ir_type, int root, MPI_Comm communicator);
    torch::jit::IValue bcastPrimitiveArray(const torch::jit::IValue& ivalue, const IRType& ir_type, int root, MPI_Comm communicator);
}


#endif //PYRANNC_SCOMMPRIMITIVE_H
