//
// Created by Masahiro Tanaka on 2019/11/19.
//

#include "SCommPrimitive.h"
#include <cuda/CudaUtil.h>

namespace rannc {

torch::jit::IValue bcastPrimitive(
    const torch::jit::IValue& ivalue, const IRType& ir_type, int root,
    MPI_Comm communicator) {
  assert(ir_type.getBaseType() == IRBaseType::SCALAR);
  switch (ir_type.getScalarType()) {
    case IRScalarType::INT:
      return ::rannc::bcastPrimitive<int64_t>(
          ivalue, root, communicator,
          [](const torch::jit::IValue& iv) { return iv.isInt(); },
          [](const torch::jit::IValue& iv) { return iv.toInt(); });
    case IRScalarType::BOOL:
      return ::rannc::bcastPrimitive<bool>(
          ivalue, root, communicator,
          [](const torch::jit::IValue& iv) { return iv.isBool(); },
          [](const torch::jit::IValue& iv) { return iv.toBool(); });
    case IRScalarType::FLOAT:
      return ::rannc::bcastPrimitive<double>(
          ivalue, root, communicator,
          [](const torch::jit::IValue& iv) { return iv.isDouble(); },
          [](const torch::jit::IValue& iv) { return iv.toDouble(); });
    case IRScalarType::DEVICE: {
      int dev_id;
      if (mpi::getRank(communicator) == root) {
        assert(ivalue.isDevice());
        dev_id = deviceToInt(ivalue.toDevice());
      }
      mpi::checkMPIResult(
          MPI_Bcast(&dev_id, 1, getMPIDataType<int>(), root, communicator));

      torch::jit::IValue dev_iv;
      if (deviceFromInt(dev_id) == c10::DeviceType::CPU &&
          torch::cuda::is_available()) {
        // cuda is NOT available at source, but available here
        dev_iv = torch::jit::IValue(
            c10::Device(c10::DeviceType::CUDA, getCurrentCudaDeviceId()));
      } else if (
          deviceFromInt(dev_id) == c10::DeviceType::CUDA &&
          !torch::cuda::is_available()) {
        // cuda is available at source, but not available here (maybe on master)
        dev_iv = torch::jit::IValue(c10::Device(c10::DeviceType::CPU, 0));
      } else {
        // not sure how to handle, just set current device id
        dev_iv = torch::jit::IValue(
            c10::Device(c10::DeviceType::CUDA, getCurrentCudaDeviceId()));
      }

      return dev_iv;
    }
    default:
      throw std::invalid_argument(
          "bcastPrimitive receives a scalar, but the type is unsupported: " +
          toString(ir_type.getScalarType()));
  }
};

torch::jit::IValue bcastPrimitiveArray(
    const torch::jit::IValue& ivalue, const IRType& ir_type, int root,
    MPI_Comm communicator) {
  assert(ir_type.getBaseType() == IRBaseType::LIST);
  switch (ir_type.getListType()) {
    case IRListType::INT:
      return vectorToList(
          doBcastPrimitiveArray<int64_t>(ivalue, ir_type, root, communicator));
    case IRListType::FLOAT:
      return vectorToList(
          doBcastPrimitiveArray<double>(ivalue, ir_type, root, communicator));
    case IRListType::BOOL:
      return vectorToList(
          doBcastPrimitiveArray<bool>(ivalue, ir_type, root, communicator));
    default:
      throw std::invalid_argument(
          "bcastPrimitiveArray receives a list, but the element type is unsupported: " +
          toString(ir_type.getListType()));
  }
}
} // namespace rannc
