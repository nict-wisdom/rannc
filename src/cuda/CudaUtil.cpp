//
// Created by Masahiro Tanaka on 2019-07-20.
//

#include "CudaUtil.h"
#include "Common.h"

#include <cuda_runtime_api.h>
#include <unistd.h>
#include <ios>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/api/include/torch/all.h>

namespace rannc {

    int HOST_NAME_MAX_LEN = 1024;

    bool isCudaAvailable() {
        static bool checked = false;
        static bool available = false;
        if (!checked) {
            available = torch::cuda::is_available();
            checked = true;
        }
        return available;
    }

    int getCudaDeviceCount() {
        static int count = -1;
        if (count >= 0) return count;

        if (cudaSuccess == cudaGetDeviceCount(&count)) {
            return count;
        }
        count = 0;
        return count;
    }

    bool isDeviceListDefined() {
        return getenv("RANNC_CUDA_DEVICES") != nullptr;
    }

    std::unordered_set<int> getCudaDeviceIds() {
        int count = ::rannc::getCudaDeviceCount();
        std::unordered_set<int> ids;
        if (!isDeviceListDefined()) {
            for (int i = 0; i < count; i++) {
                ids.insert(i);
            }
            return ids;
        }
    }

    int getCurrentCudaDeviceId() {
        int dev;
        cudaGetDevice(&dev);
        return dev;
    }

    CudaDeviceInfo getCudaDeviceInfo(int id) {
        if (getCudaDeviceCount() == 0) {
            const auto ret = CudaDeviceInfo();
            return ret;
        }

        CudaDeviceInfo info;
        info.dev_id = id;

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, id);
        info.name = prop.name;
        info.pci_bus_id = prop.pciBusID;
        info.pci_dev_id = prop.pciDeviceID;

        size_t free_mem;
        size_t total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        info.free_mem = free_mem;
        info.total_mem = total_mem;

        char hostname[HOST_NAME_MAX_LEN];
        gethostname(hostname, HOST_NAME_MAX_LEN);
        info.hostname = hostname;

        return info;
    }

    void syncStream() {
        if (torch::cuda::is_available()) {
            auto stream = c10::cuda::getCurrentCUDAStream(getCurrentCudaDeviceId());
            stream.synchronize();
        }
    }

    void syncDevice() {
        if (torch::cuda::is_available()) {
            cudaDeviceSynchronize();
        }
    }

    c10::cuda::CUDAStream getStream() {
        if (!torch::cuda::is_available()) {
            throw std::runtime_error("Failed to get CUDA stream: CUDA is not available.");
        }
        return c10::cuda::getCurrentCUDAStream(getCurrentCudaDeviceId());
    }

//    void sync() {
//        if (torch::cuda::is_available()) {
//            cudaDeviceSynchronize();
//        }
//    }

    std::string toString(const CudaDeviceInfo& cuDevInfo) {
        std::stringstream ss;
        ss << cuDevInfo;
        return ss.str();
    }

    bool isDevicePointer(const void* ptr) {
        cudaPointerAttributes attributes;
        cudaError_t ret = cudaPointerGetAttributes(&attributes, ptr);

        if (ret != cudaSuccess) {
            return false;
        }

        // CUDA 11 returns
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__UNIFIED.html#group__CUDART__UNIFIED_1gd89830e17d399c064a2f3c3fa8bb4390
        return attributes.devicePointer != nullptr;
    }
}

