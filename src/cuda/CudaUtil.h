//
// Created by Masahiro Tanaka on 2019-07-20.
//

#ifndef PYRANNC_CUDAUTIL_H
#define PYRANNC_CUDAUTIL_H

#include <string>
#include <ostream>
#include <sstream>
#include <msgpack.hpp>
#include <cuda_runtime_api.h>
#include <spdlog/spdlog.h>
#include <c10/cuda/CUDAStream.h>

namespace rannc {

    struct CudaDeviceInfo {
        int dev_id = -1;
        std::string name;
        std::string hostname;
        int pci_bus_id = -1;
        int pci_dev_id = -1;
        size_t free_mem = 0;
        size_t total_mem = 0;

        bool operator==(const CudaDeviceInfo &rhs) const {
            return hostname == rhs.hostname &&
                   pci_bus_id == rhs.pci_bus_id &&
                   pci_dev_id == rhs.pci_dev_id;
        }

        bool operator!=(const CudaDeviceInfo &rhs) const {
            return !(rhs == *this);
        }

        friend std::ostream &operator<<(std::ostream &os, const CudaDeviceInfo &info) {
            os << "id: " << info.dev_id << " name: " << info.name
            << " hostname: " << info.hostname << " pci_bus_id: " << info.pci_bus_id
               << " pci_dev_id: " << info.pci_dev_id << " free_mem: " << info.free_mem << " total_mem: "
               << info.total_mem;
            return os;
        }

        MSGPACK_DEFINE(dev_id, name, hostname, pci_bus_id, pci_dev_id, free_mem, total_mem);
    };

    struct CudaDeviceInfoHash {
        std::size_t operator()(const CudaDeviceInfo &info) const {
            std::stringstream ss;
            return std::hash<std::string>()(ss.str());
        };
    };

    using CudaDeviceInfoSet = std::unordered_set<CudaDeviceInfo, CudaDeviceInfoHash>;

    int getCudaDeviceCount();
    std::unordered_set<int> getCudaDeviceIds();
    int getCurrentCudaDeviceId();
    CudaDeviceInfo getCudaDeviceInfo(int id);
    void syncStream();
    void syncDevice();
    c10::cuda::CUDAStream getStream();
    bool isDevicePointer(const void* ptr);
}

#endif //PYRANNC_CUDAUTIL_H
