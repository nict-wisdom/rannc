//
// Created by Masahiro Tanaka on 2019-07-11.
//

#include <comm/ObjectComm.h>
#include <Config.h>
#include <comp/EventRecorder.h>
#include "SCommCommon.h"

namespace {
    std::string getSendKey(int dest, int tag, size_t count) {
        std::stringstream ss_key;
        ss_key << "SComm send dest=" << dest << "_size=" << count;
        return ss_key.str();
    }

    std::string getRecvKey(int src, int tag, size_t count) {
        std::stringstream ss_key;
        ss_key << "SComm recv src=" << src << "_count=" << count;
        return ss_key.str();
    }
}

namespace rannc {

    MPI_Datatype scalarTypeToMPIDatatype(c10::ScalarType scalarType) {
        switch (scalarType) {
            case c10::ScalarType::Int: return MPI_INT;
            case c10::ScalarType::Long: return MPI_LONG;
            case c10::ScalarType::Float: return MPI_FLOAT;
            case c10::ScalarType::Double: return MPI_DOUBLE;
            case c10::ScalarType::Half: return MPI_SHORT;
            default:
                std::stringstream ss;
                ss << "Unsupported tensor elem type: " << toString(scalarType);
                throw std::invalid_argument(ss.str());
        }
    }

    int deviceToInt(const c10::Device &dev) {
        int deviceVal;

        if (dev.is_cpu()) {
            deviceVal = -1;
        } else {
            deviceVal = dev.index();
        }
        return deviceVal;
    }

    c10::Device deviceFromInt(int val) {
        // receive the value as int
        if (val < 0) {
            return c10::Device(c10::DeviceType::CPU);
        }
        return {c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(val)};
    }

    void Blob::reserve(size_t size) {
        if (ptr_ == nullptr) {
            ptr_ = alloc(size);
        } else if (size > size_) {
            ptr_ = realloc(size);
        }
        size_ = size;
    }
    void Blob::clear() {
        if (use_cuda_) {
            cudaMemset(ptr_, 0, size_);
        } else {
            memset(ptr_, 0, size_);
        }
    }
    void* Blob::alloc(size_t size) {
        void* ptr;
        if (use_cuda_) {
            if (cudaMalloc((void**)&ptr, size) != cudaSuccess) {
                std::stringstream ss;
                ss << "Failed to allocate mem. size=" << size;
                spdlog::info(ss.str());
//                throw std::runtime_error(ss.str());
            }
        } else {
            ptr = std::malloc(size);
        }
        return ptr;
    }
    void* Blob::realloc(size_t size) {
        if (use_cuda_) {
            free();
            ptr_ = alloc(size);
        } else {
            ptr_ = std::realloc(ptr_, size);
        }
        return ptr_;
    }
    void Blob::free() {
        if (ptr_ != nullptr) {
            if (use_cuda_) {
                cudaFree(ptr_);
            } else {
                std::free(ptr_);
            }
        }
    }

    void* CommBuffer::allocate(const std::string& name, size_t size, bool clear) {
        if (!contains(buf_map_, name)) {
            buf_map_[name] = std::make_shared<Blob>(size, use_cuda_);
        }
        std::shared_ptr<Blob> &buf = buf_map_.at(name);
        buf->reserve(size);
        if (clear) {
            buf->clear();
        }
        return buf->getPtr();
    }

    void* CommBuffer::get(const std::string& name) const {
        assert(contains(buf_map_, name));
        return buf_map_.at(name)->getPtr();
    }

    size_t CommBuffer::getTotalAllocated() const {
        size_t total = 0;
        for (const auto& it: buf_map_) {
            total += it.second->getSize();
        }
        return total;
    }


    std::string toString(const CommProgress& progress) {
        switch (progress) {
            case CommProgress::NOT_STARTED: return "NOT_STARTED";
            case CommProgress::WAITING: return "WAITING";
            case CommProgress::RECEIVED: return "RECEIVED";
            case CommProgress::DONE: return "DONE";
        }
    }

    std::string toString(RouteTypeDP type) {
        switch (type) {
            case RouteTypeDP::P2P:
                return "P2P";
            case RouteTypeDP::REDIST:
                return "REDIST";
            case RouteTypeDP::SCATTER:
                return "SCATTER";
            case RouteTypeDP::GATHER:
                return "GATHER";
            case RouteTypeDP::REDUCE:
                return "REDUCE";
            case RouteTypeDP::WEIGHTED_REDUCE:
                return "WEIGHTED_REDUCE";
            case RouteTypeDP::BROADCAST:
                return "BROADCAST";
            case RouteTypeDP::ANY:
                return "ANY";
            case RouteTypeDP::ANY_TO_ALL:
                return "ANY_TO_ALL";
            case RouteTypeDP::NA:
                return "NA";
            case RouteTypeDP::SCATTER_ANY:
                return "SCATTER_ANY";
            case RouteTypeDP::ALL_GATHER:
                return "ALL_GATHER";
        }
    }

    std::unordered_set<int> getRanksInRoute(const RouteDP& route) {
        std::unordered_set<int> ret;
        for (const int r: route.sources) {
            ret.insert(r);
        }
        for (const int r: route.dests) {
            ret.insert(r);
        }
        return ret;
    }

    int getLocalRank(const std::unordered_set<int>& ranks, int rank) {
        std::vector<int> vec_ranks = setToVector(ranks);
        std::sort(vec_ranks.begin(), vec_ranks.end());

        int idx = 0;
        for (const int r: vec_ranks) {
            if (r == rank) {
                return idx;
            }
            idx++;
        }
        std::stringstream ss;
        ss << "Failed to find local rank. group_ranks=" << join_as_str(ranks)
           << " target=" << rank;
        throw std::invalid_argument(ss.str());
    }

    bool isRankOnBothSide(const RouteDP& route, int rank) {
        return contains(route.sources, rank) && contains(route.dests, rank);
    }

    template <>
    void copyFromVector(bool *dst, const std::vector<bool> &src) {
        for (size_t i=0; i<src.size(); i++) {
            dst[i] = src.at(i);
        }
    }

    template <>
    void copyToVector(std::vector<bool> &dest, const bool* src, int size) {
        dest.clear();
        dest.reserve(size);
        for (int i=0; i<size; i++) {
            dest.push_back(src[i]);
        }
    }

    int getBcastRoot(const RouteDP& route) {

//      This assert does not always hold because bcast is used also for any-to-all
//      assert(route.sources.size() == 1);

        std::vector<int> sources = route.sources;
        std::sort(sources.begin(), sources.end());
        const auto ranks = getRanksInRoute(route);
        return getLocalRank(ranks, sources.front());
    }

    int getReduceRoot(const RouteDP& route) {

        assert(route.dests.size() == 1);
        std::vector<int> dests = route.dests;
        std::sort(dests.begin(), dests.end());
        const auto ranks = getRanksInRoute(route);
        return getLocalRank(ranks, dests.front());
    }

    RouteDP createListElemRoute(const RouteDP& route, int index) {
        RouteDP elemRoute = route;
        elemRoute.location = createListElem(route.location, index);
        return elemRoute;
    }
    RouteDP createTupleElemRoute(const RouteDP& route, int index) {
        RouteDP elemRoute = route;
        elemRoute.location = createTupleElem(route.location, index);
        return elemRoute;
    }

    double getDpRatio(size_t total_batch_size, const std::unordered_set<int>& ranks, int myrank) {
        const auto dp_split_batch_sizes = getSplitBatchSizes(total_batch_size, ranks);
        assert(contains(dp_split_batch_sizes, myrank));
        return (double) dp_split_batch_sizes.at(myrank) / total_batch_size;
    }

    double getDpRatio(size_t total_batch_size, const std::vector<int>& ranks, int myrank) {
        return getDpRatio(total_batch_size, vectorToSet(ranks), myrank);
    }

    IRType getFirstValidTensorType(const std::vector<IRType>& ir_types) {
        for (const auto& type: ir_types) {
            if (type.getBaseType() == IRBaseType::TENSOR
                && type.getTensorElemType() != IRTensorElemType::UNDEF) {
                return type;
            }
        }
        throw std::invalid_argument("No tensor type found.");
    }

    IRType getFirstValidType(const std::vector<IRType>& ir_types) {
        for (const auto& type: ir_types) {
            if (type.getBaseType() == IRBaseType::NONE
                || (type.getBaseType() == IRBaseType::TENSOR && type.getTensorElemType() == IRTensorElemType::UNDEF)) {
                continue;
            }
            return type;
        }
        throw std::invalid_argument("Base types are all None.");
    }

    std::vector<IRType> getNonNoneTypes(const std::vector<IRType>& ir_types) {
        std::vector<IRType> results;
        for (const auto& type: ir_types) {
            if (type.getBaseType() == IRBaseType::NONE) {
                continue;
            }
            results.push_back(type);
        }
        return results;
    }

    bool allBaseTypesSameOrNone(const std::vector<IRType>& ir_types) {
        IRType a_type = getFirstValidType(ir_types);
        for (const auto& type: ir_types) {
            if (type.getBaseType() == IRBaseType::NONE) {
                continue;
            }
            if (a_type.getBaseType() != type.getBaseType()) {
                return false;
            }
        }
        return  true;
    }

    bool allBaseTypesNone(const std::vector<IRType>& ir_types) {
        for (const auto& type: ir_types) {
            if (type.getBaseType() != IRBaseType::NONE) {
                return false;
            }
        }
        return  true;
    }

    IRType reduceBatchTensorTypes(const std::vector<IRType>& ir_types) {
        // check number of dims
        assert(!ir_types.empty());

        IRType a_type = getFirstValidTensorType(ir_types);
        auto expected_dim = a_type.getTensorDim();
        const size_t expected_ndim = expected_dim.size();
        int64_t batch_size = 0;
        for (const auto& type: ir_types) {
            if (type.getBaseType() == IRBaseType::NONE
                || (type.getBaseType() == IRBaseType::TENSOR && type.getTensorElemType() == IRTensorElemType::UNDEF)) {
                continue;
            }

            const auto& dim = type.getTensorDim();
            assert(dim.size() == expected_ndim);

            for (size_t i=1; i<dim.size(); i++) {
                assert(dim.at(i) == expected_dim.at(i));
            }
            batch_size += (int64_t) dim.front();
        }
        expected_dim[0] = batch_size;

        return IRType::createTensorType(a_type.getTensorElemType(), expected_dim, a_type.requiresGrad());
    }

    IRType reduceLossTypes(const std::vector<IRType>& ir_types) {
        // check number of dims
        assert(!ir_types.empty());

        // just for assertion
        for (const auto& type: ir_types) {
            if (type.getBaseType() == IRBaseType::NONE) {
                continue;
            }
            assert(type.getTensorDim().empty());
        }

        return IRType::createTensorType(getFirstValidTensorType(ir_types).getTensorElemType(), {}, false);
    }

    IRType doReduceTypes(const std::vector<IRType>& ir_types) {
        if (allBaseTypesNone(ir_types)) {
            return IRType::createNoneType();
        }

        assert(allBaseTypesSameOrNone(ir_types));
        const auto target_types = getNonNoneTypes(ir_types);
        if (target_types.empty()) {
            return IRType::createNoneType();
        }

        const auto a_type = getFirstValidType(ir_types);

        switch (a_type.getBaseType()) {
            case IRBaseType::TENSOR: {
                const auto& dim = a_type.getTensorDim();
                if (dim.empty()) {
                    return reduceLossTypes(target_types);
                }
                return reduceBatchTensorTypes(target_types);
            }
            case IRBaseType::LIST: {
                const auto& a_comp_types = a_type.getCompoundTypes();
                std::vector<IRType> list_elem_types;
                for (size_t i=0; i<a_comp_types.size(); i++) {
                    std::vector<IRType> elem_types;
                    for (const auto& t: target_types) {
                        elem_types.push_back(t.getCompoundTypes().at(i));
                    }
                    list_elem_types.push_back(doReduceTypes(elem_types));
                }

                const auto list_type = a_type.getListType();
                if (list_type == IRListType::TENSOR) {
                    return IRType::createTensorListType(list_elem_types);
                } else if (list_type == IRListType::GENERIC) {
                    return IRType::createListType(list_elem_types);
                }
                break;
            }
            case IRBaseType::TUPLE: {
                const auto& a_comp_types = a_type.getCompoundTypes();

                std::vector<IRType> tuple_types;
                for (size_t i=0; i<a_comp_types.size(); i++) {
                    std::vector<IRType> elem_types;
                    for (const auto& t: target_types) {
                        elem_types.push_back(t.getCompoundTypes().at(i));
                    }
                    tuple_types.push_back(doReduceTypes(elem_types));
                }
                return IRType::createTupleType(tuple_types);
            }
            default: // expects NONE
                throw std::runtime_error("Encountered unexpected type when reducing types: " + toString(a_type));
        }
    }

    IRType reduceBatchTypes(const std::vector<IRType>& ir_types) {
        return doReduceTypes(ir_types);
    }

    IRType reduceBatchTypes(IRType type, MPI_Comm comm) {
        ObjectComm& ocomm = ObjectComm::get();
        const std::vector<IRType> all_types = ocomm.allgather(type, comm);
        return reduceBatchTypes(all_types);
    }

    IRType reduceLossTypes(IRType type, MPI_Comm comm) {
        ObjectComm& ocomm = ObjectComm::get();
        const std::vector<IRType> all_types = ocomm.allgather(type, comm);
        return reduceLossTypes(all_types);
    }

    IRType reduceTypes(IRType type, MPI_Comm comm) {
        ObjectComm& ocomm = ObjectComm::get();
        const std::vector<IRType> all_types = ocomm.allgather(type, comm);
        return doReduceTypes(all_types);
    }

    void bcast(void *buf, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
        mpi::checkMPIResult(MPI_Bcast(buf, count, datatype, root, comm));
    }

    void redist(void* sendbuf, const RedistArgs& redist_args, MPI_Datatype datatype, MPI_Comm communicator,
                void* recvbuf, bool use_cuda) {

        spdlog::trace("@redist sendcounts={} sdispl={} recvcounts={} rdispls={}",
                      join_as_str(redist_args.sendcounts), join_as_str(redist_args.sdispls),
                      join_as_str(redist_args.recvcounts), join_as_str(redist_args.rdispls));

        int recv_sum = 0;
        for (int s: redist_args.recvcounts) {
            recv_sum += s;
        }

        int DEFAULT_TAG = 10;
        bool cfg_p2p = config::Config::get().getVal<bool>(config::P2P_COMM);
        if (cfg_p2p) {
            int my_local_rank = mpi::getRank(communicator);
            size_t n_ranks = redist_args.sendcounts.size();
            int type_size = mpi::getTypeSize(datatype);


            for (size_t i=0; i<n_ranks; i++) {
                int sc = redist_args.sendcounts[i];
                int rc = redist_args.recvcounts[i];

                if (i < my_local_rank) { //send first
                    if (sc > 0) {
                        recordStart(getSendKey(i, DEFAULT_TAG, sc));
                        mpi::checkMPIResult(MPI_Send((char *) sendbuf + redist_args.sdispls[i] * type_size,
                                 sc, datatype, i, DEFAULT_TAG, communicator));
                        recordEnd(getSendKey(i, DEFAULT_TAG, sc));
                    }
                    if (rc > 0) {
                        recordStart(getRecvKey(i, DEFAULT_TAG, rc));
                        MPI_Status st;
                        mpi::checkMPIResult(MPI_Recv((char*) recvbuf + redist_args.rdispls[i]*type_size,
                                                     redist_args.recvcounts[i], datatype,
                                                     i,
                                                     DEFAULT_TAG, communicator, &st));
                        recordEnd(getRecvKey(i, DEFAULT_TAG, rc));
                    }
                } else if (i > my_local_rank) {
                    if (rc > 0) {
                        MPI_Status st;
                        recordStart(getRecvKey(i, DEFAULT_TAG, rc));
                        mpi::checkMPIResult(MPI_Recv((char*) recvbuf + redist_args.rdispls[i]*type_size,
                                                     redist_args.recvcounts[i], datatype,
                                                     i,
                                                     DEFAULT_TAG, communicator, &st));
                        recordEnd(getRecvKey(i, DEFAULT_TAG, rc));
                    }
                    if (sc > 0) {
                        recordStart(getSendKey(i, DEFAULT_TAG, sc));
                        mpi::checkMPIResult(MPI_Send((char *) sendbuf + redist_args.sdispls[i] * type_size,
                                                     sc, datatype, i, DEFAULT_TAG, communicator));
                        recordEnd(getSendKey(i, DEFAULT_TAG, sc));
                    }
                } else {
                    if (sc > 0) {
                        if (use_cuda) {
                            cudaMemcpy((char *) recvbuf + redist_args.rdispls[i] * type_size,
                                       (char *) sendbuf + redist_args.sdispls[i] * type_size,
                                       sc * type_size, cudaMemcpyDeviceToDevice);
                        } else {
                            memcpy((char *) recvbuf + redist_args.rdispls[i] * type_size,
                                   (char *) sendbuf + redist_args.sdispls[i] * type_size,
                                       sc * type_size);

                        }
                    }
                }
            }
        } else {
            mpi::checkMPIResult(MPI_Alltoallv(sendbuf, &redist_args.sendcounts[0], &redist_args.sdispls[0], datatype,
                                              recvbuf, &redist_args.recvcounts[0], &redist_args.rdispls[0], datatype,
                                              communicator));
        }
    }
}
