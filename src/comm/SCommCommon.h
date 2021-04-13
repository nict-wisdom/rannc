//
// Created by Masahiro Tanaka on 2019-07-11.
//

#ifndef PYRANNC_SCOMMCOMMON_H
#define PYRANNC_SCOMMCOMMON_H

#include <future>
#include <torch/csrc/api/include/torch/all.h>
#include <cuda_runtime_api.h>

#include <comp/TimeCounter.h>
#include <graph/ir.h>

#include "MPIUtil.h"
#include "torch/IValueLocation.h"


namespace rannc {

    enum class RouteTypeDP {
        P2P, REDIST, SCATTER, GATHER, REDUCE, WEIGHTED_REDUCE, BROADCAST,
        SCATTER_ANY, ALL_GATHER,
        ANY, ANY_TO_ALL, NA
    };

    std::string toString(RouteTypeDP type);
}
MSGPACK_ADD_ENUM(rannc::RouteTypeDP);


namespace rannc {

    MPI_Datatype scalarTypeToMPIDatatype(c10::ScalarType scalarType);
    int deviceToInt(const c10::Device &dev);
    c10::Device deviceFromInt(int val);

    template <typename T>
    std::vector<char> serialize(const T data) {
        std::stringstream buffer;
        msgpack::pack(buffer, data);

        buffer.seekg(0);
        std::string str_buf(buffer.str());

        std::vector<char> vec_buf(str_buf.size());
        memcpy(&vec_buf[0], str_buf.c_str(), str_buf.size());
        return vec_buf;
    }

    template <typename T>
    T deserialize(const std::vector<char>& data) {
        msgpack::object_handle oh = msgpack::unpack(&data[0], data.size());
        msgpack::object deserialized = oh.get();

        T obj;
        deserialized.convert(obj);
        return obj;
    }

    struct RouteDP {
        RouteDP() = default;
        RouteDP(IValueLocation location, std::vector<int> sources, std::vector<int> dests, int tag, int source_tag, RouteTypeDP type):
                location(std::move(location)), sources(std::move(sources)), dests(std::move(dests)), tag(tag), source_tag(source_tag), type(type) {
        }
        RouteDP(IValueLocation location, std::vector<int> sources, std::vector<int> dests, int tag, int source_tag, RouteTypeDP type,
                IRValue ir_value):
                location(std::move(location)), sources(std::move(sources)), dests(std::move(dests)), tag(tag), source_tag(source_tag),
                type(type), ir_value(std::move(ir_value)) {
        }

        IValueLocation location;
        std::vector<int> sources;
        std::string source_graph;
        int source_order = -1;
        std::vector<int> dests;
        std::string dest_graph;
        int dest_order = -1;
        int tag = -1;
        int source_tag = -1;
        RouteTypeDP type = RouteTypeDP::NA;
        IRValue ir_value;

        friend std::ostream &operator<<(std::ostream &os, const RouteDP &route) {
            os << "loc: " << toString(route.location)
               << " _source_graph: " << route.source_graph
               << " _sources: " << join_as_str(route.sources)
               << " _source_order: " << route.source_order
               << " _dest_graph: " << route.dest_graph
               << " _dests: " << join_as_str(route.dests)
               << " _dest_order: " << route.dest_order
               << " _tag: " << route.tag << " _source_tag: " << route.source_tag
               << " _type: " << toString(route.type);
            return os;
        }

        MSGPACK_DEFINE(location, sources, source_graph, source_order, dests, dest_graph, dest_order,
                tag, source_tag, type, ir_value);
    };

    RouteDP createListElemRoute(const RouteDP& route, int index);
    RouteDP createTupleElemRoute(const RouteDP& route, int index);

    class Blob {
    public:
        Blob(): size_(0), ptr_(nullptr), use_cuda_(false) {
        }

        Blob(size_t size, bool use_cuda): size_(size), use_cuda_(use_cuda) {
            ptr_ = alloc(size_);
        }

        void reserve(size_t size);

        void clear();

        void *getPtr() const {
            return ptr_;
        }

        size_t getSize() const {
            return size_;
        }

        ~Blob() {
            free();
        }

    private:
        void* alloc(size_t size);
        void* realloc(size_t size);
        void free();

        size_t size_;
        void* ptr_;
        bool use_cuda_;
    };

    class CommBuffer {
    public:
        CommBuffer() = default;
        explicit CommBuffer(bool use_cuda): use_cuda_(use_cuda) {}

        CommBuffer(const CommBuffer&) = delete;
        CommBuffer& operator=(const CommBuffer&) = delete;

        void* allocate(const std::string& name, size_t size, bool clear=false);
        void* get(const std::string& name) const;
        size_t getTotalAllocated() const;

        bool useCuda() {
            return use_cuda_;
        }

        void setCuda(bool use_cuda) {
            if (useCuda() != use_cuda) {
                use_cuda_ = use_cuda;
                buf_map_.clear();
            }
        }

    private:
        std::unordered_map<std::string, std::shared_ptr<Blob>> buf_map_;
        bool use_cuda_ = false;
    };

    struct MPICall {
        std::unordered_set<int> sources;
        std::unordered_set<int> dests;
        int tag;
        bool blocking;
        std::unique_ptr<MPI_Request> request;
        rannc::unique_void_ptr preserved_ptr;
        bool preserved;

        MPICall(std::unordered_set<int> a_src, std::unordered_set<int> a_dest, int a_tag, bool a_blocking)
                :sources(std::move(a_src)), dests(std::move(a_dest)),
                 preserved_ptr(rannc::unique_void((char*) nullptr)) {
            tag = a_tag;
            blocking = a_blocking;
            preserved = false;
        }
        MPICall(MPICall&&) noexcept = default;
        MPICall(const MPICall&) = delete;
    };

    struct CommRequest {
        std::vector<MPICall> calls;

        void wait() {
            for (auto& c: calls) {
                if (!c.blocking) {
                    if (c.request) {
                        MPI_Status status;
                        mpi::checkMPIResult(MPI_Wait(c.request.get(), &status));
                        assert(!c.preserved || c.preserved_ptr);
                    }
                }
            }
        }
    };

    enum class CommProgress {
        NOT_STARTED,
        WAITING,
        RECEIVED,
        DONE
    };
    std::string toString(const CommProgress& progress);

    struct CommResult {
        virtual ~CommResult() {}
    };
    template <typename T>
    struct CommResultT : public CommResult {
        T result;

        ~CommResultT() {
//            std::cout << "Releasing CommResultT: " << name << std::endl;
        }
    };

    struct RedistArgs {
        std::vector<int> sendcounts;
        std::vector<int> recvcounts;
        std::vector<int> sdispls;
        std::vector<int> rdispls;
    };

    template <typename Elem>
    MPI_Datatype getMPIDataType() {
        throw std::invalid_argument("@CommCommon: Failed to get MPI data type. Unsupported data type.");
    }

    template <>
    inline MPI_Datatype getMPIDataType<int>() {
        return MPI_INT;
    }
    template <>
    inline MPI_Datatype getMPIDataType<int64_t>() {
        return MPI_LONG;
    }
    template <>
    inline MPI_Datatype getMPIDataType<size_t>() {
        return MPI_LONG;
    }
    template <>
    inline MPI_Datatype getMPIDataType<double>() {
        return MPI_DOUBLE;
    }
    template <>
    inline MPI_Datatype getMPIDataType<bool>() {
        return MPI_CXX_BOOL;
    }
    template <>
    inline MPI_Datatype getMPIDataType<char>() {
        return MPI_CHAR;
    }

    std::unordered_set<int> getRanksInRoute(const RouteDP& route);
    int getLocalRank(const std::unordered_set<int>& ranks, int rank);

    int getBcastRoot(const RouteDP& route);
    bool isRankOnBothSide(const RouteDP& route, int rank);

    template <typename T>
    void copyFromVector(T* dst, const std::vector<T>& src) {
        memcpy(dst, &src[0], src.size()*sizeof(T));
    }
    template <>
    void copyFromVector(bool *dst, const std::vector<bool> &src);

    template <typename T>
    void copyToVector(std::vector<T>& dest, const T* src, int size) {
        memcpy(&dest[0], src, size*sizeof(T));
    }
    template <>
    void copyToVector(std::vector<bool>& dest, const bool* src, int size);

    double getDpRatio(size_t total_batch_size, const std::unordered_set<int>& ranks, int myrank, int split_index=0);
    double getDpRatio(size_t total_batch_size, const std::vector<int>& ranks, int myrank, int split_index=0);

    IRType reduceBatchTypes(IRType type, MPI_Comm comm);
    IRType reduceLossTypes(IRType type, MPI_Comm comm);
    IRType reduceTypes(IRType type, MPI_Comm comm);

    void redist(void* sendbuf, const RedistArgs& redist_args, MPI_Datatype datatype, MPI_Comm communicator,
                void* recvbuf, bool use_cuda);
}


#endif //PYRANNC_SCOMMCOMMON_H
