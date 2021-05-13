//
// Created by Masahiro Tanaka on 2019-07-05.
//

#include <comm/SComm.h>
#include <comm/ObjectComm.h>
#include <Config.h>
#include <cuda/CudaUtil.h>
#include <comp/EventRecorder.h>
#include "SCommCommon.h"
#include "SCommPrimitive.h"
#include "NCCLWrapper.h"


namespace {
    rannc::RedistArgs getRedistArgsSend(int my_rank, const std::vector<int64_t> &dim,
                                        const std::unordered_map<int, std::vector<int64_t>> &src_dist,
                                        const std::unordered_map<int, std::vector<int64_t>> &dest_dist,
                                        const std::vector<int> &src_ranks,
                                        const std::vector<int> &dest_ranks) {

        std::unordered_map<int, int64_t> sendcounts;
        std::unordered_map<int, int64_t> sdispls;

        int64_t send_idx_from = 0;
        for (int src_rank: src_ranks) {
            if (my_rank == src_rank) {
                int64_t send_idx_offset = 0;
                int64_t send_idx_to = send_idx_from + rannc::productDim(src_dist.at(my_rank));

                int64_t recv_idx_from = 0;
                for (int64_t dest_idx = 0; dest_idx < dest_ranks.size(); dest_idx++) {

                    int64_t dest_rank = dest_ranks.at(dest_idx);
                    size_t dest_size = rannc::productDim(dest_dist.at(dest_rank));
                    int64_t recv_idx_to = recv_idx_from + dest_size;

//                        std::cout  << "src_rank=" << src_rank
//                                   << " dest_idx=" << dest_idx
//                                   << " send_idx_from=" << send_idx_from
//                                   << " send_idx_to=" << send_idx_to
//                                   << " send_idx_offset=" << send_idx_offset
//                                   << " recv_idx_from=" << recv_idx_from
//                                   << " recv_idx_to=" << recv_idx_to << std::endl;

                    if (send_idx_from + send_idx_offset <= recv_idx_from + dest_size) {
                        int64_t sendcount = std::min(send_idx_to - (send_idx_from + send_idx_offset),
                                                     recv_idx_to - (send_idx_from + send_idx_offset));
                        sendcounts[dest_rank] = sendcount;
                        sdispls[dest_rank] = send_idx_offset;

                        send_idx_offset += sendcount;
                    }
                    recv_idx_from += dest_size;

                    if (send_idx_from + send_idx_offset == send_idx_to) {
                        break;
                    }
                }
            } else {
                send_idx_from += rannc::productDim(src_dist.at(src_rank));
            }
        }

        // remove duplicates
        std::unordered_set<int> group_ranks_set;
        for (int r: src_ranks) { group_ranks_set.insert(r); }
        for (int r: dest_ranks) { group_ranks_set.insert(r); }
        std::vector<int> group_ranks;
        group_ranks.reserve(group_ranks_set.size());
        for (int r: group_ranks_set) {
            group_ranks.push_back(r);
        }
        std::sort(group_ranks.begin(), group_ranks.end());

        rannc::RedistArgs args;
        for (int r: group_ranks) {
            assert(sendcounts[r] < INT_MAX);
            args.sendcounts.push_back(sendcounts[r]);
            assert(sdispls[r] < INT_MAX);
            args.sdispls.push_back(sdispls[r]);
            args.recvcounts.push_back(0);
            args.rdispls.push_back(0);
        }

        return args;
    }


    rannc::RedistArgs getRedistArgsRecv(int my_rank, const std::vector<int64_t> &dim,
                                        const std::unordered_map<int, std::vector<int64_t>> &src_dist,
                                        const std::unordered_map<int, std::vector<int64_t>> &dest_dist,
                                        const std::vector<int> &src_ranks,
                                        const std::vector<int> &dest_ranks) {

        std::unordered_map<int, int64_t> recvcounts;
        std::unordered_map<int, int64_t> rdispls;

        int64_t recv_idx_from = 0;
        for (int dest_rank: dest_ranks) {
            if (my_rank == dest_rank) {
                int64_t recv_idx_offset = 0;
                int64_t recv_idx_to = recv_idx_from + rannc::productDim(dest_dist.at(my_rank));

                int64_t send_idx_from = 0;
                for (int64_t src_idx = 0; src_idx < src_ranks.size(); src_idx++) {

                    int src_rank = src_ranks.at(src_idx);
                    size_t src_size = rannc::productDim(src_dist.at(src_rank));

                    int64_t send_idx_to = send_idx_from + src_size;

//                    std::cout << "recv_idx_from=" << recv_idx_from
//                              << " recv_idx_to=" << recv_idx_to
//                              << " recv_idx_offset=" << recv_idx_offset
//                              << " send_idx_from=" << send_idx_from
//                              << " send_idx_to=" << send_idx_to << std::endl;

                    if (recv_idx_from + recv_idx_offset <= send_idx_from + src_size) {
                        int64_t recvcount = std::min(recv_idx_to - (recv_idx_from + recv_idx_offset),
                                                     send_idx_to - (recv_idx_from + recv_idx_offset));
                        recvcounts[src_rank] = recvcount;
                        rdispls[src_rank] = recv_idx_offset;
                        recv_idx_offset += recvcount;
                    }
                    send_idx_from += src_size;

                    if (recv_idx_from + recv_idx_offset == recv_idx_to) {
                        break;
                    }
                }
            } else {
                recv_idx_from += rannc::productDim(dest_dist.at(dest_rank));
            }
        }

        // remove duplicates
        std::unordered_set<int> group_ranks_set;
        for (int r: src_ranks) { group_ranks_set.insert(r); }
        for (int r: dest_ranks) { group_ranks_set.insert(r); }
        std::vector<int> group_ranks;
        group_ranks.reserve(group_ranks_set.size());
        for (int r: group_ranks_set) {
            group_ranks.push_back(r);
        }
        std::sort(group_ranks.begin(), group_ranks.end());

        rannc::RedistArgs args;
        for (int r: group_ranks) {
            args.sendcounts.push_back(0);
            args.sdispls.push_back(0);

            assert(recvcounts[r] < INT_MAX);
            args.recvcounts.push_back(recvcounts[r]);
            assert(rdispls[r] < INT_MAX);
            args.rdispls.push_back(rdispls[r]);
        }

//        std::cout << "recvcounts=" << join_as_str(args.recvcounts) << std::endl;
//        std::cout << "rdispls=" << join_as_str(args.rdispls) << std::endl;

        return args;
    }
}

namespace rannc {
    rannc::TimeCounter scomm_time_counter(false);

    unique_group_ptr unique_group(MPI_Group *ptr) {
        return unique_group_ptr(ptr, [](MPI_Group* ptr) {
            MPI_Group_free(ptr);
            delete ptr;
        });
    }
    unique_comm_ptr unique_comm(MPI_Comm *ptr) {
        return unique_comm_ptr(ptr, [](MPI_Comm* ptr) {
            MPI_Comm_free(ptr);
            delete ptr;
        });
    }

    int TagMap::getParamTag(long param_id) {
        if (!contains(param_map_, param_id)) {
            param_map_[param_id] = generate(param_id);
        }
        return param_map_[param_id];
    }

    int TagMap::getRankSetTag(const std::unordered_set<int>& ranks) {
        if (!contains(param_comm_map_, ranks)) {
            param_comm_map_[ranks] = generate(ranks);
        }
        return param_comm_map_[ranks];
    }

    int TagMap::getValueTag(const IValueLocation& loc) {
        return getStrTag(toString(loc));
    }

    int TagMap::getRouteTag(const RouteDP& route) {
        return getStrTag(toString(route));
    }

    int TagMap::getRouteSourceTag(const RouteDP& route) {
        return getStrTag(toString(route) + join_as_str(route.sources));
    }

    void TagMap::sync() {
        ObjectComm& ocomm = ObjectComm::get();
        tags_ = ocomm.bcast(tags_);
    }

    IRType SComm::reduceRouteSourceBatchTypes(const IRType& type, const RouteDP& route) {
        return reduceBatchTypes(type, getRouteSourcesCommunicator(route));
    }

    IRType SComm::sendTypeBcast(IRType local_type, const RouteDP& route) {
        ObjectComm& ocomm = ObjectComm::get();
        int root = route.sources.front();
        return ocomm.bcast(local_type, getLocalRank(getRanksInRoute(route), root), getRouteCommunicator(route));
    }

    IRType SComm::recvTypeBcast(const RouteDP& route) {
        ObjectComm& ocomm = ObjectComm::get();
        int root = route.sources.front();
        IRType dummy_type;
        return ocomm.bcast(dummy_type, getLocalRank(getRanksInRoute(route), root), getRouteCommunicator(route));
    }

    SComm::SComm(): batch_size_(0), split_index_(0), is_bwd_(false) {}

    SComm& SComm::get() {
        static SComm instance;
        return instance;
    }

    void SComm::sendTensorRedist(const torch::jit::IValue& send_val, const RouteDP& route, const IRType& global_type) {
        assert(contains(route.sources, mpi::getRank()));

        // setup and run alltoall here
        const auto& global_dim = global_type.getTensorDim();
        if (batch_size_ < 1) {
            batch_size_ = global_dim.front();
        }
        assert(send_val.isTensor());
        at::Tensor send_ten = send_val.toTensor();

        const auto redist_args = getRedistArgs(mpi::getRank(), batch_size_, global_dim,
                                               vectorToSet(route.sources), vectorToSet(route.dests), split_index_);

        syncStream();
        assert(send_ten.is_contiguous());

        MPI_Datatype datatype = scalarTypeToMPIDatatype(send_ten.type().scalarType());
        const auto route_ranks = getRanksInRoute(route);

        // for profiling
        int size_sum = 0;
        for (size_t i=0; i<route_ranks.size(); i++) {
            size_sum += redist_args.sendcounts[i];
        }

        std::stringstream ss;
        ss << "sendTensorRedist_size=" << (size_sum * mpi::getTypeSize(datatype));
        scomm_time_counter.start(ss.str());

        bool cfg_p2p = config::Config::get().getVal<bool>(config::P2P_COMM);
        if (cfg_p2p) {
            int type_size = mpi::getTypeSize(datatype);
            std::vector<int> dests = route.dests;
            std::sort(dests.begin(), dests.end());

            for (const int dest: dests) {
                int dest_local_rank = getLocalRank(route_ranks, dest);
                mpi::checkMPIResult(MPI_Send((char*) send_ten.data_ptr() + redist_args.sdispls[dest_local_rank]*type_size,
                                             redist_args.sendcounts[dest_local_rank], datatype,
                                             dest,
                                             route.tag, MPI_COMM_WORLD));
            }
        } else {
            mpi::checkMPIResult(MPI_Alltoallv(send_ten.data_ptr(), &redist_args.sendcounts[0], &redist_args.sdispls[0], datatype,
                                              nullptr, &redist_args.recvcounts[0], &redist_args.rdispls[0], datatype,
                                              getRouteCommunicator(route)));
        }

        scomm_time_counter.stop(ss.str());
    }

    torch::jit::IValue SComm::recvTensorRedist(const RouteDP& route, const IRType& global_type) {
        // setup and run alltoall here
        const auto& global_dim = global_type.getTensorDim();
        if (batch_size_ < 1) {
            batch_size_ = global_dim.front();
        }
        const auto redist_args = getRedistArgs(mpi::getRank(), batch_size_, global_dim,
                                               vectorToSet(route.sources), vectorToSet(route.dests), split_index_);

        at::ScalarType stype = fromIRTensorElemTypeToScalarType(global_type.getTensorElemType());
        MPI_Datatype datatype = scalarTypeToMPIDatatype(stype);

        int recv_sum = 0;
        for (int s: redist_args.recvcounts) {
            recv_sum += s;
        }
        size_t sizeInByte = recv_sum * mpi::getTypeSize(datatype);

        std::stringstream ss;
        ss << "recvTensorRedist_size=" << sizeInByte;
        scomm_time_counter.start(ss.str());

        std::unordered_map<int, std::vector<int64_t>> dest_dist =
                rannc::calcDistBatchDims(batch_size_, global_dim, rannc::vectorToSet(route.dests), split_index_);
        const auto& local_dim = dest_dist.at(mpi::getRank());
        at::Tensor ret = buf_cache_.get(getKey(route), setDimToIRType(global_type, local_dim));
        void* resultBuf = ret.data_ptr();

        bool cfg_p2p = config::Config::get().getVal<bool>(config::P2P_COMM);
        if (cfg_p2p) {
            int type_size = mpi::getTypeSize(datatype);
            std::vector<int> sources = route.sources;
            std::sort(sources.begin(), sources.end());
            for (const int src: sources) {
                int src_local_rank = getLocalRank(getRanksInRoute(route), src);
                MPI_Status st;
                mpi::checkMPIResult(MPI_Recv((char *) resultBuf + redist_args.rdispls[src_local_rank] * type_size,
                                             redist_args.recvcounts[src_local_rank], datatype,
                                             src,
                                             route.tag, MPI_COMM_WORLD, &st));
            }
        } else {
            mpi::checkMPIResult(MPI_Alltoallv(nullptr, &redist_args.sendcounts[0], &redist_args.sdispls[0], datatype,
                                              resultBuf, &redist_args.recvcounts[0], &redist_args.rdispls[0], datatype,
                                              getRouteCommunicator(route)));
        }

        scomm_time_counter.stop(ss.str());

        return ret;
    }

    void SComm::sendIValue(const torch::jit::IValue& ivalue, const RouteDP& route) {
        const IRType global_type = reduceRouteSourceBatchTypes(toIRType(ivalue), route);
        sendTypeBcast(global_type, route);

        if (global_type.getBaseType() == IRBaseType::TENSOR) {
            assert(ivalue.isTensor());
            auto ten = ivalue.toTensor();
            sendTensorRedist(ivalue, route, global_type);
        } else if (global_type.getBaseType() == IRBaseType::LIST &&
                        global_type.getListType() == IRListType::TENSOR) {
            assert(ivalue.isTensorList());
            int idx = 0;
            const auto& types = global_type.getCompoundTypes();
            for (const auto& t: ivalue.toTensorVector()) {
                const auto& type = types.at(idx);
                sendTensorRedist(t, route, type);
                idx++;
            }
        } else {
            throw std::invalid_argument("sendIValue only supports tensor or tensor list. route=" + toString(route));
        }
    }

    torch::jit::IValue SComm::recvIValue(const RouteDP& route) {
        const IRType global_type = recvTypeBcast(route);

        if (global_type.getBaseType() == IRBaseType::TENSOR) {
            auto ret = recvTensorRedist(route, global_type);
            return ret;
        } else if (global_type.getBaseType() == IRBaseType::LIST &&
                   global_type.getListType() == IRListType::TENSOR) {
            int idx = 0;
            c10::List<at::Tensor> tensor_list;
            for (const auto& type: global_type.getCompoundTypes()) {
                auto ret = recvTensorRedist(route, type);

                assert(ret.isTensor());
                tensor_list.push_back(ret.toTensor());
                idx++;
            }
            return tensor_list;
        } else {
            throw std::invalid_argument("recvIValue only supports tensor or tensor list. route=" + toString(route)
                + " type=" + toString(global_type));
        }
    }

    torch::jit::IValue SComm::bcastIValue(const torch::jit::IValue& ivalue, const RouteDP& route) {
        IRType ir_type = toIRType(ivalue);
        MPI_Comm communicator = getRouteCommunicator(route);
        ObjectComm& ocomm = ObjectComm::get();
        int root = getBcastRoot(route);
        ir_type = ocomm.bcast(ir_type, root, communicator);

        if (ir_type.getBaseType() == IRBaseType::SCALAR) {
            return bcastPrimitive(ivalue, ir_type, root, communicator);
        } else if (ir_type.getBaseType() == IRBaseType::TENSOR) {
            return bcastTensor(ivalue, ir_type, route, communicator);
        } else if (ir_type.getBaseType() == IRBaseType::LIST) {
            return bcastPrimitiveArray(ivalue, ir_type, root, communicator);
        } else if (ir_type.getBaseType() == IRBaseType::NONE) {
            return torch::jit::IValue();
        }
        throw std::invalid_argument("bcastIValue does not support type: " + toString(ir_type.getBaseType()));
    }

    IValueMap SComm::bcastIValueMap(const IValueMap& ivalue_map, const RouteDP& route) {
        std::vector<IValueLocation> locs;
        locs.reserve(ivalue_map.size());
        for (const auto& it: ivalue_map) {
            locs.push_back(it.first);
        }

        ObjectComm& ocomm = ObjectComm::get();
        int root = getBcastRoot(route);
        MPI_Comm communicator = getRouteCommunicator(route);
        locs = ocomm.bcast(locs, root, communicator);

        IValueMap results;
        assert(!route.sources.empty());
        for (const auto& loc: locs) {
            if (mpi::getRank() == route.sources.front()) {
                results[loc] = bcastIValue(ivalue_map.at(loc), route);
            } else {
                results[loc] = bcastIValue(torch::jit::IValue(), route);
            }
        }
        return results;
    }

    torch::jit::IValue SComm::distributeBatchTensor(const torch::jit::IValue& val, const IRType& global_type,
            const RouteDP& route, int split_delay) {
        assert(val.isTensor() || val.isNone());

        recordStart("distributeBatchTensor_" + toString(route));

        const auto& global_dim = global_type.getTensorDim();
        if (batch_size_ < 1) {
            batch_size_ = global_dim.front();
        }

        // setup and run alltoall here
        std::unordered_map<int, std::vector<int64_t>> dest_dist =
                rannc::calcDistBatchDims(batch_size_, global_dim, rannc::vectorToSet(route.dests), split_index_-split_delay);

        at::Tensor recv_buf;
        void* send_ptr;
        void* recv_ptr = nullptr;

        if (val.isNone() || (val.isTensor() && !val.toTensor().defined())) {
            send_ptr = nullptr;
        } else if (val.isTensor()) {
            auto ten = val.toTensor();
            assert(ten.is_contiguous());
            send_ptr = ten.data_ptr();
        } else {
            throw std::runtime_error("Unsupported tensor type for distribution: " + toString(toIRType(val)));
        }

        if (contains(route.dests, mpi::getRank())) {
            assert(contains(dest_dist, mpi::getRank()));
            const auto& dim = dest_dist.at(mpi::getRank());
            if (productDim(dim) > 0) {
                recv_buf = buf_cache_.get(getKey(route), setDimToIRType(global_type, dim));
                recv_ptr = recv_buf.data_ptr();
            }
        }

        NCCLWrapper& ar = NCCLWrapper::get();
        ar.redist(send_ptr, recv_ptr, route, batch_size_, global_type, split_index_-split_delay);

        recordEnd("distributeBatchTensor_" + toString(route));

        if (!contains(dest_dist, mpi::getRank())) {
            return torch::jit::IValue();
        }

        return recv_buf;
    }

    torch::jit::IValue SComm::doDistribute(const torch::jit::IValue& val, const IRType& global_type,
                                    const RouteDP& route, bool is_bwd, int split_delay) {
        if (global_type.getBaseType() == IRBaseType::TENSOR) {
            const auto& dim = global_type.getTensorDim();
            if (dim.empty()) {
                return distributeLossTensor(val, global_type, route, is_bwd, batch_size_);
            }
            return distributeBatchTensor(val, global_type, route, split_delay);
        } else if (global_type.getBaseType() == IRBaseType::LIST) {

            const auto& elem_types = global_type.getCompoundTypes();
            assert(!elem_types.empty());
            const auto& a_type = elem_types.front();
            bool is_tensor_list = a_type.getBaseType() == IRBaseType::TENSOR;

            size_t idx = 0;
            std::vector<torch::jit::IValue> results;
            for (const auto& elem: elem_types) {
                if (val.isNone()) {
                    results.push_back(doDistribute(val, elem, createListElemRoute(route, idx), is_bwd, split_delay));
                } else {
                    results.push_back(doDistribute(val.toListRef().at(idx), elem,
                                                   createListElemRoute(route, idx), is_bwd, split_delay));
                }
                idx++;
            }

            if (!contains(route.dests, mpi::getRank())) {
                return torch::jit::IValue();
            }

            if (is_tensor_list) {
                std::vector<at::Tensor> tensors;
                tensors.reserve(results.size());
                for (const auto& iv: results) {
                    assert(iv.isTensor());
                    tensors.push_back(iv.toTensor());
                }
                return c10::List<at::Tensor>(tensors);
            }

            c10::impl::GenericList list(at::AnyType::get());
            for (const auto& iv: results) {
                    assert(iv.isTensor());
                list.push_back(iv.toTensor());
            }
            return list;

        } else if (global_type.getBaseType() == IRBaseType::TUPLE) {
            const auto& elem_types = global_type.getCompoundTypes();
            size_t idx = 0;
            std::vector<torch::jit::IValue> results;
            for (const auto& elem: elem_types) {
                if (val.isNone()) {
                    results.push_back(doDistribute(val, elem, createTupleElemRoute(route, idx), is_bwd, split_delay));
                } else {
                    results.push_back(doDistribute(val.toTuple()->elements().at(idx), elem,
                            createTupleElemRoute(route, idx), is_bwd, split_delay));
                }
                idx++;
            }
            if (!contains(route.dests, mpi::getRank())) {
                return torch::jit::IValue();
            }
            return c10::ivalue::Tuple::create(results);
        } else if (global_type.getBaseType() == IRBaseType::NONE) {
            return torch::jit::IValue();
        }
        throw std::invalid_argument("Unexpected type for distribution: " + toString(global_type));
    }

    IRType SComm::reduceRouteTypes(const IRType& type, const RouteDP& route) {
        return reduceTypes(type, getRouteCommunicator(route));
    }

    torch::jit::IValue SComm::distribute(const torch::jit::IValue& val, const RouteDP& route, bool is_bwd, int split_delay) {
        return distribute(val, route, is_bwd, reduceRouteTypes(toIRType(val), route), split_delay);
    }

    torch::jit::IValue SComm::distribute(const torch::jit::IValue& val, const RouteDP& route, bool is_bwd,
                                  const IRType& global_type, int split_delay) {
        // A subgraph may not have the value to distribute
        assert(val.isNone() || val.isTuple() || val.isTensor() || val.isTensorList());
        return doDistribute(val, global_type, route, is_bwd, split_delay);
    }

    rannc::RedistArgs getRedistArgs(int my_rank, int64_t total_batch_size, const std::vector<int64_t>& global_dim,
                                    const std::unordered_set<int>& src_ranks,
                                    const std::unordered_set<int>& dest_ranks,
                                    int split_index) {
        std::unordered_map<int, std::vector<int64_t>> src_dist = rannc::calcDistBatchDims(total_batch_size, global_dim, src_ranks, split_index);
        std::unordered_map<int, std::vector<int64_t>> dest_dist = rannc::calcDistBatchDims(total_batch_size, global_dim, dest_ranks, split_index);

//        for (const auto& it: src_dist) {
//            int rank = it.first;
//            std::cout << "src_dist: rank=" << rank << " dim=" << rannc::join_as_str(it.second) << std::endl;
//        }
//        for (const auto& it: dest_dist) {
//            int rank = it.first;
//            std::cout << "dest_dist: rank=" << rank << " dim=" << rannc::join_as_str(it.second) << std::endl;
//        }

        auto vec_src_ranks = rannc::setToVector(src_ranks);
        std::sort(vec_src_ranks.begin(), vec_src_ranks.end());
        auto vec_dest_ranks = rannc::setToVector(dest_ranks);
        std::sort(vec_dest_ranks.begin(), vec_dest_ranks.end());

        bool send = false;
        if (rannc::contains(src_ranks, my_rank)) {
            send = true;
        }
        bool recv = false;
        if (rannc::contains(dest_ranks, my_rank)) {
            recv = true;
        }

        bool both = send && recv;
        rannc::RedistArgs sendArgs, recvArgs, bothArgs;
        if (send) {
            sendArgs = getRedistArgsSend(my_rank, global_dim, src_dist, dest_dist, vec_src_ranks, vec_dest_ranks);
            if (!both) {
                return sendArgs;
            }
        }
        if (recv) {
            recvArgs = getRedistArgsRecv(my_rank, global_dim, src_dist, dest_dist, vec_src_ranks, vec_dest_ranks);
            if (!both) {
                return recvArgs;
            }
        }

        bothArgs.sendcounts = sendArgs.sendcounts;
        bothArgs.sdispls = sendArgs.sdispls;
        bothArgs.recvcounts = recvArgs.recvcounts;
        bothArgs.rdispls = recvArgs.rdispls;
        return bothArgs;
    }

    torch::jit::IValue SComm::distributeLossTensor(const torch::jit::IValue& val, const IRType& global_type,
                                            const RouteDP& route, bool weight, int64_t batch_size) {
        assert(val.isTensor() || val.isNone());
        const auto& global_dim = global_type.getTensorDim();

        auto type = global_type; // copy
        type.setRequiresGrad(false);
        at::Tensor src;

        double src_ratio = 0;
        if (val.isNone()) {
            src = buf_cache_.get(getKey(route), type);
            src.zero_();
        } else if (val.isTensor()) {
            src = val.toTensor();
            if (contains(route.sources, mpi::getRank())) {
                src_ratio = getDpRatio(batch_size, route.sources, mpi::getRank(), split_index_);
            }
            src = src_ratio * src;
        } else {
            throw std::runtime_error("Unsupported tensor type for loss distribution: " + toString(toIRType(val)));
        }

        syncStream();

        NCCLWrapper& ar = NCCLWrapper::get();
        std::vector<at::Tensor> ar_buf = {src};
        TagMap& tag_map = TagMap::get();
        const auto route_ranks = getRanksInRoute(route);

        if (contains(route_ranks, mpi::getRank())) {
            ar.allreduce(tag_map.getRankSetTag(route_ranks), route_ranks, ar_buf);
        }

        if (!contains(route.dests, mpi::getRank())) {
            return torch::jit::IValue();
        }

        double ratio = 1.0;
        if (weight) {
            ratio *= getDpRatio(batch_size, route.dests, mpi::getRank(), split_index_);
        }

        return ratio * src;
    }

    at::Tensor SComm::bcastTensor(const torch::jit::IValue& ivalue, const IRType& ir_type, const RouteDP& route, MPI_Comm comm) {

        auto scalar_type = fromIRTensorElemTypeToScalarType(ir_type.getTensorElemType());
        int count = productDim(ir_type.getTensorDim());
        MPI_Datatype datatype = scalarTypeToMPIDatatype(scalar_type);
        int root = getBcastRoot(route);

        void *ptr;
        at::Tensor ten;
        if (mpi::getRank(comm) == root) {
            assert(ivalue.isTensor());
            ten = ivalue.toTensor();
        } else {
            ten = createBufTensor(ir_type);
            syncStream();
        }

        if (count > 0) {
            ptr = ten.data_ptr();
            mpi::checkMPIResult(MPI_Bcast(ptr, count, datatype, root, comm));
        }

        return ten;
    }

    void SComm::startFwd(int64_t batch_size) {
        start("FWD", batch_size);
        is_bwd_ = false;
    }

    void SComm::startBwd(int64_t batch_size) {
        start("BWD", batch_size);
        is_bwd_ = true;
    }

    void SComm::start(std::string prefix, int64_t batch_size) {
        prefix_ = std::move(prefix);
        batch_size_ = batch_size;
    }

    void SComm::startSplit(int split_index) {
        split_index_ = split_index;
    }

    std::string SComm::getKey(const RouteDP& route) const {
        std::stringstream ss;
        ss << "[PREFIX=" << prefix_ << "]" << toString(route.location) << "[TAG=" << route.tag << "]"
           << "[SPLIT=" << split_index_ << "]";
        return ss.str();
    }

    MPI_Comm SComm::getRouteCommunicator(const RouteDP& route) {
        return getCommunicator(route.tag, getRanksInRoute(route));
    }

    MPI_Comm SComm::getRouteSourcesCommunicator(const RouteDP& route) {
        return getCommunicator(route.source_tag, vectorToSet(route.sources));
    }

    MPI_Comm SComm::getCommunicator(int tag, const std::unordered_set<int>& ranks) {
        if (contains(comm_map_, tag)) {
            return *comm_map_.at(tag);
        }

        assert(contains(ranks, mpi::getRank()));

        MPI_Group world_group;
        MPI_Comm_group(MPI_COMM_WORLD, &world_group);

        unique_group_ptr new_group = unique_group(new MPI_Group);

        std::vector<int> vec_ranks = setToVector(ranks);
        std::sort(vec_ranks.begin(), vec_ranks.end());
        MPI_Group_incl(world_group, ranks.size(), &vec_ranks[0], new_group.get());

        unique_comm_ptr new_comm = unique_comm(new MPI_Comm);
        MPI_Comm_create_group(MPI_COMM_WORLD, *new_group, tag, new_comm.get());

        group_map_[tag] = std::move(new_group);
        comm_map_[tag] = std::move(new_comm);

        return *comm_map_.at(tag);
    }

    void SComm::destroy() {
        for (auto& c: comm_map_) {
            c.second.reset();
        }
        comm_map_.clear();

        for (auto& g: group_map_) {
            g.second.reset();
        }
        group_map_.clear();
        buf_cache_.clear();
    }
}

