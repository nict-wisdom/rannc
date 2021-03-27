//
// Created by Masahiro Tanaka on 2018-12-27.
//

#ifndef PYRANNC_MPIUTIL_H
#define PYRANNC_MPIUTIL_H

#include <ostream>
#include <mpi.h>

#include "Common.h"


namespace mpi {

    class MPIException : public std::exception {
    public:
        explicit MPIException(std::string message): message_(std::move(message)) {}
        MPIException(const MPIException& e) noexcept {
            message_ = e.message_;
        }

        ~MPIException() noexcept override = default;
        const char* what() const noexcept override {
            return message_.c_str();
        }
    private:
        std::string message_;
    };


    int getRank();
    int getRank(MPI_Comm comm);
    bool isMaster();
    int getSize();
    int getSize(MPI_Comm comm);
    int getTypeSize(MPI_Datatype datatype);
    int getTagUB();

    std::string getMPILibraryVersion();

    std::string getTypeName(MPI_Datatype datatype);
    std::string getProcessorName();
    std::unordered_set<int> getRanksInComm(MPI_Comm comm);

    template <typename T>
    int generateTag(const T& n) {
        int hash = (int) std::hash<T>()(n);
        return std::abs(hash) % mpi::getTagUB();
    }
    template<> int generateTag(const std::unordered_set<int>& n);

    std::unordered_set<int> getAllRanks();
    std::unordered_set<int> getAllWorkerRanks();

    int64_t allReduceSumBatchSize(int64_t batch_size, MPI_Comm comm=MPI_COMM_WORLD);
    int64_t allReduceMaxBatchSize(int64_t batch_size, MPI_Comm comm=MPI_COMM_WORLD);

    void checkMPIResult(int code);
    void checkMPIStatusCode(int code);
}

#endif //PYRANNC_MPIUTIL_H


