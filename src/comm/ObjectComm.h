//
// Created by Masahiro Tanaka on 2019/09/20.
//

#ifndef PYRANNC_OBJECTCOMM_H
#define PYRANNC_OBJECTCOMM_H

#include "SCommCommon.h"

namespace rannc {
    class ObjectComm {

    public:
        static ObjectComm &get() {
            static ObjectComm instance;
            return instance;
        }

        template<typename T>
        void send(T& obj, int dest, int tag, MPI_Comm comm=MPI_COMM_WORLD) {
            std::vector<char> data = serialize(obj);

            size_t size = data.size();
            mpi::checkMPIResult(MPI_Send((void*) &size, 1, MPI_LONG, dest, tag, comm));
            mpi::checkMPIResult(MPI_Send(&data[0], size, MPI_BYTE, dest, tag, comm));
        }

        template<typename T>
        T recv(int src, int tag, MPI_Comm comm=MPI_COMM_WORLD) {
            size_t size;
            MPI_Status st;
            mpi::checkMPIResult(MPI_Recv((void*) &size, 1, MPI_LONG, src, tag, comm, &st));

            std::vector<char> buf(size);
            mpi::checkMPIResult(MPI_Recv(&buf[0], size, MPI_BYTE, src, tag, comm, &st));

            return deserialize<T>(buf);
        }

        template<typename T>
        std::vector<T> allgather(T& obj, MPI_Comm comm=MPI_COMM_WORLD) {
            std::vector<char> data = serialize(obj);

            std::vector<std::vector<char>> recv = doAllgather(data, comm);
            std::vector<T> results;
            results.reserve(recv.size());
            for (const auto& buf: recv) {
                results.push_back(deserialize<T>(buf));
            }

            return results;
        }

        template<typename T>
        T bcast(T& obj, int root=0, MPI_Comm comm=MPI_COMM_WORLD) {
            std::vector<char> data = serialize(obj);
            std::vector<char> recv = doBcast(data, root, comm);
            return deserialize<T>(recv);
        }

    private:
        std::vector<std::vector<char>> doAllgather(const std::vector<char>& data, MPI_Comm comm);
        std::vector<char> doBcast(const std::vector<char>& buf, int root, MPI_Comm comm);
    };
}

#endif //PYRANNC_OBJECTCOMM_H
