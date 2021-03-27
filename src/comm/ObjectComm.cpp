//
// Created by Masahiro Tanaka on 2019/09/20.
//

#include "ObjectComm.h"

namespace rannc {

    inline size_t max(const std::vector<size_t>& nums) {
        size_t max_val = 0;
        for (size_t n: nums) {
            if (max_val < n) {
                max_val = n;
            }
        }
        return max_val;
    }

    std::vector<std::vector<char>> ObjectComm::doAllgather(const std::vector<char>& data, MPI_Comm comm) {

        // get sizes
        size_t size = data.size();

        int comm_size = mpi::getSize(comm);
        auto all_sizes_buf = std::unique_ptr<size_t>(new size_t[comm_size]);
        mpi::checkMPIResult(MPI_Allgather(&size, 1, MPI_LONG, all_sizes_buf.get(), 1, MPI_LONG, comm));
        std::vector<size_t> all_sizes;
        all_sizes.reserve(comm_size);
        for (int i=0; i<comm_size; i++) {
            all_sizes.push_back(*(all_sizes_buf.get() + i));
        }
        size_t max_size = max(all_sizes);

        auto sendbuf = std::unique_ptr<char>(new char[max_size]);
        memcpy(sendbuf.get(), &data[0], data.size());
        auto recvbuf = std::unique_ptr<char>(new char[max_size * comm_size]);

        mpi::checkMPIResult(MPI_Allgather(
                sendbuf.get(), max_size, MPI_CHAR,
                recvbuf.get(), max_size, MPI_CHAR,
                comm));

        std::vector<std::vector<char>> results;
        for (int i=0; i<comm_size; i++) {
            size_t a_size = all_sizes.at(i);
            std::vector<char> buf(a_size);
            memcpy(&buf[0], recvbuf.get() + max_size*i, a_size);
            results.push_back(std::move(buf));
        }

        return results;
    }

    std::vector<char> ObjectComm::doBcast(const std::vector<char>& data, int root, MPI_Comm comm) {
        size_t size = data.size();
        mpi::checkMPIResult(MPI_Bcast((void*) &size, 1, MPI_LONG, root, comm));

        std::vector<char> buf(size);
        if (mpi::getRank(comm) == root) {
            memcpy(&buf[0], &data[0], size);
        }

        mpi::checkMPIResult(MPI_Bcast((void*) &buf[0], size, MPI_CHAR, root, comm));
        return buf;
    }
}
