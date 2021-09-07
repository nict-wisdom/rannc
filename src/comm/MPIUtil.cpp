//
// Created by Masahiro Tanaka on 2018-12-27.
//
#include <mpi.h>

#include <torch/torch.h>

#include "Common.h"
#include "MPIUtil.h"

namespace {
const int RANNC_TAG_UB = 30000;
}

namespace mpi {

int getRank() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

int getRank(MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
}

bool isMaster() {
  return getRank() == 0;
}

int getSize() {
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return size;
}

int getSize(MPI_Comm comm) {
  int size;
  MPI_Comm_size(comm, &size);
  return size;
}

int getTypeSize(MPI_Datatype datatype) {
  int size;
  MPI_Type_size(datatype, &size);
  return size;
}

int getTagUB() {
  //        int ub;
  //        int flag;
  //        MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &ub, &flag);
  //
  //        return ub;
  //// The above returns different values in different ranks.

  return RANNC_TAG_UB;
}

void checkMPIResult(int code) {
  if (code == MPI_SUCCESS)
    return;

  char err_str[MPI_MAX_ERROR_STRING];
  char err_cls_str[MPI_MAX_ERROR_STRING];

  int err_cls;
  MPI_Error_class(code, &err_cls);

  int err_len, err_cls_len;
  MPI_Error_string(code, err_str, &err_len);
  MPI_Error_string(err_cls, err_cls_str, &err_cls_len);

  std::stringstream ss;
  ss << "MPI error: " << err_str << " error_class: " << err_cls_str;
  std::string msg = ss.str();
  throw MPIException(msg);
}

std::string getMPILibraryVersion() {
  char ver_str[MPI_MAX_LIBRARY_VERSION_STRING];
  int len;
  MPI_Get_library_version(ver_str, &len);
  return ver_str;
}

std::string getTypeName(MPI_Datatype datatype) {
  char name[MPI_MAX_OBJECT_NAME];
  int namelen;
  MPI_Type_get_name(datatype, name, &namelen);
  name[namelen] = '\0';
  return name;
}

std::string getProcessorName() {
  int len;
  char name[MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name(name, &len);

  return name;
}

template <>
int generateTag(const std::unordered_set<int>& n) {
  int hash = rannc::IntSetHash()(n);
  return std::abs(hash) % mpi::getTagUB();
}

std::unordered_set<int> getRanks(int idx_start) {
  std::unordered_set<int> ranks;
  int size = mpi::getSize();
  for (int i = idx_start; i < size; i++) {
    ranks.insert(i);
  }
  return ranks;
}

std::unordered_set<int> getRanksInComm(MPI_Comm comm) {
  int size = getSize(comm);
  MPI_Group grp;
  MPI_Comm_group(comm, &grp);

  MPI_Group world_grp;
  MPI_Comm_group(MPI_COMM_WORLD, &world_grp);

  std::vector<int> grp_ranks;
  grp_ranks.reserve(size);
  for (int i = 0; i < size; i++) {
    grp_ranks.push_back(i);
  }

  std::vector<int> translated_ranks(size);
  MPI_Group_translate_ranks(
      grp, size, &grp_ranks[0], world_grp, &translated_ranks[0]);

  std::unordered_set<int> ret;
  for (const auto r : translated_ranks) {
    ret.insert(r);
  }
  return ret;
}

std::unordered_set<int> getAllRanks() {
  return getRanks(0);
}

std::unordered_set<int> getAllWorkerRanks() {
  return getRanks(1);
}

int64_t allReduceSumBatchSize(int64_t batch_size, MPI_Comm comm) {
  int64_t sum = 0;
  mpi::checkMPIResult(
      MPI_Allreduce(&batch_size, &sum, 1, MPI_LONG, MPI_SUM, comm));
  return sum;
}

int64_t allReduceMaxBatchSize(int64_t batch_size, MPI_Comm comm) {
  int64_t sum = 0;
  mpi::checkMPIResult(
      MPI_Allreduce(&batch_size, &sum, 1, MPI_LONG, MPI_MAX, comm));
  return sum;
}
} // namespace mpi
