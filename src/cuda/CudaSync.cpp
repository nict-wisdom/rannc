//
// Created by Masahiro Tanaka on 2021/12/10.
//

#include <boost/filesystem.hpp>

#include <comm/NCCLWrapper.h>
#include <Config.h>
#include "CudaSync.h"

namespace fs = boost::filesystem;

namespace {
constexpr int CHECK_INTERVAL = 10000;
constexpr char LOCK_PREFIX[] = "rannc_";
} // namespace

namespace rannc {

const std::string DEFAULT_CONF_DIR_NAME = "lock";
const std::string LOCK_PATH_VAR = "RANNC_WATCHDOG_LOCKFILE_DIR";

SyncWatchDog::SyncWatchDog() : stop_(true) {
  std::string str_lock_path;
  fs::path lock_dir_path;
  if (const char* lock_dir_var = std::getenv(LOCK_PATH_VAR.c_str())) {
    lock_dir_path = lock_dir_var;
  } else {
    config::Config& config = config::Config::get();
    std::string lock_dir_str =
        config.getVal<std::string>(config::WATCHDOG_LOCKFILE_DIR);
    if (lock_dir_str.empty()) {
      fs::path conf_dir = config.getVal<std::string>(config::CONF_DIR);
      lock_dir_path = conf_dir / LOCK_PATH_VAR;
    } else {
      lock_dir_path = lock_dir_str;
    }
  }

  int my_rank = mpi::getRank();
  if (my_rank == 0) {
    fs::create_directories(lock_dir_path);
  }

  std::stringstream my_rank_ss;
  my_rank_ss << LOCK_PREFIX << my_rank << ".lock";
  my_rank_lock_path_ = lock_dir_path / my_rank_ss.str();

  for (int i = 0; i < mpi::getSize(); i++) {
    if (i != my_rank) {
      std::stringstream ss;
      ss << LOCK_PREFIX << i << ".lock";
      other_rank_lock_paths_.push_back(lock_dir_path / ss.str());
    }
  }
}

void SyncWatchDog::run_loop() {
  while (true) {
    std::ofstream output(my_rank_lock_path_.c_str());

    std::this_thread::sleep_for(std::chrono::milliseconds(CHECK_INTERVAL));
    if (stop_) {
      break;
    }

    for (const auto& p : other_rank_lock_paths_) {
      if (!fs::exists(p)) {
        logger->warn("lockfile not found: {}", p.string());
        NCCLWrapper& nccl = NCCLWrapper::get();
        nccl.abortAllCommunicators();
        logger->warn("Aborted all communicators");
        return;
      }
    }
  }
}

void SyncWatchDog::start() {
  std::ofstream output(my_rank_lock_path_.c_str());
  logger->info("Starting watchdog ...");
  MPI_Barrier(MPI_COMM_WORLD);

  if (stop_) {
    stop_ = false;
    watch_th_ = std::thread([this]() { this->run_loop(); });
  }
}

void SyncWatchDog::stop(bool error) {
  if (!stop_) {
    stop_ = true;
    watch_th_.join();
    if (!error) {
      MPI_Barrier(MPI_COMM_WORLD);
    }
    logger->info("Removing lock file: {}", my_rank_lock_path_.c_str());
    fs::remove(my_rank_lock_path_);
    logger->info("Stopped watchdog");
  }
}

void syncWithErrorCheck() {
  NCCLWrapper& nccl = NCCLWrapper::get();

  try {
    nccl.syncWithErrorCheck();
  } catch (CommErrorException& e) {
    SyncWatchDog& watch_dog = SyncWatchDog::get();
    watch_dog.stop(true);
    throw e;
  }
}

} // namespace rannc