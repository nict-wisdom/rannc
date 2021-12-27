//
// Created by Masahiro Tanaka on 2021/12/10.
//

#ifndef PYRANNC_CUDASYNC_H
#define PYRANNC_CUDASYNC_H

#include <boost/filesystem.hpp>
#include <thread>

namespace rannc {

class SyncWatchDog {
 public:
  static SyncWatchDog& get() {
    static SyncWatchDog instance;
    return instance;
  }

  bool isRunning() const {
    return !stop_;
  }

  void start();
  void stop(bool error);

 private:
  SyncWatchDog();
  void run_loop();

  std::thread watch_th_;
  bool stop_;
  fs::path my_rank_lock_path_;
  std::vector<fs::path> other_rank_lock_paths_;

  const std::shared_ptr<spdlog::logger> logger = getLogger("SyncWatchDog");
};

void syncWithErrorCheck();

} // namespace rannc

#endif // PYRANNC_CUDASYNC_H
