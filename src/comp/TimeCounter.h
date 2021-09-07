//
// Created by Masahiro Tanaka on 2019-04-17.
//

#ifndef PYRANNC_TIMECOUNTER_H
#define PYRANNC_TIMECOUNTER_H

#include <chrono>

#include "Common.h"

namespace rannc {
class TimeCounter {
 public:
  TimeCounter() : enabled_(false) {}
  TimeCounter(bool enable) : enabled_(enable) {}

  void start(const std::string& task);
  void stop(const std::string& task);
  long getCount(const std::string& task) const;
  long long get(const std::string& task) const;
  std::vector<std::string> getTasks() const;
  bool hasRecord(const std::string& task) const;
  void clear();

  template <typename U>
  long long get(const std::string& task) const;

  template <typename U>
  std::string summary() const;

  void enable(bool enabled);
  bool isEnabled() const;

 private:
  bool enabled_;

  std::unordered_map<std::string, std::chrono::system_clock::time_point> start_;
  std::unordered_map<std::string, std::chrono::system_clock::duration>
      duration_;
  std::unordered_map<std::string, long> count_;
};

template <typename U>
long long TimeCounter::get(const std::string& task) const {
  if (enabled_) {
    if (contains(start_, task)) {
      throw std::invalid_argument("Task is still running: " + task);
    }
    if (!contains(duration_, task)) {
      throw std::invalid_argument("No record found for task: " + task);
    }
    long long average =
        std::chrono::duration_cast<U>(duration_.at(task)).count() /
        count_.at(task);
    return average;
  }
  return -1;
}

template <typename U>
std::string TimeCounter::summary() const {
  if (enabled_) {
    std::stringstream ss;
    for (const auto& task : getTasks()) {
      long count = getCount(task);
      ss << "Task " << task << ": average time=" << get<U>(task)
         << " count=" << count << std::endl;
    }
    return ss.str();
  }
  return "Profiling is disabled.";
}

template <typename T, typename F>
T measureTime(
    F f, int iter, int min_iter, TimeCounter& counter, const std::string& id) {
  if (iter >= min_iter) {
    counter.start(id);
  }
  T ret = f();
  if (iter >= min_iter) {
    counter.stop(id);
  }
  return ret;
}

template <typename F>
void measureTime(
    F f, int iter, int min_iter, TimeCounter& counter, const std::string& id) {
  if (iter >= min_iter) {
    counter.start(id);
  }
  f();
  if (iter >= min_iter) {
    counter.stop(id);
  }
}
} // namespace rannc

#endif // PYRANNC_TIMECOUNTER_H
