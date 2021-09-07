//
// Created by Masahiro Tanaka on 2020/06/02.
//

#ifndef PYRANNC_EVENTRECORDER_H
#define PYRANNC_EVENTRECORDER_H

#include <comm/SCommCommon.h>
#include <Common.h>
#include <msgpack.hpp>
#include <chrono>
#include <queue>
#include <utility>

namespace rannc {

struct Event {
  int rank;
  std::string name;
  std::string phase;
  size_t time;

  Event() {}

  Event(int rank, std::string name, std::string phase, size_t time)
      : rank(rank),
        name(std::move(name)),
        phase(std::move(phase)),
        time(time) {}

  MSGPACK_DEFINE(rank, name, phase, time);
};

class EventRecorder {
 public:
  static EventRecorder& get();

  void start(const std::string& name);
  void stop(const std::string& name);
  std::string dump();
  void dump(const std::string& path);

  bool isEnabled() const {
    return enabled_;
  }

  void enable(bool enabled) {
    enabled_ = enabled;
  }

 private:
  EventRecorder();

  std::queue<Event> events_;
  bool enabled_;

  static const size_t MAX_EVENT_NUM = 100000;
};

void recordStart(const std::string& key);
void recordEnd(const std::string& key);

std::string getFuncKey(
    const std::string& prefix, const std::string& func, const std::string& id,
    int split, bool grad);
std::string getCommKey(
    const std::string& prefix, const std::string& direction,
    const rannc::RouteDP& r, int split);
std::string getCommKey(
    const std::string& prefix, const std::string& direction,
    const rannc::RouteDP& r, int split, const IRType& type);
std::string getCopyKey(
    const std::string& prefix, const std::string& func, const std::string& name,
    const IRType& type);

struct TraceEvent {
  TraceEvent(std::string key) : key_(std::move(key)) {
    recordStart(key_);
  }

  ~TraceEvent() {
    recordEnd(key_);
  }
  std::string key_;
};
} // namespace rannc

#endif // PYRANNC_EVENTRECORDER_H
