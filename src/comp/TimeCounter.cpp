//
// Created by Masahiro Tanaka on 2019-04-17.
//

#include <Common.h>
#include "TimeCounter.h"

namespace rannc {

    void TimeCounter::start(const std::string& task) {
        if (enabled_) {
            start_[task] = std::chrono::system_clock::now();
            count_[task]++;
        }
    }

    void TimeCounter::stop(const std::string& task) {
        if (enabled_) {
            if (!contains(start_, task)) {
                throw std::invalid_argument("Task is not running: " + task);
            }
            auto &start = start_.at(task);
            auto end = std::chrono::system_clock::now();
            duration_[task] += end - start;
            start_.erase(task);
        }
    }

    long TimeCounter::getCount(const std::string& task) const {
        if (enabled_) {
            if (!contains(count_, task)) {
                throw std::invalid_argument("No record foound for task: " + task);
            }
            return count_.at(task);
        }
        return -1;
    }

    std::vector<std::string> TimeCounter::getTasks() const {
        return keys(duration_);
    }

    bool TimeCounter::hasRecord(const std::string& task) const {
        return contains(duration_, task);
    }

    void TimeCounter::clear() {
        start_.clear();
        duration_.clear();
        count_.clear();
    }

    void TimeCounter::enable(bool enable) {
        enabled_ = enable;
    }

    bool TimeCounter::isEnabled() const {
        return enabled_;
    }
}
