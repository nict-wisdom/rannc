//
// Created by Masahiro Tanaka on 2020/06/02.
//

#ifndef PYRANNC_EVENTRECORDER_H
#define PYRANNC_EVENTRECORDER_H

#include <chrono>
#include <msgpack.hpp>
#include <utility>
#include <Common.h>
#include <queue>

namespace rannc {

    struct Event {
        int rank;
        std::string name;
        std::string phase;
        size_t time;

        Event() {}

        Event(int rank, std::string name, std::string phase, size_t time)
            : rank(rank), name(std::move(name)), phase(std::move(phase)), time(time) {}

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

    private:
        EventRecorder();

        std::queue<Event> events_;
        bool enabled_;

        static const size_t MAX_EVENT_NUM = 100000;
    };

    void recordStart(const std::string &key);
    void recordEnd(const std::string &key);
}

#endif //PYRANNC_EVENTRECORDER_H
