//
// Created by Masahiro Tanaka on 2020/06/02.
//

#include "EventRecorder.h"

#include <json.hpp>
#include <comm/MPIUtil.h>
#include <comm/ObjectComm.h>
#include <Config.h>


namespace rannc {

    inline size_t getTimePoint() {
        auto t = std::chrono::system_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(t.time_since_epoch()).count();
    }

    EventRecorder &EventRecorder::get() {
        static EventRecorder instance;

        return instance;
    }

    EventRecorder::EventRecorder() {
        enabled_ = config::Config::get().getVal<bool>(config::TRACE_EVENTS);
    }

    void EventRecorder::start(const std::string &name) {
        if (enabled_) {
            if (events_.size() >= MAX_EVENT_NUM) {
                events_.pop();
            }
            events_.emplace(mpi::getRank(), name, "B", getTimePoint());
        }
    }

    void EventRecorder::stop(const std::string &name) {
        if (enabled_) {
            if (events_.size() >= MAX_EVENT_NUM) {
                events_.pop();
            }
            events_.emplace(mpi::getRank(), name, "E", getTimePoint());
        }
    }

    std::string EventRecorder::dump() {
        ObjectComm& ocomm = ObjectComm::get();

        std::vector<Event> vec_events;
        vec_events.reserve(events_.size());
        while (!events_.empty()) {
            vec_events.push_back(events_.front());
            events_.pop();
        }

        const std::vector<std::vector<Event>> gathered_events = ocomm.allgather(vec_events, MPI_COMM_WORLD);

        std::vector<nlohmann::json> js_events;
        for (const auto& evts: gathered_events) {
            for (const auto& evt: evts) {
                nlohmann::json obj;
                obj["name"] = evt.name;
                obj["ph"] = evt.phase;
                obj["pid"] = evt.rank;
                obj["tid"] = 0;
                obj["ts"] = evt.time * 1000;

                js_events.push_back(obj);
            }
        }

        nlohmann::json trace;
        trace["traceEvents"] = js_events;
        trace["displayTimeUnit"] = "ns";
        return trace.dump();
    }

    void EventRecorder::dump(const std::string& path) {
        EventRecorder& erec = EventRecorder::get();
        if (isEnabled()) {
            const auto& dump = erec.dump();

            if (mpi::getRank() == 0) {
                auto logger = getLogger("main");
                logger->info("Saving event trace to {}", path);

                std::ofstream out(path, std::ios::out);
                if (!out) {
                    throw std::invalid_argument("Failed to open file: " + path);
                }
                out << dump;
                out.close();
            }
        }
    }

    void recordStart(const std::string &key) {
        EventRecorder& erec = EventRecorder::get();
        if (erec.isEnabled()) {
            erec.start(key);
        }
    }

    void recordEnd(const std::string &key) {
        EventRecorder& erec = EventRecorder::get();
        if (erec.isEnabled()) {
            erec.stop(key);
        }
    }
}