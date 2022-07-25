//
// Created by Masahiro Tanaka on 2022/07/21.
//

#include "KinetoWrapper.h"

namespace rannc {

void KinetoWrapper::start() {
  if (!enabled_) {
    return;
  }
  torch::autograd::profiler::prepareProfiler(config_, activities_);
  torch::autograd::profiler::enableProfiler(config_, activities_, scopes_);
}

void KinetoWrapper::stop(const std::string& key) {
  if (!enabled_) {
    return;
  }
  auto profiler_results_ptr = torch::autograd::profiler::disableProfiler();

  long cuda_time_sum = 0;
  long cpu_time_sum = 0;
  for (const auto& e : profiler_results_ptr->events()) {
    if (e.deviceType() == c10::DeviceType::CUDA) {
      cuda_time_sum += e.durationUs();
    } else {
      cpu_time_sum += e.durationUs();
    }
  }

  if (counts_[key] >= warmup_) {
    cuda_times_[key] += cuda_time_sum;
    cpu_times_[key] += cpu_time_sum;
    counts_valid_[key]++;
  }

  counts_[key]++;
}
} // namespace rannc