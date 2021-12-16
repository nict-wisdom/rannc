//
// Created by Masahiro Tanaka on 2021/12/10.
//

#include "CudaSync.h"
#include <comm/NCCLWrapper.h>

namespace rannc {

void syncWithErrorCheck() {
  NCCLWrapper& nccl = NCCLWrapper::get();
  nccl.syncWithErrorCheck();
}

} // namespace rannc