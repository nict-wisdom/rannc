#include <iostream>

#include <pybind11/pybind11.h>
#include <mpi.h>

#include <comm/NCCLWrapper.h>
#include <comp/Backward.h>
#include <comm/SComm.h>
#include <bind/PybindUtil.h>
#include <comp/EventRecorder.h>

#include "Logging.h"
#include "comm/MPIUtil.h"
#include "bind/RaNNCProcess.h"
#include "comp/RaNNCModule.h"
#include "bind/Tracer.h"
#include "bind/RaNNCFactory.h"
#include "comp/DistributedParamLocator.h"


namespace py = pybind11;
using namespace rannc;


PYBIND11_MODULE(_pyrannc, m) {
    m.add_object("_cleanup", py::capsule([]() {
        EventRecorder& erec = EventRecorder::get();
        erec.dump(config::Config::get().getVal<std::string>(config::EVENT_TRACE_FILE));

        auto process = RaNNCFactory::get();
        for (auto& it: process->getModules()) {
            it.second->destroy();
        }
        process->clear();

        MPI_Finalize();
    }));

    m.def("clear", []() {
        auto process = RaNNCFactory::get();
        for (auto& it: process->getModules()) {
            it.second->destroy();
        }
        process->clear();
    });

    m.def("get_rank", []() {
        return mpi::getRank();
    });

    m.def("get_world_size", []() {
        return mpi::getSize();
    });

    m.def("barrier", []() {
        MPI_Barrier(MPI_COMM_WORLD);
    });

    m.def("delay_grad_allreduce", [](bool delay) {
        return RaNNCTensorBackward::setDelayGradAllreduce(delay);
    });

    m.def("sync_params_on_init", [](bool sync) {
        return ParamStorage::syncOnInit(sync);
    });

    m.def("allreduce_tensor", [](py::handle py_tensor, bool sum) {
        auto iv = torch::jit::_toTypeInferredIValue(py_tensor);
        assert(iv.isTensor());

        NCCLWrapper& ar = NCCLWrapper::get();
        std::vector<at::Tensor> t = {iv.toTensor()};

        TagMap& tag_map = TagMap::get();
        int tag = tag_map.getRankSetTag(mpi::getAllRanks());

        ar.createCommunicator(tag, mpi::getAllRanks());
        if (sum) {
            ar.allreduce(tag, t);
        } else {
            ar.allreduceMin(tag, t);
        }
    });

    m.def("keep_graph", [](bool keep) {
        return TorchDriver::setKeepGraph(keep);
    });

    m.def("dump_events", []() {
        EventRecorder& erec = EventRecorder::get();
        if (!erec.isEnabled()) {
            auto logger = getLogger("main");
            logger->warn("Event tracing has not been enabled. No event was output.");
            return;
        }
        erec.dump(config::Config::get().getVal<std::string>(config::EVENT_TRACE_FILE));
    });

    py::class_<RaNNCProcess, std::shared_ptr<RaNNCProcess>>(m, "RaNNCMaster")
            .def("start", [](RaNNCProcess& self) {
                self.start();
            });

    m.def("get_rannc", []() {
        return RaNNCFactory::get();
    });

    m.def("enter_rank", [](int rank) {
        enterRank(rank);
    });

    m.def("exit_rank", []() {
        exitRank();
    });

    m.def("local_pid_to_global", [](long param_id) {
        auto r = RaNNCFactory::get();
        auto param_storage = r->getParamStorage();
        return param_storage->localToGlobal(param_id);
    });

    m.def("register_amp_master_param", [](long model_param_id, py::object& param) {
        long master_param_id = getPythonObjId(param);
        const auto ten = py::cast<at::Tensor>(param);

        auto r = RaNNCFactory::get();
        auto param_storage = r->getParamStorage();
        param_storage->registerAmpMasterParam(model_param_id, master_param_id, ten);
    });

    m.def("store_dist_param", [](py::object& param) {
        long pid = getPythonObjId(param);
        const auto ten = py::cast<at::Tensor>(param);
        DistributedParamLocator& zpl = DistributedParamLocator::get();
        return zpl.store(pid, ten);
    });

    m.def("load_dist_param", [](long pid) {
        DistributedParamLocator& zpl = DistributedParamLocator::get();
        return zpl.load(pid);
    });

    py::class_<RaNNCModule, std::shared_ptr<RaNNCModule>>(m, "RaNNCModule")
            .def(py::init<bool, bool, bool, bool>())
            .def("init", [](RaNNCModule& self,
                            const py::function& fwdFunc,
                            const std::vector<py::tuple>& params, const std::vector<py::tuple>& buffers,
                            const py::function& var_lookup_fn,
                            bool gather_inputs, const py::args& args) {
                try {
                    return self.init(fwdFunc, params, buffers, var_lookup_fn, args, gather_inputs);
                } catch (c10::Error& e) {
                    std::cerr << "Torch exception caught: " << e.what() << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, -1);
                } catch (std::runtime_error& e) {
                    std::cerr << "Runtime error caught: " << e.what() << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, -2);
                } catch (std::invalid_argument& e) {
                    std::cerr << "Invalid argument exception caught: " << e.what() << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, -3);
                } catch (std::exception& e) {
                    std::cerr << "Unknown exception caught: " << e.what() << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, -4);
                }
                std::cerr << "Failed to init model. exiting." << std::endl;
                std::exit(-5);
            })
            .def("__call__", [](RaNNCModule& self, py::args args, py::kwargs kwargs) {
                try {
                    return self(args, kwargs);
                } catch (c10::Error& e) {
                    std::cerr << "Torch exception caught: " << e.what() << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, -1);
                } catch (std::runtime_error& e) {
                    std::cerr << "Runtime error caught: " << e.what() << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, -2);
                } catch (std::invalid_argument& e) {
                    std::cerr << "Invalid argument exception caught: " << e.what() << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, -3);
                } catch (std::exception& e) {
                    std::cerr << "Unknown exception caught: " << e.what() << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, -4);
                }
                std::cerr << "Failed to compute forward. exiting." << std::endl;
                std::exit(-5);
            })
            .def("allreduce_grads", [](RaNNCModule& self) {
                self.allReduceParamGrads();
            })
            .def("clip_grad_norm", [](RaNNCModule& self, float max_grad_norm) {
                self.clipGrad(max_grad_norm);
            })
            .def("calc_grad_norm", [](RaNNCModule& self) {
                return self.calcGradL2Norm();
            })
            .def("is_checkpointing_enabled", [](RaNNCModule& self) {
                return self.isCheckpointingEnabled();
            })
            .def("zero_grad", [](RaNNCModule& self) {
                self.clearParamGrads();
            })
            .def("use_amp_master_params", [](RaNNCModule& self) {
                return self.useAmpMasterParams();
            })
            .def("undeploy", [](RaNNCModule& self) {
                self.destroy();
            })
            .def("sync_param", [](RaNNCModule& self, long param_id) {
                return self.syncParam(param_id);
            })
            .def("sync_param_grad", [](RaNNCModule& self, long param_id) {
                return self.syncParamGrad(param_id);
            })
            .def("sync_param_zero", [](RaNNCModule& self, bool grad) {
                return self.syncParamZero(grad);
            })
            .def("get_local_param_range", [](RaNNCModule& self, long param_id) {
                return self.getLocalParamRange(param_id);
            })
            .def("get_local_param_segment", [](RaNNCModule& self, long param_id) {
                return self.getLocalParamSegment(param_id);
            })
            .def("load_deployment", [](RaNNCModule& self, const std::string& file) {
                self.setLoadDeployment(true);
                self.setDeploymentFile(file);
            })
            .def("save_deployment", [](RaNNCModule& self, const std::string& file) {
                self.saveDeployment(file);
            })
            .def("__del__", [](RaNNCModule& self) {
                self.destroy();
            });

    m.def("send_bytes", [](const py::bytes& data, int dest) {
        const auto str_data = static_cast<std::string>(data);

        size_t size = str_data.size();
        mpi::checkMPIResult(MPI_Send((void*) &size, 1, MPI_LONG, dest, 10, MPI_COMM_WORLD));

        size_t offset = 0;
        while (size > offset) {
            size_t remaining = size - offset;
            size_t chunk_size = std::min((size_t) INT_MAX, remaining);
            mpi::checkMPIResult(MPI_Send((void*) (str_data.c_str() + offset), chunk_size, MPI_BYTE, dest, 10, MPI_COMM_WORLD));
            offset += chunk_size;
        }
    });

    m.def("recv_bytes", [](int src) {
        size_t size;
        MPI_Status st;
        mpi::checkMPIResult(MPI_Recv((void*) &size, 1, MPI_LONG, src, 10, MPI_COMM_WORLD, &st));

        std::string buf;
        buf.resize(size);

        size_t offset = 0;
        while (size > offset) {
            size_t remaining = size - offset;
            size_t chunk_size = std::min((size_t) INT_MAX, remaining);
            mpi::checkMPIResult(MPI_Recv((void*) (buf.c_str() + offset), chunk_size, MPI_BYTE, src, 10, MPI_COMM_WORLD, &st));
            offset += chunk_size;
        }

        return py::bytes(buf);
    });


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
