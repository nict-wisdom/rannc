cmake_minimum_required(VERSION 3.9)
project(pyrannc)

#SET(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 14)
set(PYBIND11_CPP_STANDARD -std=c++11)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_PREFIX_PATH
        $ENV{BOOST_DIR}
        $ENV{LIBTORCH_DIR}
        $ENV{MPI_DIR})

set(Boost_USE_STATIC_LIBS ON)
find_package(Boost COMPONENTS filesystem graph system)
find_package(CUDA REQUIRED)
find_package(MPI REQUIRED)
find_package(Torch REQUIRED)

set(CPATH $ENV{CPATH})
IF(CPATH)
    string(REPLACE ":" ";" ENV_INCLUDE_LIST ${CPATH})
    list(APPEND INCLUDE_LIST ${ENV_INCLUDE_LIST})
ENDIF()
list(APPEND INCLUDE_LIST
        "src"
        "submodules/spdlog/include"
        "submodules/spdlog_setup/include"
        "submodules/msgpack-c/include"
        "submodules/toml11"
        "submodules/json/single_include/nlohmann"
        ${TORCH_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${MPI_INCLUDE_PATH}
        ${Boost_INCLUDE_DIR})

include_directories(${INCLUDE_LIST})

set(LIBRARY_PATH $ENV{LIBRARY_PATH})
IF(LIBRARY_PATH)
    string(REPLACE ":" ";" ENV_LIBRARY_LIST ${LIBRARY_PATH})
    list(APPEND LIBRARY_LIST ${ENV_LIBRARY_LIST})
ENDIF()
link_directories(${LIBRARY_LIST})

option(MSGPACK_BUILD_EXAMPLES OFF)

add_subdirectory(submodules/pybind11)

set(RANNC_SRCS
        src/main.cpp
        src/Common.cpp
        src/Config.cpp
        src/Logging.cpp
        src/bind/PybindUtil.cpp
        src/bind/RaNNCFactory.cpp
        src/bind/RaNNCProcess.cpp
        src/bind/Tracer.cpp
        src/comm/MPIUtil.cpp
        src/comm/ObjectComm.cpp
        src/comp/Backward.cpp
        src/comp/FunctionStorage.cpp
        src/comp/GraphConnector.cpp
        src/comp/GraphLauncher.cpp
        src/comp/GraphValueStorage.cpp
        src/comp/ParamStorage.cpp
        src/comp/NodeProfiler.cpp
        src/comp/GraphProfiler.cpp
        src/comp/RaNNCModule.cpp
        src/comp/TimeCounter.cpp
        src/comp/EventRecorder.cpp
        src/comp/Validator.cpp
        src/cuda/CudaUtil.cpp
        src/graph/ConvertGraph.cpp
        src/graph/Decomposition.cpp
        src/graph/GuessValueTypes.cpp
        src/graph/FairWeightDecomposer.cpp
        src/graph/SchedulingDecomposer.cpp
#        src/graph/ManualDecomposer.cpp
        src/graph/ProfiledWeightDecomposer.cpp
        src/graph/MLGraph.cpp
        src/graph/DPStaging.cpp
        src/graph/ProfilerUtil.cpp
        src/graph/MLPartDecomposer.cpp
        src/graph/MetaDecomposer.cpp
        src/graph/DeploymentSerializer.cpp
        src/graph/Partitioner.cpp
        src/graph/balancer/AdjacentBalancer.cpp
        src/graph/balancer/VarianceBalancer.cpp
        src/graph/ir.cpp
        src/comm/SComm.cpp
        src/comm/SCommCommon.cpp
        src/comm/SCommPrimitive.cpp
        src/torch/IValueLocation.cpp
        src/torch/TorchDriver.cpp
        src/torch/TorchUtil.cpp)

if(NO_NCCL)
    list(APPEND RANNC_SRCS src/comm/MPIAllReduceRunner.cpp)
else()
    list(APPEND RANNC_SRCS src/comm/NCCLAllReduceRunner.cpp)
endif()

pybind11_add_module(_pyrannc ${RANNC_SRCS})
target_link_libraries(_pyrannc PRIVATE "${TORCH_LIBRARIES}" "${MPI_C_LIBRARIES}" "${MPI_CXX_LIBRARIES}" "${CUDA_LIBRARIES}" "${Boost_LIBRARIES}" nccl)

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
target_link_libraries(_pyrannc PRIVATE "${TORCH_PYTHON_LIBRARY}")

if(MPI_CXX_COMPILE_FLAGS)
    set_target_properties(_pyrannc PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
endif()
if(MPI_CXX_LINK_FLAGS)
    set_target_properties(_pyrannc PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
endif()

