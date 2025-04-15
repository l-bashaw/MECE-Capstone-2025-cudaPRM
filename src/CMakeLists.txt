cmake_minimum_required(VERSION 3.14)
project(pprm_cuda LANGUAGES CXX CUDA)

# Set policy and CUDA architectures
cmake_policy(SET CMP0104 NEW)
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80)

# Fetch pybind11 automatically via FetchContent
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1  # Specify desired version
)
FetchContent_MakeAvailable(pybind11)

# Locate Python (for headers and interpreter)
find_package(Python REQUIRED COMPONENTS Interpreter)

# Locate the CUDA Toolkit (to get the include directories)
find_package(CUDAToolkit REQUIRED)

# Set CUDA standard version
set(CMAKE_CUDA_STANDARD 14)

# Include directories for your project sources
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/planning)
include_directories(${PROJECT_SOURCE_DIR}/collision)
include_directories(${Python_INCLUDE_DIRS})

# List your source files: bindings and CUDA sources
set(SOURCES
    bindings/pybindings.cpp
    planning/pprm.cu
    planning/construction.cu
    collision/cc_2D.cu
    collision/env_2D.cu
)

# Create the shared module library for Python.
add_library(pprm_cuda MODULE ${SOURCES})

# Link with pybind11 module
target_link_libraries(pprm_cuda PRIVATE pybind11::module)

# Now add the CUDA include directories to the target
target_include_directories(pprm_cuda PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# Set target properties so that Python can load the module correctly.
set_target_properties(pprm_cuda PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                          SUFFIX "${PYTHON_MODULE_EXTENSION}")
