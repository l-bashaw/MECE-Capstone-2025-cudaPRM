#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>         // optional
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>
#include <cuda_runtime.h>
#include "../planning/pprm.cuh"

namespace py = pybind11;

class PyRoadmap {
public:
    planning::Roadmap roadmap_;
    
    // Expose raw device pointer to states array
    uintptr_t d_states() const { 
        return reinterpret_cast<uintptr_t>(roadmap_.d_states); 
    }

    // Expose raw device pointer to edges array
    uintptr_t d_edges() const { 
        return reinterpret_cast<uintptr_t>(roadmap_.d_edges); 
    }

    // Expose raw device pointer to neighbors array
    uintptr_t d_neighbors() const { 
        return reinterpret_cast<uintptr_t>(roadmap_.d_neighbors); 
    }

    // Expose raw device pointer to valid nodes array
    uintptr_t d_valid_nodes() const { 
        return reinterpret_cast<uintptr_t>(roadmap_.d_validNodes); 
    }

    // Expose raw device pointer to valid edges array
    uintptr_t d_valid_edges() const { 
        return reinterpret_cast<uintptr_t>(roadmap_.d_validEdges); 
    }

    // Constructor, allocate roadmap
    PyRoadmap() {
        planning::allocateRoadmap(roadmap_);
    }

    // Destructor, free roadmap
    ~PyRoadmap() {
        planning::freeRoadmap(roadmap_);
    }
};

// Pybind11 module definition
PYBIND11_MODULE(pprm_cuda, m) {
    py::class_<PyRoadmap>(m, "PyRoadmap")
        .def(py::init<>())
        .def("d_states", &PyRoadmap::d_states)
        .def("d_edges", &PyRoadmap::d_edges)
        .def("d_neighbors", &PyRoadmap::d_neighbors)
        .def("d_valid_nodes", &PyRoadmap::d_valid_nodes)
        .def("d_valid_edges", &PyRoadmap::d_valid_edges);
}
