#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "../planning/pprm.cuh"
#include "../collision/env_2D.cuh"

namespace py = pybind11;

PYBIND11_MODULE(cuPRM, m) {

    m.doc() = "Wowowowowowow"; // Optional module docstring

    m.def("create_env", [](torch::Tensor circles, torch::Tensor rectangles, torch::Tensor bounds){
        collision::environment::Env2D_d env;
        collision::environment::buildEnvFromTensors(env, circles, rectangles, bounds);
        return env;
    });

    

    py::class_<planning::Roadmap>(m, "Roadmap")
        .def(py::init<>())  // Default constructor
        .def_readwrite("h_states", &planning::Roadmap::h_states)
        .def_readwrite("d_states", &planning::Roadmap::d_states)
        .def_readwrite("h_edges", &planning::Roadmap::h_edges)
        .def_readwrite("d_edges", &planning::Roadmap::d_edges)
        .def_readwrite("h_neighbors", &planning::Roadmap::h_neighbors)
        .def_readwrite("d_neighbors", &planning::Roadmap::d_neighbors)
        .def_readwrite("h_validNodes", &planning::Roadmap::h_validNodes)
        .def_readwrite("d_validNodes", &planning::Roadmap::d_validNodes)
        .def_readwrite("h_validEdges", &planning::Roadmap::h_validEdges)
        .def_readwrite("d_validEdges", &planning::Roadmap::d_validEdges);

    
    
    m.def("setupEnv", &planning::setupEnv, "m");
    m.def("cleanupEnv", &planning::cleanupEnv, "m");

    m.def("allocateRoadmap", &planning::allocateRoadmap, "m");
    m.def("buildRoadmap", &planning::buildRoadmap, "m");
    m.def("freeRoadmap", &planning::freeRoadmap, "m");

    m.def("copyToHost", &planning::copyToHost, "m");

}


struct MyStruct {
    int a;
    double b;
};

int process_struct(const MyStruct& s) {
    return static_cast<int>(s.a + s.b);
}

PYBIND11_MODULE(my_module, m) {
    py::class_<MyStruct>(m, "MyStruct")
        .def(py::init<>())  // Default constructor
        .def_readwrite("a", &MyStruct::a)
        .def_readwrite("b", &MyStruct::b);

    m.def("process_struct", &process_struct, "Process a MyStruct instance");
}
