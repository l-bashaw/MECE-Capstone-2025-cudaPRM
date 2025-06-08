#include "py_bind.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include "../params/hyperparameters.cuh"
#include "../planning/pprm.cuh"
#include "../collision/env_2D.cuh"

namespace prm_bindings {

    Bounds tensor_to_bounds(torch::Tensor bounds) {
        TORCH_CHECK(bounds.dim() == 2 && bounds.size(0) == 2 && bounds.size(1) == 5, 
                   "bounds must be [2, 5] tensor");
        
        auto bounds_cpu = bounds.cpu();
        Bounds result;
        
        for(int i = 0; i < 5; i++) {
            result.lower[i] = bounds_cpu[0][i].item<float>();
            result.upper[i] = bounds_cpu[1][i].item<float>();
        }
        
        return result;
    }

    collision::environment::Env2D* tensor_to_env2d_device(
        torch::Tensor circles,
        torch::Tensor rectangles
    ) {
        TORCH_CHECK(circles.device().is_cuda(), "circles must be on CUDA device");
        TORCH_CHECK(rectangles.device().is_cuda(), "rectangles must be on CUDA device");
        TORCH_CHECK(circles.is_contiguous(), "circles must be contiguous");
        TORCH_CHECK(rectangles.is_contiguous(), "rectangles must be contiguous");
        
        // Allocate the env struct on device
        collision::environment::Env2D* env_d;
        cudaMalloc(&env_d, sizeof(collision::environment::Env2D));
        
        // Create host copy to populate
        collision::environment::Env2D env_host = {};
        env_host.numCircles = circles.size(0);
        env_host.numRectangles = rectangles.size(0);
        
        // Allocate and copy circles
        if (env_host.numCircles > 0) {
            cudaMalloc(&env_host.circles, env_host.numCircles * sizeof(collision::environment::Circle));
            cudaMemcpy(env_host.circles, circles.data_ptr<float>(),
                    env_host.numCircles * sizeof(collision::environment::Circle),
                    cudaMemcpyDeviceToDevice);
        }
        
        // Allocate and copy rectangles  
        if (env_host.numRectangles > 0) {
            cudaMalloc(&env_host.rectangles, env_host.numRectangles * sizeof(collision::environment::Rectangle));
            cudaMemcpy(env_host.rectangles, rectangles.data_ptr<float>(),
                    env_host.numRectangles * sizeof(collision::environment::Rectangle),
                    cudaMemcpyDeviceToDevice);
        }
        
        // Copy the populated struct to device
        cudaMemcpy(env_d, &env_host, sizeof(collision::environment::Env2D), cudaMemcpyHostToDevice);
        
        cudaCheckErrors("Failed to setup environment");
        return env_d;
    }

    std::vector<torch::Tensor> roadmap_to_tensors(const planning::Roadmap& prm) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        auto bool_options = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);
        auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
        
        // Create tensors and copy data from device roadmap
        auto nodes = torch::from_blob(prm.d_states, {NUM_STATES, DIM}, options).clone();
        auto node_validity = torch::from_blob(prm.d_validNodes, {NUM_STATES}, bool_options).clone();
        auto neighbors = torch::from_blob(prm.d_neighbors, {NUM_STATES, K}, int_options).clone();
        auto edges = torch::from_blob(prm.d_edges, {NUM_STATES, K, INTERP_STEPS, DIM}, options).clone();
        auto edge_validity = torch::from_blob(prm.d_validEdges, {NUM_STATES, K}, bool_options).clone();
        
        return {nodes, node_validity, neighbors, edges, edge_validity};
    }

    std::vector<torch::Tensor> build_prm(
            torch::Tensor circles,
            torch::Tensor rectangles, 
            torch::Tensor bounds,
            unsigned long seed
        ) {
        // Validate inputs
        TORCH_CHECK(circles.dim() == 2 && circles.size(1) == 3, 
                   "circles must be [N, 3] tensor");
        TORCH_CHECK(rectangles.dim() == 2 && rectangles.size(1) == 4, 
                   "rectangles must be [N, 4] tensor");
        
        // Convert bounds to struct
        Bounds bounds_struct = tensor_to_bounds(bounds);
        
        // Create environment directly on device - no setupEnv needed!
        auto env_d = tensor_to_env2d_device(circles, rectangles);
        
        // Build roadmap
        planning::Roadmap prm;
        planning::allocateRoadmap(prm);
        planning::buildRoadmap(prm, env_d, bounds_struct, seed);
        cudaCheckErrors("Roadmap construction failure");
        
        // Convert roadmap to tensors
        auto result = roadmap_to_tensors(prm);
        
        // Cleanup
        planning::freeRoadmap(prm);
        planning::cleanupEnv(env_d);  
        
        return result;
    }

}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Parallel PRM Python bindings";
    
    m.def("build_prm", &prm_bindings::build_prm, 
          "Build Probabilistic Roadmap",
          pybind11::arg("circles"),
          pybind11::arg("rectangles"), 
          pybind11::arg("bounds"),
          pybind11::arg("seed") = 12345);
}