#include "python_bindings.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include "params/hyperparameters.cuh"
#include "planning/pprm.cuh"
#include "collision/env_2D.cuh"

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

    collision::environment::Env2D tensor_to_env2d(
        torch::Tensor circles, 
        torch::Tensor rectangles
    ) {
        TORCH_CHECK(circles.device().is_cuda(), "circles must be on CUDA device");
        TORCH_CHECK(rectangles.device().is_cuda(), "rectangles must be on CUDA device");
        
        collision::environment::Env2D env;
        env.numCircles = circles.size(0);
        env.numRectangles = rectangles.size(0);
        
        // Allocate device memory for obstacles
        if (env.numCircles > 0) {
            cudaMalloc(&env.circles, env.numCircles * sizeof(collision::environment::Circle));
            
            // Convert tensor data to Circle structs
            auto circles_cpu = circles.cpu();
            std::vector<collision::environment::Circle> circle_data(env.numCircles);
            
            for (unsigned int i = 0; i < env.numCircles; i++) {
                circle_data[i].x = circles_cpu[i][0].item<float>();
                circle_data[i].y = circles_cpu[i][1].item<float>();
                circle_data[i].r = circles_cpu[i][2].item<float>();
            }
            
            cudaMemcpy(env.circles, circle_data.data(), 
                      env.numCircles * sizeof(collision::environment::Circle), 
                      cudaMemcpyHostToDevice);
        } else {
            env.circles = nullptr;
        }
        
        if (env.numRectangles > 0) {
            cudaMalloc(&env.rectangles, env.numRectangles * sizeof(collision::environment::Rectangle));
            
            // Convert tensor data to Rectangle structs
            auto rectangles_cpu = rectangles.cpu();
            std::vector<collision::environment::Rectangle> rect_data(env.numRectangles);
            
            for (unsigned int i = 0; i < env.numRectangles; i++) {
                rect_data[i].x = rectangles_cpu[i][0].item<float>();
                rect_data[i].y = rectangles_cpu[i][1].item<float>();
                rect_data[i].w = rectangles_cpu[i][2].item<float>();
                rect_data[i].h = rectangles_cpu[i][3].item<float>();
            }
            
            cudaMemcpy(env.rectangles, rect_data.data(), 
                      env.numRectangles * sizeof(collision::environment::Rectangle), 
                      cudaMemcpyHostToDevice);
        } else {
            env.rectangles = nullptr;
        }
        
        cudaCheckErrors("Failed to setup environment");
        return env;
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
            int num_states,
            int k,
            unsigned long seed
        ) {
        // Validate inputs
        TORCH_CHECK(circles.dim() == 2 && circles.size(1) == 3, 
                   "circles must be [N, 3] tensor");
        TORCH_CHECK(rectangles.dim() == 2 && rectangles.size(1) == 4, 
                   "rectangles must be [N, 4] tensor");
        
        // Convert bounds to struct
        Bounds bounds_struct = tensor_to_bounds(bounds);
        
        // Convert tensors to environment
        auto env_h = tensor_to_env2d(circles, rectangles);
        collision::environment::Env2D* env_d;
        planning::setupEnv(env_d, env_h);
        
        // Allocate and build roadmap (you'll need to update this to pass bounds)
        planning::Roadmap prm;
        planning::allocateRoadmap(prm);
        planning::buildRoadmap(prm, env_d, bounds_struct, seed);  // Pass bounds_struct
        cudaCheckErrors("Roadmap construction failure");
        
        // Copy results to host for tensor conversion
        planning::copyToHost(prm);
        
        // Convert roadmap to tensors
        auto result = roadmap_to_tensors(prm);
        
        // Cleanup
        planning::freeRoadmap(prm);
        planning::cleanupEnv(env_d, env_h);
        
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
          pybind11::arg("num_states") = 5000,
          pybind11::arg("k") = 10,
          pybind11::arg("seed") = 12345);
}