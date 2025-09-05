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

    planning::Roadmap tensors_to_roadmap(
        torch::Tensor nodes,
        torch::Tensor node_validity,
        torch::Tensor neighbors,
        torch::Tensor edges,
        torch::Tensor edge_validity
    ) {
        TORCH_CHECK(nodes.device().is_cuda(), "nodes must be on CUDA device");
        TORCH_CHECK(node_validity.device().is_cuda(), "node_validity must be on CUDA device");
        TORCH_CHECK(neighbors.device().is_cuda(), "neighbors must be on CUDA device");
        TORCH_CHECK(edges.device().is_cuda(), "edges must be on CUDA device");
        TORCH_CHECK(edge_validity.device().is_cuda(), "edge_validity must be on CUDA device");

        planning::Roadmap prm;

        // Allocate memory for the roadmap
        planning::allocateRoadmap(prm);

        // Copy data from tensors to the roadmap
        cudaMemcpy(prm.d_states, nodes.data_ptr<float>(), nodes.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(prm.d_validNodes, node_validity.data_ptr<bool>(), node_validity.numel() * sizeof(bool), cudaMemcpyDeviceToDevice);
        cudaMemcpy(prm.d_neighbors, neighbors.data_ptr<int>(), neighbors.numel() * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(prm.d_edges, edges.data_ptr<float>(), edges.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(prm.d_validEdges, edge_validity.data_ptr<bool>(), edge_validity.numel() * sizeof(bool), cudaMemcpyDeviceToDevice);

        cudaCheckErrors("Failed to copy tensors to roadmap");

        return prm;
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

    std::vector<torch::Tensor> addStartAndGoal_(
        torch::Tensor nodes,
        torch::Tensor node_validity,
        torch::Tensor neighbors,
        torch::Tensor edges,
        torch::Tensor edge_validity,
        torch::Tensor start,
        torch::Tensor goal,
        torch::Tensor circles,
        torch::Tensor rectangles
    ) {
        // Validate inputs
        TORCH_CHECK(start.dim() == 1 && start.size(0) == DIM, "start must be [DIM] tensor");   
        TORCH_CHECK(goal.dim() == 1 && goal.size(0) == DIM, "goal must be [DIM] tensor");
        TORCH_CHECK(circles.dim() == 2 && circles.size(1) == 3, "circles must be [N, 3] tensor");
        TORCH_CHECK(rectangles.dim() == 2 && rectangles.size(1) == 4, "rectangles must be [N, 4] tensor");

        // Convert tensors to roadmap
        planning::Roadmap prm = tensors_to_roadmap(nodes, node_validity, neighbors, edges, edge_validity);

        // Create environment on device
        auto env_d = tensor_to_env2d_device(circles, rectangles);

        // Allocate memory for start and goal
        planning::Start start_struct;
        planning::Goal goal_struct;
        planning::allocateStartAndGoal(start_struct, goal_struct);

        // Copy start and goal states to device
        cudaMemcpy(start_struct.d_state, start.data_ptr<float>(), start.numel() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(goal_struct.d_state, goal.data_ptr<float>(), goal.numel() * sizeof(float), cudaMemcpyHostToDevice);

        // Connect start and goal to the roadmap
        planning::connectStartAndGoal(start_struct, goal_struct, prm, env_d);
        
        // Print start and goal for debugging
        std::vector<float> start_host(DIM);
        std::vector<float> goal_host(DIM);
        cudaMemcpy(start_host.data(), start_struct.d_state, DIM * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(goal_host.data(), goal_struct.d_state, DIM * sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "Start state: ";
        for (const auto& val : start_host) std::cout << val << " ";
        std::cout << std::endl;
        std::cout << "Goal state: ";
        for (const auto& val : goal_host) std::cout << val << " ";
        std::cout << std::endl; 


        // Convert start and goal structs to tensors
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        auto bool_options = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);
        auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

        auto start_state = torch::from_blob(start_struct.d_state, {DIM}, options).clone();
        auto start_neighbors = torch::from_blob(start_struct.d_neighbors, {K}, int_options).clone();
        auto start_edges = torch::from_blob(start_struct.d_edges, {K, INTERP_STEPS, DIM}, options).clone();
        auto start_valid_edges = torch::from_blob(start_struct.d_validEdges, {K}, bool_options).clone();

        auto goal_state = torch::from_blob(goal_struct.d_state, {DIM}, options).clone();
        auto goal_neighbors = torch::from_blob(goal_struct.d_neighbors, {K}, int_options).clone();
        auto goal_edges = torch::from_blob(goal_struct.d_edges, {K, INTERP_STEPS, DIM}, options).clone();
        auto goal_valid_edges = torch::from_blob(goal_struct.d_validEdges, {K}, bool_options).clone();

        // Cleanup
        planning::freeStartAndGoal(start_struct, goal_struct);
        planning::cleanupEnv(env_d);
        planning::freeRoadmap(prm);

        return {start_state, start_neighbors, start_edges, start_valid_edges,
                goal_state, goal_neighbors, goal_edges, goal_valid_edges};
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

    m.def("addStartAndGoal_", &prm_bindings::addStartAndGoal_,
          "Add start and goal to existing roadmap",
          pybind11::arg("nodes"),
          pybind11::arg("node_validity"),
          pybind11::arg("neighbors"),
          pybind11::arg("edges"),
          pybind11::arg("edge_validity"),
          pybind11::arg("start"),
          pybind11::arg("goal"),
          pybind11::arg("circles"),
          pybind11::arg("rectangles"));
}