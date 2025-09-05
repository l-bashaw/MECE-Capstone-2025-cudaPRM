#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include "../collision/env_2D.cuh"
#include "../planning/pprm.cuh"

namespace prm_bindings {
    
    // Main function to build PRM from PyTorch tensors
    std::vector<torch::Tensor> build_prm(
        torch::Tensor circles,      // [num_circles, 3] (x, y, radius)
        torch::Tensor rectangles,   // [num_rectangles, 4] (x, y, width, height)  
        torch::Tensor bounds,       // [2, 5] (lower_bounds, upper_bounds)
        unsigned long seed = 12345
    );
    
    // Helper functions
    collision::environment::Env2D tensor_to_env2d(
        torch::Tensor circles, 
        torch::Tensor rectangles
    );
    
    std::vector<torch::Tensor> roadmap_to_tensors(
        const planning::Roadmap& prm
    );
    
    void setup_bounds(torch::Tensor bounds);


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
    );

    planning::Roadmap tensors_to_roadmap(
        torch::Tensor nodes,
        torch::Tensor node_validity,
        torch::Tensor neighbors,
        torch::Tensor edges,
        torch::Tensor edge_validity
    );

}