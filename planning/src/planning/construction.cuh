#pragma once
#include "../collision/cc_2D.cuh"
#include "../collision/env_2D.cuh"

namespace prm::construction
{
    // __global__ void generateStates(
    //     float* states, 
    //     unsigned long seed
    // );

    __global__ void generateStates(
        float* states, 
        bool* valid, 
        collision::environment::Env2D &env, 
        Bounds bounds,
        unsigned long seed
    );

    __global__ void findNeighbors(
        const float* states, 
        int* neighbors
    ); 

    __global__ void findNeighborsForSingleState(
        const float* roadmap_states,
        const float* single_state,
        int* neighbors
    );

    __global__ void interpolate(
        const float* states, 
        int* neighbors, 
        float* edges
    );

    __global__ void interpolateForSingleState(
        const float* single_state,  
        const int* neighbors,       
        const float* roadmap_states,
        float* edges                
    );

} // namespace prm::construction
