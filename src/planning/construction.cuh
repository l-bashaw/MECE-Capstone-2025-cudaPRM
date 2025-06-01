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
        unsigned long seed
    );

    __global__ void findNeighbors(
        const float* states, 
        int* neighbors
    ); 

    __global__ void lin_interp(
        const float* states, 
        int* neighbors, 
        float* edges
    );

} // namespace prm::construction
