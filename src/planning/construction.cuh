#pragma once

namespace prm::construction
{
    __global__ void generateStates(
        float* states, 
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
