#pragma once
#include "env_2D.cuh"

namespace collision::environment{

    __device__ bool inCollisionCircle(
        float x,  
        float y, 
        circle &c
    );

    __device__ bool inCollisionRectangle(
        float x,  
        float y,
        rectangle &rect
    );

    __global__ void nodesInCollision(
        float* states, 
        bool* valid, 
        env_2D &env
    );

    __global__ void edgesInCollision(
        float* edges, 
        bool* valid, 
        env_2D &env
    );



}