#pragma once
#include "env_2D.cuh"

namespace collision::environment{

    __device__ bool inCollisionCircle(
        float x,  
        float y, 
        Circle &c
    );

    __device__ bool inCollisionRectangle(
        float x,  
        float y,
        Rectangle &rect
    );

    __global__ void nodesInCollision(
        float* states, 
        bool* valid, 
        Env2D &env
    );

    __global__ void edgesInCollision(
        float* edges, 
        bool* valid, 
        Env2D &env
    );



}