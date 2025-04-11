#include <cuda_runtime.h>
#include <iostream>

#include "params.cuh"
#include "env_2D.cuh"
#include "cc_2D.cuh"

namespace collision::environment{

    // Device functions to check for collisions between a circular robot and a circle or rectangle
    // These are used in the collision checking kernel
    __device__ bool inCollisionCircle(
        float x,  // Location of robot
        float y, 
        circle &c
    )
    {
        float distance_sq = (x - c.x) * (x - c.x) + (y - c.y) * (y - c.y); 
        return distance_sq <= (R_ROBOT + c.r) * (R_ROBOT + c.r) + CC_E;
    }

    __device__ bool inCollisionRectangle(
        float x,  // Location of robot
        float y,
        rectangle &rect)
    {
        float closest_x = fmax(rect.x - rect.w/2, fmin(x, rect.x + rect.w/2));  // Determine the closest point on the rectangle to the robot
        float closest_y = fmax(rect.y - rect.h/2, fmin(y, rect.y + rect.h/2));

        float distance_x = x - closest_x;
        float distance_y = y - closest_y;
        float distance_sq = (distance_x * distance_x) + (distance_y * distance_y);

        return distance_sq <= (R_ROBOT * R_ROBOT) + CC_E;
    }

    // Kernel to check if a state is in collision with the environment
    //  Each thread checks collision of one state with the entire environment.
    __global__ void nodeInCollision(float* states, bool* valid, env_2D &env)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= NUM_STATES) return;

        // Load state into registers
        float x = states[idx * 5 + 0];       // x position
        float y = states[idx * 5 + 1];       // y position
        // float theta = states[idx * 5 + 2];   // theta

        // Check circular obstacles for collisions
        for (int i = 0; i < env.numCircles; i++){
            if (inCollisionCircle(x, y, env.circles[i])){
                valid[idx] = false;
                return;
            }
        }

        // Check rectangular obstacles for collisions
        for (int i = 0; i < env.numRectangles; i++){
            if (inCollisionRectangle(x, y, env.rectangles[i])){
                valid[idx] = false;
                return;
            }
        }
        valid[idx] = true;
    }
}


// // Each thread checks collision of one edge with the entire environment.
// __global__ void edgeInCollision(float* edges, bool* valid, env_2D env)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
// }