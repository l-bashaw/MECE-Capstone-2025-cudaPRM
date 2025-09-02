#pragma once
#include "env_2D.cuh"

namespace collision::environment{
    // Device functions to check for collisions between a circular robot and a circle or rectangle
    // These are used in the collision checking kernel
    __device__ inline bool inCollisionCircle(
        float x,  // Location of robot
        float y, 
        Circle &c
    )
    {
        float distance_sq = (x - c.x) * (x - c.x) + (y - c.y) * (y - c.y); 
        return distance_sq <= (R_ROBOT + c.r) * (R_ROBOT + c.r) + CC_E;
    }

    __device__ inline bool inCollisionRectangle(
        float x,  // Location of robot
        float y,
        Rectangle &rect)
    {
        float closest_x = fmax(rect.x - rect.w/2, fmin(x, rect.x + rect.w/2));  // Determine the closest point on the rectangle to the robot
        float closest_y = fmax(rect.y - rect.h/2, fmin(y, rect.y + rect.h/2));

        float distance_x = x - closest_x;
        float distance_y = y - closest_y;
        float distance_sq = (distance_x * distance_x) + (distance_y * distance_y);

        return distance_sq <= (R_ROBOT * R_ROBOT) + CC_E;
    }

    // __global__ void nodesInCollision(
    //     float* states, 
    //     bool* valid, 
    //     Env2D &env
    // );

    __global__ void edgesInCollision(
        float* edges, 
        bool* valid, 
        Env2D &env
    );
}