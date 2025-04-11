#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "hyperparameters.cuh"

namespace collision::environment{

    // Structs for circluar and rectangular obstacles
    struct circle{
        float x;  // (x,y) defines the center
        float y;
        float r;
    };

    struct rectangle{
        float x;  // (x,y) defines the center
        float y;
        float h;
        float w;
    };

    // Environment struct that contains the world bounds and obstacles
    struct env_2D{
        float x_min;
        float x_max;
        float y_min;
        float y_max;

        circle* circles;
        rectangle* rectangles;
        int numCircles;
        int numRectangles;
    };

    // Function to create an environment with given bounds and obstacles
    env_2D constructEnvironment(
        float x_min,
        float x_max,
        float y_min,
        float y_max,
        circle* circles,
        int numCircles,
        rectangle* rectangles,
        int numRectangles)
    {
        env_2D env;
        env.x_min = x_min;
        env.x_max = x_max;
        env.y_min = y_min;
        env.y_max = y_max;
        env.circles = circles;
        env.rectangles = rectangles;
        env.numCircles = numCircles;
        env.numRectangles = numRectangles;

        // Ensure that the environment is valid (using the function implemented below) before returning
        if (isValidEnvironment(env) == false){
            printf("Invalid environment parameters\n");
            exit(1);
        }
        return env;
    }


    // Function to check if a created environment is valid
    bool isValidEnvironment(const env_2D& env){
        // Check if bounds are valid
        if (env.x_min >= env.x_max || env.y_min >= env.y_max){
            printf("Environment bounds are invalid\n");
            return false;
        }

        // Check if circles are within bounds
        for (int i = 0; i < env.numCircles; i++){
            if (env.circles[i].x  < env.x_min ||
                env.circles[i].x  > env.x_max ||
                env.circles[i].y  < env.y_min ||
                env.circles[i].y  > env.y_max)
            {
                printf("Circle %d is out of environment bounds\n", i);
                return false;
            }
        }

        // Check if rectangles are within bounds
        for (int i = 0; i < env.numRectangles; i++){
            if (env.rectangles[i].x < env.x_min ||
                env.rectangles[i].x > env.x_max ||
                env.rectangles[i].y < env.y_min ||
                env.rectangles[i].y > env.y_max)
            {   
                printf("Rectangle %d is out of environment bounds\n", i);
                return false;
            }
        }
        return true;
    }

    env_2D setupEnv1(){
        env_2D env;
        env.numCircles = 2;
        env.numRectangles = 2;
    
        // Define circular obstacles
        env.circles[0].x = 0.0f;
        env.circles[0].y = 0.0f;
        env.circles[0].r = 1.0f;
        
        env.circles[1].x = 3.0f;
        env.circles[1].y = 3.0f;
        env.circles[1].r = 1.5f;
    
        // Define rectangular obstacles
        env.rectangles[0].x = 5.0f;
        env.rectangles[0].y = 5.0f;
        env.rectangles[0].w = 2.0f;
        env.rectangles[0].h = 2.0f;
        
        env.rectangles[1].x = -5.0f;
        env.rectangles[1].y = -5.0f;
        env.rectangles[1].w = 2.0f;
        env.rectangles[1].h = 2.0f;
    
        // Define the environment bounds
        env.x_min = -10.0f;
        env.x_max = 10.0f;
        env.y_min = -10.0f;
        env.y_max = 10.0f;
    
        // Ensure that the environment is valid (using the function implemented below) before returning
        if (isValidEnvironment(env) == false){
            printf("Invalid environment parameters\n");
            exit(1);
        }
    
        return env;
    }


    // Device functions to check for collisions between a circular robot and a circle or rectangle
    // These are used in the collision checking kernel
    __device__ bool inCollisionCircle(
        float x,  // Location of robot
        float y, 
        circle c
    )
    {
        float distance_sq = (x - c.x) * (x - c.x) + (y - c.y) * (y - c.y); 
        return distance_sq <= (R_ROBOT + c.r) * (R_ROBOT + c.r) + CC_E;
    }

    __device__ bool inCollisionRectangle(
        float x,  // Location of robot
        float y,
        rectangle rect)
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
    __global__ void nodeInCollision(float* states, bool* valid, env_2D env)
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