#include "../params/hyperparameters.cuh"
#include "env_2D.cuh"
#include "cc_2D.cuh"

namespace collision::environment{

    

    // Kernel to check if a state is in collision with the environment
    // Each thread checks collision of one state with the entire environment.
    // __global__ void nodesInCollision(float* states, bool* valid, Env2D &env)
    // {
    //     int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //     if (idx >= NUM_STATES) return;

    //     // Load state into registers
    //     float x = states[idx * 5 + 0];       // x position
    //     float y = states[idx * 5 + 1];       // y position
    //     // float theta = states[idx * 5 + 2];   // theta

    //     // Check circular obstacles for collisions
    //     for (int i = 0; i < env.numCircles; i++){
    //         if (inCollisionCircle(x, y, env.circles[i])){
    //             valid[idx] = false;
    //             return;
    //         }
    //     }

    //     // Check rectangular obstacles for collisions
    //     for (int i = 0; i < env.numRectangles; i++){
    //         if (inCollisionRectangle(x, y, env.rectangles[i])){
    //             valid[idx] = false;
    //             return;
    //         }
    //     }
    //     valid[idx] = true;
    // }

    
    // Each thread checks collision of one edge with the entire environment.
    // Launching this kernel with NUM_STATES * K threads.
    __global__ void edgesInCollision(float* edges, bool* valid, Env2D &env)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= NUM_STATES * K) return;
        int start_of_edge = idx  * DIM * INTERP_STEPS; 

        float edge_i[2*INTERP_STEPS];
        for (int t = 0; t < INTERP_STEPS; t++) {
            edge_i[t*2]   = edges[start_of_edge + t * DIM + 0]; // x position
            edge_i[t*2+1] = edges[start_of_edge + t * DIM + 1]; // y position
        }

        for (int i=0; i<2*INTERP_STEPS; i++){
            // Check circular obstacles for collisions
            for (int j = 0; j < env.numCircles; j++){
                if (inCollisionCircle(edge_i[i*2], edge_i[i*2+1], env.circles[j])){
                    valid[idx] = false;
                    return;
                }
            }

            // Check rectangular obstacles for collisions
            for (int j = 0; j < env.numRectangles; j++){
                if (inCollisionRectangle(edge_i[i*2], edge_i[i*2+1], env.rectangles[j])){
                    valid[idx] = false;
                    return;
                }
            }
        }
        valid[idx] = true;
    }
}

// edges[state i][neighbor m][interp step t][dimension d]