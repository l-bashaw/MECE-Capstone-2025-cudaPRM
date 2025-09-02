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

        // Check if theta from start to end wraps around more than pi/6, which is too much rotation for one edge
        // float start_theta = edges[start_of_edge + 2];
        // float end_theta = edges[start_of_edge + (INTERP_STEPS - 1) * DIM + 2];
        // float d_theta = end_theta - start_theta;    

        // if (d_theta > M_PI){
        //     d_theta -= 2 * M_PI;
        // }
        // if (d_theta < -M_PI){
        //     d_theta += 2 * M_PI;
        // }

        // float angle_limit = M_PI / 12;

        // // Invalidate the edge if the theta difference exceeds the threshold
        // if (fabs(d_theta) > angle_limit) {
        //     valid[idx] = false;
        //     return;
        // }

        // // Check if the angle from edge start to edge end is too far from start theta
        // float x1 = edges[start_of_edge + 0];
        // float y1 = edges[start_of_edge + 1];
        // float x2 = edges[start_of_edge + (INTERP_STEPS - 1) * DIM + 0];
        // float y2 = edges[start_of_edge + (INTERP_STEPS - 1) * DIM + 1];
        // float edge_angle = atan2(y2 - y1, x2 - x1);
        // float angle_diff = edge_angle - start_theta;

        // // Normalize angle_diff to [-pi, pi]
        // if (angle_diff > M_PI){
        //     angle_diff -= 2 * M_PI;
        // }
        // if (angle_diff < -M_PI){
        //     angle_diff += 2 * M_PI;
        // }
        
        // if (fabs(angle_diff) > angle_limit) {
        //     valid[idx] = false;
        //     return;
        // }

        valid[idx] = true;
    }
}

// edges[state i][neighbor m][interp step t][dimension d]