#include <curand_kernel.h>
#include <float.h>
#include "../params/hyperparameters.cuh"
#include "construction.cuh"
#include "../collision/cc_2D.cuh"
#include "../local_planning/reedsshepp.cuh"


namespace prm::construction{

    // __global__ void generateStates(float* states, unsigned long seed) {
    //     int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //     if (idx >= NUM_STATES) return;
    
    //     // Init RNG
    //     curandStateXORWOW state;
    //     curand_init(seed, idx, 0, &state);
    //     int base = idx * 5;
    
    //     // Generate random values in the range [LOWER_BOUNDS, UPPER_BOUNDS)
    //     states[base + 0] = LOWER_BOUNDS[0] + (UPPER_BOUNDS[0]-LOWER_BOUNDS[0]) * curand_uniform(&state); 
    //     states[base + 1] = LOWER_BOUNDS[1] + (UPPER_BOUNDS[1]-LOWER_BOUNDS[1]) * curand_uniform(&state);
    //     states[base + 2] = LOWER_BOUNDS[2] + (UPPER_BOUNDS[2]-LOWER_BOUNDS[2]) * curand_uniform(&state);
    //     states[base + 3] = 0.0f;   // Leave camera pan and tilt angles as 0
    //     states[base + 4] = 0.0f;
    // }

    __global__ void generateStates(
        float* states, 
        bool* valid, 
        collision::environment::Env2D &env, 
        Bounds bounds,
        unsigned long seed
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= NUM_STATES) return;

        // Init RNG with different seed for each iteration
        curandStateXORWOW state;
        curand_init(seed, idx, 0, &state);

        //curand_init(seed, idx + iteration * NUM_STATES, 0, &state);
        
        int base = idx * DIM;
        const int max_attempts = 50;
        bool collision_free = false;
        float x, y, theta;
        
        for (int attempt = 0; attempt < max_attempts; attempt++) {
            // Generate new random state
            x = bounds.lower[0] + (bounds.upper[0]-bounds.lower[0]) * curand_uniform(&state);
            y = bounds.lower[1] + (bounds.upper[1]-bounds.lower[1]) * curand_uniform(&state);
            theta = bounds.lower[2] + (bounds.upper[2]-bounds.lower[2]) * curand_uniform(&state);

            // Check if this new state is collision-free
            collision_free = true;
            
            // Check circular obstacles
            for (int i = 0; i < env.numCircles && collision_free; i++) {
                if (collision::environment::inCollisionCircle(x, y, env.circles[i])) {
                    collision_free = false;
                    break;
                }
            }

            if (!collision_free) continue; // If collision with circles, skip to next attempt
            
            // Check rectangular obstacles
        
            for (int i = 0; i < env.numRectangles && collision_free; i++) {
                if (collision::environment::inCollisionRectangle(x, y, env.rectangles[i])) {
                    collision_free = false;
                    break;
                }
            }
            

            if (!collision_free) continue; // If collision with rectangles, skip to next attempt

            // If we reach here, the state is valid, so we store it and return
            states[base + 0] = x;
            states[base + 1] = y;
            states[base + 2] = theta;
            states[base + 3] = 0.0f;
            states[base + 4] = 0.0f;
            valid[idx] = true;
            return;
            
        }
        // If we reach here in regeneration rounds, the state remains invalid and we store it
        states[base + 0] = x;
        states[base + 1] = y;
        states[base + 2] = theta;
        states[base + 3] = 0.0f;
        states[base + 4] = 0.0f;
        valid[idx] = false;
    }



    __global__ void findNeighbors(
        const float* states,   // Input states (row-major format), (numstates * dim)
        int* neighbors          // Output array of neighbor indices (numstates x k)
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= NUM_STATES) return;

        // Stack allocate arrays to store top-k distances and indices
        float topDistances[K];
        int topIndices[K];

        for (int s = 0; s < K; s++) {
            topDistances[s] = FLT_MAX;  // Initialize to overwriteable values
            topIndices[s] = -1;
        }

        // Load vector i into registers
        float vec_i[DIM];
        for (int d = 0; d < DIM; d++) {
            vec_i[d] = states[i * DIM + d];
        }

        for (int j = 0; j < NUM_STATES; j++) {
            
            if (i == j) continue; // Skip self
            float distance = 0.0f;
            
            // Compute Euclidean distance between states i and j
            for (int d = 0; d < DIM; d++) {
                float diff = vec_i[d] - states[j * DIM + d];
                distance += diff * diff;
            }

            for (int m = 0; m < K; m++) {
                if (distance < topDistances[m]) {

                    // Shift elements to make room         
                    for (int n = K - 1; n > m; n--) {                      
                        topDistances[n] = topDistances[n - 1];
                        topIndices[n] = topIndices[n - 1];
                    }
                    
                    // Insert new element
                    topDistances[m] = distance;
                    topIndices[m] = j;
                    break;
                }
            }  
        }

        for (int j = 0; j < K; j++) {
            neighbors[i * K + j] = topIndices[j];
        }    
    }

    // Each thread interpolates between one vector and its neighbors
    __global__ void interpolate(
        const float* states,  // Input states
        int* neighbors,        // Neighbors for each vector
        float* edges           // Output containing edges
    ) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= NUM_STATES) return;

        // Allocate shared memory for the interpolation step values
        // __shared__ float interp[INTERP_STEPS];
        // if(threadIdx.x == 0) {  // Only the first thread initializes this
        //     for (int t = 0; t < INTERP_STEPS; t++) {
        //         interp[t] = (1.0f / (INTERP_STEPS-1)) * t;  
        //     }
        // }
        // __syncthreads();

        // Load vec_i into registers
        float vec_i[DIM];
        for (int d = 0; d < DIM; d++) {
            vec_i[d] = states[i * DIM + d];
        }

        // Load indices of the neighbors of vec_i into registers
        int neighbors_i[K];
        for (int m = 0; m < K; m++) {
            neighbors_i[m] = neighbors[i * K + m];
        }

        // Calculate interpolated states between vec_i and each neighbor
        for (int m = 0; m < K; m++) {
            int neighborIndex = neighbors_i[m];
            
            // Load neighbor vector directly from global memory
            float vec_n[DIM];
            for (int d = 0; d < DIM; d++) {
                vec_n[d] = states[neighborIndex * DIM + d];
            }
            
            float path[INTERP_STEPS * DIM]; // used to store reedsshepp path
            // find the reedsshepp path between the two states
            // populated the edges array with the path
            

            lp::reedsshepp::computeReedsSheppPath(vec_i, vec_n, path);
            for (int t = 0; t < INTERP_STEPS; t++) {
                for (int d = 0; d < DIM; d++) {
                    edges[i * K * INTERP_STEPS * DIM + m * INTERP_STEPS * DIM + t * DIM + d] = path[t * DIM + d];
                }
            }
        }
    }
}