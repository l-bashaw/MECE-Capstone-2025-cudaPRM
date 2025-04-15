#include <curand_kernel.h>
#include <float.h>
#include "../params/hyperparameters.cuh"
#include "construction.cuh"


namespace prm::construction{

    __global__ void generateStates(float* states, unsigned long seed) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= NUM_STATES) return;
    
        // Init RNG
        curandStateXORWOW state;
        curand_init(seed, idx, 0, &state);
        int base = idx * 5;
    
        // Generate random values in the range [LOWER_BOUNDS, UPPER_BOUNDS)
        states[base + 0] = LOWER_BOUNDS[0] + (UPPER_BOUNDS[0]-LOWER_BOUNDS[0]) * curand_uniform(&state); 
        states[base + 1] = LOWER_BOUNDS[1] + (UPPER_BOUNDS[1]-LOWER_BOUNDS[1]) * curand_uniform(&state);
        states[base + 2] = LOWER_BOUNDS[2] + (UPPER_BOUNDS[2]-LOWER_BOUNDS[2]) * curand_uniform(&state);
        states[base + 3] = 0.0f;   // Leave camera pan and tilt angles as 0
        states[base + 4] = 0.0f;
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
    __global__ void lin_interp(
        const float* states,  // Input states
        int* neighbors,        // Neighbors for each vector
        float* edges           // Output containing edges
    ) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= NUM_STATES) return;

        // Allocate shared memory for the interpolation step values
        __shared__ float interp[INTERP_STEPS];
        if(threadIdx.x == 0) {  // Only the first thread initializes this
            for (int t = 0; t < INTERP_STEPS; t++) {
                interp[t] = (1.0f / (INTERP_STEPS-1)) * t;  
            }
        }
        __syncthreads();

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
            
            // Interpolate between vec_i and vec_n
            for (int t = 0; t < INTERP_STEPS; t++) {
                float t_val = interp[t];
                for (int d = 0; d < DIM; d++) {
                    // Proper linear interpolation: (1-t)*start + t*end
                    float interpolated = vec_i[d] * (1.0f - t_val) + vec_n[d] * t_val;
                    edges[i * K * INTERP_STEPS * DIM + m * INTERP_STEPS * DIM + t * DIM + d] = interpolated;
                }
            }
        }
    }
}