#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "hyperparameters.cuh"

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
}

// #include <iostream>
// #include <fstream>
// #include <stdio.h>

int main(){

    int numPoses = 100000; 
    int dim = 5; 
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0); 
    int blocksize = 256; //can increase depending on gpu
    int gridsize = (numPoses + blocksize - 1)/blocksize;

    float *h_poses, *d_poses;
    cudaMallocHost(&h_poses, numPoses * dim * sizeof(float));
    cudaCheckErrors("cudaMalloc failure"); 

    cudaMalloc(&d_poses, numPoses * dim * sizeof(float));
    cudaCheckErrors("cudaMalloc failure"); 

    unsigned long seed = 12345UL;
    prm::construction::generateStates<<<gridsize, blocksize>>>(d_poses, seed);
    cudaCheckErrors("kernel launch failure");
    
    cudaFreeHost(h_poses);
    cudaCheckErrors("cudaFreeHost failure");
    cudaFree(d_poses);
    cudaCheckErrors("cudaFree failure");

    return 0;
}


    
