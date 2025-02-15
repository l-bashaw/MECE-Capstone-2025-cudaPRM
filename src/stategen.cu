#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "kNN-CUDA/code/knncuda.h"


// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)




__global__ void initRandStates(curandState* states, unsigned int seed){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void generatePoses(float *poses, curandState *states, int numPoses, int dim){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx < numPoses){
        curandState localState = states[idx];

        for (int d = 0; d < dim; d++){
            poses[idx * dim + d] = curand_uniform(&localState);
        }

        states[idx] = localState;
    }
}

void displayPoses(float *poses, int numPoses, int dim){
    for (int i = 0; i < 20; i++){
        printf("Pose %d: ", i);
        for (int j = 0; j < dim; j++){
            printf("%f ", poses[i*dim + j]);
        }
        printf("\n");
    }
}


int main(){
    // Number of poses to generate and their dimension
    int numPoses = 10000; 
    int dim = 7; // x, y, z, qx, qy, qz, qw

    // Allocate host and device memory for vectors of poses
    // Poses are stored as one sequences are stored in row-major order (pose 0, pose 1, ..., pose N)
    float *h_poses, *d_poses;
    h_poses = new float[numPoses*dim];
    cudaMalloc(&d_poses, numPoses*dim*sizeof(float));
    cudaCheckErrors("cudaMalloc failure"); 

    // Allocate device memory for the random number generators (curandStates) that we need
    // to generate the poses in parallel
    curandState *d_states;
    cudaMalloc(&d_states, numPoses*sizeof(curandState));
    cudaCheckErrors("cudaMalloc failure"); 

    ///int blocks = 320;
    int blocksize = 256;
    int gridsize = (numPoses + blocksize - 1)/blocksize;

    unsigned int seed = 123;
    initRandStates<<<gridsize, blocksize>>>(d_states, seed);
    cudaCheckErrors("kernel launch failure");

    generatePoses<<<gridsize, blocksize>>>(d_poses, d_states, numPoses, dim);
    cudaCheckErrors("kernel launch failure");
/*
    // Perform kNN search on pose list
    int k = 10; 
    
    // Allocate host memory for indices and distances of nearest neighbors
    int* h_indices;
    h_indices = new int[numPoses*k]; 
    float* h_distances;
    h_distances = new float[numPoses*k];

    // Allocate device memory for indices and lengths of and to nearest neighbors
    int* d_indices;
    cudaMalloc(&d_indices, numPoses*k*sizeof(int));
    float* d_distances;
    cudaMalloc(&d_distances, numPoses*k*sizeof(float)); // Allocate memory for distances to nearest neighbors

    // Copy poses to constant memory for use with knn_cublas
    const float *d_poses_const;
    cudaMalloc((void**)&d_poses_const, numPoses*dim*sizeof(float));
    cudaMemcpy((void*)d_poses_const, d_poses, numPoses*dim*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaCheckErrors("cudaMemcpy D2D failure");

    // Perform kNN search
   // bool nn_success = knn_cublas(d_poses_const, numPoses, d_poses_const, numPoses, dim, k, d_distances, d_indices);

    cudaMemcpy(h_indices, d_indices, numPoses*k*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_distances, d_distances, numPoses*k*sizeof(float), cudaMemcpyDeviceToHost);

    */
    cudaMemcpy(h_poses, d_poses, numPoses*dim*sizeof(float), cudaMemcpyDeviceToHost);
    //displayPoses(h_poses, numPoses, dim);

    delete[] h_poses;
  //  delete[] h_indices;
  //  delete[] h_distances;
    cudaFree(d_poses);
    cudaFree(d_states);
 //   cudaFree(d_indices);
  //  cudaFree(d_distances);


    return 0;
}