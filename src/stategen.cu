#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "knncuda.h"
#include <chrono>

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






__global__ void generatePoses(float *poses, unsigned long seed, int numPoses, int dim){
    
    extern __shared__ curandState sharedStates[];
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int tid = threadIdx.x;
    
    // If it is first thread in block, initialize curand state
    if (tid == 0){
        curand_init(seed, blockIdx.x, 0, &sharedStates[0]);
    }
    __syncthreads(); // Ensure all threads finish initializing curand state

    if(idx < numPoses){
        for (int d = 0; d < dim; d++){
            // Skip ahead based on thread ID to maintain randomness
            for (int j = 0; j < tid; j++){
                curand_uniform(&sharedStates[0]);
            }
            poses[idx * dim + d] = curand_uniform(&sharedStates[0]);
            __syncthreads(); // Ensure all threads finish using the state before next iteration
        }
    }

    
}

void displayPoses(float *poses, int numPoses, int dim){
    for (int i = 0; i < numPoses; i++){
        printf("Pose %d: ", i);
        for (int j = 0; j < dim; j++){
            printf("%f ", poses[i*dim + j]);
        }
        printf("\n");
    }
}


int main(){
    // Number of poses to generate and their dimension
    auto st = std::chrono::high_resolution_clock::now();

    auto sx = std::chrono::high_resolution_clock::now();
   
    int numPoses = 10000; 
    int dim = 6; // x, y, z, qx, qy, qz, qw
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0); // dev_ID = 0
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    int multiProcessorCount = deviceProp.multiProcessorCount;
    int blocksize = 256; //can increase depending on gpu
    int gridsize = (numPoses + blocksize - 1)/blocksize;
    auto ex = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsedx = ex - sx;
    printf("Device properties time: %f seconds\n", elapsedx.count());

    auto start = std::chrono::high_resolution_clock::now();

    // Pin host memory for faster memory transfers
    float *h_poses, *d_poses;
    cudaMallocHost(&h_poses, numPoses * dim * sizeof(float));
    auto se = std::chrono::high_resolution_clock::now();
    cudaCheckErrors("cudaMalloc failure"); 
    auto fe = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsede = fe - se;
    printf("Error check time: %f seconds\n", elapsede.count());
    // Poses are stored as one sequence in row-major order 
    cudaMalloc(&d_poses, numPoses * dim * sizeof(float));
    cudaCheckErrors("cudaMalloc failure"); 
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("Malloc time: %f seconds\n", elapsed.count());
    unsigned long seed = 12345UL;

    // Using one curandState per block, store it in shared memory
    int sharedMemSize = sizeof(curandState);

    auto start1 = std::chrono::high_resolution_clock::now();
    generatePoses<<<gridsize, blocksize, sharedMemSize>>>(d_poses, seed, numPoses, dim);
    auto end1 = std::chrono::high_resolution_clock::now();
    cudaCheckErrors("kernel launch failure");
    std::chrono::duration<double> elapsed1 = end1 - start1;
    printf("PoseGen kernel time: %f seconds\n", elapsed1.count());

    auto start2 = std::chrono::high_resolution_clock::now();
    cudaMemcpyAsync(h_poses, d_poses, numPoses * dim * sizeof(float), cudaMemcpyDeviceToHost);
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end2 - start2;
    printf("Memcpy time: %f seconds\n", elapsed2.count());

    cudaCheckErrors("cudaMemcpyAsync failure");

    // Display first 20 poses
    //displayPoses(h_poses, numPoses, dim);
    auto start3 = std::chrono::high_resolution_clock::now();
    cudaFreeHost(h_poses);
    cudaFree(d_poses);
    auto end3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed3 = end3 - start3;
    printf("Free time: %f seconds\n", elapsed3.count());

    auto et = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedt = et - st;
    printf("Total time: %f seconds\n", elapsedt.count());

    /*
    // Perform kNN search on pose list
    int k = 10; 
    // Allocate pinned host memory for indices and distances of nearest neighbors
    int* h_indices;
    float* h_distances;
    cudaMallocHost(&h_indices, numPoses * k * sizeof(int));
    cudaCheckErrors("cudaMallocHost failure");
    cudaMallocHost(&h_distances, numPoses * k * sizeof(float));
    cudaCheckErrors("cudaMallocHost failure");

    // Allocate device memory for indices and lengths of and to nearest neighbors
    int* d_indices;
    float* d_distances;
    cudaMalloc(&d_indices, numPoses * k * sizeof(int));
    cudaCheckErrors("cudaMalloc failure");
    cudaMalloc(&d_distances, numPoses * k * sizeof(float)); 
    cudaCheckErrors("cudaMalloc failure");


    // Perform kNN search on the poses
    bool nn_success = knn_cublas(h_poses, numPoses, h_poses, numPoses, dim, k, h_distances, h_indices);

    if (!nn_success){
        printf("kNN search failed\n");
    } else {
        printf("kNN search succeeded\n");
    }
    printf("Distance to nearest neighbors for first 20 poses:\n");
    for (int i = 0; i < 20; i++){
        printf("Pose %d: ", i);
        for (int j = 0; j < k; j++){
            printf("%f ", h_distances[i*k + j]);
        }
        printf("\n");
    }
    printf("Nearest neighbors for first 20 poses:\n");
    for (int i = 0; i < 20; i++){
        printf("Pose %d: ", i);
        for (int j = 0; j < k; j++){
            printf("%d ", h_indices[i*k + j]);
        }
        printf("\n");
    }

    // Use cudaFreeHost for pinned memory
    cudaFreeHost(h_indices);
    cudaFreeHost(h_distances);
    cudaFree(d_indices);
    cudaFree(d_distances);
    */
    
    
    
    return 0;
}