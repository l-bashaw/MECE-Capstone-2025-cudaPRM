#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "knncuda.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>

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

// Takes a row major array of data and writes it to a CSV file
void saveToCSV(const std::string& filename, float* data, int numPoses, int dim) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < numPoses; i++) {
        for (int j = 0; j < dim; j++) {
            file << data[i * dim + j];
            if (j < dim - 1) file << ",";  // Comma between values
        }
        file << "\n";  // Newline after each pose
    }

    file.close();
    std::cout << "Data successfully written to " << filename << std::endl;
}




__global__ void generatePoses(float *poses, unsigned long seed, int numPoses, int dim){
    
    extern __shared__ curandState sharedStates[];
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int tid = threadIdx.x;
    float lower_bound = 0.0;  // change these to satisfy specific joint limits
    float upper_bound = 1.0;

    // If it is first thread in block, initialize curand state
    if (tid == 0){
        curand_init(seed, blockIdx.x, 0, &sharedStates[0]);
    }
    
    // Remove this line??????
    __syncthreads(); // Ensure all threads finish initializing curand state


    if(idx < numPoses){
        for (int d = 0; d < dim; d++){
            // Skip ahead based on thread ID to maintain randomness
            for (int j = 0; j < tid; j++){
                curand_uniform(&sharedStates[0]);

            }
    
            poses[idx * dim + d] = lower_bound + (upper_bound - lower_bound)*curand_uniform(&sharedStates[0]);
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

   
    int numPoses = 100000; 
    int dim = 7; 
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0); // dev_ID = 0
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    int multiProcessorCount = deviceProp.multiProcessorCount;
    int blocksize = 256; //can increase depending on gpu
    int gridsize = (numPoses + blocksize - 1)/blocksize;

    bool time = true;
    bool pose = false;

    // Pin host memory for faster memory transfers
    float *h_poses, *d_poses;
    cudaMallocHost(&h_poses, numPoses * dim * sizeof(float));
    cudaCheckErrors("cudaMalloc failure"); 

    // Poses are stored as one sequence in row-major order 
    cudaMalloc(&d_poses, numPoses * dim * sizeof(float));
    cudaCheckErrors("cudaMalloc failure"); 
    unsigned long seed = 12345UL;

    // Using one curandState per block, store it in shared memory
    int sharedMemSize = sizeof(curandState);

    if(time && pose || !time && !pose){
        printf("Pick one bozo");
        return 1;
    }

    if(time){
        printf("Timing the kernel\n");
        int num_iters = 10000;
        float* times;
        cudaMallocHost(&times, num_iters * sizeof(float));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for (int i=0; i<num_iters; i++){

            cudaEventRecord(start);
            generatePoses<<<gridsize, blocksize, sharedMemSize>>>(d_poses, seed, numPoses, dim);
            cudaCheckErrors("kernel launch failure");
            cudaEventRecord(stop);
            
            cudaEventSynchronize(stop);
            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);
            times[i] = ms;
        }
        
        saveToCSV("times_7_100k_ms.csv", times, num_iters, 1);
        cudaFreeHost(times);
    }
    if(pose){
        printf("Generating 2D poses\n");
        generatePoses<<<gridsize, blocksize, sharedMemSize>>>(d_poses, seed, numPoses, dim);
        cudaCheckErrors("kernel launch failure");
        cudaDeviceSynchronize();
        cudaMemcpy(h_poses, d_poses, numPoses * dim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaCheckErrors("cudaMemcpy H2D failure");
        saveToCSV("posesImproved.csv", h_poses, numPoses, dim);
    }
    
    cudaFreeHost(h_poses);
    cudaFree(d_poses);
  

    return 0;
}

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
    
    
    
