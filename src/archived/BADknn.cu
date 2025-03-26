#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <fstream>


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

    
// Kernel to compute pairwise distances between vectors
__global__ void findNeighbors(
    const float* vectors,   // Input vectors (row-major format)
    int numVectors,         // Number of vectors
    int dimensions,         // Dimensions per vector
    float* distances,       // Output distance matrix (numVectors x numVectors)
    int* neighbors          // Output array of neighbor indices (numVectors x k)
) {
    const int k = 10;  // Number of neighbors to find
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < numVectors) {
        for (int j = 0; j < numVectors; j++) {
            float distance = 0.0f;
            
            // Compute Euclidean distance between vectors i and j
            for (int d = 0; d < dimensions; d++) {
                float diff = vectors[i * dimensions + d] - vectors[j * dimensions + d];
                distance += diff * diff;
            }
            
            // Store the squared distance (can be square-rooted later if needed)
            distances[i * numVectors + j] = distance;
        }

        __syncthreads();
        
        // Stack allocate arrays to store top-k distances and indices
        float topDistances[k];
        int topIndices[k];

        for (int s = 0; s < k; s++) {
            topDistances[s] = FLT_MAX;  // Initialize to overwriteable values
            topIndices[s] = -1;
        }

        for (int j = 0; j < numVectors; j++) {
            if (i == j) continue; // Skip self
            
            float distance = distances[i * numVectors + j];
            
            // Check if this distance is smaller than any in our top-k
            for (int m = 0; m < k; m++) {
                if (distance < topDistances[m]) {
                    // Shift elements to make room
                    for (int n = k - 1; n > m; n--) {
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
        for (int j = 0; j < k; j++) {
            neighbors[i * k + j] = topIndices[j];
        }
    
    }
}




// Host function to execute the kernels
void findAllNearestNeighbors(
    const float* h_vectors,  // Host vectors
    int numVectors,          // Number of vectors
    int dimensions,          // Dimensions per vector
    const int numNeighbors,  // Number of neighbors to find
    int* h_neighbors         // Host output array (numVectors x k)
) {
    // Allocate device memory
    float *d_vectors, *d_distances;
    int *d_neighbors;
    
    size_t vectorsSize = numVectors * dimensions * sizeof(float);
    size_t distancesSize = numVectors * numVectors * sizeof(float);
    size_t neighborsSize = numVectors * numNeighbors * sizeof(int);
    
    cudaMalloc((void**)&d_vectors, vectorsSize);
    cudaMalloc((void**)&d_distances, distancesSize);
    cudaMalloc((void**)&d_neighbors, neighborsSize);
    
    // Copy vectors to device
    cudaMemcpy(d_vectors, h_vectors, vectorsSize, cudaMemcpyHostToDevice);
    
    // Configure grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (numVectors + threadsPerBlock - 1) / threadsPerBlock;
    
    findNeighbors<<<blocksPerGrid, threadsPerBlock>>>(d_vectors, numVectors, dimensions, d_distances, d_neighbors);
    cudaCheckErrors("kernel launch failure");
    /*
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i=0; i<5; i++){

        cudaEventRecord(start);
        cudaCheckErrors("kernel launch failure");
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        printf("Time: %f milliseconds\n", ms);
    }
    cudaCheckErrors("findNeighbors kernel execution failed");
    */
 
    // Copy results back to host
    cudaMemcpy(h_neighbors, d_neighbors, neighborsSize, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_vectors);
    cudaFree(d_distances);
    cudaFree(d_neighbors);
}


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
void saveToCSV(const std::string& filename, int* data, int numPoses, int dim) {
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

int main() {
    const int numVectors = 10000;
    const int dimensions = 15;
    const int numNeighbors = 10;

    // Allocate and initialize host memory
    float* h_vectors = new float[numVectors * dimensions];
    int* h_neighbors = new int[numVectors * numNeighbors];
    
    // Initialize vectors with random data
    for (int i = 0; i < numVectors * dimensions; i++) {
        h_vectors[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Find nearest neighbors
    findAllNearestNeighbors(h_vectors, numVectors, dimensions, numNeighbors, h_neighbors);
    
    // Print some results
    for (int i = 387; i < 391; i++) {  // Print for first 5 vectors
        printf("Nearest neighbors for vector %d: ", i);
        for (int j = 0; j < numNeighbors; j++) {
            printf("%d ", h_neighbors[i * numNeighbors + j]);
        }
        printf("\n");
    }
    

    bool write = false;
    if (write){
        // Save vectors to CSV file
        saveToCSV("vectors.csv", h_vectors, numVectors, dimensions);

        //save neighbors to CSV file
        saveToCSV("neighbors.csv", h_neighbors, numVectors, numNeighbors);
    }
  

    // Free host memory
    delete[] h_vectors;
    delete[] h_neighbors;
    
    return 0;
}