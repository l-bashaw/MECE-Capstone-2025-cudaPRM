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

#define K 10
#define NUMVECTORS 10000
#define DIMENSIONS 15


// Kernel to compute pairwise distances between vectors
__global__ void findNeighbors(
            const float* vectors,   // Input vectors (row-major format)
            int* neighbors          // Output array of neighbor indices (numVectors x k)
) {

   

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i >= NUMVECTORS) return;
    
    // Stack allocate arrays to store top-k distances and indices
    float topDistances[K];
    int topIndices[K];

    for (int s = 0; s < K; s++) {
        topDistances[s] = FLT_MAX;  // Initialize to overwriteable values
        topIndices[s] = -1;
    }

    // Load vector i into registers
    float vec_i[DIMENSIONS];
    for (int d = 0; d < DIMENSIONS; d++) {
        vec_i[d] = vectors[i * DIMENSIONS + d];
    }

    for (int j = 0; j < NUMVECTORS; j++) {
        
        if (i == j) continue; // Skip self
        float distance = 0.0f;
        
        // Compute Euclidean distance between vectors i and j
        for (int d = 0; d < DIMENSIONS; d++) {
            float diff = vec_i[d] - vectors[j * DIMENSIONS + d];
            distance += diff * diff;
        }

        // TODO: try to get rid of this loop - may be a simpler way to store these values
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




// Host function to execute the kernels
void findAllNearestNeighbors(
    const float* h_vectors,  // Host vectors
    int* h_neighbors         // Host output array (numVectors x k)
) {


    // Allocate device memory
    float *d_vectors;
    int *d_neighbors;
    
    size_t vectorsSize = NUMVECTORS * DIMENSIONS * sizeof(float);
    size_t neighborsSize = NUMVECTORS * K * sizeof(int);
    
    cudaMalloc((void**)&d_vectors, vectorsSize);
    cudaMalloc((void**)&d_neighbors, neighborsSize);
    
    // Copy vectors to device
    cudaMemcpy(d_vectors, h_vectors, vectorsSize, cudaMemcpyHostToDevice);
    
    // Configure grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (NUMVECTORS + threadsPerBlock - 1) / threadsPerBlock;
    
    findNeighbors<<<blocksPerGrid, threadsPerBlock>>>(d_vectors, d_neighbors);
    cudaCheckErrors("kernel launch failure");
    
    // Copy results back to host
    cudaMemcpy(h_neighbors, d_neighbors, neighborsSize, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_vectors);
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

    // Allocate and initialize host memory
    float* h_vectors = new float[NUMVECTORS * DIMENSIONS];
    int* h_neighbors = new int[NUMVECTORS * K];
    
    // Initialize vectors with random data
    for (int i = 0; i < NUMVECTORS * DIMENSIONS; i++) {
        h_vectors[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Find nearest neighbors
    findAllNearestNeighbors(h_vectors, h_neighbors);
    
    
    

    bool verbose = false;
    if (verbose){

        // Print some results
        for (int i = 387; i < 391; i++) {  
            printf("Nearest neighbors for vector %d: ", i);
            for (int j = 0; j < K; j++) {
                printf("%d ", h_neighbors[i * K + j]);
            }
            printf("\n");
        }

        // Save vectors to CSV file
        saveToCSV("vectors.csv", h_vectors, NUMVECTORS, DIMENSIONS);

        //save neighbors to CSV file
        saveToCSV("neighbors.csv", h_neighbors, NUMVECTORS, K);
    }
  

    // Free host memory
    delete[] h_vectors;
    delete[] h_neighbors;
    
    return 0;
}