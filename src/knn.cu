#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <fstream>

// TODO:
//      Try to move vectors to L2 cache
//      

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
#define DIMENSIONS 11
#define INTERP_STEPS 5



// Kernel to compute pairwise distances between vectors
__global__ void findNeighbors(
            const float* vectors,   // Input vectors (row-major format), (numVectors * dim)
            int* neighbors          // Output array of neighbor indices (numVectors x k)
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i >= NUMVECTORS) return;
    
    // Stack allocate arrays to store top-k distances and indices
    float topDistances[K];
    int topIndices[K];

    // Allocate these in shared memory?? Every thread uses these...
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
    const float* vectors,  // Input vectors
    int* neighbors,         
    float* edges
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= NUMVECTORS) return;

    // Allocate shared memory for the interpolation step values
    __shared__ float interp[INTERP_STEPS];
    if(threadIdx.x == 0) {  // Only the first thread initializes this
        for (int t = 0; t < INTERP_STEPS; t++) {
            interp[t] = (1.0f / (INTERP_STEPS-1)) * t;  
        }
    }
    __syncthreads();

    // Load vec_i into registers
    float vec_i[DIMENSIONS];
    for (int d = 0; d < DIMENSIONS; d++) {
        vec_i[d] = vectors[i * DIMENSIONS + d];
    }

    // Load indices of the neighbors of vec_i into registers
    int neighbors_i[K];
    for (int m = 0; m < K; m++) {
        neighbors_i[m] = neighbors[i * K + m];
    }

    // Calculate interpolated vectors between vec_i and each neighbor
    for (int m = 0; m < K; m++) {
        int neighborIndex = neighbors_i[m];
        
        // Load neighbor vector directly from global memory
        float vec_n[DIMENSIONS];
        for (int d = 0; d < DIMENSIONS; d++) {
            vec_n[d] = vectors[neighborIndex * DIMENSIONS + d];
        }
        
        // Interpolate between vec_i and vec_n
        for (int t = 0; t < INTERP_STEPS; t++) {
            float t_val = interp[t];
            for (int d = 0; d < DIMENSIONS; d++) {
                // Proper linear interpolation: (1-t)*start + t*end
                float interpolated = vec_i[d] * (1.0f - t_val) + vec_n[d] * t_val;
                edges[i * K * INTERP_STEPS * DIMENSIONS + m * INTERP_STEPS * DIMENSIONS + t * DIMENSIONS + d] = interpolated;
            }
        }
    }
}



// Host function to execute the kernels
void findAllNearestNeighbors(
    const float* h_vectors,  // Host vectors
    int* h_neighbors,         // Host output array (numVectors x k)
    float* h_edges          // Host output array (numVectors x k x dim x interp_steps)
) {


    // Allocate device memory
    float *d_vectors;
    int *d_neighbors;
    float *d_edges;
    
    size_t vectorsSize = NUMVECTORS * DIMENSIONS * sizeof(float);
    size_t neighborsSize = NUMVECTORS * K * sizeof(int);
    size_t edgesSize = NUMVECTORS * DIMENSIONS * K * INTERP_STEPS * sizeof(float);

    cudaMalloc((void**)&d_vectors, vectorsSize);
    cudaMalloc((void**)&d_neighbors, neighborsSize);
    cudaMalloc((void**)&d_edges, edgesSize);
    
    // Copy vectors to device
    cudaMemcpy(d_vectors, h_vectors, vectorsSize, cudaMemcpyHostToDevice);
    
    // Configure grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (NUMVECTORS + threadsPerBlock - 1) / threadsPerBlock;
    
    findNeighbors<<<blocksPerGrid, threadsPerBlock>>>(d_vectors, d_neighbors);
    cudaCheckErrors("kernel launch failure");

    cudaDeviceSynchronize();

    lin_interp<<<blocksPerGrid, threadsPerBlock>>>(d_vectors, d_neighbors, d_edges);
    cudaCheckErrors("kernel launch failure");

    // Copy results back to host
    cudaMemcpy(h_neighbors, d_neighbors, neighborsSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_edges, d_edges, edgesSize, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_vectors);
    cudaFree(d_neighbors);
    cudaFree(d_edges);
}


int main() {

    // Allocate and initialize host memory
    float* h_vectors = new float[NUMVECTORS * DIMENSIONS];
    int* h_neighbors = new int[NUMVECTORS * K];
    float* h_edges = new float[NUMVECTORS * DIMENSIONS * K * INTERP_STEPS];
    
    // Initialize vectors with random data
    for (int i = 0; i < NUMVECTORS * DIMENSIONS; i++) {
        h_vectors[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Find nearest neighbors
    findAllNearestNeighbors(h_vectors, h_neighbors, h_edges);
    bool verbose_ = true;
    if (verbose_){

        // Print a vector, one of its neighbors, and the interpolations between them
        int vecIndex = 0; // Index of the vector to print
        int b = 7;
        int neighborIndex = h_neighbors[vecIndex * K + b]; // Index of the nearest neighbor
        printf("Vector %d: ", vecIndex);
        for (int d = 0; d < DIMENSIONS; d++) {
            printf("%f ", h_vectors[vecIndex * DIMENSIONS + d]);
        }
        printf("\nNearest neighbor %d: ", neighborIndex);
        for (int d = 0; d < DIMENSIONS; d++) {
            printf("%f ", h_vectors[neighborIndex * DIMENSIONS + d]);
        }
        printf("\nInterpolated vectors between %d and %d:\n", vecIndex, neighborIndex);
        int m = b; // First neighbor
        for (int t = 0; t < INTERP_STEPS; t++) {
            printf("t = %f: ", (1.0f / (INTERP_STEPS-1)) * t);
            for (int d = 0; d < DIMENSIONS; d++) {
                printf("%f ", h_edges[vecIndex * K * INTERP_STEPS * DIMENSIONS + 
                                    m * INTERP_STEPS * DIMENSIONS + 
                                    t * DIMENSIONS + d]);
            }
            printf("\n");
        }

    }
    // Free host memory
    delete[] h_vectors;
    delete[] h_neighbors;
    delete[] h_edges;
    
    return 0;
}




// // Takes a row major array of data and writes it to a CSV file
// void saveToCSV(const std::string& filename, float* data, int numPoses, int dim) {
//     std::ofstream file(filename);
//     if (!file.is_open()) {
//         std::cerr << "Error opening file: " << filename << std::endl;
//         return;
//     }

//     for (int i = 0; i < numPoses; i++) {
//         for (int j = 0; j < dim; j++) {
//             file << data[i * dim + j];
//             if (j < dim - 1) file << ",";  // Comma between values
//         }
//         file << "\n";  // Newline after each pose
//     }

//     file.close();
//     std::cout << "Data successfully written to " << filename << std::endl;
// }
// void saveToCSV(const std::string& filename, int* data, int numPoses, int dim) {
//     std::ofstream file(filename);
//     if (!file.is_open()) {
//         std::cerr << "Error opening file: " << filename << std::endl;
//         return;
//     }

//     for (int i = 0; i < numPoses; i++) {
//         for (int j = 0; j < dim; j++) {
//             file << data[i * dim + j];
//             if (j < dim - 1) file << ",";  // Comma between values
//         }
//         file << "\n";  // Newline after each pose
//     }

//     file.close();
//     std::cout << "Data successfully written to " << filename << std::endl;
// }





// bool verbose = false;
// if (verbose){

//     // Print some results
//     for (int i = 387; i < 391; i++) {  
//         printf("Nearest neighbors for vector %d: ", i);
//         for (int j = 0; j < K; j++) {
//             printf("%d ", h_neighbors[i * K + j]);
//         }
//         printf("\n");
//     }

//     // Save vectors to CSV file
//     saveToCSV("vectors.csv", h_vectors, NUMVECTORS, DIMENSIONS);

//     //save neighbors to CSV file
//     saveToCSV("neighbors.csv", h_neighbors, NUMVECTORS, K);
// }
