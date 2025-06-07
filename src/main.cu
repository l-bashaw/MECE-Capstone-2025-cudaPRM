#include "params/hyperparameters.cuh"
#include "planning/pprm.cuh"

// Define the constant arrays
__constant__ float LOWER_BOUNDS[5] = { -10.0f, -10.0f, 0.0f  , 0.0, 0.0  };  // x, y, theta
__constant__ float UPPER_BOUNDS[5] = { 10.0f ,  10.0f, 2*M_PI, 0.0, 0.0  };

void saveResults(planning::Roadmap &prm, collision::environment::Env2D &env_h){
    FILE *file = fopen("roadmap.txt", "w");
    if (file == NULL) {
        std::cerr << "Error opening file for writing" << std::endl;
    }
    for (int i = 0; i < NUM_STATES; i++) {
        fprintf(file, "State %d: ", i);
        for (int j = 0; j < DIM; j++) {
            fprintf(file, "%f ", prm.h_states[i * DIM + j]);
        }
        fprintf(file, "\n");
    }
    for (int i = 0; i < NUM_STATES; i++) {
        fprintf(file, "Neighbors %d: ", i);
        for (int j = 0; j < K; j++) {
            fprintf(file, "%d ", prm.h_neighbors[i * K + j]);
        }
        fprintf(file, "\n");
    }
    for (int i = 0; i < NUM_STATES; i++) {
        fprintf(file, "Valid Node %d: %d\n", i, prm.h_validNodes[i]);
    }
    for (int i = 0; i < NUM_STATES * K; i++) {
        fprintf(file, "Valid Edge %d: %d\n", i, prm.h_validEdges[i]);
    }
    // Add the environment information
    fprintf(file, "Environment:\n");
    fprintf(file, "Bounds: [%f, %f] x [%f, %f]\n", env_h.x_min, env_h.x_max, env_h.y_min, env_h.y_max);
    fprintf(file, "Circles:\n");
    for (unsigned int i = 0; i < env_h.numCircles; i++) {
        fprintf(file, "Circle %d: Center (%f, %f), Radius %f\n", i, env_h.circles[i].x, env_h.circles[i].y, env_h.circles[i].r);
    }
    fprintf(file, "Rectangles:\n");
    for (unsigned int i = 0; i < env_h.numRectangles; i++) {
        fprintf(file, "Rectangle %d: Center (%f, %f), Width %f, Height %f\n", i, env_h.rectangles[i].x, env_h.rectangles[i].y, env_h.rectangles[i].w, env_h.rectangles[i].h);
    }
    fclose(file);
    std::cout << "Roadmap saved to roadmap.txt" << std::endl;
}

void displayStateAndNeighbors(int stateIndex, const planning::Roadmap& prm, int numStates, int interpSteps, int dim) {
    float* h_states = prm.h_states;
    float* h_edges = prm.h_edges;
    int* h_neighbors = prm.h_neighbors;
    bool* h_validNodes = prm.h_validNodes;
    
    // Print state information
    std::cout << "--- State " << stateIndex << " ---\n";
    std::cout << "Position: (";
    for (int d = 0; d < dim; d++) {
        std::cout << h_states[stateIndex * dim + d];
        if (d < dim - 1) std::cout << ", ";
    }
    std::cout << ")\n";
    std::cout << "Valid: " << (h_validNodes[stateIndex] ? "Yes" : "No") << "\n\n";
    
    // Print neighbors and interpolation paths
    std::cout << "Neighbors of State " << stateIndex << ":\n";
    for (int i = 0; i < K; i++) {
        int neighborIdx = h_neighbors[stateIndex * K + i];
        
        // Check if it's a valid neighbor index
        if (neighborIdx >= 0 && neighborIdx < numStates) {
            std::cout << "  Neighbor " << i << " (State " << neighborIdx << "): (";
            
            // Print neighbor position
            for (int d = 0; d < dim; d++) {
                std::cout << h_states[neighborIdx * dim + d];
                if (d < dim - 1) std::cout << ", ";
            }
            std::cout << ")\n";
            
            // Print interpolation path
            std::cout << "    Interpolation path:\n";
            for (int step = 0; step < interpSteps; step++) {
                int edgeIndex = (stateIndex * K + i) * interpSteps * dim + step * dim;
                std::cout << "      Step " << step << ": (";
                
                for (int d = 0; d < dim; d++) {
                    std::cout << h_edges[edgeIndex + d];
                    if (d < dim - 1) std::cout << ", ";
                }
                std::cout << ")\n";
            }
            std::cout << "\n";
        } else {
            std::cout << "  Neighbor " << i << ": Invalid index " << neighborIdx << "\n";
        }
    }
    std::cout << "----------------------\n";
}

__global__ void warmupKernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_STATES) {
        // Perform a simple operation to avoid compiler optimization
        float x = static_cast<float>(idx);
        float y = x * x;
    }
}

// nvcc main.cu collision/cc_2D.cu collision/env_2D.cu planning/construction.cu planning/pprm.cu -o main
// time each kernel for k=10 and num_states=5000


int main(){

    // Set up the CUDA device and RNG seed
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    unsigned long seed = 12345UL;
    
    // Set up the environment
    collision::environment::Env2D env_h = collision::environment::setupEnv2();
    collision::environment::Env2D* env_d;
    planning::setupEnv(env_d, env_h);

    // Set up the roadmap
    planning::Roadmap prm;
    planning::allocateRoadmap(prm);
    
    // Create timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup GPU (doesn't really work)
    warmupKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    cudaCheckErrors("Warmup kernel launch failure");
 
    // Start timing
    cudaEventRecord(start);

    // Build the roadmap

    planning::buildRoadmap(prm, env_d, seed);  // TODO: fix Bounds/seed
    cudaCheckErrors("Roadmap construction failure");

    // Stop timing and calculate elapsed time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);  // Wait for all kernels to finish
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernels completed in %.3f ms\n", milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy results back to host
    planning::copyToHost(prm);

    // Save the results to a file
    saveResults(prm, env_h);

    // Clean up memory
    planning::freeRoadmap(prm);
    planning::cleanupEnv(env_d, env_h);

    return 0;

}