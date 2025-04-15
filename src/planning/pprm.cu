#include "../params/hyperparameters.cuh"
#include "pprm.cuh"

void setupEnv(collision::environment::Env2D *&env_d, const collision::environment::Env2D &env_h){
    cudaMalloc(&env_d, sizeof(collision::environment::Env2D));
    cudaCheckErrors("cudaMalloc failure");
    cudaMemset(env_d, 0, sizeof(collision::environment::Env2D));  // Set to zeros first
    cudaCheckErrors("cudaMemset failure");

    if (env_h.numCircles > 0) {
        collision::environment::Circle *d_circles;
        cudaMalloc(&d_circles, env_h.numCircles * sizeof(collision::environment::Circle));
        cudaCheckErrors("cudaMalloc failure");

        cudaMemcpy(
            d_circles, 
            env_h.circles, 
            env_h.numCircles * sizeof(collision::environment::Circle), 
            cudaMemcpyHostToDevice
        );
        cudaCheckErrors("cudaMemcpy failure");

        cudaMemcpy(
            &(env_d->circles),
            &d_circles,
            sizeof(collision::environment::Circle*),
            cudaMemcpyHostToDevice
        );
        cudaCheckErrors("cudaMemcpy failure");

        cudaMemcpy(
            &(env_d->numCircles),
            &env_h.numCircles,
            sizeof(unsigned int),
            cudaMemcpyHostToDevice
        );
    }

    if (env_h.numRectangles > 0) {
        collision::environment::Rectangle* d_rectangles;
        cudaMalloc(&d_rectangles, env_h.numRectangles * sizeof(collision::environment::Rectangle));
        cudaCheckErrors("cudaMalloc failure");

        cudaMemcpy(
            d_rectangles, 
            env_h.rectangles, 
            env_h.numRectangles * sizeof(collision::environment::Rectangle), 
            cudaMemcpyHostToDevice
        );
        cudaCheckErrors("cudaMemcpy failure");

        cudaMemcpy(
            &(env_d->rectangles),
            &d_rectangles,
            sizeof(collision::environment::Rectangle*),
            cudaMemcpyHostToDevice
        );
        cudaCheckErrors("cudaMemcpy failure");

        cudaMemcpy(
            &(env_d->numRectangles),
            &env_h.numRectangles,
            sizeof(unsigned int),
            cudaMemcpyHostToDevice
        );
    } // Environment setup on device.
}

void cleanupEnv(collision::environment::Env2D *env_d, const collision::environment::Env2D &env_h){
    
    collision::environment::Circle *d_circles = nullptr;
    collision::environment::Rectangle *d_rectangles = nullptr;

    if(env_h.numCircles > 0){
        cudaMemcpy(&d_circles, &(env_d->circles), sizeof(collision::environment::Circle*), cudaMemcpyDeviceToHost);
        cudaCheckErrors("cudaMemcpy failure");
        cudaFree(d_circles);
        cudaCheckErrors("cudaFree failure");
    }
    if(env_h.numRectangles > 0){
        cudaMemcpy(&d_rectangles, &(env_d->rectangles), sizeof(collision::environment::Rectangle*), cudaMemcpyDeviceToHost);
        cudaCheckErrors("cudaMemcpy failure");
        cudaFree(d_rectangles);
        cudaCheckErrors("cudaFree failure");
    }  
    cudaFree(env_d);
    cudaCheckErrors("cudaFree failure");
}



void allocateRoadmap(Roadmap &map) {
    size_t size_states    = NUM_STATES * DIM * sizeof(float);
    size_t size_edges     = NUM_STATES * DIM * K * INTERP_STEPS * sizeof(float);
    size_t size_neighbors = NUM_STATES * K * sizeof(int);
    size_t size_validNodes     = NUM_STATES * sizeof(bool);
    size_t size_validEdges = NUM_STATES * K * sizeof(bool);

    cudaMallocHost(&map.h_states, size_states);
    cudaCheckErrors("cudaMallocHost h_states");
    cudaMalloc(&map.d_states, size_states);
    cudaCheckErrors("cudaMalloc d_states");

    cudaMallocHost(&map.h_edges, size_edges);
    cudaCheckErrors("cudaMallocHost h_edges");
    cudaMalloc(&map.d_edges, size_edges);
    cudaCheckErrors("cudaMalloc d_edges");

    cudaMallocHost(&map.h_neighbors, size_neighbors);
    cudaCheckErrors("cudaMallocHost h_neighbors");
    cudaMalloc(&map.d_neighbors, size_neighbors);
    cudaCheckErrors("cudaMalloc d_neighbors");

    cudaMallocHost(&map.h_validNodes, size_validNodes);
    cudaCheckErrors("cudaMallocHost h_valid");
    cudaMalloc(&map.d_validNodes, size_validNodes);
    cudaCheckErrors("cudaMalloc d_valid");

    cudaMallocHost(&map.h_validEdges, size_validEdges);
    cudaCheckErrors("cudaMallocHost h_valid");
    cudaMalloc(&map.d_validEdges, size_validEdges);
    cudaCheckErrors("cudaMalloc d_valid");
}

void freeRoadmap(Roadmap &map) {
    cudaFreeHost(map.h_states);
    cudaFree(map.d_states);
    cudaFreeHost(map.h_edges);
    cudaFree(map.d_edges);
    cudaFreeHost(map.h_neighbors);
    cudaFree(map.d_neighbors);
    cudaFreeHost(map.h_validNodes);
    cudaFree(map.d_validNodes);
    cudaFreeHost(map.h_validEdges);
    cudaFree(map.d_validEdges);
    cudaCheckErrors("cudaFree failure");
}

__global__ void warmupKernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_STATES) {
        // Perform a simple operation to avoid compiler optimization
        float x = static_cast<float>(idx);
        float y = x * x;
    }
}

void displayStateAndNeighbors(int stateIndex, const Roadmap& prm, int numStates, int interpSteps, int dim) {
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
