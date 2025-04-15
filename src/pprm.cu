#include <iostream>
#include <device_launch_parameters.h>

#include "planning/construction.cuh"
#include "collision/cc_2D.cuh"
#include "collision/env_2D.cuh"
#include "params/hyperparameters.cuh"

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

struct Roadmap {
    float *h_states, *d_states;
    float *h_edges,  *d_edges;
    int   *h_neighbors, *d_neighbors;
    bool  *h_validNodes, *d_validNodes, 
          *h_validEdges, *d_validEdges;
};

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

int main(){
    
    // Set up the environment
    collision::environment::Env2D env_h = collision::environment::setupEnv1();
    collision::environment::Env2D* env_d;

    setupEnv(env_d, env_h);


    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int blocksize = 256;
    int gridsize = (NUM_STATES + blocksize - 1) / blocksize;
    int gridsize1 = (NUM_STATES * K + blocksize - 1) / blocksize;

    unsigned long seed = 12345UL;


    Roadmap prm;
    allocateRoadmap(prm);
    float *d_states = prm.d_states;
    float *d_edges = prm.d_edges;
    int *d_neighbors = prm.d_neighbors;
    bool *d_validNodes = prm.d_validNodes;
    bool *d_validEdges = prm.d_validEdges;

    float *h_states = prm.h_states;
    float *h_edges = prm.h_edges;
    int *h_neighbors = prm.h_neighbors;
    bool *h_validNodes = prm.h_validNodes;
    bool *h_validEdges = prm.h_validEdges;

       

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup kernel
    warmupKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    cudaCheckErrors("Warmup kernel launch failure");
 
    cudaEventRecord(start);

    // Generate states
    prm::construction::generateStates<<<gridsize, blocksize>>>(d_states, seed);
    cudaDeviceSynchronize();
    cudaCheckErrors("Stategen kernel launch failure");

    prm::construction::findNeighbors<<<gridsize, blocksize>>>(d_states, d_neighbors);
    cudaDeviceSynchronize();
    cudaCheckErrors("kNN kernel launch failure");

    prm::construction::lin_interp<<<gridsize, blocksize>>>(d_states, d_neighbors, d_edges);
    cudaDeviceSynchronize();
    cudaCheckErrors("Interpolation kernel launch failure");

    collision::environment::nodesInCollision<<<gridsize, blocksize>>>(d_states, d_validNodes, *env_d);
    cudaDeviceSynchronize();
    cudaCheckErrors("Collision kernel launch failure");

    collision::environment::edgesInCollision<<<gridsize1, blocksize>>>(d_edges, d_validEdges, *env_d);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);  // Wait for all kernels to finish
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernels completed in %.3f ms\n", milliseconds);

    // Copy results back to host
    cudaMemcpy(h_states, d_states, NUM_STATES * DIM * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy h_states failure");
    cudaMemcpy(h_edges, d_edges, NUM_STATES * DIM * K * INTERP_STEPS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy h_edges failure");
    cudaMemcpy(h_neighbors, d_neighbors, NUM_STATES * K * sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy h_neighbors failure");
    cudaMemcpy(h_validNodes, d_validNodes, NUM_STATES * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy h_valid failure");
    cudaMemcpy(h_validEdges, d_validEdges, NUM_STATES * K * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy h_valid failure");

    // Display the first state and its neighbors
    // displayStateAndNeighbors(0, prm, NUM_STATES, INTERP_STEPS, DIM);

    // Save the results to a file
    FILE *file = fopen("roadmap.txt", "w");
    if (file == NULL) {
        std::cerr << "Error opening file for writing" << std::endl;
        return 1;
    }
    for (int i = 0; i < NUM_STATES; i++) {
        fprintf(file, "State %d: ", i);
        for (int j = 0; j < DIM; j++) {
            fprintf(file, "%f ", h_states[i * DIM + j]);
        }
        fprintf(file, "\n");
    }
    for (int i = 0; i < NUM_STATES; i++) {
        fprintf(file, "Neighbors %d: ", i);
        for (int j = 0; j < K; j++) {
            fprintf(file, "%d ", h_neighbors[i * K + j]);
        }
        fprintf(file, "\n");
    }
    for (int i = 0; i < NUM_STATES; i++) {
        fprintf(file, "Valid Node %d: %d\n", i, h_validNodes[i]);
    }
    for (int i = 0; i < NUM_STATES * K; i++) {
        fprintf(file, "Valid Edge %d: %d\n", i, h_validEdges[i]);
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

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    freeRoadmap(prm);
    cleanupEnv(env_d, env_h);

    return 0;

}