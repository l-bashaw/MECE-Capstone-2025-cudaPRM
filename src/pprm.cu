#include <iostream>
#include <device_launch_parameters.h>

#include "planning/construction.cuh"
#include "collision/cc_2D.cuh"
#include "collision/env_2D.cuh"
#include "planning/hyperparameters.cuh"

void setupEnv(collision::environment::env_2D* &env_d, const collision::environment::env_2D& env_h){
    cudaMalloc(&env_d, sizeof(collision::environment::env_2D));
    cudaCheckErrors("cudaMalloc failure");
    cudaMemset(env_d, 0, sizeof(collision::environment::env_2D));  // Set to zeros first
    cudaCheckErrors("cudaMemset failure");

    if (env_h.numCircles > 0) {
        collision::environment::circle* d_circles;
        cudaMalloc(&d_circles, env_h.numCircles * sizeof(collision::environment::circle));
        cudaCheckErrors("cudaMalloc failure");

        cudaMemcpy(
            d_circles, 
            env_h.circles, 
            env_h.numCircles * sizeof(collision::environment::circle), 
            cudaMemcpyHostToDevice
        );
        cudaCheckErrors("cudaMemcpy failure");

        cudaMemcpy(
            &(env_d->circles),
            &d_circles,
            sizeof(collision::environment::circle*),
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
        collision::environment::rectangle* d_rectangles;
        cudaMalloc(&d_rectangles, env_h.numRectangles * sizeof(collision::environment::rectangle));
        cudaCheckErrors("cudaMalloc failure");

        cudaMemcpy(
            d_rectangles, 
            env_h.rectangles, 
            env_h.numRectangles * sizeof(collision::environment::rectangle), 
            cudaMemcpyHostToDevice
        );
        cudaCheckErrors("cudaMemcpy failure");

        cudaMemcpy(
            &(env_d->rectangles),
            &d_rectangles,
            sizeof(collision::environment::rectangle*),
            cudaMemcpyHostToDevice
        );
        cudaCheckErrors("cudaMemcpy failure");

        cudaMemcpy(
            &(env_d->rectangles),
            &env_h.numRectangles,
            sizeof(unsigned int),
            cudaMemcpyHostToDevice
        );
    } // Environment setup on device.
}

void cleanupEnv(collision::environment::env_2D* &env_d, const collision::environment::env_2D& env_h){
    return;
}

struct Roadmap {
    float *h_states, *d_states;
    float *h_edges,  *d_edges;
    int   *h_neighbors, *d_neighbors;
    bool  *h_valid, *d_valid;
};

void allocateRoadmap(Roadmap &map) {
    size_t size_states    = NUM_STATES * DIM * sizeof(float);
    size_t size_edges     = NUM_STATES * DIM * K * INTERP_STEPS * sizeof(float);
    size_t size_neighbors = NUM_STATES * K * sizeof(int);
    size_t size_valid     = NUM_STATES * sizeof(bool);

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

    cudaMallocHost(&map.h_valid, size_valid);
    cudaCheckErrors("cudaMallocHost h_valid");
    cudaMalloc(&map.d_valid, size_valid);
    cudaCheckErrors("cudaMalloc d_valid");
}

void freeRoadmap(Roadmap &map) {
    cudaFreeHost(map.h_states);
    cudaFree(map.d_states);
    cudaFreeHost(map.h_edges);
    cudaFree(map.d_edges);
    cudaFreeHost(map.h_neighbors);
    cudaFree(map.d_neighbors);
    cudaFreeHost(map.h_valid);
    cudaFree(map.d_valid);
}

int main(){
    printf("1");
    
    // Set up the environment
    collision::environment::env_2D env_h = collision::environment::setupEnv1();
    collision::environment::env_2D* env_d;

    setupEnv(env_d, env_h);


    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int blocksize = 256;
    int gridsize = (NUM_STATES + blocksize - 1) / blocksize;
    unsigned long seed = 12345UL;


    Roadmap prm;
    allocateRoadmap(prm);
    float *d_states = prm.d_states;
    float *d_edges = prm.d_edges;
    int *d_neighbors = prm.d_neighbors;
    bool *d_valid = prm.d_valid;
    float *h_states = prm.h_states;
    float *h_edges = prm.h_edges;
    int *h_neighbors = prm.h_neighbors;
    bool *h_valid = prm.h_valid;

    // Generate states
    prm::construction::generateStates<<<gridsize, blocksize>>>(d_states, seed);
    cudaCheckErrors("Stategen kernel launch failure");

    prm::construction::findNeighbors<<<gridsize, blocksize>>>(d_states, d_neighbors);
    cudaCheckErrors("kNN kernel launch failure");

    prm::construction::lin_interp<<<gridsize, blocksize>>>(d_states, d_neighbors, d_edges);
    cudaCheckErrors("Interpolation kernel launch failure");

    collision::environment::nodeInCollision<<<gridsize, blocksize>>>(d_states, d_valid, *env_d);
    cudaCheckErrors("Collision kernel launch failure");

    
    freeRoadmap(prm);

    // Free environment memory

    collision::environment::destroyEnv1(env_h);

    return 0;

}