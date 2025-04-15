#include "params/hyperparameters.cuh"
#include "planning/pprm.cuh"
#include "collision/cc_2D.cuh"
#include "planning/construction.cuh"

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