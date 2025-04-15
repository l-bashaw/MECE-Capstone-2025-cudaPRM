#include "params/hyperparameters.cuh"
#include "planning/pprm.cuh"
#include "collision/cc_2D.cuh"
#include "planning/construction.cuh"

void copyToHost(Roadmap &prm){
    cudaMemcpy(prm.h_states, prm.d_states, NUM_STATES * DIM * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy h_states failure");
    cudaMemcpy(prm.h_edges, prm.d_edges, NUM_STATES * DIM * K * INTERP_STEPS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy h_edges failure");
    cudaMemcpy(prm.h_neighbors, prm.d_neighbors, NUM_STATES * K * sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy h_neighbors failure");
    cudaMemcpy(prm.h_validNodes, prm.d_validNodes, NUM_STATES * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy h_valid failure");
    cudaMemcpy(prm.h_validEdges, prm.d_validEdges, NUM_STATES * K * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy h_valid failure");
}

void saveResults(Roadmap &prm, collision::environment::Env2D &env_h){
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

    // Generate states
    prm::construction::generateStates<<<gridsize, blocksize>>>(prm.d_states, seed);
    cudaDeviceSynchronize();
    cudaCheckErrors("Stategen kernel launch failure");

    prm::construction::findNeighbors<<<gridsize, blocksize>>>(prm.d_states, prm.d_neighbors);
    cudaDeviceSynchronize();
    cudaCheckErrors("kNN kernel launch failure");

    prm::construction::lin_interp<<<gridsize, blocksize>>>(prm.d_states, prm.d_neighbors, prm.d_edges);
    cudaDeviceSynchronize();
    cudaCheckErrors("Interpolation kernel launch failure");

    collision::environment::nodesInCollision<<<gridsize, blocksize>>>(prm.d_states, prm.d_validNodes, *env_d);
    cudaDeviceSynchronize();
    cudaCheckErrors("Collision kernel launch failure");

    collision::environment::edgesInCollision<<<gridsize1, blocksize>>>(prm.d_edges, prm.d_validEdges, *env_d);
    cudaDeviceSynchronize();

    // Stop timing and calculate elapsed time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);  // Wait for all kernels to finish
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernels completed in %.3f ms\n", milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy results back to host
    copyToHost(prm);

    // Display the first state and its neighbors
    // displayStateAndNeighbors(0, prm, NUM_STATES, INTERP_STEPS, DIM);

    // Save the results to a file
    saveResults(prm, env_h);

    // Clean up
    
    freeRoadmap(prm);
    cleanupEnv(env_d, env_h);

    return 0;

}