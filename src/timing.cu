#include <fstream>
#include <string>
#include <vector>
#include <sstream>

#include "params/hyperparameters.cuh"
#include "planning/pprm.cuh"


__global__ void warmupKernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_STATES) {
        // Perform a simple operation to avoid compiler optimization
        float x = static_cast<float>(idx);
        float y = x * x;
    }
}

int main() {
    // Set up the CUDA device and RNG seed
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    unsigned long seed = 12345UL;
    
    // Set up the environment
    collision::environment::Env2D env_h = collision::environment::setupEnv1();
    collision::environment::Env2D* env_d;
    planning::setupEnv(env_d, env_h);
    
    // Vector to store all timing results
    std::vector<float> timings;
    timings.reserve(100);
    
    // Create timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup GPU (doesn't really work)
    warmupKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    cudaCheckErrors("Warmup kernel launch failure");
    
    printf("Running buildRoadmap 100 times...\n");
    
    // Run 100 times
    for (int i = 0; i < 100; i++) {
        // Set up the roadmap for this iteration
        planning::Roadmap prm;
        planning::allocateRoadmap(prm);
        
        // Different seed for each run to avoid caching effects (optional)
        unsigned long run_seed = seed + i;
        
        // Start timing
        cudaEventRecord(start);
        
        // Build the roadmap
        planning::buildRoadmap(prm, env_d, run_seed);
        cudaCheckErrors("Roadmap construction failure");
        
        // Stop timing and calculate elapsed time
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);  // Wait for all kernels to finish
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        timings.push_back(milliseconds);
        
        // printf("Run %d: %.3f ms\n", i+1, milliseconds);
        
        // Clean up roadmap memory for this iteration
        planning::freeRoadmap(prm);
    }

    std::stringstream filename;
    filename << "times_" << NUM_STATES << "_states_" << K << "_K.txt";
    std::string output_file = filename.str();
    
    // Write timing results to a file
    std::ofstream outfile(output_file);
    if (outfile.is_open()) {
        outfile << "Time(ms)\n";
        for (int i = 0; i < timings.size(); i++) {
            outfile << timings[i] << "\n";
        }
        
        outfile.close();
        printf("Timing results written to roadmap_timings.txt\n");
    } else {
        printf("Failed to open output file\n");
    }
    
    // Clean up timing events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Clean up environment
    planning::cleanupEnv(env_d, env_h);
    
    return 0;
}