#include "params/hyperparameters.cuh"
#include "planning/pprm.cuh"


int main(){

    // Set up the CUDA device and RNG seed
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    unsigned long seed = 12345UL;

    int numStates = 1000;
    int k = 5;
    
    
    // Set up the environment
    collision::environment::Env2D env_h = collision::environment::setupEnv1();
    collision::environment::Env2D* env_d;
    planning::setupEnv(env_d, env_h);

    // Set up the roadmap
    planning::Roadmap prm;
    planning::allocateRoadmap(prm);

    // Build the roadmap
    planning::buildRoadmap(prm, env_d, seed); // TODO: fix Bounds/seed
    cudaCheckErrors("Roadmap construction failure");

    // Copy results back to host
    planning::copyToHost(prm);

    // Clean up memory
    planning::freeRoadmap(prm);
    planning::cleanupEnv(env_d, env_h);

    return 0;

}