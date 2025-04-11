#include "hyperparameters.cuh"
#include "stategen.cu"
#include "knn.cu"
#include "cc_2D.cu"

#define CC_E 5e-3
#define R_ROBOT 0.5f

// PRM parameters
#define K 10
#define NUM_STATES 10000
#define DIM 11
#define INTERP_STEPS 5

namespace pprm{

    



}



int main(){

    // Set up the environment
    collision::environment::env_2D env_h = collision::environment::setupEnv1();
    collision::environment::env_2D* env_d;

    cudaMalloc(&env_d, sizeof(collision::environment::env_2D));
    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(env_d, &env_h, sizeof(collision::environment::env_2D), cudaMemcpyHostToDevice);
    

    // Allocate memory for states
    float* d_states;
    cudaMalloc(&d_states, NUM_STATES * DIM * sizeof(float));
    cudaCheckErrors("cudaMalloc failure");

    // Generate random states
    unsigned long seed = 12345UL;
    prm::construction::generateRandomVectors<<<(NUM_STATES + 255) / 256, 256>>>(d_states, seed, NUM_STATES);
    cudaCheckErrors("kernel launch failure");

    // Free allocated memory
    cudaFree(d_states);
    cudaCheckErrors("cudaFree failure");

    return 0;

}