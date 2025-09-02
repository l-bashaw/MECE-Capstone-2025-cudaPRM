#pragma once
#include <iostream>

// CUDA error-checking macro
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

// // Robot c-space bounds
// extern __constant__ float LOWER_BOUNDS[5];  // x, y, theta
// extern __constant__ float UPPER_BOUNDS[5];  

// Collision checking buffer and robot radius
constexpr float CC_E = 5e-3;
constexpr float R_ROBOT = 0.1f;

// PRM parameters
constexpr unsigned int K = 10;   
constexpr unsigned int NUM_STATES = 2000;  
constexpr unsigned int DIM = 5;
constexpr unsigned int INTERP_STEPS = 10;
constexpr unsigned int R_TURNING = 0.05;  // half of R_ROBOT

// Bounds structure to pass to kernels
struct Bounds {
    float lower[5];
    float upper[5];
};

// K = 5, 10, 20
// NUM_STATES = 1000, 2000, 5000, 10000, 20000

