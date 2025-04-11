#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

#ifndef HYPERPARAMS
#define HYPERPARAMS

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

// Robot c-space bounds
__constant__ float LOWER_BOUNDS[5] = { -10.0f, -10.0f, 0.0f  , 0.0, 0.0  };  // x, y, theta
__constant__ float UPPER_BOUNDS[5] = { 10.0f ,  10.0f, 2*M_PI, 0.0, 0.0  };  

// Collision checking buffer and robot radius
#define CC_E 5e-3
#define R_ROBOT 0.5f

// PRM parameters
#define K 10
#define NUM_STATES 10000
#define DIM 11
#define INTERP_STEPS 5

#endif