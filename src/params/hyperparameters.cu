#include "hyperparameters.cuh"

// DEFINITIONS of the constant memory variables
// These are the actual memory allocations on the GPU
__constant__ float LOWER_BOUNDS[5];
__constant__ float UPPER_BOUNDS[5];