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
constexpr unsigned int NUM_STATES = 1500;  
constexpr unsigned int DIM = 5;
constexpr unsigned int INTERP_STEPS = 10;
constexpr float R_TURNING = 1;  // half of R_ROBOT


__device__ constexpr float PI = 3.14159265358979323846f;
__device__ constexpr float TWO_PI = 2.0f * PI;
__device__ constexpr float HALF_PI = 0.5f * PI;
__device__ constexpr float ZERO_THRESHOLD = 10.0f * 1e-7f;
__device__ constexpr float INVALID_PATH_VALUE = 999.0f;

__device__ constexpr unsigned int RS_LEFT  = 0;
__device__ constexpr unsigned int RS_RIGHT = 1;
__device__ constexpr unsigned int RS_STRAIGHT = 2;
__device__ constexpr unsigned int RS_NOP = 3;

__device__ constexpr int REEDS_SHEPP_PATH_TYPES[18][5] = {
    {RS_LEFT, RS_RIGHT, RS_LEFT, RS_NOP, RS_NOP},         // 0
    {RS_RIGHT, RS_LEFT, RS_RIGHT, RS_NOP, RS_NOP},        // 1
    {RS_LEFT, RS_RIGHT, RS_LEFT, RS_RIGHT, RS_NOP},       // 2
    {RS_RIGHT, RS_LEFT, RS_RIGHT, RS_LEFT, RS_NOP},       // 3
    {RS_LEFT, RS_RIGHT, RS_STRAIGHT, RS_LEFT, RS_NOP},    // 4
    {RS_RIGHT, RS_LEFT, RS_STRAIGHT, RS_RIGHT, RS_NOP},   // 5
    {RS_LEFT, RS_STRAIGHT, RS_RIGHT, RS_LEFT, RS_NOP},    // 6
    {RS_RIGHT, RS_STRAIGHT, RS_LEFT, RS_RIGHT, RS_NOP},   // 7
    {RS_LEFT, RS_RIGHT, RS_STRAIGHT, RS_RIGHT, RS_NOP},   // 8
    {RS_RIGHT, RS_LEFT, RS_STRAIGHT, RS_LEFT, RS_NOP},    // 9
    {RS_RIGHT, RS_STRAIGHT, RS_RIGHT, RS_LEFT, RS_NOP},   // 10
    {RS_LEFT, RS_STRAIGHT, RS_LEFT, RS_RIGHT, RS_NOP},    // 11
    {RS_LEFT, RS_STRAIGHT, RS_RIGHT, RS_NOP, RS_NOP},     // 12
    {RS_RIGHT, RS_STRAIGHT, RS_LEFT, RS_NOP, RS_NOP},     // 13
    {RS_LEFT, RS_STRAIGHT, RS_LEFT, RS_NOP, RS_NOP},      // 14
    {RS_RIGHT, RS_STRAIGHT, RS_RIGHT, RS_NOP, RS_NOP},    // 15
    {RS_LEFT, RS_RIGHT, RS_STRAIGHT, RS_LEFT, RS_RIGHT},  // 16
    {RS_RIGHT, RS_LEFT, RS_STRAIGHT, RS_RIGHT, RS_LEFT}   // 17
};


// Bounds structure to pass to kernels
struct Bounds {
    float lower[5];
    float upper[5];
};

// K = 5, 10, 20
// NUM_STATES = 1000, 2000, 5000, 10000, 20000

