#pragma once
#include "../collision/env_2D.cuh"
void setupEnv(collision::environment::Env2D *&env_d, const collision::environment::Env2D &env_h);
void cleanupEnv(collision::environment::Env2D *env_d, const collision::environment::Env2D &env_h);

struct Roadmap {
    float *h_states, *d_states;
    float *h_edges,  *d_edges;
    int   *h_neighbors, *d_neighbors;
    bool  *h_validNodes, *d_validNodes, 
          *h_validEdges, *d_validEdges;
};

void allocateRoadmap(Roadmap &map);

void freeRoadmap(Roadmap &map);

__global__ void warmupKernel();
void displayStateAndNeighbors(int stateIndex, const Roadmap& prm, int numStates, int interpSteps, int dim);

