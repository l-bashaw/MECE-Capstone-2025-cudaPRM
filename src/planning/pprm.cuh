#pragma once
#include "../collision/env_2D.cuh"

namespace planning{
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
    void buildRoadmap(Roadmap &prm, collision::environment::Env2D *env_d, unsigned long seed);

    void copyToHost(Roadmap &prm);
}






