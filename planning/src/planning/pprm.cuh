#pragma once
#include "../params/hyperparameters.cuh"
#include "../collision/env_2D.cuh"

namespace planning{
    void setupEnv(collision::environment::Env2D *&env_d, const collision::environment::Env2D &env_h);
    void cleanupEnv(collision::environment::Env2D *env_d);
    
    struct Roadmap {
        float *h_states, *d_states;
        float *h_edges,  *d_edges;
        int   *h_neighbors, *d_neighbors;
        bool  *h_validNodes, *d_validNodes, 
              *h_validEdges, *d_validEdges;
    };

    struct Roadmap_d {
        float *d_states, *d_edges;
        int   *d_neighbors;
        bool  *d_validNodes, *d_validEdges;
    };

    struct Start{
        float *h_state, *d_state;
        float *h_edges,  *d_edges;
        int *h_neighbors, *d_neighbors;
        bool *h_validNodes, *d_validNodes,
             *h_validEdges, *d_validEdges;
    };  

    struct Goal{
        float *h_state, *d_state;
        float *h_edges,  *d_edges;
        int *h_neighbors, *d_neighbors;
        bool *h_validNodes, *d_validNodes,
             *h_validEdges, *d_validEdges;
    };

    void allocateStartAndGoal(
        Start &start, Goal &goal
    );
    void freeStartAndGoal(
        Start &start, Goal &goal
    );

    void allocateRoadmap(Roadmap &map);
    void freeRoadmap(Roadmap &map);
    void buildRoadmap(Roadmap &prm, collision::environment::Env2D *env_d, Bounds bounds, unsigned long seed);

    void copyToHost(Roadmap &prm);
    void connectStartAndGoal(
        Start &start, Goal &goal, const Roadmap &prm, collision::environment::Env2D *env_d
    );
    // void roadmapToTensors(Roadmap &prm);
}






