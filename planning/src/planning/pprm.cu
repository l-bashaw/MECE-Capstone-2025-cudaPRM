#include "../params/hyperparameters.cuh"
#include "pprm.cuh"
#include "../collision/cc_2D.cuh"
#include "construction.cuh"

// __constant__ float LOWER_BOUNDS[5];
// __constant__ float UPPER_BOUNDS[5];

namespace planning {
    void setupEnv(collision::environment::Env2D *&env_d, const collision::environment::Env2D &env_h){
        cudaMalloc(&env_d, sizeof(collision::environment::Env2D));
        cudaCheckErrors("cudaMalloc failure");
        cudaMemset(env_d, 0, sizeof(collision::environment::Env2D));  // Set to zeros first
        cudaCheckErrors("cudaMemset failure");

        if (env_h.numCircles > 0) {
            collision::environment::Circle *d_circles;
            cudaMalloc(&d_circles, env_h.numCircles * sizeof(collision::environment::Circle));
            cudaCheckErrors("cudaMalloc failure");

            cudaMemcpy(
                d_circles, 
                env_h.circles, 
                env_h.numCircles * sizeof(collision::environment::Circle), 
                cudaMemcpyHostToDevice
            );
            cudaCheckErrors("cudaMemcpy failure");

            cudaMemcpy(
                &(env_d->circles),
                &d_circles,
                sizeof(collision::environment::Circle*),
                cudaMemcpyHostToDevice
            );
            cudaCheckErrors("cudaMemcpy failure");

            cudaMemcpy(
                &(env_d->numCircles),
                &env_h.numCircles,
                sizeof(unsigned int),
                cudaMemcpyHostToDevice
            );
        }

        if (env_h.numRectangles > 0) {
            collision::environment::Rectangle* d_rectangles;
            cudaMalloc(&d_rectangles, env_h.numRectangles * sizeof(collision::environment::Rectangle));
            cudaCheckErrors("cudaMalloc failure");

            cudaMemcpy(
                d_rectangles, 
                env_h.rectangles, 
                env_h.numRectangles * sizeof(collision::environment::Rectangle), 
                cudaMemcpyHostToDevice
            );
            cudaCheckErrors("cudaMemcpy failure");

            cudaMemcpy(
                &(env_d->rectangles),
                &d_rectangles,
                sizeof(collision::environment::Rectangle*),
                cudaMemcpyHostToDevice
            );
            cudaCheckErrors("cudaMemcpy failure");

            cudaMemcpy(
                &(env_d->numRectangles),
                &env_h.numRectangles,
                sizeof(unsigned int),
                cudaMemcpyHostToDevice
            );
        } // Environment setup on device.
    }

    void cleanupEnv(collision::environment::Env2D *env_d) {
        // Copy the entire env structure from device to host to get pointers and counts
        collision::environment::Env2D env_host;
        cudaMemcpy(&env_host, env_d, sizeof(collision::environment::Env2D), cudaMemcpyDeviceToHost);
        cudaCheckErrors("cudaMemcpy env struct failure");
        
        // Free the arrays using the pointers from the host copy
        if(env_host.numCircles > 0) {
            cudaFree(env_host.circles);
            cudaCheckErrors("cudaFree circles failure");
        }
        if(env_host.numRectangles > 0) {
            cudaFree(env_host.rectangles);
            cudaCheckErrors("cudaFree rectangles failure");
        }
        
        // Free the env struct itself
        cudaFree(env_d);
        cudaCheckErrors("cudaFree env_d failure");
    }



    void allocateRoadmap(Roadmap &map) {
        size_t size_states     = NUM_STATES * DIM * sizeof(float);
        size_t size_edges      = NUM_STATES * DIM * K * INTERP_STEPS * sizeof(float);
        size_t size_neighbors  = NUM_STATES * K * sizeof(int);
        size_t size_validNodes = NUM_STATES * sizeof(bool);
        size_t size_validEdges = NUM_STATES * K * sizeof(bool);

        cudaMallocHost(&map.h_states, size_states);
        cudaCheckErrors("cudaMallocHost h_states");
        cudaMalloc(&map.d_states, size_states);
        cudaCheckErrors("cudaMalloc d_states");

        cudaMallocHost(&map.h_edges, size_edges);
        cudaCheckErrors("cudaMallocHost h_edges");
        cudaMalloc(&map.d_edges, size_edges);
        cudaCheckErrors("cudaMalloc d_edges");

        cudaMallocHost(&map.h_neighbors, size_neighbors);
        cudaCheckErrors("cudaMallocHost h_neighbors");
        cudaMalloc(&map.d_neighbors, size_neighbors);
        cudaCheckErrors("cudaMalloc d_neighbors");

        cudaMallocHost(&map.h_validNodes, size_validNodes);
        cudaCheckErrors("cudaMallocHost h_valid");
        cudaMalloc(&map.d_validNodes, size_validNodes);
        cudaCheckErrors("cudaMalloc d_valid");

        cudaMallocHost(&map.h_validEdges, size_validEdges);
        cudaCheckErrors("cudaMallocHost h_valid");
        cudaMalloc(&map.d_validEdges, size_validEdges);
        cudaCheckErrors("cudaMalloc d_valid");
    }

    void freeRoadmap(Roadmap &map) {
        cudaFreeHost(map.h_states);
        cudaFree(map.d_states);
        cudaFreeHost(map.h_edges);
        cudaFree(map.d_edges);
        cudaFreeHost(map.h_neighbors);
        cudaFree(map.d_neighbors);
        cudaFreeHost(map.h_validNodes);
        cudaFree(map.d_validNodes);
        cudaFreeHost(map.h_validEdges);
        cudaFree(map.d_validEdges);
        cudaCheckErrors("cudaFree failure");
    }



    void copyToHost(Roadmap &prm){
        cudaMemcpy(prm.h_states, prm.d_states, NUM_STATES * DIM * sizeof(float), cudaMemcpyDeviceToHost);
        cudaCheckErrors("cudaMemcpy h_states failure");
        cudaMemcpy(prm.h_edges, prm.d_edges, NUM_STATES * DIM * K * INTERP_STEPS * sizeof(float), cudaMemcpyDeviceToHost);
        cudaCheckErrors("cudaMemcpy h_edges failure");
        cudaMemcpy(prm.h_neighbors, prm.d_neighbors, NUM_STATES * K * sizeof(int), cudaMemcpyDeviceToHost);
        cudaCheckErrors("cudaMemcpy h_neighbors failure");
        cudaMemcpy(prm.h_validNodes, prm.d_validNodes, NUM_STATES * sizeof(bool), cudaMemcpyDeviceToHost);
        cudaCheckErrors("cudaMemcpy h_valid failure");
        cudaMemcpy(prm.h_validEdges, prm.d_validEdges, NUM_STATES * K * sizeof(bool), cudaMemcpyDeviceToHost);
        cudaCheckErrors("cudaMemcpy h_valid failure");
    }
    

    void buildRoadmap(Roadmap &prm, collision::environment::Env2D *env_d, Bounds bounds, unsigned long seed){
        int blocksize = 256;
        int gridsize = (NUM_STATES + blocksize - 1) / blocksize;
        int gridsize1 = (NUM_STATES * K + blocksize - 1) / blocksize;
        prm::construction::generateStates<<<gridsize, blocksize>>>(prm.d_states, prm.d_validNodes, *env_d, bounds, seed);
        cudaDeviceSynchronize();
        cudaCheckErrors("Stategen kernel launch failure");

        prm::construction::findNeighbors<<<gridsize, blocksize>>>(prm.d_states, prm.d_neighbors);
        cudaDeviceSynchronize();
        cudaCheckErrors("kNN kernel launch failure");

        prm::construction::interpolate<<<gridsize, blocksize>>>(prm.d_states, prm.d_neighbors, prm.d_edges);
        cudaDeviceSynchronize();
        cudaCheckErrors("Interpolation kernel launch failure");

        // collision::environment::nodesInCollision<<<gridsize, blocksize>>>(prm.d_states, prm.d_validNodes, *env_d);
        // cudaDeviceSynchronize();
        // cudaCheckErrors("Collision kernel launch failure");

        collision::environment::edgesInCollision<<<gridsize1, blocksize>>>(prm.d_edges, prm.d_validEdges, *env_d);
        cudaDeviceSynchronize();
    }



    void allocateStartAndGoal(
        Start &start, Goal &goal
    ){
        size_t size_states     = 1 * DIM * sizeof(float);
        size_t size_edges      = 1 * DIM * K * INTERP_STEPS * sizeof(float);
        size_t size_neighbors  = 1 * K * sizeof(int);
        size_t size_validNodes = 1 * sizeof(bool);
        size_t size_validEdges = 1 * K * sizeof(bool);

        // Start allocations
        cudaMallocHost(&start.h_state, size_states);
        cudaCheckErrors("cudaMallocHost h_states");
        cudaMalloc(&start.d_state, size_states);
        cudaCheckErrors("cudaMalloc d_states");

        cudaMallocHost(&start.h_edges, size_edges);
        cudaCheckErrors("cudaMallocHost h_edges");
        cudaMalloc(&start.d_edges, size_edges);
        cudaCheckErrors("cudaMalloc d_edges");

        cudaMallocHost(&start.h_neighbors, size_neighbors);
        cudaCheckErrors("cudaMallocHost h_neighbors");
        cudaMalloc(&start.d_neighbors, size_neighbors);
        cudaCheckErrors("cudaMalloc d_neighbors");

        cudaMallocHost(&start.h_validNodes, size_validNodes);
        cudaCheckErrors("cudaMallocHost h_valid");
        cudaMalloc(&start.d_validNodes, size_validNodes);
        cudaCheckErrors("cudaMalloc d_valid");

        cudaMallocHost(&start.h_validEdges, size_validEdges);
        cudaCheckErrors("cudaMallocHost h_valid");
        cudaMalloc(&start.d_validEdges, size_validEdges);
        cudaCheckErrors("cudaMalloc d_valid");

        // Goal allocations
        cudaMallocHost(&goal.h_state, size_states);
        cudaCheckErrors("cudaMallocHost h_states");
        cudaMalloc(&goal.d_state, size_states);
        cudaCheckErrors("cudaMalloc d_states");

        cudaMallocHost(&goal.h_edges, size_edges);
        cudaCheckErrors("cudaMallocHost h_edges");
        cudaMalloc(&goal.d_edges, size_edges);
        cudaCheckErrors("cudaMalloc d_edges");

        cudaMallocHost(&goal.h_neighbors, size_neighbors);
        cudaCheckErrors("cudaMallocHost h_neighbors");
        cudaMalloc(&goal.d_neighbors, size_neighbors);
        cudaCheckErrors("cudaMalloc d_neighbors");

        cudaMallocHost(&goal.h_validNodes, size_validNodes);
        cudaCheckErrors("cudaMallocHost h_valid");
        cudaMalloc(&goal.d_validNodes, size_validNodes);
        cudaCheckErrors("cudaMalloc d_valid");

        cudaMallocHost(&goal.h_validEdges, size_validEdges);
        cudaCheckErrors("cudaMallocHost h_valid");
        cudaMalloc(&goal.d_validEdges, size_validEdges);
        cudaCheckErrors("cudaMalloc d_valid");
    }

    void freeStartAndGoal(
        Start &start, Goal &goal
    ){
        // Free start memory
        cudaFreeHost(start.h_state);
        cudaFree(start.d_state);
        cudaFreeHost(start.h_edges);
        cudaFree(start.d_edges);
        cudaFreeHost(start.h_neighbors);
        cudaFree(start.d_neighbors);
        cudaFreeHost(start.h_validNodes);
        cudaFree(start.d_validNodes);
        cudaFreeHost(start.h_validEdges);
        cudaFree(start.d_validEdges);
        cudaCheckErrors("cudaFree failure");

        // Free goal memory
        cudaFreeHost(goal.h_state);
        cudaFree(goal.d_state);
        cudaFreeHost(goal.h_edges);
        cudaFree(goal.d_edges);
        cudaFreeHost(goal.h_neighbors);
        cudaFree(goal.d_neighbors);
        cudaFreeHost(goal.h_validNodes);
        cudaFree(goal.d_validNodes);
        cudaFreeHost(goal.h_validEdges);
        cudaFree(goal.d_validEdges);
        cudaCheckErrors("cudaFree failure");
        
    }

    void connectStartAndGoal(
        Start &start, Goal &goal, const Roadmap &prm, collision::environment::Env2D *env_d
    ) {
        int blocksize = 256;
        int gridsize = (NUM_STATES + blocksize - 1) / blocksize;
        int gridsize1 = (NUM_STATES * K + blocksize - 1) / blocksize;
                
        // Debug: Print start and goal states before processing
        printf("Start state: %f, %f, %f\n", start.h_state[0], start.h_state[1], start.h_state[2]);
        printf("Goal state: %f, %f, %f\n", goal.h_state[0], goal.h_state[1], goal.h_state[2]);

        // Find nearest neighbors for start
        prm::construction::findNeighborsForSingleState<<<gridsize, blocksize>>>(
            prm.d_states, start.d_state, start.d_neighbors
        );
        cudaDeviceSynchronize();
        cudaCheckErrors("Start neighbor finding kernel failure");

        // Interpolate edges for start
        prm::construction::interpolateForSingleState<<<gridsize, blocksize>>>(
            start.d_state, start.d_neighbors, prm.d_states, start.d_edges
        );
        cudaDeviceSynchronize();
        cudaCheckErrors("Start interpolation kernel failure");

        // Check edges for collision for start
        collision::environment::edgesInCollision<<<gridsize1, blocksize>>>(
            start.d_edges, start.d_validEdges, *env_d
        );
        cudaDeviceSynchronize();
        cudaCheckErrors("Start edge collision kernel failure");

        // Find nearest neighbors for goal
        prm::construction::findNeighborsForSingleState<<<gridsize, blocksize>>>(
            prm.d_states, goal.d_state, goal.d_neighbors
        );
        cudaDeviceSynchronize();
        cudaCheckErrors("Goal neighbor finding kernel failure");

        // Interpolate edges for goal
        prm::construction::interpolateForSingleState<<<gridsize, blocksize>>>(
            goal.d_state, goal.d_neighbors, prm.d_states, goal.d_edges
        );
        cudaDeviceSynchronize();
        cudaCheckErrors("Goal interpolation kernel failure");

        // Check edges for collision for goal
        collision::environment::edgesInCollision<<<gridsize1, blocksize>>>(
            goal.d_edges, goal.d_validEdges, *env_d
        );
        cudaDeviceSynchronize();
        cudaCheckErrors("Goal edge collision kernel failure");

        // Debug: Print goal state after processing
        printf("Processed goal state: %f, %f, %f\n", goal.h_state[0], goal.h_state[1], goal.h_state[2]);
    }

} // namespace planning

// Will need a host function to add the start and goal states to the roadmap after it is built
// This function will need to:
// 1. Find the nearest neighbor in the roadmap to the start and goal states
// 2. Connect start/goal to their nearest neighbors if the edge is collision-free, using the
//    same reeds-shepp interpolation and collision-checking methods as in the roadmap construction

// Will need to allocate extra memory for the start and goal states, neighbor validity indices, and edges
// Allocate a small chunk of memory for all start and goal connection data to avoid
// having to copy the entire roadmap or allocate number of states + 2 on the device

// Function to add start and goal memory should allow for multiple calls to addStartAndGoal
// and overwrite the previous start and goal data

// Start and goal states, neighbor relations, and edges will be stored independently from the roadmap
// but will use the same data structures (float arrays) and will be concatenated to the roadmap on the host
// when converting to a networkx graph (they will be the last two nodes in the graph)