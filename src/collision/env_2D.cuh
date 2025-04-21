#pragma once
#include <cuda_runtime.h>
// #include <torch/extension.h>

namespace collision::environment{

    // Structs for circluar and rectangular obstacles
    struct Circle{
        float x;  // (x,y) defines the center
        float y;
        float r;
    };

    struct Rectangle{
        float x;  // (x,y) defines the center
        float y;
        float h;
        float w;
    };

    // Environment struct that contains the world bounds and obstacles
    struct Env2D{
        float x_min, x_max, y_min, y_max;

        Circle *circles;
        Rectangle *rectangles;

        unsigned int numCircles;
        unsigned int numRectangles;

        Env2D() = default;

        bool ownsMemory = false;
        ~Env2D() { 
            if (ownsMemory){
                delete[] circles;
                delete[] rectangles;
            }
        }
    };

    struct Env2D_d{
        float x_min, x_max, y_min, y_max;

        Circle *circles = nullptr;
        Rectangle *rectangles = nullptr;

        unsigned int numCircles = 0;
        unsigned int numRectangles = 0;

        ~Env2D_d() { 
            if (circles != nullptr){
                cudaFree(circles);
            }
            if (rectangles != nullptr){
                cudaFree(rectangles);
            }
        }
    };

    // void buildEnvFromTensors(
    //     Env2D_d &env,
    //     torch::Tensor circles,     // (x, y, r)
    //     torch::Tensor rectangles,  // (x, y, w, h)
    //     torch::Tensor bounds       // (x_min, x_max, y_min, y_max)
    // );

    bool isValidEnvironment(const Env2D& env);

    Env2D setupEnv1();
    Env2D setupEnv2();
}