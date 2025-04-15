#pragma once

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
        float x_min;
        float x_max;
        float y_min;
        float y_max;

        Circle *circles;
        Rectangle *rectangles;

        unsigned int numCircles;
        unsigned int numRectangles;

        Env2D() = default;

        ~Env2D() {
            delete[] circles;
            delete[] rectangles;
        }
    };


    bool isValidEnvironment(const Env2D& env);

    Env2D setupEnv1();


    // void setupDeviceEnv(env_2D* &env_d, const env_2D& env_h);

    // void destroyDeviceEnv(env_2D* env_d, const env_2D& env_h);

}