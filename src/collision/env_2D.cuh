#pragma once

namespace collision::environment{

    // Structs for circluar and rectangular obstacles
    struct circle{
        float x;  // (x,y) defines the center
        float y;
        float r;
    };

    struct rectangle{
        float x;  // (x,y) defines the center
        float y;
        float h;
        float w;
    };

    // Environment struct that contains the world bounds and obstacles
    struct env_2D{
        float x_min;
        float x_max;
        float y_min;
        float y_max;

        circle *circles;
        rectangle *rectangles;

        unsigned int numCircles;
        unsigned int numRectangles;

        env_2D() = default;

        ~env_2D() {
            delete[] circles;
            delete[] rectangles;
        }
    };


    bool isValidEnvironment(const env_2D& env);

    env_2D setupEnv1();


    // void setupDeviceEnv(env_2D* &env_d, const env_2D& env_h);

    // void destroyDeviceEnv(env_2D* env_d, const env_2D& env_h);

}