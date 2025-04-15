#include <iostream>
#include "env_2D.cuh"

namespace collision::environment{
    
    // Function to check if a created environment is valid
    bool isValidEnvironment(const Env2D& env){
        // Check if bounds are valid
        if (env.x_min >= env.x_max || env.y_min >= env.y_max){
            printf("Environment bounds are invalid\n");
            return false;
        }

        // Check if circles are within bounds
        for (int i = 0; i < env.numCircles; i++){
            if (env.circles[i].x  < env.x_min ||
                env.circles[i].x  > env.x_max ||
                env.circles[i].y  < env.y_min ||
                env.circles[i].y  > env.y_max)
            {
                printf("Circle %d is out of environment bounds\n", i);
                return false;
            }
        }

        // Check if rectangles are within bounds
        for (int i = 0; i < env.numRectangles; i++){
            if (env.rectangles[i].x < env.x_min ||
                env.rectangles[i].x > env.x_max ||
                env.rectangles[i].y < env.y_min ||
                env.rectangles[i].y > env.y_max)
            {   
                printf("Rectangle %d is out of environment bounds\n", i);
                return false;
            }
        }
        return true;
    }

    Env2D setupEnv1(){
        Env2D env;
        env.numCircles = 2;
        env.numRectangles = 2;

        // Allocate memory for circles and rectangles
        env.circles = new Circle[env.numCircles];
        env.rectangles = new Rectangle[env.numRectangles];
    
        // Define circular obstacles
        env.circles[0].x = 0.0f;
        env.circles[0].y = 0.0f;
        env.circles[0].r = 1.0f;
        
        env.circles[1].x = 3.0f;
        env.circles[1].y = 3.0f;
        env.circles[1].r = 1.5f;
    
        // Define rectangular obstacles
        env.rectangles[0].x = 5.0f;
        env.rectangles[0].y = 5.0f;
        env.rectangles[0].w = 2.0f;
        env.rectangles[0].h = 2.0f;
        
        env.rectangles[1].x = -5.0f;
        env.rectangles[1].y = -5.0f;
        env.rectangles[1].w = 2.0f;
        env.rectangles[1].h = 2.0f;
    
        // Define the environment bounds
        env.x_min = -10.0f;
        env.x_max = 10.0f;
        env.y_min = -10.0f;
        env.y_max = 10.0f;
    
        // Ensure that the environment is valid before returning
        if (isValidEnvironment(env) == false){
            printf("Invalid environment parameters\n");
            exit(1);
        }
    
        return env;
    }

  
}