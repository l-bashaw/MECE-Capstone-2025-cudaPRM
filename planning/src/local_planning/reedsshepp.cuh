namespace lp::reedsshepp{

    // Reeds-Shepp path segment types
    enum ReedsSheppSegmentType {
        RS_LEFT = 0,
        RS_RIGHT = 1,
        RS_STRAIGHT = 2,
        RS_NOP = 3
    };

    // Device constant for path types (18 paths, up to 5 segments each)
    __device__ __constant__ int d_reedsSheppPathTypes[18][5] = {
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

    // Constants
    __device__ const float PI = 3.14159265358979323846f;
    __device__ const float TWO_PI = 2.0f * PI;
    __device__ const float HALF_PI = 0.5f * PI;
    __device__ const float ZERO_THRESHOLD = 10.0f * 1e-7f; // float epsilon equivalent
    __device__ const float INVALID_PATH_VALUE = 999.0f;

    // Simple path structure for CUDA
    struct ReedsSheppPath {
        int pathTypeIndex;
        float lengths[5];
        float totalLength;
        
        __device__ ReedsSheppPath() : pathTypeIndex(-1), totalLength(INVALID_PATH_VALUE) {
            for(int i = 0; i < 5; i++) lengths[i] = 0.0f;
        }
        
        __device__ ReedsSheppPath(int typeIndex, float t, float u, float v, float w = 0.0f, float x = 0.0f) 
            : pathTypeIndex(typeIndex) {
            lengths[0] = t;
            lengths[1] = u;
            lengths[2] = v;
            lengths[3] = w;
            lengths[4] = x;
            totalLength = fabsf(t) + fabsf(u) + fabsf(v) + fabsf(w) + fabsf(x);
        }
    };

    // Utility functions
    __device__ inline float mod2pi(float x);

    __device__ inline void polar(float x, float y, float &r, float &theta);

    __device__ inline void tauOmega(float u, float v, float xi, float eta, float phi, float &tau, float &omega);

    // Path computation functions
    __device__ inline bool LpSpLp(float x, float y, float phi, float &t, float &u, float &v);

    __device__ inline bool LpSpRp(float x, float y, float phi, float &t, float &u, float &v);

    __device__ inline bool LpRmL(float x, float y, float phi, float &t, float &u, float &v);

    __device__ inline bool LpRupLumRm(float x, float y, float phi, float &t, float &u, float &v);

    __device__ inline bool LpRumLumRp(float x, float y, float phi, float &t, float &u, float &v);

    __device__ inline bool LpRmSLmRp(float x, float y, float phi, float &t, float &u, float &v);

    // Path family computation functions
    __device__ void CSC(float x, float y, float phi, ReedsSheppPath &path);

    __device__ void CCC(float x, float y, float phi, ReedsSheppPath &path);

    __device__ void CCCC(float x, float y, float phi, ReedsSheppPath &path);

    __device__ void CCSC(float x, float y, float phi, ReedsSheppPath &path);

    __device__ void CCSCC(float x, float y, float phi, ReedsSheppPath &path);

    __device__ inline bool LpRmSmLm(float x, float y, float phi, float &t, float &u, float &v);

    __device__ inline bool LpRmSmRm(
        float x,
         float y, 
         float phi, 
         float &t, 
         float &u, 
         float &v
    ); 

    // Main Reeds-Shepp path computation
    __device__ ReedsSheppPath computeReedsSheppPathInternal(
        float x, 
        float y, 
        float phi,
        ReedsSheppPath& path
    );
    

    // Interpolate along a Reeds-Shepp path
    __device__ void interpolateReedsSheppPath(
        const float* start, 
        const ReedsSheppPath& path,                                   
        float t, 
        float* result
    );

    // Main function to compute Reeds-Shepp path between two states
    __device__ void computeReedsSheppPath(
        const float* start, 
        const float* end, 
        float* path
    );



}
