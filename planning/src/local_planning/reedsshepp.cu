#include "reedsshepp.cuh"
#include "../params/hyperparameters.cuh"
//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>


namespace lp::reedsshepp{




    // // Reeds-Shepp path segment types
    // enum ReedsSheppSegmentType {
    //     RS_LEFT = 0,
    //     RS_RIGHT = 1,
    //     RS_STRAIGHT = 2,
    //     RS_NOP = 3
    // };

    // Device constant for path types (18 paths, up to 5 segments each)
    // __device__ static const int d_reedsSheppPathTypes[18][5] = {
    //     {RS_LEFT, RS_RIGHT, RS_LEFT, RS_NOP, RS_NOP},         // 0
    //     {RS_RIGHT, RS_LEFT, RS_RIGHT, RS_NOP, RS_NOP},        // 1
    //     {RS_LEFT, RS_RIGHT, RS_LEFT, RS_RIGHT, RS_NOP},       // 2
    //     {RS_RIGHT, RS_LEFT, RS_RIGHT, RS_LEFT, RS_NOP},       // 3
    //     {RS_LEFT, RS_RIGHT, RS_STRAIGHT, RS_LEFT, RS_NOP},    // 4
    //     {RS_RIGHT, RS_LEFT, RS_STRAIGHT, RS_RIGHT, RS_NOP},   // 5
    //     {RS_LEFT, RS_STRAIGHT, RS_RIGHT, RS_LEFT, RS_NOP},    // 6
    //     {RS_RIGHT, RS_STRAIGHT, RS_LEFT, RS_RIGHT, RS_NOP},   // 7
    //     {RS_LEFT, RS_RIGHT, RS_STRAIGHT, RS_RIGHT, RS_NOP},   // 8
    //     {RS_RIGHT, RS_LEFT, RS_STRAIGHT, RS_LEFT, RS_NOP},    // 9
    //     {RS_RIGHT, RS_STRAIGHT, RS_RIGHT, RS_LEFT, RS_NOP},   // 10
    //     {RS_LEFT, RS_STRAIGHT, RS_LEFT, RS_RIGHT, RS_NOP},    // 11
    //     {RS_LEFT, RS_STRAIGHT, RS_RIGHT, RS_NOP, RS_NOP},     // 12
    //     {RS_RIGHT, RS_STRAIGHT, RS_LEFT, RS_NOP, RS_NOP},     // 13
    //     {RS_LEFT, RS_STRAIGHT, RS_LEFT, RS_NOP, RS_NOP},      // 14
    //     {RS_RIGHT, RS_STRAIGHT, RS_RIGHT, RS_NOP, RS_NOP},    // 15
    //     {RS_LEFT, RS_RIGHT, RS_STRAIGHT, RS_LEFT, RS_RIGHT},  // 16
    //     {RS_RIGHT, RS_LEFT, RS_STRAIGHT, RS_RIGHT, RS_LEFT}   // 17
    // };

    
    // // Simple path structure for CUDA
    // struct ReedsSheppPath {
    //     int pathTypeIndex;
    //     float lengths[5];
    //     float totalLength;
        
    //     __device__ ReedsSheppPath() : pathTypeIndex(-1), totalLength(INVALID_PATH_VALUE) {
    //         for(int i = 0; i < 5; i++) lengths[i] = 0.0f;
    //     }
        
    //     __device__ ReedsSheppPath(int typeIndex, float t, float u, float v, float w = 0.0f, float x = 0.0f) 
    //         : pathTypeIndex(typeIndex) {
    //         lengths[0] = t;
    //         lengths[1] = u;
    //         lengths[2] = v;
    //         lengths[3] = w;
    //         lengths[4] = x;
    //         totalLength = fabsf(t) + fabsf(u) + fabsf(v) + fabsf(w) + fabsf(x);
    //     }
    // };

    // Utility functions
    __device__ inline float mod2pi(float x) {
        float v = fmodf(x, TWO_PI);
        if (v < -PI)
            v += TWO_PI;
        else if (v > PI)
            v -= TWO_PI;
        return v;
    }

    __device__ inline void polar(float x, float y, float &r, float &theta) {
        r = sqrtf(x * x + y * y);
        theta = atan2f(y, x);
    }

    __device__ inline void tauOmega(float u, float v, float xi, float eta, float phi, float &tau, float &omega) {
        float delta = mod2pi(u - v);
        float A = sinf(u) - sinf(delta);
        float B = cosf(u) - cosf(delta) - 1.0f;
        float t1 = atan2f(eta * A - xi * B, xi * A + eta * B);
        float t2 = 2.0f * (cosf(delta) - cosf(v) - cosf(u)) + 3.0f;
        tau = (t2 < 0) ? mod2pi(t1 + PI) : mod2pi(t1);
        omega = mod2pi(tau - u + v - phi);
    }

    // Path computation functions
    __device__ inline bool LpSpLp(float x, float y, float phi, float &t, float &u, float &v) {
        polar(x - sinf(phi), y - 1.0f + cosf(phi), u, t);
        if (t >= -ZERO_THRESHOLD) {
            v = mod2pi(phi - t);
            if (v >= -ZERO_THRESHOLD) {
                return true;
            }
        }
        return false;
    }

    __device__ inline bool LpSpRp(float x, float y, float phi, float &t, float &u, float &v) {
        float t1, u1;
        polar(x + sinf(phi), y - 1.0f - cosf(phi), u1, t1);
        u1 = u1 * u1;
        if (u1 >= 4.0f) {
            float theta;
            u = sqrtf(u1 - 4.0f);
            theta = atan2f(2.0f, u);
            t = mod2pi(t1 + theta);
            v = mod2pi(t - phi);
            return t >= -ZERO_THRESHOLD && v >= -ZERO_THRESHOLD;
        }
        return false;
    }

    __device__ inline bool LpRmL(float x, float y, float phi, float &t, float &u, float &v) {
        float xi = x - sinf(phi);
        float eta = y - 1.0f + cosf(phi);
        float u1, theta;
        polar(xi, eta, u1, theta);
        if (u1 <= 4.0f) {
            u = -2.0f * asinf(0.25f * u1);
            t = mod2pi(theta + 0.5f * u + PI);
            v = mod2pi(phi - t + u);
            return t >= -ZERO_THRESHOLD && u <= ZERO_THRESHOLD;
        }
        return false;
    }

    __device__ inline bool LpRupLumRm(float x, float y, float phi, float &t, float &u, float &v) {
        float xi = x + sinf(phi);
        float eta = y - 1.0f - cosf(phi);
        float rho = 0.25f * (2.0f + sqrtf(xi * xi + eta * eta));
        if (rho <= 1.0f) {
            u = acosf(rho);
            tauOmega(u, -u, xi, eta, phi, t, v);
            return t >= -ZERO_THRESHOLD && v <= ZERO_THRESHOLD;
        }
        return false;
    }

    __device__ inline bool LpRumLumRp(float x, float y, float phi, float &t, float &u, float &v) {
        float xi = x + sinf(phi);
        float eta = y - 1.0f - cosf(phi);
        float rho = (20.0f - xi * xi - eta * eta) / 16.0f;
        if (rho >= 0.0f && rho <= 1.0f) {
            u = -acosf(rho);
            if (u >= -HALF_PI) {
                tauOmega(u, u, xi, eta, phi, t, v);
                return t >= -ZERO_THRESHOLD && v >= -ZERO_THRESHOLD;
            }
        }
        return false;
    }

    __device__ inline bool LpRmSLmRp(float x, float y, float phi, float &t, float &u, float &v) {
        float xi = x + sinf(phi);
        float eta = y - 1.0f - cosf(phi);
        float rho, theta;
        polar(xi, eta, rho, theta);
        if (rho >= 2.0f) {
            u = 4.0f - sqrtf(rho * rho - 4.0f);
            if (u <= ZERO_THRESHOLD) {
                t = mod2pi(atan2f((4.0f - u) * xi - 2.0f * eta, -2.0f * xi + (u - 4.0f) * eta));
                v = mod2pi(t - phi);
                return t >= -ZERO_THRESHOLD && v >= -ZERO_THRESHOLD;
            }
        }
        return false;
    }

    // Path family computation functions
    __device__ void CSC(float x, float y, float phi, ReedsSheppPath &path) {
        float t, u, v, L;
        float Lmin = path.totalLength;
        
        if (LpSpLp(x, y, phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(14, t, u, v);
            Lmin = L;
        }
        if (LpSpLp(-x, y, -phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(14, -t, -u, -v);
            Lmin = L;
        }
        if (LpSpLp(x, -y, -phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(15, t, u, v);
            Lmin = L;
        }
        if (LpSpLp(-x, -y, phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(15, -t, -u, -v);
            Lmin = L;
        }
        if (LpSpRp(x, y, phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(12, t, u, v);
            Lmin = L;
        }
        if (LpSpRp(-x, y, -phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(12, -t, -u, -v);
            Lmin = L;
        }
        if (LpSpRp(x, -y, -phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(13, t, u, v);
            Lmin = L;
        }
        if (LpSpRp(-x, -y, phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(13, -t, -u, -v);
        }
    }

    __device__ void CCC(float x, float y, float phi, ReedsSheppPath &path) {
        float t, u, v, L;
        float Lmin = path.totalLength;
        
        if (LpRmL(x, y, phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(0, t, u, v);
            Lmin = L;
        }
        if (LpRmL(-x, y, -phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(0, -t, -u, -v);
            Lmin = L;
        }
        if (LpRmL(x, -y, -phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(1, t, u, v);
            Lmin = L;
        }
        if (LpRmL(-x, -y, phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(1, -t, -u, -v);
            Lmin = L;
        }

        // backwards
        float xb = x * cosf(phi) + y * sinf(phi);
        float yb = x * sinf(phi) - y * cosf(phi);
        if (LpRmL(xb, yb, phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(0, v, u, t);
            Lmin = L;
        }
        if (LpRmL(-xb, yb, -phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(0, -v, -u, -t);
            Lmin = L;
        }
        if (LpRmL(xb, -yb, -phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(1, v, u, t);
            Lmin = L;
        }
        if (LpRmL(-xb, -yb, phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(1, -v, -u, -t);
        }
    }

    __device__ void CCCC(float x, float y, float phi, ReedsSheppPath &path) {
        float t, u, v, L;
        float Lmin = path.totalLength;
        
        if (LpRupLumRm(x, y, phi, t, u, v) && Lmin > (L = fabsf(t) + 2.0f * fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(2, t, u, -u, v);
            Lmin = L;
        }
        if (LpRupLumRm(-x, y, -phi, t, u, v) && Lmin > (L = fabsf(t) + 2.0f * fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(2, -t, -u, u, -v);
            Lmin = L;
        }
        if (LpRupLumRm(x, -y, -phi, t, u, v) && Lmin > (L = fabsf(t) + 2.0f * fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(3, t, u, -u, v);
            Lmin = L;
        }
        if (LpRupLumRm(-x, -y, phi, t, u, v) && Lmin > (L = fabsf(t) + 2.0f * fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(3, -t, -u, u, -v);
            Lmin = L;
        }

        if (LpRumLumRp(x, y, phi, t, u, v) && Lmin > (L = fabsf(t) + 2.0f * fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(2, t, u, u, v);
            Lmin = L;
        }
        if (LpRumLumRp(-x, y, -phi, t, u, v) && Lmin > (L = fabsf(t) + 2.0f * fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(2, -t, -u, -u, -v);
            Lmin = L;
        }
        if (LpRumLumRp(x, -y, -phi, t, u, v) && Lmin > (L = fabsf(t) + 2.0f * fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(3, t, u, u, v);
            Lmin = L;
        }
        if (LpRumLumRp(-x, -y, phi, t, u, v) && Lmin > (L = fabsf(t) + 2.0f * fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(3, -t, -u, -u, -v);
        }
    }

    __device__ void CCSC(float x, float y, float phi, ReedsSheppPath &path) {
        float t, u, v, L;
        float Lmin = path.totalLength - HALF_PI;
        
        if (LpRmSmLm(x, y, phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(4, t, -HALF_PI, u, v);
            Lmin = L;
        }
        if (LpRmSmLm(-x, y, -phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(4, -t, HALF_PI, -u, -v);
            Lmin = L;
        }
        if (LpRmSmLm(x, -y, -phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(5, t, -HALF_PI, u, v);
            Lmin = L;
        }
        if (LpRmSmLm(-x, -y, phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(5, -t, HALF_PI, -u, -v);
            Lmin = L;
        }

        if (LpRmSmRm(x, y, phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(8, t, -HALF_PI, u, v);
            Lmin = L;
        }
        if (LpRmSmRm(-x, y, -phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(8, -t, HALF_PI, -u, -v);
            Lmin = L;
        }
        if (LpRmSmRm(x, -y, -phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(9, t, -HALF_PI, u, v);
            Lmin = L;
        }
        if (LpRmSmRm(-x, -y, phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(9, -t, HALF_PI, -u, -v);
            Lmin = L;
        }

        // backwards
        float xb = x * cosf(phi) + y * sinf(phi);
        float yb = x * sinf(phi) - y * cosf(phi);
        if (LpRmSmLm(xb, yb, phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(6, v, u, -HALF_PI, t);
            Lmin = L;
        }
        if (LpRmSmLm(-xb, yb, -phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(6, -v, -u, HALF_PI, -t);
            Lmin = L;
        }
        if (LpRmSmLm(xb, -yb, -phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(7, v, u, -HALF_PI, t);
            Lmin = L;
        }
        if (LpRmSmLm(-xb, -yb, phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(7, -v, -u, HALF_PI, -t);
            Lmin = L;
        }

        if (LpRmSmRm(xb, yb, phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(10, v, u, -HALF_PI, t);
            Lmin = L;
        }
        if (LpRmSmRm(-xb, yb, -phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(10, -v, -u, HALF_PI, -t);
            Lmin = L;
        }
        if (LpRmSmRm(xb, -yb, -phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(11, v, u, -HALF_PI, t);
            Lmin = L;
        }
        if (LpRmSmRm(-xb, -yb, phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(11, -v, -u, HALF_PI, -t);
        }
    }

    __device__ void CCSCC(float x, float y, float phi, ReedsSheppPath &path) {
        float t, u, v, L;
        float Lmin = path.totalLength - PI;
        
        if (LpRmSLmRp(x, y, phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(16, t, -HALF_PI, u, -HALF_PI, v);
            Lmin = L;
        }
        if (LpRmSLmRp(-x, y, -phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(16, -t, HALF_PI, -u, HALF_PI, -v);
            Lmin = L;
        }
        if (LpRmSLmRp(x, -y, -phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(17, t, -HALF_PI, u, -HALF_PI, v);
            Lmin = L;
        }
        if (LpRmSLmRp(-x, -y, phi, t, u, v) && Lmin > (L = fabsf(t) + fabsf(u) + fabsf(v))) {
            path = ReedsSheppPath(17, -t, HALF_PI, -u, HALF_PI, -v);
        }
    }

    // Missing helper functions for CCSC and CCSCC
    __device__ inline bool LpRmSmLm(float x, float y, float phi, float &t, float &u, float &v) {
        float xi = x - sinf(phi);
        float eta = y - 1.0f + cosf(phi);
        float rho, theta;
        polar(xi, eta, rho, theta);
        if (rho >= 2.0f) {
            float r = sqrtf(rho * rho - 4.0f);
            u = 2.0f - r;
            t = mod2pi(theta + atan2f(r, -2.0f));
            v = mod2pi(phi - HALF_PI - t);
            return t >= -ZERO_THRESHOLD && u <= ZERO_THRESHOLD && v <= ZERO_THRESHOLD;
        }
        return false;
    }

    __device__ inline bool LpRmSmRm(float x, float y, float phi, float &t, float &u, float &v) {
        float xi = x + sinf(phi);
        float eta = y - 1.0f - cosf(phi);
        float rho, theta;
        polar(-eta, xi, rho, theta);
        if (rho >= 2.0f) {
            t = theta;
            u = 2.0f - rho;
            v = mod2pi(t + HALF_PI - phi);
            return t >= -ZERO_THRESHOLD && u <= ZERO_THRESHOLD && v <= ZERO_THRESHOLD;
        }
        return false;
    }

    // Main Reeds-Shepp path computation
    __device__ ReedsSheppPath computeReedsSheppPathInternal(float x, float y, float phi, ReedsSheppPath& path) {
        path = ReedsSheppPath(); // Initialized with invalid path
        
        CSC(x, y, phi, path);
        CCC(x, y, phi, path);
        CCCC(x, y, phi, path);
        CCSC(x, y, phi, path);
        CCSCC(x, y, phi, path);
        
        return path;
    }

    // Interpolate along a Reeds-Shepp path
    __device__ void interpolateReedsSheppPath(const float* start, const ReedsSheppPath& path, 
                                            float t, float* result) {
        if (path.pathTypeIndex == -1) {
            // Invalid path, set to large values
            result[0] = INVALID_PATH_VALUE;
            result[1] = INVALID_PATH_VALUE;
            result[2] = INVALID_PATH_VALUE;
            result[3] = 0.0f;
            result[4] = 0.0f;
            return;
        }
        
        float seg = t * path.totalLength;
        float phi = start[2]; // Current orientation
        float x = 0.0f, y = 0.0f; // Current position in local coordinates
        
        for (int i = 0; i < 5 && seg > 0.0f; ++i) {
            float v;
            if (path.lengths[i] < 0.0f) {
                v = fmaxf(-seg, path.lengths[i]);
                seg += v;
            } else {
                v = fminf(seg, path.lengths[i]);
                seg -= v;
            }
            
            float currentPhi = phi;
            switch (REEDS_SHEPP_PATH_TYPES[path.pathTypeIndex][i]) {
                case RS_LEFT:
                    x += sinf(currentPhi + v) - sinf(currentPhi);
                    y += -cosf(currentPhi + v) + cosf(currentPhi);
                    phi = currentPhi + v;
                    break;
                case RS_RIGHT:
                    x += -sinf(currentPhi - v) + sinf(currentPhi);
                    y += cosf(currentPhi - v) - cosf(currentPhi);
                    phi = currentPhi - v;
                    break;
                case RS_STRAIGHT:
                    x += v * cosf(currentPhi);
                    y += v * sinf(currentPhi);
                    break;
                case RS_NOP:
                    break;
            }
        }
        
        // Transform back to world coordinates
        result[0] = x * R_TURNING + start[0];
        result[1] = y * R_TURNING + start[1];
        result[2] = phi;
        result[3] = 0.0f;
        result[4] = 0.0f;
    }

    // Main function to compute Reeds-Shepp path between two states
    __device__ void computeReedsSheppPath(const float* start, const float* end, float* path) {
        // Transform end state to local coordinate frame of start state
        float dx = end[0] - start[0];
        float dy = end[1] - start[1];
        float c = cosf(start[2]);
        float s = sinf(start[2]);
        float x = c * dx + s * dy;
        float y = -s * dx + c * dy;
        float phi = end[2] - start[2];
        
        // Normalize by turning radius
        x /= R_TURNING;
        y /= R_TURNING;
        
        // Compute the optimal Reeds-Shepp path
        ReedsSheppPath rsPath;
        computeReedsSheppPathInternal(x, y, phi, rsPath);
        
        // Generate INTERP_STEPS waypoints along the path
        for (int t = 0; t < INTERP_STEPS; t++) {
            float t_val = (float)t / (float)(INTERP_STEPS - 1);
            float waypoint[5];
            interpolateReedsSheppPath(start, rsPath, t_val, waypoint);
            
            // Store in path array
            for (int d = 0; d < DIM; d++) {
                path[t * DIM + d] = waypoint[d];
            }
        }
    }

    __global__ void dummyKernel() {
        float start[3], end[3], path[5];
        computeReedsSheppPath(start, end, path);
    }
}