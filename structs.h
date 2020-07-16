#pragma once
#include <cmath>
#include <stdlib.h>
#include <algorithm>
using namespace std;

struct __host__ __device__ Camera {
    float3 pos;
    float horAngle;
    float verAngle;
    float fov;
    float3 dir;

    float3 LD, LU, RD, RU;

    float3 left;
    float3 right;
};

enum __host__ __device__ Primitive {
    SPHERE,
    PLANE,
    TRIANGLE,
};

struct __host__ __device__ Object
{
    Primitive type;
    float shine;
    float specular;
    float mirror;
    float3 color;
    float3 pos;
    float3 size;
    float3 third;
};

struct __host__ __device__ Ray
{
    float3 origin;
    float3 dir;
};

struct __host__ __device__ Light
{
    float3 pos;
    float3 color;
    float intensity;
};

typedef struct {
    double r;       // a fraction between 0 and 1
    double g;       // a fraction between 0 and 1
    double b;       // a fraction between 0 and 1
} rgb;

typedef struct {
    double h;       // angle in degrees
    double s;       // a fraction between 0 and 1
    double v;       // a fraction between 0 and 1
} hsv;

static hsv   rgb2hsv(rgb in);
static rgb   hsv2rgb(hsv in);

hsv rgb2hsv(rgb in)
{
    hsv         out;
    double      min, max, delta;

    min = in.r < in.g ? in.r : in.g;
    min = min < in.b ? min : in.b;

    max = in.r > in.g ? in.r : in.g;
    max = max > in.b ? max : in.b;

    out.v = max;                                // v
    delta = max - min;
    if (delta < 0.00001)
    {
        out.s = 0;
        out.h = 0; // undefined, maybe nan?
        return out;
    }
    if (max > 0.0) { // NOTE: if Max is == 0, this divide would cause a crash
        out.s = (delta / max);                  // s
    }
    else {
        // if max is 0, then r = g = b = 0              
        // s = 0, h is undefined
        out.s = 0.0;
        out.h = 0.0;                            // its now undefined
        return out;
    }
    if (in.r >= max)                           // > is bogus, just keeps compilor happy
        out.h = (in.g - in.b) / delta;        // between yellow & magenta
    else
        if (in.g >= max)
            out.h = 2.0 + (in.b - in.r) / delta;  // between cyan & yellow
        else
            out.h = 4.0 + (in.r - in.g) / delta;  // between magenta & cyan

    out.h *= 60.0;                              // degrees

    if (out.h < 0.0)
        out.h += 360.0;

    return out;
}

rgb hsv2rgb(hsv in)
{
    double      hh, p, q, t, ff;
    long        i;
    rgb         out;

    if (in.s <= 0.0) {       // < is bogus, just shuts up warnings
        out.r = in.v;
        out.g = in.v;
        out.b = in.v;
        return out;
    }
    hh = in.h;
    if (hh >= 360.0) hh = 0.0;
    hh /= 60.0;
    i = (long)hh;
    ff = hh - i;
    p = in.v * (1.0 - in.s);
    q = in.v * (1.0 - (in.s * ff));
    t = in.v * (1.0 - (in.s * (1.0 - ff)));

    switch (i) {
    case 0:
        out.r = in.v;
        out.g = t;
        out.b = p;
        break;
    case 1:
        out.r = q;
        out.g = in.v;
        out.b = p;
        break;
    case 2:
        out.r = p;
        out.g = in.v;
        out.b = t;
        break;

    case 3:
        out.r = p;
        out.g = q;
        out.b = in.v;
        break;
    case 4:
        out.r = t;
        out.g = p;
        out.b = in.v;
        break;
    case 5:
    default:
        out.r = in.v;
        out.g = p;
        out.b = q;
        break;
    }
    return out;
}

// vector operations
__host__ __device__ float3 inline operator+(const float3& v1, const float3& v2) {
    return { v1.x + v2.x, v1.y + v2.y, v1.z + v2.z };
}
__host__ __device__ float3 inline operator-(const float3& v1, const float3& v2) {
    return { v1.x - v2.x, v1.y - v2.y, v1.z - v2.z };
}
__host__ __device__ float inline operator*(const float3& v1, const float3& v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}
__host__ __device__ float3 inline operator*(const float3& v1, const float& a) {
    return { v1.x * a, v1.y * a, v1.z * a };
}
__host__ __device__ float3 inline operator*(const float& a, const float3& v1) {
    return { v1.x * a, v1.y * a, v1.z * a };
}
__host__ __device__ float3 inline operator^(const float3& v1, const float3& v2) {
    return { v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x };
}
__host__ __device__ float3 inline operator|(const float3& v1, const float3& v2) {
    return { v1.x * v2.x, v1.y * v2.y, v1.z * v2.z };
}
__host__ __device__ float inline norm(const float3& v) {
    #ifdef __CUDA_ARCH__
        return norm3df(v.x, v.y, v.z);
    #else
        return sqrt(v * v);
    #endif
}
__host__ __device__ float3 inline normalize(const float3& v) {
    return v * (1.0 / norm(v));
}

__host__ __device__ float inline clamp(float x, float a, float b)
{
    return max(a, min(b, x));
}

__host__ __device__ int inline clamp(int x, int a, int b)
{
    return max(a, min(b, x));
}

// transformations