#pragma once
#include <cmath>
#include <stdlib.h>
#include <algorithm>
using namespace std;

// objects
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
    bool light;
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

__host__ __device__ uchar4 inline operator*(const uchar4& v1, const float& a) {
    return { (unsigned char)(v1.x * a), (unsigned char)(v1.y * a), (unsigned char)(v1.z * a), (unsigned char)(v1.w * a) };
}
__host__ __device__ uchar4 inline operator+(const uchar4& v1, const uchar4& v2) {
    return { (unsigned char)(v1.x + v2.x), (unsigned char)(v1.y + v2.y), (unsigned char)(v1.z + v2.z), (unsigned char)(v1.w + v2.w) };
}

__host__ __device__ float inline clamp(float x, float a, float b)
{
    return max(a, min(b, x));
}

__host__ __device__ int inline clamp(int x, int a, int b)
{
    return max(a, min(b, x));
}


