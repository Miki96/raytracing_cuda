/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Utilities and system includes

#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <structs.h>
#include "vector_functions.h"
using namespace std;
typedef unsigned char GLubyte;

const int SPHERE_NUMBER = 100;
__constant__ sphere positionsGPU[SPHERE_NUMBER];

//float min(float a, float b);
//float max(float a, float b);
//#define 	max(a, b)   ((a) > (b) ? (a) : (b))
//#define 	min(a, b)   ((a) < (b) ? (a) : (b))
//#define 	make_uchar4(a, b, c, d)   {a,b,c,d}

// HELPER FUNCTIONS
// clamp x to range [a, b]
__device__ float inline clamp(float x, float a, float b)
{
    return max(a, min(b, x));
}

__device__ int inline clamp(int x, int a, int b)
{
    return max(a, min(b, x));
}

// convert floating point rgb color to 8-bit integer
__device__ int inline rgbToInt(float r, float g, float b)
{
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);
    return (int(b)<<16) | (int(g)<<8) | int(r);
}

// convert 8-bit integer to floating point rgb color
__device__ void inline intToRgb(int c, int &r, int &g, int &b)
{
    b = (c >> 16) & 255;
    g = (c >> 8) & 255;
    r = c & 255;
}

// vector operations
__device__ float3 inline operator+(const float3& v1,const float3& v2) {
    return { v1.x + v2.x, v1.y + v2.y, v1.z + v2.z };
}
__device__ float3 inline operator-(const float3& v1, const float3& v2) {
    return { v1.x - v2.x, v1.y - v2.y, v1.z - v2.z };
}
__device__ float inline operator*(const float3& v1, const float3& v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}
__device__ float3 inline operator*(const float3& v1, const float& a) {
    return { v1.x * a, v1.y * a, v1.z * a };
}
__device__ float inline norm(const float3& v) {
    //return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return norm3df(v.x, v.y, v.z);
    //return exp10f(3);
}
//__device__ float inline norm2(const float3& v) {
//    return v.x * v.x + v.y * v.y + v.z * v.z;
//}
__device__ float3 inline normalize(const float3& v) {
    return v * (1.0 / norm(v));
}

__device__ bool inline checkHitSphere(const float3& ray, const float3& sphere, float3& normal, float& rayDist) {
    float sr = 0.5;
    float sr2 = sr * sr;

    // calc if hit
    float image = (sphere * ray);
    float3 d = ray * image;
    float dist = norm(d - sphere);
    // hit pos
    float tdist = sqrtf(sr2 - dist * dist);
    float3 t = ray * (image - tdist);
    // normal
    normal = normalize(t - sphere);

    // save distance from ray
    rayDist = norm(t);
    
    // hit
    return dist < sr;
}

__device__ bool inline checkShadow(const float3& origin, const float3& ray, const float3& sphere) {
    float sr = 0.5;
    float sr2 = sr * sr;

    // calc if hit
    float3 L = sphere - origin;
    float image = (L * ray);
    if (image < 0) return false;

    float d = L * L - image * image;
    if (d > sr2 || d < 0) return false;

    // hit
    return true;
}


// GPU
__global__ void
cudaProcess(unsigned int *g_odata, int imgw, int imgh, float3 pos, float3 sun, sphere *positions, int n)
{
    positions = positionsGPU;
    // get location
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    // coordinates
    int x = blockIdx.x*bw + tx;
    int y = blockIdx.y*bh + ty;

    if (x >= imgw || y >= imgh) return;
    float aspect = (1.0 * imgw) / imgh;

    // hit spheres
    sun = normalize(sun);
    float sr = 0.5;
    float sr2 = sr * sr;

    float3 mnormal;
    float mdist = -1;
    float3 ray = normalize({ 
        ((float)x / (float)(imgw - 1) - 0.5) * 2.0 * aspect,
        -((float)y / (float)(imgh - 1) - 0.5) * 2.0,
        -1 
    });

    // spheres
    bool hit = false;
    float3 normal;
    float rayDist;
    int selected = -1;
    for (int i = 0; i < n; i++) {
        pos = { positions[i].x, positions[i].y, positions[i].z };
        if (checkHitSphere(ray, pos, normal, rayDist)) {
            if (mdist == -1 || mdist > rayDist) {
                mdist = rayDist;
                mnormal = normal;
                selected = i;
            }
            hit = true;
        }
    }

    // plane
    bool planeHit = false;
    float3 planeNormal = { 0, 1, 0 };
    float3 planeOffset = { 0, -1, 0 };
    float denom = ray * planeNormal;
    float t;
    if (denom * denom > 0.00001) {
        t = (planeOffset * planeNormal) / denom;
        if (t > 0 && (mdist == -1 || t < mdist)) {
            mnormal = planeNormal;
            hit = true;
            planeHit = true;
            mdist = t;
        }
    }

    // draw shadow on plane
    bool shadow = false;
    if (planeHit) {
        float3 p = ray * (mdist);
        for (int i = 0; i < n; i++) {
            pos = { positions[i].x, positions[i].y, positions[i].z };
            if (checkShadow(p, sun * -1, pos)) {
                shadow = true;
                break;
            }
        }
    }

    // draw shadow on ball
    bool shadowBall = false;
    if (!planeHit && hit) {
        float3 p = ray * (mdist);
        for (int i = 0; i < n; i++) {
            pos = { positions[i].x, positions[i].y, positions[i].z };
            if (checkShadow(p, sun * -1, pos)) {
                shadowBall = true;
                break;
            }
        }
    }

    float3 color = { 255, 255, 255 };
    if (selected != -1 && !planeHit) {
        color = {
            positions[selected].r,
            positions[selected].g,
            positions[selected].b
        };
    }

    // light
    uchar4 c4;
    if (hit && !shadow && !shadowBall) {
        float part = max(0.0, -(mnormal * sun));
        unsigned char c = part * 255;
        c4 = { color.x * part, color.y * part, color.z * part, 1 };
    }
    /*else if (shadow) {
        c4 = { 30, 30, 30, 0 };
    }
    else if (hit && shadowBall) {
        c4 = { 30, 30, 30, 0 };
    }*/
    else {
        c4 = { 0, 0, 0, 0 };
    }
    g_odata[y*imgw+x] = rgbToInt(c4.x, c4.y, c4.z);

    //g_odata[y * imgw + x] = rgbToInt(244, 100, 40);

    /*
    //if (hit) {
    //}
    //else {
    //    // back
    //    c4 = { 130, 130, 130, 0 };
    //}

    if (!hit && (
           y == imgw * 1 / 4
        || y == imgw * 3 / 4
        || y == imgw * 1 / 2
        || x == imgh * 1 / 4
        || x == imgh * 3 / 4
        || x == imgh * 1 / 2)) {
        c4 = { 0, 0, 0, 0 };
    }*/
}

// AntiAliasing
__global__ void
cudaAlias(unsigned int* g_odata, int imgw, int imgh)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;

    int x = blockIdx.x * bw + tx;
    int y = blockIdx.y * bh + ty;

    if (x >= imgw || y >= imgh) return;

    //__shared__ unsigned int blockData[32][32];
    // read
    //blockData[tx][ty] = g_odata[y * imgw + x];
    //__syncthreads();

    // calc

    int rgb[9][3];
    float r = 0, g = 0, b = 0;
    if (x + 1 < imgw && y + 1 < imgh && x > 0 && y > 0 && tx != 0 && ty != 0 && tx != bw-1 && ty != bh-1) {
        
        /*intToRgb(blockData[tx][ty], rgb[0][0], rgb[0][1], rgb[0][2]);
        intToRgb(blockData[tx][ty+1], rgb[1][0], rgb[1][1], rgb[1][2]);
        intToRgb(blockData[tx][ty-1], rgb[2][0], rgb[2][1], rgb[2][2]);
        intToRgb(blockData[tx+1][ty], rgb[3][0], rgb[3][1], rgb[3][2]);
        intToRgb(blockData[tx+1][ty+1], rgb[4][0], rgb[4][1], rgb[4][2]);
        intToRgb(blockData[tx+1][ty-1], rgb[5][0], rgb[5][1], rgb[5][2]);
        intToRgb(blockData[tx-1][ty], rgb[6][0], rgb[6][1], rgb[6][2]);
        intToRgb(blockData[tx-1][ty+1], rgb[7][0], rgb[7][1], rgb[7][2]);
        intToRgb(blockData[tx-1][ty-1], rgb[8][0], rgb[8][1], rgb[8][2]);*/

        intToRgb(g_odata[(y + 0) * imgw + x - 1], rgb[0][0], rgb[0][1], rgb[0][2]);
        intToRgb(g_odata[(y + 0) * imgw + x + 0], rgb[1][0], rgb[1][1], rgb[1][2]);
        intToRgb(g_odata[(y + 0) * imgw + x + 1], rgb[2][0], rgb[2][1], rgb[2][2]);
        intToRgb(g_odata[(y - 1) * imgw + x - 1], rgb[3][0], rgb[3][1], rgb[3][2]);
        intToRgb(g_odata[(y - 1) * imgw + x + 0], rgb[4][0], rgb[4][1], rgb[4][2]);
        intToRgb(g_odata[(y - 1) * imgw + x + 1], rgb[5][0], rgb[5][1], rgb[5][2]);
        intToRgb(g_odata[(y + 1) * imgw + x - 1], rgb[6][0], rgb[6][1], rgb[6][2]);
        intToRgb(g_odata[(y + 1) * imgw + x + 0], rgb[7][0], rgb[7][1], rgb[7][2]);
        intToRgb(g_odata[(y + 1) * imgw + x + 1], rgb[8][0], rgb[8][1], rgb[8][2]);

        float k = 9;

        for (int i = 0; i < k; i++) {
            r += rgb[i][0];
            g += rgb[i][1];
            b += rgb[i][2];
        }
        r /= k;
        g /= k;
        b /= k;

        g_odata[y * imgw + x] = rgbToInt(r, g, b);
    }

    //intToRgb(g_odata[(y) * imgw + x], r, g, b);
    //g_odata[y * imgw + x] = rgbToInt(r, g, b);
    
}


// Launcher
extern "C" void
launch_cudaProcess(dim3 grid, dim3 block, int sbytes,
                   unsigned int *g_odata,
                   int imgw, int imgh, float3 pos, float3 sun, sphere *positions, int n)
{

    //cudaMemcpyToSymbol(coeffs1, cff1, 8 * sizeof(float));
    //cout << "test" << positions[12].g << endl;
    getLastCudaError("ERROR TEST\n");

    cudaMemcpyToSymbol(positionsGPU, positions, sizeof(sphere) * SPHERE_NUMBER);

    getLastCudaError("ERROR COPy\n");

    // raytracing

    cudaProcess<<< grid, block >>>(g_odata, imgw, imgh, pos, sun, positionsGPU, n);

    //getLastCudaError("ERROR FIRST\n");

    //cudaAlias<<< grid, block >>> (g_odata, imgw, imgh); 

    // mandelbrot
    //cudaMandel << < grid, block >> > (g_odata, imgw, imgh, offsetX, offsetY, zoom);

    getLastCudaError("ERROR MIKi\n");
    //cpuCalc(imgw, imgh, g_odata, pos, sun);
}





















