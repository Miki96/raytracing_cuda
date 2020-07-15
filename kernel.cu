#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <structs.h>
#include "vector_functions.h"
#include <scene.h>
using namespace std;
typedef unsigned char GLubyte;

#define MAX_DEPTH 3
#define PI 3.141592

__constant__ Object objectsGPU[OBJECTS_NUMBER];
texture<uchar4, 2, cudaReadModeElementType> tex;
bool init = false;

__device__ int inline rgbToInt(float r, float g, float b)
{
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);
    return (int(b)<<16) | (int(g)<<8) | int(r);
}

__device__ void inline intToRgb(int c, int &r, int &g, int &b)
{
    b = (c >> 16) & 255;
    g = (c >> 8) & 255;
    r = c & 255;
}


__device__ bool inline checkHit(const Ray& ray, int index, float3& hitPos, float3& hitNormal, float& hitDist) {

    Object& object = objectsGPU[index];

    switch (object.type)
    {
    case Primitive::SPHERE:

        float sr = object.size.x;
        float sr2 = sr * sr;

        float3 L = object.pos - ray.origin;
        float tca = L * ray.dir;
        // wrong side of ray
        if (tca < 0) return false;
        float d2 = L * L - tca * tca;
        // ray missed
        if (d2 > sr2 || d2 < 0) return false;
        float thc = sqrtf(sr2 - d2);

        // hit distance
        hitDist = tca - thc;
        // hit position
        hitPos = ray.origin + ray.dir * hitDist;
        // hit normal
        hitNormal = normalize(hitPos - object.pos);

        return true;
        break;

    case Primitive::PLANE:
        bool planeHit = false;
        float3 planeNormal = object.size;
        float3 planeOffset = object.pos;
        float denom = ray.dir * planeNormal;
        float t;
        if (denom * denom > 0.00001) {
            t = ((planeOffset - ray.origin) * planeNormal) / denom;

            if (t < 0) return false;

            // hit distance
            hitDist = t;
            // hit position
            hitPos = ray.origin + ray.dir * t;
            // hit normal
            hitNormal = planeNormal;

            return true;
        }
        else {
            return false;
        }
        break;
    case Primitive::TRIANGLE:
        float3 v0 = object.pos;
        float3 v1 = object.size;
        float3 v2 = object.third;
        // NEW
        float3 v0v1 = v1 - v0;
        float3 v0v2 = v2 - v0;
        float3 pvec = ray.dir^v0v2;
        float det = v0v1*pvec;
        if (det < 0.001) return false;
        float invDet = 1 / det;

        float3 tvec = ray.origin - v0;
        float u = (tvec*pvec) * invDet;
        if (u < 0 || u > 1) return false;

        float3 qvec = tvec^(v0v1);
        float v = (ray.dir*qvec) * invDet;
        if (v < 0 || u + v > 1) return false;

        t = (v0v2*qvec) * invDet;
        if (t < 0) return false;

        // hit distance
        hitDist = t;
        // hit position
        hitPos = ray.origin + ray.dir * t;
        // hit normal
        hitNormal = v0v1^v0v2;

        return true;
    }

    return false;
}

template<int depth>
__device__ float3 inline trace(const Ray& ray, const float3& sun, int n) {

    float minHitDist = -1;
    float3 minHitPos;
    float3 minHitNormal;
    int index = -1;

    float3 hitPos;
    float3 hitNormal;
    float hitDist;

    // find closest hit
    for (int i = 0; i < n; i++) {
        if (checkHit(ray, i, hitPos, hitNormal, hitDist) && (hitDist < minHitDist || minHitDist == -1)) {
            minHitDist = hitDist;
            minHitPos = hitPos;
            minHitNormal = hitNormal;
            index = i;
        }
    }

    // color
    if (index == -1/* || index == 5*/) {

        // sky texture
        float y = 1 - (asinf(ray.dir.y) + PI / 2.0f) / PI;
        float x = (atan2f(ray.dir.x, ray.dir.z) + PI) / (2.0f * PI);

        uchar4 v = tex2D(tex, x, y);
        return { v.x, v.y, v.z };
    }
    else if ((index == 5 || index == 2) && depth < MAX_DEPTH) {
        // MIRROR
        // diffuse color
        Ray shadow = { minHitPos, sun * -1 };
        shadow.origin = shadow.origin + shadow.dir * 0.001;
        float part = -1;
        // check if in shadow
        for (int i = 0; i < n; i++) {
            if (checkHit(shadow, i, hitPos, hitNormal, hitDist)) {
                //return { 0, 0, 0 };
                part = 0;
                break;
            }
        }
        if (part == -1) {
            // not in shadow
            part = max(0.0, -(minHitNormal * sun));
        }

        // glass color
        Ray reflection = { minHitPos, normalize(ray.dir - 2 * (minHitNormal * ray.dir) * minHitNormal) };
        reflection.origin = reflection.origin + reflection.dir * 0.001;
        float3 refColor = trace<depth + 1>(reflection, sun, n);
        float kR = 0.7;
        return refColor * kR + objectsGPU[index].color * part * (1 - kR);
    }
    else {
        // diffuse object
        Ray shadow = { minHitPos, sun * -1 };
        shadow.origin = shadow.origin + shadow.dir * 0.0001;

        float part = -1;

        // check if in shadow
        for (int i = 0; i < n; i++) {
            if (checkHit(shadow, i, hitPos, hitNormal, hitDist)) {
                //return { 0, 0, 0 };
                part = 0;
                break;
            }
        }

        if (part == -1) {
            // not in shadow
            part = max(0.0, -(minHitNormal * sun));
        }

        // specular
        float3 specDir = normalize(sun - 2 * (minHitNormal * sun) * minHitNormal);
        float partSpec = __powf(max(0.0, -(specDir * ray.dir)), 256);
        //partSpec = 0;

        // ambient light
        part += 0.22;
        float3 white = { 255, 255, 255 };
        return objectsGPU[index].color * part + white * partSpec;
    }
}

template<>
__device__ float3 inline trace<MAX_DEPTH + 1>(const Ray& ray, const float3& sun, int n) {
    return { 0, 0, 0 };
}

// GPU
__global__ void
raytracing(unsigned int* image, int imgw, int imgh, Camera cam, float3 sun, int n)
{
    // get location
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    // coordinates
    int x = blockIdx.x * bw + tx;
    int y = blockIdx.y * bh + ty;
    // exit if out of image
    if (x >= imgw || y >= imgh) return;
    float aspect = (1.0 * imgw) / imgh;

    //// show image
    //if (x >= 1436 || y >= 357) {
    //    image[y * imgw + x] = rgbToInt(220, 0, 0);
    //}
    //else {
    //    int i = (y * 1436 + x) * 3;
    //    image[y * imgw + x] = rgbToInt(tex[i], tex[i+1], tex[i+2]);
    //}
    //return;

    // position sun
    sun = normalize(sun);

    // create camera ray
    float partX = (float)x / (float)(imgw - 1);
    float partY = (float)y / (float)(imgh - 1);
    float3 vd = cam.LD + (cam.RD - cam.LD) * partX;
    float3 vu = cam.LU + (cam.RU - cam.LU) * partX;
    float3 target = vu - (vu - vd) * partY;

    Ray ray = {
        cam.pos,
        normalize(target),
    };

    // color pixel
    float3 c = trace<0>(ray, sun, n);
    image[y * imgw + x] = rgbToInt(c.x, c.y, c.z);
    return;
}

// AntiAliasing
__global__ void
antialiasing(unsigned int* image, int imgw, int imgh)
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

        intToRgb(image[(y + 0) * imgw + x - 1], rgb[0][0], rgb[0][1], rgb[0][2]);
        intToRgb(image[(y + 0) * imgw + x + 0], rgb[1][0], rgb[1][1], rgb[1][2]);
        intToRgb(image[(y + 0) * imgw + x + 1], rgb[2][0], rgb[2][1], rgb[2][2]);
        intToRgb(image[(y - 1) * imgw + x - 1], rgb[3][0], rgb[3][1], rgb[3][2]);
        intToRgb(image[(y - 1) * imgw + x + 0], rgb[4][0], rgb[4][1], rgb[4][2]);
        intToRgb(image[(y - 1) * imgw + x + 1], rgb[5][0], rgb[5][1], rgb[5][2]);
        intToRgb(image[(y + 1) * imgw + x - 1], rgb[6][0], rgb[6][1], rgb[6][2]);
        intToRgb(image[(y + 1) * imgw + x + 0], rgb[7][0], rgb[7][1], rgb[7][2]);
        intToRgb(image[(y + 1) * imgw + x + 1], rgb[8][0], rgb[8][1], rgb[8][2]);

        float k = 9;

        for (int i = 0; i < k; i++) {
            r += rgb[i][0];
            g += rgb[i][1];
            b += rgb[i][2];
        }
        r /= k;
        g /= k;
        b /= k;

        image[y * imgw + x] = rgbToInt(r, g, b);
    }

    //intToRgb(g_odata[(y) * imgw + x], r, g, b);
    //g_odata[y * imgw + x] = rgbToInt(r, g, b);
}

// launcher
extern "C" void
launch_cudaProcess(dim3 grid, dim3 block, int sbytes,
                   unsigned int *g_odata,
                   int imgw, int imgh, Camera cam, float3 sun, Object *objects, int n, unsigned char* h_tex, int w, int h)
{
    if (!init) {
        init = true;

        // 2D
        int mem_size = sizeof(uchar4) * w * h;
        cudaArray *d_array;
        cudaMallocArray(&d_array, &tex.channelDesc, w, h);
        cudaMemcpyToArray(d_array, 0, 0, h_tex, mem_size, cudaMemcpyHostToDevice);

        // Set texture parameters
        tex.normalized = true;
        cudaBindTextureToArray(tex, d_array, tex.channelDesc);
    }

    //cudaFuncSetCacheConfig(raytracing, cudaFuncCachePreferL1);
    getLastCudaError("ERROR TEST\n");

    cudaMemcpyToSymbol(objectsGPU, objects, sizeof(Object) * OBJECTS_NUMBER);

    getLastCudaError("ERROR COPy\n");

    // raytracing

    raytracing<<< grid, block >>>(g_odata, imgw, imgh, cam, sun, n);

    getLastCudaError("ERROR FIRST\n");

    antialiasing<<< grid, block >>> (g_odata, imgw, imgh); 

    getLastCudaError("ERROR MIKi\n");
}

