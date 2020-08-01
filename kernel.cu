#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "vector_functions.h"
#include <scene.h>
#include <structs.h>
using namespace std;
typedef unsigned char GLubyte;

#define MAX_DEPTH 4
#define PI 3.141592

__constant__ Object objectsGPU[OBJECTS_NUMBER];
__constant__ Light lightsGPU[LIGHTS_NUMBER];
__constant__ float skyVars[4];
__constant__ float dayTime;
__constant__ float3 ambientColor;
texture<uchar4, 2, cudaReadModeElementType> tex1;
texture<uchar4, 2, cudaReadModeElementType> tex2;
texture<uchar4, 2, cudaReadModeElementType> tex3;
texture<uchar4, 2, cudaReadModeElementType> tex4;
bool init = false;
__device__ unsigned int* baseImage; // helper buffer

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
        if (tca <= 0) return false;
        float d2 = L * L - tca * tca;
        // ray missed
        if (d2 >= sr2 || d2 <= -0.01) return false;
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
        hitNormal = normalize(v0v1^v0v2);

        return true;
    }

    return false;
}

template<int depth>
__device__ float3 inline trace(const Ray& ray) {

    float minHitDist = -1;
    float3 minHitPos;
    float3 minHitNormal;
    int index = -1;

    float3 hitPos;
    float3 hitNormal;
    float hitDist;

    // find closest hit
    for (int i = 0; i < OBJECTS_NUMBER; i++) {
        if (checkHit(ray, i, hitPos, hitNormal, hitDist) && (hitDist < minHitDist || minHitDist == -1)) {
            minHitDist = hitDist;
            minHitPos = hitPos;
            minHitNormal = hitNormal;
            index = i;
        }
    }

    // calculate pixel color
    if (index == -1) {
        // ray miss - sky
        float y = 1 - (asinf(ray.dir.y) + PI / 2.0f) / PI;
        float x = fmodf((atan2f(ray.dir.x, ray.dir.z) + PI) / (2.0f * PI) + dayTime, 1);
        uchar4 v = 
            tex2D(tex1, x, y) * skyVars[0] + 
            tex2D(tex2, x, y) * skyVars[1] +
            tex2D(tex3, x, y) * skyVars[2] +
            tex2D(tex4, x, y) * skyVars[3];
        return float3{ v.x, v.y, v.z } * (1.0f / 255.0f);
    }
    else {
        // ray hit - object
        Object& o = objectsGPU[index];

        if (o.light) return o.color;

        float ambient = 0.2;
        float3 phongColor = o.color | ambientColor;

        // calcuate phong color
        for (int i = 0; i < LIGHTS_NUMBER; i++) {
            Light l = lightsGPU[i];

            // direct color
            float3 vec = l.pos - minHitPos;
            float shadowDist = norm(vec);
            float3 shadowDir = normalize(vec);
            Ray shadow = { minHitPos, shadowDir };
            shadow.origin = shadow.origin + shadow.dir * 0.001;

            float angle = max(0.0, (minHitNormal * shadowDir));

            // check if in shadow
            for (int k = 0; k < OBJECTS_NUMBER; k++) {
                if (!objectsGPU[k].light && checkHit(shadow, k, hitPos, hitNormal, hitDist) && hitDist < shadowDist) {
                    angle = 0;
                    break;
                }
            }

            phongColor = phongColor + (o.color | l.color) * ((angle * l.intensity) / 1);

            // specular
            float3 specColor = { 0, 0, 0 };
            if (o.shine > 0) {
                float3 lightDir = shadowDir * -1;
                float3 specDir = normalize(lightDir - 2 * (minHitNormal * lightDir) * minHitNormal);
                specColor = __powf(max(0.0, -(specDir * ray.dir)), o.specular) * float3{1, 1, 1} * o.shine * angle; // 256
            }

            phongColor = phongColor + specColor;
        }

        // calculate reflective color
        float3 refColor = { 0, 0, 0 };
        float kR = o.mirror;
        if (kR > 0) {
            Ray reflection = { minHitPos, normalize(ray.dir - 2 * (minHitNormal * ray.dir) * minHitNormal) };
            reflection.origin = reflection.origin + reflection.dir * 0.001;
            refColor = trace<depth + 1>(reflection);
        }

        // result
        return (refColor * kR + phongColor * (1 - kR));
    }
}

template<>
__device__ float3 inline trace<MAX_DEPTH + 1>(const Ray& ray) {
    return { 0, 0, 0 };
}

// Raytracing
__global__ void
raytracing(unsigned int* image, int imgw, int imgh, Camera cam)
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
    float3 c = trace<0>(ray) * 255;
    image[y * imgw + x] = rgbToInt(c.x, c.y, c.z);
    return;
}

// Antialiasing FXAA
__global__ void
antialiasing(unsigned int* image, unsigned int* result, int imgw, int imgh, bool alias)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;

    int x = blockIdx.x * bw + tx;
    int y = blockIdx.y * bh + ty;

    if (x >= imgw || y >= imgh) return;

    if (!alias) {
        result[y * imgw + x] = image[y * imgw + x];
        return;
    }

    // show egdes
    /*if (tx == 0 || ty == 0 || tx == bw-1 || ty == bh-1) {
        result[y * imgw + x] = rgbToInt(250, 0, 0);
        return;
    }*/

    __shared__ float block[32 + 2][32 + 2];

    // threshhold
    float contrastThreshold = 0.0312;
    float relativeThreshold = 0.063;

    // calculate luminance
    float c1 = 0.2126729f, c2 = 0.7151522f, c3 = 0.0721750f;
    int3 rgb;

    // calc center
    intToRgb(image[(y + 0) * imgw + x + 0], rgb.x, rgb.y, rgb.z);
    block[ty + 1][tx + 1] = min(255.0f, rgb.x * c1 + rgb.y * c2 + rgb.z * c3) / 255.0f;

    // calc border
    int d[8][2] = {
        {-1, 0}, // U
        {1, 0}, // D
        {0, -1}, // L
        {0, 1}, // R
        {-1, -1}, // UL
        {-1, 1}, // UR
        {1, -1}, // DL
        {1, 1}, // DR
    };

    int ix, iy, bx, by;
    for (int i = 0; i < 8; i++) {
        iy = y + d[i][0];
        ix = x + d[i][1];
        by = ty + d[i][0];
        bx = tx + d[i][1];
        if (iy >= 0 && iy < imgh && ix >= 0 && ix < imgw &&
           (by == -1 || by == bh) || (bx == -1 || bx == bw)) {
            // calc lum
            intToRgb(image[(iy) * imgw + ix], rgb.x, rgb.y, rgb.z);
            block[by + 1][bx + 1] = min(255.0f, rgb.x * c1 + rgb.y * c2 + rgb.z * c3) / 255.0f;
        }
    }

    // sync all
    __syncthreads();

    // calculate contrast
    if (x + 1 < imgw && y + 1 < imgh && x > 0 && y > 0) {

        int3 pInt;
        intToRgb(image[y * imgw + x], pInt.x, pInt.y, pInt.z);
        float3 p = { pInt.x, pInt.y, pInt.z, };

        // calc constrast
        float lumE = block[ty + 0 + 1][tx + 1 + 1];
        float lumW = block[ty + 0 + 1][tx - 1 + 1];
        float lumN = block[ty - 1 + 1][tx + 0 + 1];
        float lumS = block[ty + 1 + 1][tx + 0 + 1];
        float lumM = block[ty + 0 + 1][tx + 0 + 1];

        float high = max(max(max(max(lumE, lumW), lumN), lumS), lumM);
        float low = min(min(min(min(lumE, lumW), lumN), lumS), lumM);

        float contrast = high - low;

        // check if should skip
        float threshold = max(contrastThreshold, relativeThreshold * high);
        if (contrast < threshold) {
            result[y * imgw + x] = image[y * imgw + x];
            //result[y * imgw + x] = rgbToInt(0, 0, 0);
            return;
        }

        // COOL EFFECT
        //contrast *= 255;
        //contrast *= contrast;
        //if (contrast > 255) contrast = 255;
        // write
        //result[y * imgw + x] = rgbToInt(contrast, contrast, contrast);

        // calc blend
        float lumNE = block[ty - 1 + 1][tx + 1 + 1];
        float lumNW = block[ty - 1 + 1][tx - 1 + 1];
        float lumSE = block[ty + 1 + 1][tx + 1 + 1];
        float lumSW = block[ty + 1 + 1][tx - 1 + 1];

        float filter = (2 * (lumE + lumW + lumS + lumN) + lumNE + lumNW + lumSE + lumSW) / 12.0;
        filter = min(1.0f, abs(filter - lumM) / contrast);
        // smoothstep
        filter = filter * filter * (3.0f - 2.0f * filter);
        //filter *= filter;
        float b = filter;

        // calc egde
        float hor = abs(lumN + lumS - 2 * lumM) * 2 +
            abs(lumNE + lumSE - 2 * lumE) +
            abs(lumNW + lumSW - 2 * lumW);
        float ver = abs(lumE + lumW - 2 * lumM) * 2 +
            abs(lumNE + lumNW - 2 * lumN) +
            abs(lumSE + lumSW - 2 * lumS);

        // show result
        int dy = 0;
        int dx = 0;
        if (hor >= ver) {
            dy = abs(lumN - lumM) >= abs(lumS - lumM) ? -1 : 1;
        }
        else {
            dx = abs(lumE - lumM) >= abs(lumW - lumM) ? 1 : -1;
        }
        int3 sInt;
        intToRgb(image[(y + dy) * imgw + (x + dx)], sInt.x, sInt.y, sInt.z);
        float3 s = { sInt.x * b + p.x * (1 - b), sInt.y * b + p.y * (1 - b), sInt.z * b + p.z * (1 - b) };
        result[y * imgw + x] = rgbToInt(s.x, s.y, s.z);
        return;
    }
    else {
        // image border
        result[y * imgw + x] = image[y * imgw + x];
    }
}

// Main Kernel
extern "C" void
launchKernel(
                   unsigned int *result,
                   int imgw, int imgh, Camera cam, Object *objects, Light * lights, float3 ambient, float* h_skyVars,
                   unsigned char* h_tex1, unsigned char* h_tex2, unsigned char* h_tex3, unsigned char* h_tex4, int w, int h,
                   float h_dayTime, bool alias)
{
    // texture
    if (!init) 
    {
        init = true;

        int mem_size = sizeof(uchar4) * w * h;
        cudaArray* d_array1, * d_array2, * d_array3, * d_array4;

        cudaMallocArray(&d_array1, &tex1.channelDesc, w, h);
        cudaMallocArray(&d_array2, &tex2.channelDesc, w, h);
        cudaMallocArray(&d_array3, &tex3.channelDesc, w, h);
        cudaMallocArray(&d_array4, &tex4.channelDesc, w, h);
        cudaMemcpyToArray(d_array1, 0, 0, h_tex1, mem_size, cudaMemcpyHostToDevice);
        cudaMemcpyToArray(d_array2, 0, 0, h_tex2, mem_size, cudaMemcpyHostToDevice);
        cudaMemcpyToArray(d_array3, 0, 0, h_tex3, mem_size, cudaMemcpyHostToDevice);
        cudaMemcpyToArray(d_array4, 0, 0, h_tex4, mem_size, cudaMemcpyHostToDevice);
        tex1.normalized = true;
        tex2.normalized = true;
        tex3.normalized = true;
        tex4.normalized = true;
        cudaBindTextureToArray(tex1, d_array1, tex1.channelDesc);
        cudaBindTextureToArray(tex2, d_array2, tex2.channelDesc);
        cudaBindTextureToArray(tex3, d_array3, tex3.channelDesc);
        cudaBindTextureToArray(tex4, d_array4, tex4.channelDesc);

        getLastCudaError("Error texture\n");

        // helper for antialiasing
        cudaMalloc(&baseImage, w * h * sizeof(unsigned int));
    }

    //cudaFuncSetCacheConfig(raytracing, cudaFuncCachePreferL1);

    // constants
    cudaMemcpyToSymbol(skyVars, h_skyVars, sizeof(float) * 4);
    cudaMemcpyToSymbol(objectsGPU, objects, sizeof(Object) * OBJECTS_NUMBER);
    cudaMemcpyToSymbol(lightsGPU, lights, sizeof(Light) * LIGHTS_NUMBER);
    cudaMemcpyToSymbol(ambientColor, &ambient, sizeof(float3));
    cudaMemcpyToSymbol(dayTime, &h_dayTime, sizeof(float));
    getLastCudaError("Error constants\n");

    // grid
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid((imgw + block.x - 1) / block.x, (imgh + block.y - 1) / block.y, 1);
    // raytracing
    raytracing<<< grid, block >>>(baseImage, imgw, imgh, cam);
    getLastCudaError("Error raytracing\n");
    antialiasing<<< grid, block >>> (baseImage, result, imgw, imgh, alias); 
    getLastCudaError("Error antialiasing\n");
}

