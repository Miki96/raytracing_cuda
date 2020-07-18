#pragma once
#include "vector_functions.h"
#include "structs.h"
using namespace std;

// transformations
float3 trans(float3 vec, float3* matrix) {
    float3 res = { 0, 0, 0 };
    res.x = matrix[0] * vec;
    res.y = matrix[1] * vec;
    res.z = matrix[2] * vec;
    return res;
}

float3 rotY(float3 vec, float a) {
    float3 matrix[3] = {
        {cos(a), 0, sin(a)},
        {0, 1, 0},
        {-sin(a), 0, cos(a)},
    };
    return trans(vec, matrix);
}

float3 rotX(float3 vec, float a) {
    float3 matrix[3] = {
        {1, 0, 0},
        {0, cos(a), -sin(a)},
        {0, sin(a), cos(a)},
    };
    return trans(vec, matrix);
}

float3 rotZ(float3 vec, float a) {
    float3 matrix[3] = {
        {cos(a), -sin(a), 0},
        {sin(a), cos(a), 0},
        {0, 0, 1},
    };
    return trans(vec, matrix);
}
