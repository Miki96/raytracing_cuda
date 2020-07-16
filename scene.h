#pragma once
#include <vector_types.h>
#include <structs.h>

// time sync
extern float deltaTime;

// constants
const int OBJECTS_NUMBER = 6;
const int LIGHTS_NUMBER = 2;

// functions
void initScene();
void animate();
void mouseMotion(int x, int y, int windowWidth, int windowHeight);
void launch(dim3 grid, dim3 block, unsigned int* out_data, int imgw, int imgh);
