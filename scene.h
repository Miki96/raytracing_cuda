#pragma once
#include <vector_types.h>
#include <structs.h>

#define PI 3.141592

// time sync
extern float deltaTime;

// constants
const int OBJECTS_NUMBER = 131 + 2;
const int LIGHTS_NUMBER = 3;
const int BLOCK_SIZE = 32;

// functions
void initScene();
void animate();
void mouseMotion(int x, int y, int windowWidth, int windowHeight);
void launch(unsigned int* out_data, int imgw, int imgh);
