#include <scene.h>
#include <windows.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <time.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define PI 3.141592

float3 sunPos = { -1, -1, -1 };

Camera cam;
float moveSpeed = 0.3f;
float camViewDelta = 0.02;
float camViewLimit = 44;
int lastMouseX = -1;
int lastMouseY = -1;
float aspect = 1.7777f;
float runSpeedUp = 2;

// Globals
float alpha = 180;
float alphaDelta = 1;

Object* objects;

unsigned char* texture;
int texW;
int texH;

// declarations
extern "C" void
launch_cudaProcess(dim3 grid, dim3 block, int sbytes,
    unsigned int* g_odata,
    int imgw, int imgh,
    Camera cam, float3 sun, Object * positions, int n, unsigned char* tex, int w, int h);
void cameraHelperAngles();

void initCamera() {
    cam = {
        {0, 0, 0}, // position
        270, // horizontal angle
        0, // vertical angle
        40 // field of view
    };
    cam.horAngle = fmod(cam.horAngle + camViewDelta * 0 + 360.0f, 360.0f);
    cam.verAngle = clamp(cam.verAngle + camViewDelta * 0, -camViewLimit, camViewLimit);
    cameraHelperAngles();
}

// spin temp vars
float dist[] = { 0, 0, 0, 0, 0 };
float angle[] = { 110, 80, 90, 70, 60 };
float speed[] = { -0.015, 0.01, -0.02, 0.005, -0.012 };

void initObjects() {
    // RED
    {
        int i = 0;
        objects[i] = {
            Primitive::SPHERE,
            float3{-5, -2, -13},
            float3{0.91 * 255, 0.1 * 255, 0.0 * 255},
            float3{2, 2, 2}
        };
    }
    // GREEN
    {
        int i = 1;
        objects[i] = {
            Primitive::SPHERE,
            float3{2.5, -2.5, -12},
            float3{0 * 255, 1 * 255, 0.1 * 255},
            float3{1.5, 1.5, 1.5}
        };
    }
    // MIRROR
    {
        int i = 2;
        objects[i] = {
            Primitive::SPHERE,
            float3{0, 1, -20},
            float3{0, 0, 0},
            float3{5, 5, 5}
        };
    }
    // YELLOW
    {
        int i = 3;
        objects[i] = {
            Primitive::SPHERE,
            float3{15, -1, -40},
            float3{0.9 * 255, 0.9 * 255, 0.1 * 255},
            float3{3, 3, 3}
        };
    }
    // BLUE GLASS
    {
        int i = 4;
        objects[i] = {
            Primitive::SPHERE,
            float3{10, -2, -20},
            float3{0, 128, 255},
            float3{2, 2, 2}
        };
    }
    // FLOOR
    /*{
        int i = 5;
        objects[i] = {
            Primitive::SPHERE,
            float3{0, -10004, -20},
            float3{99, 93, 226},
            float3{10000, 10000, 10000}
        };
    }*/

    // PLANE
    {
        int i = 5;
        // type
        objects[i].type = Primitive::PLANE;
        // position
        objects[i].pos.x = 0;
        objects[i].pos.y = -4;
        objects[i].pos.z = 0;
        // color
        rgb c = hsv2rgb({ (double)(rand() % 360), 0.9, 0.9 });
        objects[i].color.x = 0;
        objects[i].color.y = 0;
        objects[i].color.z = 0;
        // normal
        objects[i].size.x = 0;
        objects[i].size.y = 1;
        objects[i].size.z = 0;
    }

    // calc distances
    for (int i = 0; i < 5; i++) {
        float3 v = objects[i].pos;
        v.y = 0;
        dist[i] = norm(v);
    }

    // add triangles
    {
        float y = 0.86f;
        float x = 0.5f;
        float h = y * 2.0f / 3.0f;
        float v = y * 1.0f / 3.0f;

        float3 tris[] = {
            // down
            {0, 0, 0}, // 0
            {1, 0, 0}, // 1
            {x, 0, y}, // 2
            // front
            {0, 0, 0}, // 0
            {x, 1, v}, // 3
            {1, 0, 0}, // 1
            // left
            {0, 0, 0}, // 0
            {x, 0, y}, // 2
            {x, 1, v}, // 3
            // right
            {1, 0, 0}, // 1
            {x, 1, v}, // 3
            {x, 0, y}, // 2
        };

        for (int i = 0; i < 4 * 3; i++) {
            tris[i].x += 3;
            tris[i].y -= 1;
        }

        for (int i = 0; i < 4 * 3; i++) {
            tris[i].x *= 2;
            tris[i].y *= 2;
            tris[i].z *= 2;
        }

        for (int i = 0; i < 4; i++) {
            objects[i + 6] = {
                Primitive::TRIANGLE,
                tris[i * 3],
                float3{0, 0, 200},
                tris[i * 3 + 1],
                tris[i * 3 + 2]
            };
        }
    }
}

void initTexture() {
    int n;
    texture = stbi_load("backgrounds/park.jpg", &texW, &texH, &n, 4);
    printf("%d %d %d\n", texW, texH, n);
}

void initScene() {
    // create spheres
    srand(time(NULL));

    objects = (Object*)malloc(sizeof(Object) * OBJECTS_NUMBER);

    initCamera();
    initObjects();
    initTexture();
}

void launch(dim3 grid, dim3 block, unsigned int* out_data, int imgw, int imgh) {
    aspect = (1.0f * imgw) / imgh;
    launch_cudaProcess(grid, block, 0, out_data, imgw, imgh, cam, sunPos, objects, OBJECTS_NUMBER, texture, texW, texH);
}

float3 transform(float3& vec, float3 *matrix) {
    float3 res = { 0, 0, 0 };
    res.x = matrix[0] * vec;
    res.y = matrix[1] * vec;
    res.z = matrix[2] * vec;
    return res;
}

float3 rotateY(float3& vec, float a) {
    float3 matrix[3] = {
        {cos(a), 0, sin(a)},
        {0, 1, 0},
        {-sin(a), 0, cos(a)},
    };
    return transform(vec, matrix);
}

float3 rotateX(float3& vec, float a) {
    float3 matrix[3] = {
        {1, 0, 0},
        {0, cos(a), -sin(a)},
        {0, sin(a), cos(a)},
    };
    return transform(vec, matrix);
}

float3 rotateZ(float3& vec, float a) {
    float3 matrix[3] = {
        {cos(a), -sin(a), 0},
        {sin(a), cos(a), 0},
        {0, 0, 1},
    };
    return transform(vec, matrix);
}


void cameraHelperAngles() {
    float rad = PI / 180.0f;

    float dirRad = rad * cam.horAngle;
    cam.dir = { cos(dirRad), 0, sin(dirRad) };

    float a = rad * cam.fov / 2.0;
    //float w = tan(a);
    //float h = w / aspect;
    float h = tan(a);
    float w = h * aspect;

    cam.LD = { 1, -h, -w };
    cam.RD = { 1, -h, w };
    cam.LU = { 1, h, -w };
    cam.RU = { 1, h, w };

    // rotate ver
    float av = rad * (-cam.verAngle);
    cam.LD = rotateZ(cam.LD, av);
    cam.RD = rotateZ(cam.RD, av);
    cam.LU = rotateZ(cam.LU, av);
    cam.RU = rotateZ(cam.RU, av);

    // rotate hor
    float ah = rad * (-cam.horAngle);
    cam.LD = rotateY(cam.LD, ah);
    cam.RD = rotateY(cam.RD, ah);
    cam.LU = rotateY(cam.LU, ah);
    cam.RU = rotateY(cam.RU, ah);
}

void mouseMotion(int x, int y, int windowWidth, int windowHeight) {
    if (lastMouseX != -1 && lastMouseX != -1) {
        int deltaX = x - lastMouseX;
        int deltaY = y - lastMouseY;

        // rotate camera
        cam.horAngle = fmod(cam.horAngle + camViewDelta * deltaX + 360.0f, 360.0f);
        cam.verAngle = clamp(cam.verAngle + camViewDelta * deltaY, -camViewLimit, camViewLimit);
        cameraHelperAngles();
    }
    lastMouseX = windowWidth / 2;
    lastMouseY = windowHeight / 2;
}

void moveCamera() {
    // move cam
    float3 camMove = { 0, 0, 0 };
    float3 camForw = cam.dir;
    float3 camSide = { -cam.dir.z, 0, cam.dir.x };

    int verMove = (bool)GetAsyncKeyState(0x44) - (bool)GetAsyncKeyState(0x41);
    camMove = camMove + camSide * verMove;
    int horMove = (bool)GetAsyncKeyState(0x57) - (bool)GetAsyncKeyState(0x53);
    camMove = camMove + camForw * horMove;
    // run
    float run = (bool)GetAsyncKeyState(VK_SHIFT) ? runSpeedUp : 1;

    if (verMove || horMove) {
        camMove = normalize(camMove);
        cam.pos = cam.pos + camMove * (moveSpeed * run);
    }
}


void animate() {
    // spin

    /*const float a = (3.14 / 180.0) * alpha;
    sunPos.x = sin(a);
    sunPos.z = cos(a);
    alpha = fmod(alpha + alphaDelta, 360.0);*/

    float del = 0.02;
    /*objects[0].pos.x += del;
    objects[0].pos.z += del / 2;

    objects[1].pos.x += -del / 2;*/
    //objects[1].pos.z += -del / 2;

    // move objects


    //objects[0].pos.x += del;
    /*const float rad = PI / 180.0f;
    for (int i = 0; i < 5; i++) {
        angle[i] = fmod(angle[i] + speed[i], 360.0f);
        float3 v = { cos(angle[i] * rad), 0, -sin(angle[i] * rad) };
        v = v * dist[i];
        v.y = objects[i].pos.y;
        objects[i].pos = v;
    }*/

    moveCamera();
    
    // reset
    bool reset = (GetKeyState(0x52) & 0x8000);
    if (reset) {
        initObjects();
    }

}

