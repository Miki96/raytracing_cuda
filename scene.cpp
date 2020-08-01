#include <scene.h>
#include <structs.h>
#include <windows.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <time.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <transforms.h>
#include <vector>

// Camera control
Camera cam;
float moveSpeed = 50;
float camViewDelta = 0.02;
float camViewLimit = 44;
int lastMouseX = -1;
int lastMouseY = -1;
float runSpeedUp = 2;
float aspect = 1.7777f;

// Globals
bool play = true;
bool antialiasing = true;
float seaSpeed = 2;
char timeDay[5];

// Day & Night cycle
float dayNightTime = 6;
float dayNightSpeed = 0.5 * 1;
float dayNightDistance = 500;
float datNightControlSpeed = 4;
// 1. MORNING
// 2. DAY
// 3. EVENING
// 4. NIGHT
float skyVars[4] = { 0, 0, 0, 1 };

// Materials
vector<int> vecTree;
vector<int> vecMount;
vector<int> vecLight;
float3 ambient = float3{ 0.1, 0.2, 0.4 };
float3 matTree[] = {
    float3{158, 114, 250} *(1.0 / 255),
    float3{218, 222, 255} *(1.0 / 255),
    float3{255, 166, 82} *(1.0 / 255),
    {0.31, 0.25, 0.62},
};
float3 matMount[] = {
    float3{224, 205, 255} *(1.0 / 255),
    float3{75, 111, 255} *(1.0 / 255),
    float3{255, 230, 103} *(1.0 / 255),
    {0.02, 0.04, 0.09},
};
float3 matLake[] = {
    float3{155, 4, 136} *(1.0 / 255),
    float3{20, 143, 248} *(1.0 / 255) * 0.9,
    float3{255, 20, 20} *(1.0 / 255),
    {0, 0, 0},
};
float3 matAmbient[] = {
    float3{139, 129, 197} *(1.0 / 255),
    float3{115, 136, 178} *(1.0 / 255) * 0.7,
    float3{164, 132, 121} * (1.0 / 255),
    { 0.1, 0.2, 0.4 },
};

// Lights and Models
Object* objects;
Light* lights;

// Textures
unsigned char* texture1, *texture2, *texture3, *texture4;
int texW;
int texH;

// Declarations
extern "C" void
launchKernel(
    unsigned int* g_odata,
    int imgw, int imgh,
    Camera cam, Object * objectPositions, Light * lightPositions, float3 ambient, float* skyVars,
    unsigned char* h_tex1, unsigned char* h_tex2, unsigned char* h_tex3, unsigned char* h_tex4, int w, int h, float dayTime,
    bool antialiasing);

// HELPER FUNCTIONS

float toRad(float angle) {
    return (PI / 180.0f) * angle;
}

char* getTime()
{
    return timeDay;
}

// CAMERA FUNCTIONS

void cameraHelperAngles() {
    float dirRad = toRad(cam.horAngle);
    cam.dir = { cos(dirRad), 0, sin(dirRad) };

    float a = toRad(cam.fov / 2.0);
    float h = tan(a);
    float w = h * aspect;

    cam.LD = { 1, -h, -w };
    cam.RD = { 1, -h, w };
    cam.LU = { 1, h, -w };
    cam.RU = { 1, h, w };

    // rotate ver
    float av = toRad(-cam.verAngle);
    cam.LD = rotZ(cam.LD, av);
    cam.RD = rotZ(cam.RD, av);
    cam.LU = rotZ(cam.LU, av);
    cam.RU = rotZ(cam.RU, av);

    // rotate hor
    float ah = toRad(-cam.horAngle);
    cam.LD = rotY(cam.LD, ah);
    cam.RD = rotY(cam.RD, ah);
    cam.LU = rotY(cam.LU, ah);
    cam.RU = rotY(cam.RU, ah);
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
    float3 camUp = { 0, 1, 0 };
    float3 camSide = { -cam.dir.z, 0, cam.dir.x };

    int verMove = (bool)GetAsyncKeyState(0x44) - (bool)GetAsyncKeyState(0x41);
    camMove = camMove + camSide * verMove;
    int horMove = (bool)GetAsyncKeyState(0x57) - (bool)GetAsyncKeyState(0x53);
    camMove = camMove + camForw * horMove;
    int upMove = (bool)GetAsyncKeyState(0x51) - (bool)GetAsyncKeyState(0x45);
    camMove = camMove + camUp * upMove;
    // run
    float run = (bool)GetAsyncKeyState(VK_SHIFT) ? runSpeedUp : 1;

    if (verMove || horMove || upMove) {
        camMove = normalize(camMove);
        cam.pos = cam.pos + camMove * (moveSpeed * run) * deltaTime;
        //printf("CAM: %f %f %f VIEW: %f %f\n", cam.pos.x, cam.pos.y, cam.pos.z, cam.horAngle, cam.verAngle);
    }
}

void initCamera() {
    cam = {
        {-56, 2.2, 72}, // position
        309, // horizontal angle
        -7.07, // vertical angle
        40 // field of view
    };
    cameraHelperAngles();
}

// SCENE FUNCTIONS

void createSphere(int& i, float3 color, float mirror, float specular, float shine, float3 pos, float size) {
    objects[i].type = Primitive::SPHERE;
    objects[i].color = color;
    objects[i].mirror = mirror;
    objects[i].specular = specular;
    objects[i].shine = shine;
    objects[i].pos = pos;
    objects[i].size = { size, size, size };
    objects[i].light = false;
    i++;
}

void createSnowman(int& i, float3 offset, float a) {
    float3 color = float3{ 1, 1, 1 } * 0.8;
    float mirror = 0;
    float specular = 1;
    float shine = 0.05;
    float3 pos = { 0, 0, 0 };
    float size = 1;

    // BELLY
    pos = rotY({ 0, 0, 0 }, a);
    size = 2;
    createSphere(i, color, mirror, specular, shine, pos + offset, size);

    // HEAD
    pos = rotY({ 0, 3, 0 }, a);
    size = 1.3;
    createSphere(i, color, mirror, specular, shine, pos + offset, size);

    // EYES
    color = { 0, 0, 0 };
    size = 0.2;
    pos = { 0.35, 3.2, 1.15 };
    pos = rotY(pos, a);
    createSphere(i, color, mirror, specular, shine, pos + offset, size);
    pos = { -0.35, 3.2, 1.15 };
    pos = rotY(pos, a);
    createSphere(i, color, mirror, specular, shine, pos + offset, size);

    // MOUTH
    size = 0.1;
    pos = { 0.2, 2.3, 1.05 };
    pos = rotY(pos, a);
    createSphere(i, color, mirror, specular, shine, pos + offset, size);
    pos = { -0.2, 2.3, 1.05 };
    pos = rotY(pos, a);
    createSphere(i, color, mirror, specular, shine, pos + offset, size);
    pos = { 0.55, 2.3 + 0.2, 1.05 };
    pos = rotY(pos, a);
    createSphere(i, color, mirror, specular, shine, pos + offset, size);
    pos = { -0.55, 2.3 + 0.2, 1.05 };
    pos = rotY(pos, a);
    createSphere(i, color, mirror, specular, shine, pos + offset, size);

    // BUTTONS
    size = 0.2;
    pos = { 0, 1, 1.6 };
    pos = rotY(pos, a);
    createSphere(i, color, mirror, specular, shine, pos + offset, size);
    pos = { 0, 0.3, 1.85 };
    pos = rotY(pos, a);
    createSphere(i, color, mirror, specular, shine, pos + offset, size);
    pos = { 0, -0.5, 1.8 };
    pos = rotY(pos, a);
    createSphere(i, color, mirror, specular, shine, pos + offset, size);
}

void createPyramid(int& i, float3 color, float mirror, float specular, float shine, float3 pos, float base, float height, float angle) {
    float y = 0.86f;
    float x = 0.5f;
    float h = y * 2.0f / 3.0f;
    float v = y * 1.0f / 3.0f;
    float t = 0.5f;

    float3 tris[] = {
        // down
        {0, 0, 0}, // 0
        {1, 0, 0}, // 1
        {x, 0, y}, // 2
        // front
        {0, 0, 0}, // 0
        {x, t, v}, // 3
        {1, 0, 0}, // 1
        // left
        {0, 0, 0}, // 0
        {x, 0, y}, // 2
        {x, t, v}, // 3
        // right
        {x, 0, y}, // 2
        {1, 0, 0}, // 1
        {x, t, v}, // 3
    };

    // center
    for (int k = 0; k < 4 * 3; k++) {
        tris[k].x -= x;
        tris[k].z -= v;
    }

    for (int k = 0; k < 4 * 3; k++) {
        // rotate
        tris[k] = rotY(tris[k], toRad(angle));
        // scale
        tris[k].x *= base;
        tris[k].y *= height;
        tris[k].z *= base;
        // offset
        tris[k] = tris[k] + pos;
    }

    for (int k = 0; k < 4; k++) {
        objects[i++] = {
            Primitive::TRIANGLE, shine, specular, mirror, color,
            tris[k * 3],
            tris[k * 3 + 1],
            tris[k * 3 + 2], false
        };
    }
}

void createTree(int& i, float3 offset, float angle) {
    float3 color1 = float3{ 100, 80, 200 } *(1.0f / 255.0f) * 0.8;
    float3 color2 = { 0.5, 0, 0 };
    float mirror = 0.1;
    float specular = 1;
    float shine = 0;
    float3 pos = { 0, 0, 0 };
    float size = 1;

    // UP
    pos = { 0, -1, 0 };
    size = 2;
    float base = 7;
    float heigh = 19;
    createPyramid(i, color1, mirror, specular, shine, pos + offset, base, heigh, angle);
    vecTree.push_back(i - 1);
    vecTree.push_back(i - 2);
    vecTree.push_back(i - 3);
    vecTree.push_back(i - 4);

    // DOWN
    pos = { 0, -2, 0 };
    size = 2;
    base = 4;
    heigh = 8;
    createPyramid(i, color2, mirror, specular, shine, pos + offset, base, heigh, angle);
}

void createGround(int& i, float3 offset) {
    objects[i].type = Primitive::PLANE;
    objects[i].color = float3{ 0, 0, 30 } *(1.0f / 255.0f);
    objects[i].mirror = 0.6;
    objects[i].specular = 256;
    objects[i].shine = 0;
    objects[i].pos = offset;
    objects[i].size = { 0, 1, 0 };
    objects[i].light = false;
    i++;
}

void createMountain(int& i, float3 offset, float size, float angle) {
    float3 color = float3{ 18, 31, 60 } * (1.0f / 255.0f) * 0.4;
    float mirror = 0;
    float specular = 256;
    float shine = 0;

    // UP
    createPyramid(i, color, mirror, specular, shine, offset, size, 1.5f * size, angle);
    vecMount.push_back(i - 1);
    vecMount.push_back(i - 2);
    vecMount.push_back(i - 3);
    vecMount.push_back(i - 4);
}

void createIsland(int& i, float3 offset, float size, float d) {
    float3 color = float3{ 100, 80, 200 } *(1.0f / 255.0f) * 0.8;
    float mirror = 0.1;
    float specular = 1;
    float shine = 0;

    float3 p[]{
        {0, 0, 0}, // 0
        {1, 0, 0}, // 1
        {1, 0, 1}, // 2
        {0, 0, 1}, // 3
        {0, -d, 0}, // 4
        {1, -d, 0}, // 5
        {1, -d, 1}, // 6
        {0, -d, 1}, // 7
    };

    float3 tris[] = {
        // up1
        p[0], p[2], p[1],
        // up2
        p[0], p[3], p[2],
        // front1
        p[4], p[1], p[5],
        // front2
        p[4], p[0], p[1],
        // back1
        p[6], p[3], p[7],
        // back2
        p[6], p[2], p[3],
        // right1
        p[5], p[2], p[6],
        // right2
        p[5], p[1], p[2],
        // left1
        p[7], p[0], p[4],
        // left2
        p[7], p[3], p[0]
    };

    for (int k = 0; k < 10 * 3; k++) {
        // center
        tris[k].x -= 0.5;
        tris[k].z -= 0.5;
        // scale
        tris[k].x *= size;
        tris[k].y *= 1;
        tris[k].z *= size;
        // offset
        tris[k] = tris[k] + offset;
    }

    for (int k = 0; k < 10; k++) {
        vecTree.push_back(i);
        objects[i++] = {
            Primitive::TRIANGLE, shine, specular, mirror, color,
            tris[k * 3],
            tris[k * 3 + 1],
            tris[k * 3 + 2],
            false
        };
    }
}

void createIgloo(int& i, float3 offset, float size1, float size2) {
    float3 color = float3{ 1, 1, 1 } * 0.8;
    float mirror = 0;
    float specular = 1;
    float shine = 0.05;
    float3 pos = { 0, 0, 0 };
    float size = 1;

    // MAIN
    pos = { 0, 0, 0 };
    createSphere(i, color, mirror, specular, shine, pos + offset, size1);

    // ENTRY
    pos = { -6, 0, 6 };
    createSphere(i, color, mirror, specular, shine, pos + offset, size2);
}

void createLightObjects(int& i) {
    // sun
    createSphere(i, { 1, 0.8, 0.05 }, 0, 0, 0, lights[0].pos, 50);
    objects[i - 1].light = true;
    vecLight.push_back(i - 1);
    // moon
    createSphere(i, { 0.9, 0.9, 1 }, 0, 0, 0, lights[1].pos, 50);
    objects[i - 1].light = true;
    vecLight.push_back(i - 1);
}

void initObjects() {
    objects = (Object*)malloc(sizeof(Object) * OBJECTS_NUMBER);
    int i = 0;

    float level = -4.5;
    
    createGround(i, { 0, level, 0 });
    createIsland(i, { 0, -4, 0 }, 50, 2);

    createSnowman(i, { -4, -2, 17 }, toRad(-50));
    createSnowman(i, { -15, -2, 5 }, toRad(-20));

    createTree(i, { -22, -2, -10 }, 90);
    createTree(i, { -10, -2, -20 }, 90);
    createTree(i, { 0, -2, -20 }, 80);

    createTree(i, { 17, -2, 2 }, 90);
    createTree(i, { 20, -2, 9 }, 80);
    createTree(i, { 12, -2, 22 }, 70);

    // big mountains
    float d = 4;
    createMountain(i, float3{170, level, 0 } *d, 100 * d, 0);
    createMountain(i, float3{ 90, level, -100 } *d, 110 * d, 45);
    createMountain(i, float3{ -35, level, -90 } *d, 100 * d, 0);
    createMountain(i, float3{ -100, level, 65 } *d, 100 * d, 0); //sunset
    createMountain(i, float3{ 25, level, 140 } *d, 100 * d, 0); //sunrise
    createMountain(i, float3{ 130, level, 90 } *d, 100 * d, 0);

    // small mountains
    createMountain(i, float3{ 100, level, 30 } *d, 70 * d, 0);
    createMountain(i, float3{ 100, level, -40 } *d, 50 * d, 30);
    createMountain(i, float3{ 20, level, -100 } *d, 70 * d, 0);
    createMountain(i, float3{ -80, level, -40 } *d, 80 * d, 0); // sunset
    createMountain(i, float3{ -70, level, 100 } *d, 90 * d, 0); //sunrise
    createMountain(i, float3{ 60, level, 90 } *d, 50 * d, 0);

    // igloo
    createIgloo(i, { 4, -4, -4 }, 10, 6);

    // add lights
    createLightObjects(i);

    printf("OBJECTS: %d\n", i);
}

void oldStaticScene() {
    // RED
    {
        int i = 0;
        objects[i].type = Primitive::SPHERE;
        objects[i].color = { 0.91, 0.1, 0.1 };
        objects[i].mirror = 0;
        objects[i].specular = 256;
        objects[i].shine = 1;
        objects[i].pos = { -5, -2, -13 };
        objects[i].size = { 2, 2, 2 };
    }
    // GREEN
    {
        int i = 1;
        objects[i].type = Primitive::SPHERE;
        objects[i].color = { 0, 1, 0.1 };
        objects[i].mirror = 0;
        objects[i].specular = 256;
        objects[i].shine = 0;
        objects[i].pos = { 2.5, -2.5, -12 };
        objects[i].size = { 1.5, 1.5, 1.5 };
    }
    // MIRROR
    {
        int i = 2;
        objects[i].type = Primitive::SPHERE;
        objects[i].color = { 0, 0, 0 };
        objects[i].mirror = 1;
        objects[i].specular = 256;
        objects[i].shine = 1;
        objects[i].pos = { 0, 1, -20 };
        objects[i].size = { 5, 5, 5 };
    }
    // YELLOW
    {
        int i = 3;
        objects[i].type = Primitive::SPHERE;
        objects[i].color = { 0.9, 0.9, 0.1 };
        objects[i].mirror = 0;
        objects[i].specular = 1256;
        objects[i].shine = 1;
        objects[i].pos = { 15, -1, -40 };
        objects[i].size = { 3, 3, 3 };
    }
    // BLUE GLASS
    {
        int i = 4;
        objects[i].type = Primitive::SPHERE;
        objects[i].color = { 0, 0.5, 1 };
        objects[i].mirror = 0;
        objects[i].specular = 16;
        objects[i].shine = 0.1;
        objects[i].pos = { 10, -2, -20 };
        objects[i].size = { 2, 2, 2 };
    }

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
        //rgb c = hsv2rgb({ (double)(rand() % 360), 0.9, 0.9 });
        objects[i].color.x = 1;
        objects[i].color.y = 1;
        objects[i].color.z = 1;
        objects[i].mirror = 0.2;
        objects[i].specular = 256;
        objects[i].shine = 0;
        // normal
        objects[i].size.x = 0;
        objects[i].size.y = 1;
        objects[i].size.z = 0;
    }

    //// calc distances
    //for (int i = 0; i < 5; i++) {
    //    float3 v = objects[i].pos;
    //    v.y = 0;
    //    dist[i] = norm(v);
    //}

    //// add triangles
    //{
    //    float y = 0.86f;
    //    float x = 0.5f;
    //    float h = y * 2.0f / 3.0f;
    //    float v = y * 1.0f / 3.0f;

    //    float3 tris[] = {
    //        // down
    //        {0, 0, 0}, // 0
    //        {1, 0, 0}, // 1
    //        {x, 0, y}, // 2
    //        // front
    //        {0, 0, 0}, // 0
    //        {x, 1, v}, // 3
    //        {1, 0, 0}, // 1
    //        // left
    //        {0, 0, 0}, // 0
    //        {x, 0, y}, // 2
    //        {x, 1, v}, // 3
    //        // right
    //        {1, 0, 0}, // 1
    //        {x, 1, v}, // 3
    //        {x, 0, y}, // 2
    //    };

    //    for (int i = 0; i < 4 * 3; i++) {
    //        tris[i].x += 3;
    //        tris[i].y -= 1;
    //    }

    //    for (int i = 0; i < 4 * 3; i++) {
    //        tris[i].x *= 2;
    //        tris[i].y *= 2;
    //        tris[i].z *= 2;
    //    }

    //    for (int i = 0; i < 4; i++) {
    //        objects[i + 6] = {
    //            Primitive::TRIANGLE,
    //            tris[i * 3],
    //            float3{0, 0, 200},
    //            tris[i * 3 + 1],
    //            tris[i * 3 + 2]
    //        };
    //    }
    //}
}

void initTexture() {
    int n;
    texture1 = stbi_load("backgrounds/morning.png", &texW, &texH, &n, 4);
    texture2 = stbi_load("backgrounds/day.png", &texW, &texH, &n, 4);
    texture3 = stbi_load("backgrounds/evening.png", &texW, &texH, &n, 4);
    texture4 = stbi_load("backgrounds/night.png", &texW, &texH, &n, 4);
}

void initLights() {
    lights = (Light*)malloc(sizeof(Light) * LIGHTS_NUMBER);

    // SUN
    {
        int i = 0;
        lights[i].color = { 1, 1, 1 };
        lights[i].intensity = 1;
        lights[i].pos = { -1000, 1000, 1000 };
    }

    // MOON
    {
        int i = 1;
        lights[i].color = { 1, 1, 1 };
        lights[i].intensity = 1;
        lights[i].pos = { -1000, 1000, 1000 };
    }
}

void initScene() {
    // create spheres
    srand(time(NULL));

    initCamera();
    initLights();
    initObjects();
    initTexture();
}

// ANIMATION FUNCTIONS

float3 getColorByTime(float3* mats) {
    float3 c = { 0, 0, 0 };
    for (int i = 0; i < 4; i++) {
        c = c + mats[i] * skyVars[i];
    }
    return c;
}

void recolorObjects() {
    // trees
    for (int i = 0; i < vecTree.size(); i++) {
        objects[vecTree[i]].color = getColorByTime(matTree);
    }
    // mountains
    for (int i = 0; i < vecMount.size(); i++) {
        objects[vecMount[i]].color = getColorByTime(matMount);
    }
    // lake
    objects[0].color = getColorByTime(matLake);
    // ambient
    ambient = getColorByTime(matAmbient);
}

void controls() {
    // time control
    int timeControl = (bool)GetAsyncKeyState(VK_RIGHT) - (bool)GetAsyncKeyState(VK_LEFT);
    if (timeControl) {
        dayNightTime = fmodf(dayNightTime + dayNightSpeed * deltaTime * timeControl * datNightControlSpeed + 24, 24);
    }
    else if (play) {
        dayNightTime = fmodf(dayNightTime + dayNightSpeed * deltaTime + 24, 24);
    }

    // play/pause
    if ((bool)GetAsyncKeyState(0x50)) {
        play = true;
    }
    if ((bool)GetAsyncKeyState(0x4F)) {
        play = false;
    }

    // sea control
    int seaControl = (bool)GetAsyncKeyState(VK_UP) - (bool)GetAsyncKeyState(VK_DOWN);
    objects[0].pos.y += seaControl * seaSpeed * deltaTime;

    // time of day control
    bool change = false;
    if ((bool)GetAsyncKeyState(0x31)) {
        dayNightTime = 6; // morning
        change = true;
    }
    if ((bool)GetAsyncKeyState(0x32)) {
        dayNightTime = 14; // day
        change = true;
    }
    if ((bool)GetAsyncKeyState(0x33)) {
        dayNightTime = 18; // evening
        change = true;
    }
    if ((bool)GetAsyncKeyState(0x34)) {
        dayNightTime = 1; // night
        change = true;
    }

    // print time
    if (timeControl || change || play) {
        sprintf(timeDay, "%02d:%02d\n", (int)dayNightTime, (int)(((int)(dayNightTime * 100) % 100) / 100.0 * 60));
    }

    // set camera to scene
    if ((bool)GetAsyncKeyState(0x35)) {
        // main scene
        cam.pos = { -56, 2.2, 72 };
        cam.horAngle = 309;
        cam.verAngle = -7.07;
    }
    if ((bool)GetAsyncKeyState(0x36)) {
        // day cycle
        cam.pos = { 324.4, 12.41, -84 };
        cam.horAngle = 141.2;
        cam.verAngle = -12.65;
    }

    // antialiasing
    if ((bool)GetAsyncKeyState(0x42)) {
        antialiasing = true;
    }
    if ((bool)GetAsyncKeyState(0x56)) {
        antialiasing = false;
    }
}

void moveLights() {
    // lights position
    float a = toRad(fmodf((dayNightTime / 24.0f) * 360 - 120, 360));
    // rotate
    lights[0].pos = rotY(float3{ cosf(a), sinf(a), 0 } *dayNightDistance, toRad(-45));
    lights[1].pos = lights[0].pos * -1;
    // move
    float o = 500;
    float3 offset = { -o, 0, o };
    lights[0].pos = lights[0].pos + offset;
    lights[1].pos = lights[1].pos + offset;
    // move objects
    objects[vecLight[0]].pos = lights[0].pos;
    objects[vecLight[1]].pos = lights[1].pos;
    // light intensity
    float val = fabs(lights[0].pos.y) / dayNightDistance;
    lights[0].color = float3{ 1, 1, 1 } *val;
    lights[1].color = lights[0].color * 1;
}

void calcSkyVars() {
    // calculate sky vars
    for (int i = 0; i < 4; i++) skyVars[i] = 0;

    float d = dayNightTime;
    if (d >= 6 && d <= 8) skyVars[0] = 1; // morning
    if (d >= 10 && d <= 16) skyVars[1] = 1; // day
    if (d >= 18 && d <= 20) skyVars[2] = 1; // evening
    if (d >= 22 || d <= 4) skyVars[3] = 1; // night

    if (d > 8 && d < 10) {
        skyVars[1] = (d - 8) / 2;
        skyVars[0] = 1.0f - skyVars[1];
    }
    if (d > 16 && d < 18) {
        skyVars[2] = (d - 16) / 2;
        skyVars[1] = 1.0f - skyVars[2];
    }
    if (d > 20 && d < 22) {
        skyVars[3] = (d - 20) / 2;
        skyVars[2] = 1.0f - skyVars[3];
    }
    if (d > 4 && d < 6) {
        skyVars[0] = (d - 4) / 2;
        skyVars[3] = 1.0f - skyVars[0];
    }
}

void animate() {
    // move player camera
    moveCamera();
    // read controls input
    controls();
    // change colors based on time
    recolorObjects();
    calcSkyVars();
    // change sun and moon position
    moveLights();
}

// KERNEL LAUNCH

void launch(unsigned int* out_data, int imgw, int imgh) {
    aspect = (1.0f * imgw) / imgh;
    float dayProgress = (dayNightTime / 24.0f);
    launchKernel(out_data, imgw, imgh, cam, objects, lights, ambient, skyVars,
        texture1, texture2, texture3, texture4, texW, texH, dayProgress, antialiasing);
    cudaDeviceSynchronize();
}