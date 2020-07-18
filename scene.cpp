#include <scene.h>
#include <structs.h>
#include <windows.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <time.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <transforms.h>

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
Light* lights;

unsigned char* texture;
int texW;
int texH;

// declarations
extern "C" void
launch_cudaProcess(dim3 grid, dim3 block, int sbytes,
    unsigned int* g_odata,
    int imgw, int imgh,
    Camera cam, float3 sun, Object * objectPositions, Light* lightPositions,
    unsigned char* tex, int w, int h);
void cameraHelperAngles();

float toRad(float angle) {
    return (PI / 180.0f) * angle;
}

void initCamera() {
    cam = {
        {-40, 10, 50}, // position
        310, // horizontal angle
        12, // vertical angle
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

void createSphere(int& i, float3 color, float mirror, float specular, float shine, float3 pos, float size) {
    objects[i].type = Primitive::SPHERE;
    objects[i].color = color;
    objects[i].mirror = mirror;
    objects[i].specular = specular;
    objects[i].shine = shine;
    objects[i].pos = pos;
    objects[i].size = { size, size, size };
    i++;
}

void createSnowman(int& i, float3 offset, float a) {
    float3 color = { 1, 1, 1 };
    float mirror = 0;
    float specular = 156;
    float shine = 0;
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

void createPyramid(int& i, float3 color, float mirror, float specular, float shine, float3 pos, float base, float height) {
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

    for (int k = 0; k < 4 * 3; k++) {
        // center
        tris[k].x -= x;
        tris[k].z -= v;
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
            tris[k * 3 + 2]
        };
    }
}

void createTree(int& i, float3 offset) {
    float3 color1 = { 1, 1, 1 };
    float3 color2 = { 1, 0, 0 };
    float mirror = 0;
    float specular = 256;
    float shine = 0;
    float3 pos = { 0, 0, 0 };
    float size = 1;

    // UP
    pos = { 0, -1, 0 };
    size = 2;
    float base = 7;
    float heigh = 19;
    createPyramid(i, color1, mirror, specular, shine, pos + offset, base, heigh);

    // DOWN
    pos = { 0, -2, 0 };
    size = 2;
    base = 4;
    heigh = 8;
    createPyramid(i, color2, mirror, specular, shine, pos + offset, base, heigh);
}

void createGround(int& i, float3 offset) {
    objects[i].type = Primitive::PLANE;
    objects[i].color = { 1, 0.1, 1 };
    objects[i].mirror = 0.4;
    objects[i].specular = 256;
    objects[i].shine = 0;
    objects[i].pos = offset;
    objects[i].size = { 0, 1, 0 };
    i++;
}

void createMountain(int& i, float3 offset, float size) {
    float3 color = { 1, 1, 1 };
    float mirror = 0.8;
    float specular = 256;
    float shine = 0;

    // UP
    createPyramid(i, color, mirror, specular, shine, offset, size, 1.5f * size);
}

void createIsland(int& i, float3 offset, float size, float d) {
    float3 color = { 1, 1, 1 };
    float mirror = 0.1;
    float specular = 256;
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
        objects[i++] = {
            Primitive::TRIANGLE, shine, specular, mirror, color,
            tris[k * 3],
            tris[k * 3 + 1],
            tris[k * 3 + 2]
        };
    }
}

void createIgloo(int& i, float3 offset, float size1, float size2) {
    float3 color = { 1, 1, 1 };
    float mirror = 0;
    float specular = 256;
    float shine = 0;
    float3 pos = { 0, 0, 0 };
    float size = 1;

    // MAIN
    pos = { 0, 0, 0 };
    createSphere(i, color, mirror, specular, shine, pos + offset, size1);

    // ENTRY
    pos = { -6, 0, 6 };
    createSphere(i, color, mirror, specular, shine, pos + offset, size2);
}

void initObjects() {
    objects = (Object*)malloc(sizeof(Object) * OBJECTS_NUMBER);
    int i = 0;

    float level = -4.5;
    
    createGround(i, { 0, level, 0 });

    createSnowman(i, { -4, -2, 17 }, toRad(-50));
    createSnowman(i, { -15, -2, 5 }, toRad(-20));

    createTree(i, { -20, -2, -10 });
    createTree(i, { -10, -2, -20 });
    createTree(i, { 0, -2, -22 });

    createTree(i, { 20, -2, 10 });
    createTree(i, { 17, -2, 0 });
    createTree(i, { 10, -2, 20 });

    // big mountains
    float d = 4;
    createMountain(i, float3{170, level, 0 } *d, 100 * d);
    createMountain(i, float3{ 90, level, -100 } *d, 100 * d);
    createMountain(i, float3{ -35, level, -90 } *d, 100 * d);
    createMountain(i, float3{ -120, level, 35 } *d, 100 * d);
    createMountain(i, float3{ 5, level, 130 } *d, 100 * d);
    createMountain(i, float3{ 130, level, 90 } *d, 100 * d);

    // small mountains
    createMountain(i, float3{ 100, level, 30 } *d, 70 * d);
    createMountain(i, float3{ 100, level, -40 } *d, 50 * d);
    createMountain(i, float3{ 20, level, -100 } *d, 70 * d);
    createMountain(i, float3{ -80, level, -30 } *d, 70 * d);
    createMountain(i, float3{ -70, level, 80 } *d, 70 * d);
    createMountain(i, float3{ 60, level, 90 } *d, 50 * d);

    createIsland(i, { 0, -4, 0 }, 50, 2);

    createIgloo(i, { 4, -4, -4 }, 10, 6);

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
        rgb c = hsv2rgb({ (double)(rand() % 360), 0.9, 0.9 });
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
    texture = stbi_load("backgrounds/snow.jpg", &texW, &texH, &n, 4);
    printf("%d %d %d\n", texW, texH, n);
}

void initLights() {
    lights = (Light*)malloc(sizeof(Light) * LIGHTS_NUMBER);

    // SUN
    {
        int i = 0;
        lights[i].color = { 1, 1, 1 };
        lights[i].intensity = 1000;
        lights[i].pos = { -30, 30, 30 };
    }

    // MOON
    {
        int i = 1;
        lights[i].color = { 1, 1, 1 };
        lights[i].intensity = 1000;
        lights[i].pos = { -30, 30, -30 };
    }
}

void initScene() {
    // create spheres
    srand(time(NULL));

    initCamera();
    initObjects();
    initLights();
    initTexture();
}

void launch(dim3 grid, dim3 block, unsigned int* out_data, int imgw, int imgh) {
    aspect = (1.0f * imgw) / imgh;
    launch_cudaProcess(grid, block, 0, out_data, imgw, imgh, cam, sunPos, objects, lights, texture, texW, texH);
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
    cam.LD = rotZ(cam.LD, av);
    cam.RD = rotZ(cam.RD, av);
    cam.LU = rotZ(cam.LU, av);
    cam.RU = rotZ(cam.RU, av);

    // rotate hor
    float ah = rad * (-cam.horAngle);
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
        cam.pos = cam.pos + camMove * (moveSpeed * run);

        printf("CAM: %f %f %f VIEW: %f %f\n", cam.pos.x, cam.pos.y, cam.pos.z, cam.horAngle, cam.verAngle);
    }
}

void moveSun() {
    int verMove = (bool)GetAsyncKeyState(0x49) - (bool)GetAsyncKeyState(0x4B);
    int horMove = (bool)GetAsyncKeyState(0x4C) - (bool)GetAsyncKeyState(0x4A);
    int upMove = (bool)GetAsyncKeyState(0x4F) - (bool)GetAsyncKeyState(0x55);

    lights[0].pos.x += verMove;
    lights[0].pos.z += horMove;
    lights[0].pos.y += upMove;
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
    moveSun();
    
    // reset
    bool reset = (GetKeyState(0x52) & 0x8000);
    if (reset) {
        //initObjects();
    }

}

