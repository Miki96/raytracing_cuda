// USE_TEXSUBIMAGE2D uses glTexSubImage2D() to update the final result
// commenting it will make the sample use the other way :
// map a texture in CUDA and blit the result into it
// --use_fast_math

#include <iostream>
using namespace std;

#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

// OpenGL Graphics includes
#include <helper_gl.h>
#include <GL/freeglut.h>

// CUDA includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA utilities and system includes
#include <helper_cuda.h>
#include <helper_functions.h>
#include <rendercheck_gl.h>

#include <time.h>
//#include <structs.h>
#include <scene.h>

// Shared Library Test Functions
#define MAX_EPSILON 10
#define REFRESH_DELAY 500 //ms

const char *sSDKname = "Raytracing";
unsigned int g_TotalErrors = 0;


////////////////////////////////////////////////////////////////////////////////
// constants / global variables
int factor = 1;
int window_factor = 1;
int width = 1280;
int height = 720;
unsigned int window_width = width * window_factor;
unsigned int window_height = height * window_factor;
unsigned int image_width = width * factor;
unsigned int image_height = height * factor;
int iGLUTWindowHandle = 0;  // handle to the GLUT window
bool fullscreen = false;

// delta time
int timeStart = 0;
float deltaTime = 1;

// pbo and fbo variables
GLuint pbo_dest;
struct cudaGraphicsResource *cuda_pbo_dest_resource;

GLuint fbo_source;
struct cudaGraphicsResource *cuda_tex_screen_resource;

unsigned int size_tex_data;
unsigned int num_texels;
unsigned int num_values;

// (offscreen) render target fbo variables
GLuint tex_screen;      // where we render the image
GLuint tex_cudaResult;  // where we will copy the CUDA result

bool enable_cuda     = true;

int   *pArgc = NULL;
char **pArgv = NULL;

//GLuint shDraw;

////////////////////////////////////////////////////////////////////////////////

// Forward declarations
void runStdProgram(int argc, char **argv);
void FreeResource();
void Cleanup(int iExitCode);

// GL functionality
bool initGL(int *argc, char **argv);

void createPBO(GLuint *pbo, struct cudaGraphicsResource **pbo_resource);
void deletePBO(GLuint *pbo);

void createTextureDst(GLuint *tex_cudaResult, unsigned int size_x, unsigned int size_y);
void deleteTexture(GLuint *tex);

// rendering callbacks
void display();
void idle();
void keyboard(unsigned char key, int x, int y);
void reshape(int w, int h);
void animate();

////////////////////////////////////////////////////////////////////////////////
//! Create PBO
////////////////////////////////////////////////////////////////////////////////
void
createPBO(GLuint *pbo, struct cudaGraphicsResource **pbo_resource)
{
    // set up vertex data parameter
    num_texels = image_width * image_height;
    num_values = num_texels * 4;
    size_tex_data = sizeof(GLubyte) * num_values;
    void *data = malloc(size_tex_data);

    // create buffer object
    glGenBuffers(1, pbo);
    glBindBuffer(GL_ARRAY_BUFFER, *pbo);
    glBufferData(GL_ARRAY_BUFFER, size_tex_data, data, GL_DYNAMIC_DRAW);
    free(data);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(pbo_resource, *pbo, cudaGraphicsMapFlagsNone));

    SDK_CHECK_ERROR_GL();
}

void
deletePBO(GLuint *pbo)
{
    glDeleteBuffers(1, pbo);
    SDK_CHECK_ERROR_GL();
    *pbo = 0;
}

const GLenum fbo_targets[] =
{
    GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT,
    GL_COLOR_ATTACHMENT2_EXT, GL_COLOR_ATTACHMENT3_EXT
};

// copy image and process using CUDA
void generateCUDAImage()
{
    // run the Cuda kernel
    unsigned int *out_data;

    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_dest_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&out_data, &num_bytes,
                                                         cuda_pbo_dest_resource));

    // execute CUDA kernel
    launch(out_data, image_width, image_height);

    // CUDA generated data in cuda memory or in a mapped PBO made of BGRA 8 bits
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_dest_resource, 0));
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_dest);

    glBindTexture(GL_TEXTURE_2D, tex_cudaResult);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    image_width, image_height,
                    GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    SDK_CHECK_ERROR_GL();
    glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
}

// display current time
void drawTime() {
    glRasterPos2f(0.9, -0.92);
    glColor3f(1.0f, 1.0f, 1.0f);
    string text(getTime());
    glutBitmapString(GLUT_BITMAP_HELVETICA_18, (const unsigned char*)text.c_str());
}

// display image to the screen as textured quad
void displayImage(GLuint texture)
{
    glBindTexture(GL_TEXTURE_2D, texture);
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, window_width, window_height);

    float edge = 1;
    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 1.0);
    glVertex3f(-edge, -edge, 0.5);
    glTexCoord2f(1.0, 1.0);
    glVertex3f(edge, -edge, 0.5);
    glTexCoord2f(1.0, 0.0);
    glVertex3f(edge, edge, 0.5);
    glTexCoord2f(0.0, 0.0);
    glVertex3f(-edge, edge, 0.5);
    glEnd();

    glDisable(GL_TEXTURE_2D);

    //drawTime();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    SDK_CHECK_ERROR_GL();
}



////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    // draw scene
    generateCUDAImage();
    displayImage(tex_cudaResult);

    glutSwapBuffers();
}

// custom
void timerEvent(int value)
{
    // Update FPS
    char cTitle[256];
    sprintf(cTitle, "Raytracing (%d x %d): %.1f fps", window_width, window_height, 1 / deltaTime);
    glutSetWindowTitle(cTitle);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

void mouseInput() {
    POINT pos;
    GetCursorPos(&pos);
    int x = pos.x;
    int y = pos.y;
    int wx = glutGet(GLUT_WINDOW_X);
    int wy = glutGet(GLUT_WINDOW_Y);

    int w = window_width;
    int h = window_height;

    mouseMotion(x - wx, y - wy, w, h);

    glutWarpPointer(w / 2, h / 2);
}

void updateDelta() {
    int realTimeStart = glutGet(GLUT_ELAPSED_TIME);
    deltaTime = (realTimeStart - timeStart) * 1.0 / 1000.0;
    timeStart = realTimeStart;
}

void idle() {
    // calc delta time
    updateDelta();

    mouseInput();
    animate();
    glutPostRedisplay();
}

void createTextureDst(GLuint* tex_cudaResult, unsigned int size_x, unsigned int size_y);
////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case 'f':
            fullscreen = !fullscreen;
            if (fullscreen) {
                glutFullScreen();
            }
            else {
                glutReshapeWindow(1280, 720);
            }
            break;
        case (27) :
            Cleanup(EXIT_SUCCESS);
            break;
    }
}

void reshape(int w, int h)
{
    window_width = w;
    window_height = h;
    image_width = w * factor;
    image_height = h * factor;
    //printf("shape %d %d\n", w, h);

    createPBO(&pbo_dest, &cuda_pbo_dest_resource);
    createTextureDst(&tex_cudaResult, image_width, image_height);
}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
void createTextureDst(GLuint *tex_cudaResult, unsigned int size_x, unsigned int size_y)
{
    // create a texture
    glGenTextures(1, tex_cudaResult);
    glBindTexture(GL_TEXTURE_2D, *tex_cudaResult);

    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, size_x, size_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
void deleteTexture(GLuint *tex)
{
    glDeleteTextures(1, tex);
    SDK_CHECK_ERROR_GL();

    *tex = 0;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    initScene();

    runStdProgram(argc, argv);
    exit(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
void FreeResource()
{
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_dest_resource));
    deletePBO(&pbo_dest);
    deleteTexture(&tex_screen);
    deleteTexture(&tex_cudaResult);

    if (iGLUTWindowHandle)
    {
        glutDestroyWindow(iGLUTWindowHandle);
    }

    printf("Raytracing Exiting...\n");
}

void Cleanup(int iExitCode)
{
    FreeResource();
    exit(iExitCode);
}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
void initGLBuffers()
{
    // create pbo
    createPBO(&pbo_dest, &cuda_pbo_dest_resource);
    // create texture that will receive the result of CUDA
    createTextureDst(&tex_cudaResult, image_width, image_height);

    SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Run standard demo loop with or without GL verification
////////////////////////////////////////////////////////////////////////////////
void runStdProgram(int argc, char **argv)
{
    initGL(&argc, argv);

    // Now initialize CUDA context (GL context has been created already)
    findCudaDevice(argc, (const char **)argv);

    // register callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);

    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    glutIdleFunc(idle);

    initGLBuffers();

    // start rendering mainloop
    glutMainLoop();

    // Normally unused return path
    Cleanup(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv)
{
    // Create GL context
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(window_width, window_height);
    iGLUTWindowHandle = glutCreateWindow("Raytracing");

    // glew
    glewInit();

    // default initialization
    glClearColor(1, 0, 0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1f, 10.0f);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    // custom
    glutSetCursor(GLUT_CURSOR_NONE);

    SDK_CHECK_ERROR_GL();
    return true;
}
