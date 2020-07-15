// USE_TEXSUBIMAGE2D uses glTexSubImage2D() to update the final result
// commenting it will make the sample use the other way :
// map a texture in CUDA and blit the result into it

// --use_fast_math 

#pragma warning(disable:4996)
#pragma warning(disable:26440)
#pragma warning(disable:26496)
#pragma warning(disable:26493)
#pragma warning(disable:26812)
#pragma warning(disable:26497)
#pragma warning(disable:26481)


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

const char *sSDKname = "simpleCUDA2GL";

unsigned int g_TotalErrors = 0;


////////////////////////////////////////////////////////////////////////////////
// constants / global variables
int factor = 5;
int window_factor = 1;
int width = 1280;
int height = 720;
unsigned int window_width = width * window_factor;
unsigned int window_height = height * window_factor;
unsigned int image_width = width * factor;
unsigned int image_height = height * factor;
int iGLUTWindowHandle = 0;  // handle to the GLUT window

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

//char *ref_file       = NULL;
bool enable_cuda     = true;

int   *pArgc = NULL;
char **pArgv = NULL;


// Timer
static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface *timer = NULL;


GLuint shDraw;

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


static const char *glsl_draw_fragshader_src =
    //WARNING: seems like the gl_FragColor doesn't want to output >1 colors...
    //you need version 1.3 so you can define a uvec4 output...
    //but MacOSX complains about not supporting 1.3 !!
    // for now, the mode where we use RGBA8UI may not work properly for Apple : only RGBA16F works (default)
    "#version 130\n"
    "out uvec4 FragColor;\n"
    "void main()\n"
    "{"
    "  FragColor = uvec4(gl_Color.xyz * 255.0, 255.0);\n"
    "}\n";

// copy image and process using CUDA
void generateCUDAImage()
{
    // run the Cuda kernel
    unsigned int *out_data;

    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_dest_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&out_data, &num_bytes,
                                                         cuda_pbo_dest_resource));
    //printf("CUDA mapped pointer of pbo_out: May access %ld bytes, expected %d\n", num_bytes, size_tex_data);
    // calculate grid size
    //dim3 block(32, 32, 1);
    int dm = 32;
    dim3 block(dm, dm, 1);
    dim3 grid(image_width / block.x + 1, image_height / block.y + 1, 1);
    //cout << grid.x << endl;
    //cout << grid.y << endl;
    //cout << grid.z << endl;
    // execute CUDA kernel
    

    launch(grid, block, out_data, image_width, image_height);


    // CUDA generated data in cuda memory or in a mapped PBO made of BGRA 8 bits
    // 2 solutions, here :
    // - use glTexSubImage2D(), there is the potential to loose performance in possible hidden conversion
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

    //glRotatef(45, 0, 0, 1);
    //glTranslatef(1, 0, 0);

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

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glDisable(GL_TEXTURE_2D);

    SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    sdkStartTimer(&timer);

    if (enable_cuda)
    {
        generateCUDAImage();
        displayImage(tex_cudaResult);
    }

    // NOTE: I needed to add this call so the timing is consistent.
    // Need to investigate why
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    //getLastCudaError("ERROR FIRST\n");

    // flip backbuffer
    glutSwapBuffers();

}

// miki
void timerEvent(int value)
{
    // Update fps
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
            glutFullScreen();
            //initGLBuffers();
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
    printf("shape %d %d\n", w, h);

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
    sdkDeleteTimer(&timer);

    // unregister this buffer object with CUDA
    //    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_tex_screen_resource));
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_dest_resource));
    deletePBO(&pbo_dest);
    deleteTexture(&tex_screen);
    deleteTexture(&tex_cudaResult);

    if (iGLUTWindowHandle)
    {
        glutDestroyWindow(iGLUTWindowHandle);
    }

    // finalize logs and leave
    printf("simpleCUDA2GL Exiting...\n");
}

void Cleanup(int iExitCode)
{
    FreeResource();
    printf("PPM Images are %s\n", (iExitCode == EXIT_SUCCESS) ? "Matching" : "Not Matching");
    exit(iExitCode);
}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
GLuint compileGLSLprogram(const char *vertex_shader_src, const char *fragment_shader_src)
{
    GLuint v, f, p = 0;

    p = glCreateProgram();

    if (vertex_shader_src)
    {
        v = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(v, 1, &vertex_shader_src, NULL);
        glCompileShader(v);

        // check if shader compiled
        GLint compiled = 0;
        glGetShaderiv(v, GL_COMPILE_STATUS, &compiled);

        if (!compiled)
        {
            //#ifdef NV_REPORT_COMPILE_ERRORS
            char temp[256] = "";
            glGetShaderInfoLog(v, 256, NULL, temp);
            printf("Vtx Compile failed:\n%s\n", temp);
            //#endif
            glDeleteShader(v);
            return 0;
        }
        else
        {
            glAttachShader(p,v);
        }
    }

    if (fragment_shader_src)
    {
        f = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(f, 1, &fragment_shader_src, NULL);
        glCompileShader(f);

        // check if shader compiled
        GLint compiled = 0;
        glGetShaderiv(f, GL_COMPILE_STATUS, &compiled);

        if (!compiled)
        {
            //#ifdef NV_REPORT_COMPILE_ERRORS
            char temp[256] = "";
            glGetShaderInfoLog(f, 256, NULL, temp);
            printf("frag Compile failed:\n%s\n", temp);
            //#endif
            glDeleteShader(f);
            return 0;
        }
        else
        {
            glAttachShader(p,f);
        }
    }

    glLinkProgram(p);

    int infologLength = 0;
    int charsWritten  = 0;

    glGetProgramiv(p, GL_INFO_LOG_LENGTH, (GLint *)&infologLength);

    if (infologLength > 0)
    {
        char *infoLog = (char *)malloc(infologLength);
        glGetProgramInfoLog(p, infologLength, (GLsizei *)&charsWritten, infoLog);
        printf("Shader compilation error: %s\n", infoLog);
        free(infoLog);
    }

    return p;
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
    // load shader programs
    shDraw = compileGLSLprogram(NULL, glsl_draw_fragshader_src);

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

    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    // register callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    //glutPassiveMotionFunc()
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
    iGLUTWindowHandle = glutCreateWindow("CUDA OpenGL post-processing");

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

    // miki
    glutSetCursor(GLUT_CURSOR_NONE);
    // vsync
    //((BOOL(WINAPI*)(int))wglGetProcAddress("wglSwapIntervalEXT"))(1);

    SDK_CHECK_ERROR_GL();
    return true;
}
