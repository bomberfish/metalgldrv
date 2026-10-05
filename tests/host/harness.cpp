#include "harness.h"
#include <cstdlib>
#include <cstdio>

std::vector<TestCase>& testRegistry() {
    static std::vector<TestCase> r;
    return r;
}
int g_failures = 0;
const char* g_currentTest = "";
static int g_testFailures = 0;

void testFail(const char* file, int line, const std::string& msg) {
    g_failures++;
    g_testFailures++;
    fprintf(stderr, "  FAIL %s (%s:%d): %s\n", g_currentTest, file, line, msg.c_str());
}

void readPixel(int x, int y, uint8_t out[4]) { glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, out); }

GLuint compileProgram(const char* vs, const char* fs, std::string* log) {
    GLuint p = glCreateProgram();
    const char* srcs[2] = {vs, fs};
    GLenum types[2] = {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER};
    for (int i = 0; i < 2; i++) {
        GLuint s = glCreateShader(types[i]);
        glShaderSource(s, 1, &srcs[i], nullptr);
        glCompileShader(s);
        GLint ok = 0;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char buf[4096] = {};
            glGetShaderInfoLog(s, sizeof buf, nullptr, buf);
            fprintf(stderr, "  shader compile failed: %s\n", buf);
            if (log) *log = buf;
        }
        glAttachShader(p, s);
        glDeleteShader(s);
    }
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char buf[4096] = {};
        glGetProgramInfoLog(p, sizeof buf, nullptr, buf);
        fprintf(stderr, "  link failed: %s\n", buf);
        if (log) *log = buf;
    }
    return p;
}

int mglRunTests(int argc, char** argv) {
    EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (!eglInitialize(dpy, nullptr, nullptr)) {
        fprintf(stderr, "eglInitialize failed\n");
        return 2;
    }
    eglBindAPI(EGL_OPENGL_API);
    EGLint cfgAttr[] = {EGL_RED_SIZE, 8, EGL_DEPTH_SIZE, 24, EGL_STENCIL_SIZE, 8, EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_NONE};
    EGLConfig cfg;
    EGLint n;
    eglChooseConfig(dpy, cfgAttr, &cfg, 1, &n);
    int run = 0, failedTests = 0;
    for (auto& t : testRegistry()) {
        if (argc > 1) {
            bool match = false;
            for (int i = 1; i < argc; i++)
                if (strstr(t.name, argv[i])) match = true;
            if (!match) continue;
        }
        EGLint pb[] = {EGL_WIDTH, t.width, EGL_HEIGHT, t.height, EGL_NONE};
        EGLSurface s = eglCreatePbufferSurface(dpy, cfg, pb);
        EGLint ctxAttr[] = {EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 6, EGL_CONTEXT_OPENGL_PROFILE_MASK,
                            EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT, EGL_NONE};
        EGLContext c = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ctxAttr);
        eglMakeCurrent(dpy, s, s, c);
        g_currentTest = t.name;
        g_testFailures = 0;
        t.fn();
        glFinish();
        GLenum e = glGetError();
        if (e) testFail(__FILE__, __LINE__, "trailing GL error " + std::to_string(e));
        eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglDestroyContext(dpy, c);
        eglDestroySurface(dpy, s);
        run++;
        if (g_testFailures) failedTests++;
        printf("%s %s\n", g_testFailures ? "FAIL" : "ok  ", t.name);
    }
    printf("\n%d tests, %d failed\n", run, failedTests);
    return failedTests ? 1 : 0;
}
