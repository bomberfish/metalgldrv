// Minimal test harness: each test runs in a fresh EGL pbuffer context.
#pragma once
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include "glapi.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <strings.h>
#include <functional>
#include <string>
#include <vector>

struct TestCase {
    const char* name;
    void (*fn)();
    int width, height;
};
std::vector<TestCase>& testRegistry();
struct TestRegistrar {
    TestRegistrar(const char* n, void (*f)(), int w = 64, int h = 64) { testRegistry().push_back({n, f, w, h}); }
};
#define TEST(name) \
    static void test_##name(); \
    static TestRegistrar reg_##name(#name, test_##name); \
    static void test_##name()
#define TEST_SIZE(name, w, h) \
    static void test_##name(); \
    static TestRegistrar reg_##name(#name, test_##name, w, h); \
    static void test_##name()

extern int g_failures;
extern const char* g_currentTest;
void testFail(const char* file, int line, const std::string& msg);

#define CHECK(cond) do { if (!(cond)) testFail(__FILE__, __LINE__, "CHECK(" #cond ")"); } while (0)
#define CHECK_EQ(a, b) do { auto _a = (a); auto _b = (b); if (!(_a == _b)) testFail(__FILE__, __LINE__, std::string(#a " == " #b " (") + std::to_string((long long)_a) + " vs " + std::to_string((long long)_b) + ")"); } while (0)
#define CHECK_GL() do { GLenum _e = glGetError(); if (_e) testFail(__FILE__, __LINE__, "GL error 0x" + std::to_string(_e)); } while (0)

// Reads one RGBA8 pixel from the read framebuffer.
void readPixel(int x, int y, uint8_t out[4]);
bool pixelNear(int x, int y, int r, int g, int b, int a, int tol = 2);
#define CHECK_PIXEL(x, y, r, g, b, a) do { uint8_t _p[4]; readPixel(x, y, _p); if (abs(_p[0]-(r))>2||abs(_p[1]-(g))>2||abs(_p[2]-(b))>2||abs(_p[3]-(a))>2) testFail(__FILE__, __LINE__, "pixel (" + std::to_string(x) + "," + std::to_string(y) + ") = " + std::to_string(_p[0]) + "," + std::to_string(_p[1]) + "," + std::to_string(_p[2]) + "," + std::to_string(_p[3]) + " expected " + std::to_string(r) + "," + std::to_string(g) + "," + std::to_string(b) + "," + std::to_string(a)); } while (0)

// The accuracy level this process runs at, parsed like the driver (DeviceCaps::strictLevel, parseAccuracyLevel):
// 0 default, 1 strict, 2 super strict, -1 TURBOHAX, -2 SPEEDDEMON (strict takes precedence).
inline int strictLevelFromEnv() {
    auto on = [](const char* n) {
        const char* v = getenv(n);
        return v && *v && strcmp(v, "0") && strcasecmp(v, "false") && strcasecmp(v, "no");
    };
    const char* v = getenv("METALGLDRV_STRICT");
    if (on("METALGLDRV_STRICT")) {
        if (!strcasecmp(v, "super") || !strcasecmp(v, "superstrict") || !strcasecmp(v, "super-strict")) return 2;
        return atoi(v) >= 2 ? 2 : 1;
    }
    if (on("METALGL_SPEEDDEMON") || on("METALGLDRV_SPEEDDEMON")) return -2;
    if (on("METALGL_TURBOHAX") || on("METALGLDRV_TURBOHAX")) return -1;
    return 0;
}

GLuint compileProgram(const char* vs, const char* fs, std::string* log = nullptr);

int mglRunTests(int argc, char** argv);
