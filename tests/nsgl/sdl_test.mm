// sdltest: the macOS replacement mode under SDL 3's Cocoa OpenGL backend, which
// Minecraft 26.3+ uses instead of GLFW (through LWJGL's libSDL3.dylib).
//
//   sdltest /path/to/libSDL3.dylib     (e.g. unzipped from lwjgl-sdl-<ver>-natives-macos-arm64.jar)
//
// Like nsgltest it turns the hooks on (METALGLDRV_HOOK_NSOPENGL=1) before loading
// libmetalgldrv, then does what Minecraft's GL backend does: SDL_GL_LoadLibrary of the
// driver, a 3.3 core forward-compatible sRGB context on an SDL window (hidden, never
// shown), the glGetError identity check between LWJGL's lookup (eglGetProcAddress in
// LWJGL 3.4) and SDL_GL_GetProcAddress, and vsync'd swaps: SDL waits for its
// CVDisplayLink, which it points at the context's display through
// CVDisplayLinkSetCurrentCGDisplayFromOpenGLContext (redirected by the hooks).
#define GL_SILENCE_DEPRECATION 1
#import <Foundation/Foundation.h>
#include <OpenGL/gl3.h>
#include <dlfcn.h>
#include <unistd.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static int g_failures = 0;
#define CHECK(c) \
    do { \
        if (!(c)) { \
            printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            g_failures++; \
        } \
    } while (0)

// SDL 3 API subset (SDL_video.h / SDL_init.h, release-3.4.x).
constexpr uint32_t SDL_INIT_VIDEO = 0x20;
constexpr uint64_t SDL_WINDOW_OPENGL = 0x2, SDL_WINDOW_HIDDEN = 0x8;
enum { SDL_GL_CONTEXT_MAJOR_VERSION = 17, SDL_GL_CONTEXT_MINOR_VERSION = 18, SDL_GL_CONTEXT_FLAGS = 19,
       SDL_GL_CONTEXT_PROFILE_MASK = 20, SDL_GL_FRAMEBUFFER_SRGB_CAPABLE = 22 };
constexpr int SDL_GL_CONTEXT_PROFILE_CORE = 1, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG = 2;

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s /path/to/libSDL3.dylib\n", argv[0]);
        return 2;
    }
    alarm(60);  // a vsync wait that never ends fails the test instead of hanging it
    setenv("METALGLDRV_HOOK_NSOPENGL", "1", 1);
    void* drv = dlopen(MGL_DYLIB_PATH, RTLD_NOW | RTLD_LOCAL);
    if (!drv) {
        printf("FAIL: dlopen %s: %s\n", MGL_DYLIB_PATH, dlerror());
        return 1;
    }
    void* sdl = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!sdl) {
        printf("FAIL: dlopen %s: %s\n", argv[1], dlerror());
        return 1;
    }
#define SDLFN(ret, name, ...) auto name = (ret(*)(__VA_ARGS__))dlsym(sdl, #name)
    SDLFN(bool, SDL_Init, uint32_t);
    SDLFN(void, SDL_Quit, void);
    SDLFN(const char*, SDL_GetError, void);
    SDLFN(bool, SDL_GL_LoadLibrary, const char*);
    SDLFN(void*, SDL_GL_GetProcAddress, const char*);
    SDLFN(bool, SDL_GL_SetAttribute, int, int);
    SDLFN(void*, SDL_CreateWindow, const char*, int, int, uint64_t);
    SDLFN(void, SDL_DestroyWindow, void*);
    SDLFN(void*, SDL_GL_CreateContext, void*);
    SDLFN(bool, SDL_GL_MakeCurrent, void*, void*);
    SDLFN(bool, SDL_GL_SetSwapInterval, int);
    SDLFN(bool, SDL_GL_SwapWindow, void*);
    SDLFN(bool, SDL_GL_DestroyContext, void*);
#undef SDLFN
    if (!SDL_Init || !SDL_GL_LoadLibrary || !SDL_CreateWindow || !SDL_GL_CreateContext || !SDL_GL_SwapWindow ||
        !SDL_GL_DestroyContext) {
        printf("FAIL: %s is not SDL 3\n", argv[1]);
        return 1;
    }
    @autoreleasepool {
        CHECK(SDL_Init(SDL_INIT_VIDEO));
        bool loaded = SDL_GL_LoadLibrary(MGL_DYLIB_PATH);
        if (!loaded) printf("  SDL_GL_LoadLibrary: %s\n", SDL_GetError());
        CHECK(loaded);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
        SDL_GL_SetAttribute(SDL_GL_FRAMEBUFFER_SRGB_CAPABLE, 1);
        void* win = SDL_CreateWindow("sdltest", 320, 240, SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
        if (!win) printf("  SDL_CreateWindow: %s\n", SDL_GetError());
        CHECK(win);
        void* ctx = win ? SDL_GL_CreateContext(win) : nullptr;
        if (!ctx) printf("  SDL_GL_CreateContext: %s\n", SDL_GetError());
        CHECK(ctx);
        if (ctx) {
            CHECK(SDL_GL_MakeCurrent(win, ctx));
            // Minecraft's GlBackend: LWJGL's provider and SDL must agree on glGetError.
            auto eglGetProcAddress = (void* (*)(const char*))dlsym(drv, "eglGetProcAddress");
            void* viaSDL = SDL_GL_GetProcAddress("glGetError");
            CHECK(viaSDL && viaSDL == dlsym(drv, "glGetError"));
            CHECK(eglGetProcAddress && eglGetProcAddress("glGetError") == viaSDL);
            auto getString = (const GLubyte* (*)(GLenum))SDL_GL_GetProcAddress("glGetString");
            auto clearColor = (void (*)(GLfloat, GLfloat, GLfloat, GLfloat))SDL_GL_GetProcAddress("glClearColor");
            auto clear = (void (*)(GLbitfield))SDL_GL_GetProcAddress("glClear");
            auto getError = (GLenum(*)())viaSDL;
            const char* version = getString ? (const char*)getString(GL_VERSION) : nullptr;
            printf("  GL_VERSION %s\n", version ? version : "(null)");
            CHECK(version && strstr(version, "metalgldrv"));
            CHECK(SDL_GL_SetSwapInterval(1));
            const int frames = 20;
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < frames; i++) {
                clearColor(i / (float)frames, 0.2f, 0.4f, 1.0f);
                clear(GL_COLOR_BUFFER_BIT);
                CHECK(SDL_GL_SwapWindow(win));
            }
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            printf("  %d vsync'd swaps in %.1f ms (%.1f Hz)\n", frames, ms, frames * 1000.0 / ms);
            CHECK(ms > frames * 1000.0 / 500.0);  // they did wait for the display link
            CHECK(getError() == GL_NO_ERROR);
            CHECK(SDL_GL_SetSwapInterval(0));
            for (int i = 0; i < 3; i++) CHECK(SDL_GL_SwapWindow(win));
            SDL_GL_MakeCurrent(win, nullptr);
            CHECK(SDL_GL_DestroyContext(ctx));
        }
        if (win) SDL_DestroyWindow(win);
        SDL_Quit();
    }
    printf("%s (%d failure%s)\n", g_failures ? "FAIL" : "PASS", g_failures, g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
