// Window-surface tests (iOS runner only: uses the runner's CAMetalLayer).
#include "harness.h"
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#import <QuartzCore/CAMetalLayer.h>
#include <chrono>
extern CAMetalLayer* g_runnerLayer;

TEST(window_swap) {
    if (!g_runnerLayer) return;
    EGLDisplay dpy = eglGetCurrentDisplay();
    EGLContext ctx = eglGetCurrentContext();
    EGLint cfgAttr[] = {EGL_RED_SIZE, 8, EGL_DEPTH_SIZE, 24, EGL_SURFACE_TYPE, EGL_WINDOW_BIT, EGL_NONE};
    EGLConfig cfg;
    EGLint n;
    eglChooseConfig(dpy, cfgAttr, &cfg, 1, &n);
    EGLSurface win = eglCreateWindowSurface(dpy, cfg, (__bridge EGLNativeWindowType)g_runnerLayer, nullptr);
    CHECK(win != EGL_NO_SURFACE);
    if (!win) return;
    EGLSurface oldDraw = eglGetCurrentSurface(EGL_DRAW);
    CHECK(eglMakeCurrent(dpy, win, win, ctx));
    EGLint w = 0, h = 0;
    eglQuerySurface(dpy, win, EGL_WIDTH, &w);
    eglQuerySurface(dpy, win, EGL_HEIGHT, &h);
    printf("  window surface %dx%d\n", w, h);
    CHECK(w > 0 && h > 0);
    glViewport(0, 0, w, h);
    auto t0 = std::chrono::steady_clock::now();
    const int frames = 240;
    for (int i = 0; i < frames; i++) {
        float t = i / (float)frames;
        glClearColor(t, 0.3f, 1 - t, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        // a scissored band moving up the screen exercises the quad-clear path every frame
        glEnable(GL_SCISSOR_TEST);
        glScissor(0, (int)(t * h), w, h / 10);
        glClearColor(1, 1, 1, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        glDisable(GL_SCISSOR_TEST);
        CHECK(eglSwapBuffers(dpy, win));
    }
    auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("  %d frames in %.2fs (%.1f fps)\n", frames, dt, frames / dt);
    CHECK_PIXEL(0, 0, 255, 77, 0, 255);  // last frame bottom-left: blue faded to ~0, red ~1
    eglMakeCurrent(dpy, oldDraw, oldDraw, ctx);
    eglDestroySurface(dpy, win);
    CHECK_GL();
}
#endif
