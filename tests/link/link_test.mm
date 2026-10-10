// An app ported by linking libmetalgldrv in place of OpenGL.framework (docs/porting.md): ordinary NSOpenGL and
// GL code, no metalgldrv-specific calls, no environment. The driver activates its NSOpenGL replacement when it
// sees the main executable link it, so the context below must be metalgldrv's.
#import <AppKit/AppKit.h>
#define GL_GLEXT_PROTOTYPES 1
#include <GL/glcorearb.h>
#include <cstdio>
#include <cstring>
#include "metalgldrv/metalgldrv.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); failures++; } } while (0)

int main() {
    @autoreleasepool {
        [NSApplication sharedApplication];
        printf("  metalgldrv %s, supported %d, active %d\n", metalgldrv_version(), metalgldrv_is_supported(), metalgldrv_is_active());
        CHECK(metalgldrv_is_supported());
        CHECK(metalgldrv_is_active());  // activated at load: the main executable links the driver
        CHECK(metalgldrv_activate() == 1);  // idempotent

        NSOpenGLPixelFormatAttribute attrs[] = {NSOpenGLPFAOpenGLProfile, NSOpenGLProfileVersion4_1Core, NSOpenGLPFADoubleBuffer,
                                                NSOpenGLPFAColorSize, 24, NSOpenGLPFADepthSize, 24, 0};
        NSOpenGLPixelFormat* pf = [[NSOpenGLPixelFormat alloc] initWithAttributes:attrs];
        CHECK(pf != nil);
        NSOpenGLContext* ctx = [[NSOpenGLContext alloc] initWithFormat:pf shareContext:nil];
        CHECK(ctx != nil);
        if (!ctx) return 1;
        NSWindow* win = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, 160, 120) styleMask:NSWindowStyleMaskTitled
                                                      backing:NSBackingStoreBuffered defer:NO];
        NSView* view = win.contentView;
        view.wantsBestResolutionOpenGLSurface = YES;
        [ctx setView:view];
        [ctx makeCurrentContext];

        const char* renderer = (const char*)glGetString(GL_RENDERER);
        const char* version = (const char*)glGetString(GL_VERSION);
        printf("  GL_RENDERER %s\n  GL_VERSION %s\n", renderer ? renderer : "(null)", version ? version : "(null)");
        CHECK(renderer && strstr(renderer, "metalgldrv"));
        CHECK(version && strncmp(version, "4.6", 3) == 0);

        glClearColor(0, 0, 1, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        unsigned char px[4] = {};
        glReadPixels(2, 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
        CHECK(px[0] == 0 && px[1] == 0 && px[2] == 255);
        [ctx flushBuffer];
        CHECK(glGetError() == GL_NO_ERROR);
        [NSOpenGLContext clearCurrentContext];
    }
    printf(failures ? "FAILED (%d failures)\n" : "ok\n", failures);
    return failures ? 1 : 0;
}
