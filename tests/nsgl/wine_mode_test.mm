// winemodetest: Wine mode (src/macos/hooks.mm) without Wine. Loads a fake
// winemac.so (fake_winemac.mm) that links libmetalgldrv like a patched CrossOver
// one; libmetalgldrv must notice it, redirect its dlopen of OpenGL.framework to
// itself and hook NSOpenGLContext, with no environment variable or agent. The
// window is never shown.
#define GL_SILENCE_DEPRECATION 1
#import <AppKit/AppKit.h>
#include <dlfcn.h>
#include <cstdio>

int main() {
    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyProhibited];
        NSWindow* win = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, 200, 100)
                                                    styleMask:NSWindowStyleMaskBorderless
                                                      backing:NSBackingStoreBuffered
                                                        defer:NO];
        void* wm = dlopen(MGL_FAKE_WINEMAC_PATH, RTLD_NOW | RTLD_LOCAL);
        if (!wm) {
            printf("FAILED: cannot load %s: %s\n", MGL_FAKE_WINEMAC_PATH, dlerror());
            return 1;
        }
        auto run = (int (*)(NSView*, void**))dlsym(wm, "fake_winemac_run");
        void* glHandle = nullptr;
        int step = run(win.contentView, &glHandle);
        void* drv = dlopen(MGL_DYLIB_PATH, RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD);
        bool redirected = glHandle && glHandle == drv;
        printf("  winemac dlopen(OpenGL.framework) -> %s\n", redirected ? "metalgldrv" : glHandle ? "something else" : "NULL");
        [win close];
        if (step || !redirected) {
            printf("FAILED (step %d)\n", step);
            return 1;
        }
    }
    printf("ok\n");
    return 0;
}
