// A stand-in for CrossOver's winemac.so (wine mode test, see wine_mode_test.mm):
// linked against libmetalgldrv the way scripts/crossover-install.sh relinks the real
// one, and doing what Wine's dlls/winemac.drv does with OpenGL: CGL imports,
// dlopen(OpenGL.framework, RTLD_NOLOAD) + dlsym for gl*, and an NSOpenGLContext
// subclass made with -initWithCGLContextObj:. Built without ARC, like Wine.
#define GL_SILENCE_DEPRECATION 1
#import <AppKit/AppKit.h>
#include <OpenGL/OpenGL.h>
#include <OpenGL/gl3.h>
#include <dlfcn.h>
#include <cstring>

#pragma clang diagnostic ignored "-Wdeprecated-declarations"

@interface FakeWineOpenGLContext : NSOpenGLContext {
    GLint backing_size[2];
}
@end
@implementation FakeWineOpenGLContext
- (void)setView:(NSView*)v {
    if ([NSThread isMainThread]) [super setView:v];
    else dispatch_sync(dispatch_get_main_queue(), ^{ [super setView:v]; });
}
@end

// Returns 0 on success, else the number of the failed step.
extern "C" __attribute__((visibility("default"))) int fake_winemac_run(NSView* view, void** outOpenGLHandle) {
    // macdrv_OpenGLInit
    void* h = dlopen("/System/Library/Frameworks/OpenGL.framework/OpenGL", RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD);
    *outOpenGLHandle = h;
    if (!h) return 1;
    auto pglGetString = (const GLubyte* (*)(GLenum))dlsym(h, "glGetString");
    auto pglClear = (void (*)(GLbitfield))dlsym(h, "glClear");
    auto pglClearColor = (void (*)(GLfloat, GLfloat, GLfloat, GLfloat))dlsym(h, "glClearColor");
    auto pglReadPixels = (void (*)(GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void*))dlsym(h, "glReadPixels");
    if (!pglGetString || !pglClear || !pglClearColor || !pglReadPixels || !dlsym(h, "glCopyColorTable")) return 2;

    // init_gl_info: a GL4 core probe context.
    CGLPixelFormatAttribute attribs[] = {kCGLPFADisplayMask, (CGLPixelFormatAttribute)CGDisplayIDToOpenGLDisplayMask(CGMainDisplayID()),
                                         kCGLPFAAccelerated, kCGLPFAOpenGLProfile, (CGLPixelFormatAttribute)kCGLOGLPVersion_GL4_Core,
                                         (CGLPixelFormatAttribute)0};
    CGLPixelFormatObj pix = nullptr;
    GLint screens = 0;
    if (CGLChoosePixelFormat(attribs, &pix, &screens) || !pix) return 3;
    CGLContextObj cgl = nullptr;
    if (CGLCreateContext(pix, nullptr, &cgl) || !cgl) return 4;
    CGLReleasePixelFormat(pix);
    CGLContextObj old = CGLGetCurrentContext();
    if (CGLSetCurrentContext(cgl)) return 5;
    const char* ver = (const char*)pglGetString(GL_VERSION);
    int major = 0, minor = 0;
    if (!ver || sscanf(ver, "%d.%d", &major, &minor) != 2 || major != 4 || minor < 5) return 6;
    CGLSetCurrentContext(old);

    // macdrv_create_opengl_context + macdrv_make_context_current (on a "Win32" thread).
    FakeWineOpenGLContext* ctx = [[FakeWineOpenGLContext alloc] initWithCGLContextObj:cgl];
    if (!ctx || ctx.CGLContextObj != cgl) return 7;
    __block int result = 0;
    dispatch_group_t done = dispatch_group_create();
    dispatch_group_async(done, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        GLint size[2] = {64, 32};
        CGLSetParameter(cgl, kCGLCPSurfaceBackingSize, size);
        CGLEnable(cgl, kCGLCESurfaceBackingSize);
        [ctx setView:view];
        [ctx makeCurrentContext];
        CGLSetCurrentContext(cgl);
        pglClearColor(1, 0, 1, 1);
        pglClear(GL_COLOR_BUFFER_BIT);
        unsigned char px[4] = {};
        pglReadPixels(63, 31, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
        if (px[0] != 255 || px[1] != 0 || px[2] != 255) result = 8;
        [ctx flushBuffer];
        [NSOpenGLContext clearCurrentContext];
        CGLSetCurrentContext(nullptr);
    });
    // The main thread services setView: while the GL thread waits on it.
    while (dispatch_group_wait(done, DISPATCH_TIME_NOW))
        [[NSRunLoop currentRunLoop] runMode:NSDefaultRunLoopMode beforeDate:[NSDate dateWithTimeIntervalSinceNow:0.01]];
    dispatch_release(done);
    if (result) return result;
    [ctx clearDrawable];
    [ctx release];
    CGLReleaseContext(cgl);
    return 0;
}
