// macOS system-GL replacement: CGL (src/macos/cgl.mm), Mach-O import rebinding
// (src/macos/macho_rebind.cpp) and GL entry point exports. The NSOpenGL hooks and
// the JVMTI agent are not exercised here: they must stay off in mgltest.
#include <TargetConditionals.h>
#if !TARGET_OS_IPHONE
#define GL_SILENCE_DEPRECATION 1
#include "harness.h"
#include <OpenGL/OpenGL.h>
#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <mach-o/loader.h>
#include <unistd.h>
#include "gl/procs.h"
#include "macos/macho_rebind.h"

#pragma clang diagnostic ignored "-Wdeprecated-declarations"

extern "C" GLboolean gluCheckExtension(const GLubyte* extName, const GLubyte* extString);  // OpenGL.framework's GLU

namespace {

// Saves the harness's EGL binding; CGL tests make their own contexts current.
struct EGLRestore {
    EGLDisplay dpy = eglGetCurrentDisplay();
    EGLContext ctx = eglGetCurrentContext();
    EGLSurface draw = eglGetCurrentSurface(EGL_DRAW), read = eglGetCurrentSurface(EGL_READ);
    ~EGLRestore() {
        CGLSetCurrentContext(nullptr);
        eglMakeCurrent(dpy, draw, read, ctx);
    }
};

CGLPixelFormatObj choose(std::initializer_list<int> attrs, CGLError* err = nullptr) {
    std::vector<CGLPixelFormatAttribute> a;
    for (int v : attrs) a.push_back((CGLPixelFormatAttribute)v);
    a.push_back((CGLPixelFormatAttribute)0);
    CGLPixelFormatObj pix = nullptr;
    GLint n = -1;
    CGLError e = CGLChoosePixelFormat(a.data(), &pix, &n);
    if (err) *err = e;
    return pix;
}

GLint describe(CGLPixelFormatObj pix, int attr) {
    GLint v = -1;
    CHECK_EQ(CGLDescribePixelFormat(pix, 0, (CGLPixelFormatAttribute)attr, &v), kCGLNoError);
    return v;
}

}  // namespace

TEST(cgl_pixel_format) {
    CGLPixelFormatObj pix = choose({kCGLPFAAccelerated, kCGLPFADoubleBuffer, kCGLPFAColorSize, 24, kCGLPFAAlphaSize, 8,
                                    kCGLPFADepthSize, 24, kCGLPFAStencilSize, 8, kCGLPFAOpenGLProfile, kCGLOGLPVersion_GL4_Core});
    CHECK(pix);
    if (!pix) return;
    CHECK_EQ(describe(pix, kCGLPFAColorSize), 32);
    CHECK_EQ(describe(pix, kCGLPFAAlphaSize), 8);
    CHECK_EQ(describe(pix, kCGLPFADepthSize), 24);
    CHECK_EQ(describe(pix, kCGLPFAStencilSize), 8);
    CHECK_EQ(describe(pix, kCGLPFADoubleBuffer), 1);
    CHECK_EQ(describe(pix, kCGLPFAAccelerated), 1);
    CHECK_EQ(describe(pix, kCGLPFASampleBuffers), 0);
    CHECK_EQ(describe(pix, kCGLPFAOpenGLProfile), kCGLOGLPVersion_GL4_Core);
    CHECK_EQ(CGLGetPixelFormatRetainCount(pix), 1u);
    CGLRetainPixelFormat(pix);
    CHECK_EQ(CGLGetPixelFormatRetainCount(pix), 2u);
    CGLReleasePixelFormat(pix);
    CGLReleasePixelFormat(pix);

    // No alpha, 16-bit depth rounds up, 3 samples rounds to 4 (Wine's enumeration and GLFW's requests).
    pix = choose({kCGLPFAColorSize, 24, kCGLPFAAlphaSize, 0, kCGLPFADepthSize, 16, kCGLPFASampleBuffers, 1, kCGLPFASamples, 3});
    CHECK(pix);
    if (pix) {
        CHECK_EQ(describe(pix, kCGLPFAColorSize), 24);
        CHECK_EQ(describe(pix, kCGLPFAAlphaSize), 0);
        CHECK_EQ(describe(pix, kCGLPFADepthSize), 16);
        CHECK_EQ(describe(pix, kCGLPFAStencilSize), 0);
        CHECK_EQ(describe(pix, kCGLPFASamples), 4);
        CHECK_EQ(describe(pix, kCGLPFAOpenGLProfile), kCGLOGLPVersion_Legacy);
        CGLReleasePixelFormat(pix);
    }

    // No match is "no error, NULL" (CGL convention); unknown attributes are errors.
    CGLError err;
    CHECK(!choose({kCGLPFAStereo}, &err));
    CHECK_EQ(err, kCGLNoError);
    CHECK(!choose({kCGLPFAMinimumPolicy, kCGLPFAAccumSize, 32}, &err));
    CHECK_EQ(err, kCGLNoError);
    CHECK(!choose({kCGLPFARendererID, 0x00020400 /* generic float */}, &err));
    CHECK_EQ(err, kCGLNoError);
    CHECK(!choose({9999}, &err));
    CHECK_EQ(err, kCGLBadAttribute);
    // Valued attributes with value 0 don't end the list early.
    pix = choose({kCGLPFAAuxBuffers, 0, kCGLPFAStencilSize, 8, kCGLPFAClosestPolicy});
    CHECK(pix);
    if (pix) {
        CHECK_EQ(describe(pix, kCGLPFAStencilSize), 8);
        CGLReleasePixelFormat(pix);
    }
    CHECK(strcmp(CGLErrorString(kCGLBadContext), "invalid context") == 0);
}

TEST(cgl_renderer_info) {
    CGLRendererInfoObj info = nullptr;
    GLint n = 0;
    CHECK_EQ(CGLQueryRendererInfo(0xffffffff, &info, &n), kCGLNoError);
    CHECK_EQ(n, 1);
    if (!info) return;
    GLint v = 0;
    CHECK_EQ(CGLDescribeRenderer(info, 0, kCGLRPAccelerated, &v), kCGLNoError);
    CHECK_EQ(v, 1);
    CHECK_EQ(CGLDescribeRenderer(info, 0, kCGLRPMajorGLVersion, &v), kCGLNoError);
    CHECK_EQ(v, 4);
    CHECK_EQ(CGLDescribeRenderer(info, 0, kCGLRPColorModes, &v), kCGLNoError);
    CHECK(v & kCGLARGB8888Bit);
    CHECK_EQ(CGLDescribeRenderer(info, 0, kCGLRPBufferModes, &v), kCGLNoError);
    CHECK(v & kCGLDoubleBufferBit);
    CHECK_EQ(CGLDescribeRenderer(info, 0, kCGLRPVideoMemoryMegabytes, &v), kCGLNoError);
    CHECK(v > 0);
    GLint id = 0;
    CHECK_EQ(CGLDescribeRenderer(info, 0, kCGLRPRendererID, &id), kCGLNoError);
    CHECK_EQ(CGLDescribeRenderer(info, 0, (CGLRendererProperty)12345, &v), kCGLBadProperty);
    CHECK_EQ(CGLDestroyRendererInfo(info), kCGLNoError);
    // A pixel format for that renderer, as Wine asks for it.
    CGLPixelFormatObj pix = choose({kCGLPFARendererID, id & kCGLRendererIDMatchingMask, kCGLPFASingleRenderer,
                                    kCGLPFAOpenGLProfile, kCGLOGLPVersion_3_2_Core});
    CHECK(pix);
    if (pix) CGLReleasePixelFormat(pix);
}

// Wine's init_gl_info: legacy, then GL3 / GL4 core contexts, parsing GL_VERSION.
TEST(cgl_context_versions) {
    EGLRestore restore;
    for (int profile : {kCGLOGLPVersion_Legacy, kCGLOGLPVersion_GL3_Core, kCGLOGLPVersion_GL4_Core}) {
        CGLPixelFormatObj pix = choose({kCGLPFAAccelerated, kCGLPFAOpenGLProfile, profile});
        CHECK(pix);
        if (!pix) continue;
        CGLContextObj ctx = nullptr;
        CHECK_EQ(CGLCreateContext(pix, nullptr, &ctx), kCGLNoError);
        CGLReleasePixelFormat(pix);
        if (!ctx) continue;
        CGLContextObj old = CGLGetCurrentContext();
        CHECK_EQ(CGLSetCurrentContext(ctx), kCGLNoError);
        CHECK(CGLGetCurrentContext() == ctx);
        int major = 0, minor = 0;
        sscanf((const char*)glGetString(GL_VERSION), "%d.%d", &major, &minor);
        CHECK_EQ(major, 4);
        CHECK_EQ(minor, 6);
        GLint mask = 0;
        glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &mask);
        if (profile == kCGLOGLPVersion_Legacy) {
            CHECK(mask & GL_CONTEXT_COMPATIBILITY_PROFILE_BIT);
            CHECK(glGetString(GL_EXTENSIONS) != nullptr);  // Wine reads it from the legacy context
        } else {
            CHECK(mask & GL_CONTEXT_CORE_PROFILE_BIT);
        }
        CHECK_GL();
        GLint hasDrawable = -1;
        CHECK_EQ(CGLGetParameter(ctx, kCGLCPHasDrawable, &hasDrawable), kCGLNoError);
        CHECK_EQ(hasDrawable, 0);
        CGLSetCurrentContext(old);
        CHECK_EQ(CGLGetContextRetainCount(ctx), 1u);
        CGLReleaseContext(ctx);
    }
}

TEST(cgl_parameters) {
    CGLPixelFormatObj pix = choose({kCGLPFADoubleBuffer});
    CGLContextObj ctx = nullptr;
    CHECK_EQ(CGLCreateContext(pix, nullptr, &ctx), kCGLNoError);
    CGLReleasePixelFormat(pix);
    if (!ctx) return;
    GLint v = -1;
    CHECK_EQ(CGLGetParameter(ctx, kCGLCPSwapInterval, &v), kCGLNoError);
    CHECK_EQ(v, 0);
    GLint one = 1;
    CHECK_EQ(CGLSetParameter(ctx, kCGLCPSwapInterval, &one), kCGLNoError);
    CHECK_EQ(CGLGetParameter(ctx, kCGLCPSwapInterval, &v), kCGLNoError);
    CHECK_EQ(v, 1);
    GLint size[2] = {640, 480}, got[2] = {0, 0};
    CHECK_EQ(CGLSetParameter(ctx, kCGLCPSurfaceBackingSize, size), kCGLNoError);
    CHECK_EQ(CGLIsEnabled(ctx, kCGLCESurfaceBackingSize, &v), kCGLNoError);
    CHECK_EQ(v, 0);
    CHECK_EQ(CGLEnable(ctx, kCGLCESurfaceBackingSize), kCGLNoError);
    CHECK_EQ(CGLIsEnabled(ctx, kCGLCESurfaceBackingSize, &v), kCGLNoError);
    CHECK_EQ(v, 1);
    CHECK_EQ(CGLGetParameter(ctx, kCGLCPSurfaceBackingSize, got), kCGLNoError);
    CHECK_EQ(got[0], 640);
    CHECK_EQ(got[1], 480);
    CHECK_EQ(CGLDisable(ctx, kCGLCESurfaceBackingSize), kCGLNoError);
    GLint zero = 0;
    CHECK_EQ(CGLSetParameter(ctx, kCGLCPSurfaceOpacity, &zero), kCGLNoError);
    CHECK_EQ(CGLGetParameter(ctx, kCGLCPSurfaceOpacity, &v), kCGLNoError);
    CHECK_EQ(v, 0);
    CHECK_EQ(CGLGetParameter(ctx, kCGLCPCurrentRendererID, &v), kCGLNoError);
    CHECK_EQ(CGLGetVirtualScreen(ctx, &v), kCGLNoError);
    CHECK_EQ(v, 0);
    CHECK_EQ(CGLLockContext(ctx), kCGLNoError);
    CHECK_EQ(CGLUnlockContext(ctx), kCGLNoError);
    CHECK_EQ(CGLSetCurrentContext((CGLContextObj)&v), kCGLBadContext);  // not ours
    CGLReleaseContext(ctx);
}

// Wine's pbuffers: render into one, then bind it as a texture in a shared context.
TEST(cgl_pbuffer) {
    EGLRestore restore;
    CGLPixelFormatObj pix = choose({kCGLPFAAccelerated, kCGLPFADoubleBuffer, kCGLPFAColorSize, 32, kCGLPFAAlphaSize, 8,
                                    kCGLPFADepthSize, 24, kCGLPFAStencilSize, 8});
    CHECK(pix);
    if (!pix) return;
    CGLContextObj ctx = nullptr, ctx2 = nullptr;
    CHECK_EQ(CGLCreateContext(pix, nullptr, &ctx), kCGLNoError);
    CHECK_EQ(CGLCreateContext(pix, ctx, &ctx2), kCGLNoError);
    CGLReleasePixelFormat(pix);
    CGLPBufferObj pb = nullptr;
    CHECK_EQ(CGLCreatePBuffer(32, 16, GL_TEXTURE_2D, GL_RGBA, 0, &pb), kCGLNoError);
    if (!ctx || !ctx2 || !pb) return;
    GLsizei w = 0, h = 0;
    GLenum target = 0, fmt = 0;
    GLint mip = -1;
    CHECK_EQ(CGLDescribePBuffer(pb, &w, &h, &target, &fmt, &mip), kCGLNoError);
    CHECK_EQ(w, 32);
    CHECK_EQ(h, 16);
    CHECK_EQ(target, (GLenum)GL_TEXTURE_2D);

    CHECK_EQ(CGLSetCurrentContext(ctx), kCGLNoError);
    CHECK_EQ(CGLSetPBuffer(ctx, pb, 0, 0, 0), kCGLNoError);
    GLint hasDrawable = 0;
    CGLGetParameter(ctx, kCGLCPHasDrawable, &hasDrawable);
    CHECK_EQ(hasDrawable, 1);
    GLint vp[4] = {};
    glGetIntegerv(GL_VIEWPORT, vp);
    CHECK_EQ(vp[2], 32);
    CHECK_EQ(vp[3], 16);
    glClearColor(1, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_SCISSOR_TEST);
    glScissor(0, 0, 8, 8);
    glClearColor(0, 0, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_SCISSOR_TEST);
    CHECK_PIXEL(20, 10, 255, 0, 0, 255);
    CHECK_PIXEL(2, 2, 0, 0, 255, 255);
    CHECK_GL();
    CHECK_EQ(CGLFlushDrawable(ctx), kCGLNoError);

    CHECK_EQ(CGLSetCurrentContext(ctx2), kCGLNoError);
    GLuint tex = 0, fbo = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    CHECK_EQ(CGLTexImagePBuffer(ctx2, pb, GL_FRONT), kCGLNoError);
    GLint tw = 0;
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &tw);
    CHECK_EQ(tw, 32);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    CHECK_EQ(glCheckFramebufferStatus(GL_FRAMEBUFFER), (GLenum)GL_FRAMEBUFFER_COMPLETE);
    CHECK_PIXEL(20, 10, 255, 0, 0, 255);
    CHECK_PIXEL(2, 2, 0, 0, 255, 255);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    CHECK_GL();

    CHECK_EQ(CGLClearDrawable(ctx), kCGLNoError);
    CGLSetCurrentContext(nullptr);
    CHECK(CGLGetCurrentContext() == nullptr);
    CGLReleasePBuffer(pb);
    CGLReleaseContext(ctx2);
    CGLReleaseContext(ctx);
}

TEST(glu_check_extension) {
    const GLubyte* s = (const GLubyte*)"GL_ARB_foo GL_ARB_foo_bar GL_EXT_x";
    CHECK(gluCheckExtension((const GLubyte*)"GL_ARB_foo", s));
    CHECK(gluCheckExtension((const GLubyte*)"GL_ARB_foo_bar", s));
    CHECK(gluCheckExtension((const GLubyte*)"GL_EXT_x", s));
    CHECK(!gluCheckExtension((const GLubyte*)"GL_ARB_fo", s));
    CHECK(!gluCheckExtension((const GLubyte*)"GL_EXT", s));
}

// Every GL entry point in the GetProcAddress table is exported by name, so dlsym on
// our handle (Wine, LWJGL's org.lwjgl.opengl.libname) finds the same function.
TEST(gl_exports_dlsym) {
    Dl_info info;
    CHECK(dladdr((const void*)&glGetString, &info) && info.dli_fname);
    void* h = dlopen(info.dli_fname, RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD);
    CHECK(h);
    if (!h) return;
    unsigned missing = 0, mismatched = 0;
    for (unsigned i = 0; i < mgl_gl_procs_count; i++) {
        void* p = dlsym(h, mgl_gl_procs[i].name);
        if (!p) {
            if (missing++ < 5) printf("  not exported: %s\n", mgl_gl_procs[i].name);
        } else if (p != (void*)mgl_gl_procs[i].fn) {
            if (mismatched++ < 5) printf("  dlsym mismatch: %s\n", mgl_gl_procs[i].name);
        }
    }
    CHECK_EQ(missing, 0u);
    CHECK_EQ(mismatched, 0u);
    CHECK(mgl_gl_procs_count > 2000);
    for (const char* n : {"glDrawArraysIndirect", "glTexStorage2D", "glClipControl", "glCreateBuffers", "glSpecializeShader",
                          "glMultiDrawArraysIndirectCount", "glPolygonOffsetClamp", "glBufferStorage", "glDispatchCompute",
                          "glGetTextureSubImage", "glCopyColorTable", "CGLChoosePixelFormat", "CGLTexImagePBuffer",
                          "gluCheckExtension", "eglGetProcAddress", "Agent_OnLoad"})
        CHECK(dlsym(h, n) != nullptr);
    dlclose(h);
}

namespace {
pid_t fakeGetppid() { return 424242; }
struct RebindProbe {
    void* replacement;
    void* original = nullptr;
    const char* library = nullptr;
    int seen = 0;
};
void* probeResolver(const char* symbol, const char* library, void* current, void* user) {
    auto* p = (RebindProbe*)user;
    if (strcmp(symbol, "getppid") != 0) return nullptr;
    p->seen++;
    p->original = current;
    p->library = library;
    return p->replacement;
}
}  // namespace

// Rebinds an import of this executable (a __got slot in read-only __DATA_CONST with
// chained fixups on arm64) and back.
TEST(macho_rebind_import) {
    const mach_header* exe = nullptr;
    intptr_t slide = 0;
    for (uint32_t i = 0; i < _dyld_image_count(); i++)
        if (_dyld_get_image_header(i)->filetype == MH_EXECUTE) {
            exe = _dyld_get_image_header(i);
            slide = _dyld_get_image_vmaddr_slide(i);
            break;
        }
    CHECK(exe);
    if (!exe) return;
    pid_t real = getppid();
    CHECK(real != 424242);
    RebindProbe p{(void*)&fakeGetppid};
    int n = mgl::macho::rebindImage(exe, slide, probeResolver, &p);
    CHECK(n >= 1);
    CHECK(p.seen >= 1);
    CHECK(p.library && strstr(p.library, "libSystem"));
    CHECK_EQ(getppid(), 424242);
    RebindProbe back{p.original};
    CHECK(mgl::macho::rebindImage(exe, slide, probeResolver, &back) >= 1);
    CHECK_EQ(getppid(), real);
}

TEST(macho_paths) {
    using namespace mgl::macho;
    CHECK(isOpenGLFrameworkPath("/System/Library/Frameworks/OpenGL.framework/OpenGL"));
    CHECK(isOpenGLFrameworkPath("/System/Library/Frameworks/OpenGL.framework/Versions/A/OpenGL"));
    CHECK(isOpenGLFrameworkPath("/System/Library/Frameworks/OpenGL.framework/Versions/Current/OpenGL"));
    CHECK(isOpenGLFrameworkPath("/System/Library/Frameworks/OpenGL.framework/Libraries/libGL.dylib"));
    CHECK(!isOpenGLFrameworkPath("/System/Library/Frameworks/OpenGL.framework/Resources/Info.plist"));
    CHECK(!isOpenGLFrameworkPath("/usr/lib/libGL.dylib"));
    CHECK(!isOpenGLFrameworkPath("OpenGL"));
    CHECK(!isOpenGLFrameworkPath(nullptr));
    CHECK(isSystemImagePath("/System/Library/Frameworks/AppKit.framework/Versions/C/AppKit"));
    CHECK(isSystemImagePath("/usr/lib/libobjc.A.dylib"));
    CHECK(!isSystemImagePath("/Users/x/Library/Application Support/metalgldrv/libmetalgldrv.dylib"));
    CHECK(!isSystemImagePath("/var/folders/ab/T/lwjgl_x/3.3.3/libglfw.dylib"));
}

#endif  // !TARGET_OS_IPHONE
