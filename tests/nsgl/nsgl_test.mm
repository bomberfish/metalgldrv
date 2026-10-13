// nsgltest: the macOS replacement mode (src/macos) against real AppKit. Separate
// from mgltest, which must run with the hooks off: this process turns them on
// (METALGLDRV_HOOK_NSOPENGL=1) before loading libmetalgldrv with dlopen, then uses
// NSOpenGLPixelFormat / NSOpenGLContext the way stock GLFW and Wine do, with GL
// entry points looked up the way GLFW does (CFBundle on com.apple.opengl, which
// the hooks redirect for this executable). The window is never shown.
//
// Built without ARC, like Wine's cocoa_opengl.m (MGLTestContext mirrors
// WineOpenGLContext: a subclass with its own ivars and -dealloc).
#define GL_SILENCE_DEPRECATION 1
#import <AppKit/AppKit.h>
#include <algorithm>
#import <QuartzCore/QuartzCore.h>
#include <OpenGL/gl3.h>
#include <dlfcn.h>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>

#pragma clang diagnostic ignored "-Wdeprecated-declarations"

static int g_failures = 0;
#define CHECK(c) \
    do { \
        if (!(c)) { \
            printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            g_failures++; \
        } \
    } while (0)

static int g_deallocs = 0;

@interface MGLTestContext : NSOpenGLContext {
    NSView* retainedView;
    GLint backing[2];
}
@end

@implementation MGLTestContext
- (void)dealloc {
    [retainedView release];
    g_deallocs++;
    [super dealloc];
}
- (void)setView:(NSView*)v {
    [super setView:v];
    [v retain];
    [retainedView release];
    retainedView = v;
}
@end

template <typename F>
static F gl(const char* name) {
    static CFBundleRef bundle = CFBundleGetBundleWithIdentifier(CFSTR("com.apple.opengl"));
    CFStringRef s = CFStringCreateWithCString(nullptr, name, kCFStringEncodingASCII);
    void* p = bundle ? CFBundleGetFunctionPointerForName(bundle, s) : nullptr;
    CFRelease(s);
    return (F)p;
}

// CGL comes from the driver (this test doesn't link OpenGL.framework or the driver).
static CGLContextObj (*pCGLGetCurrentContext)(void);
static CGLError (*pCGLSetParameter)(CGLContextObj, CGLContextParameter, const GLint*);
static CGLError (*pCGLEnable)(CGLContextObj, CGLContextEnable);
static CGLError (*pCGLDisable)(CGLContextObj, CGLContextEnable);

static CAMetalLayer* ourLayer(NSView* v) {
    for (CALayer* l in v.layer.sublayers)
        if ([l isKindOfClass:[CAMetalLayer class]] && [l.name isEqualToString:@"metalgldrv-nsgl"]) return (CAMetalLayer*)l;
    return nil;
}

// --- fake JavaVM / JVMTI / JNI function tables for Agent_OnLoad (slots per jni.h, jvmti.h) ---
namespace fakejvm {
void* jvmtiTable[160];
void* jniTable[240];
void* vmTable[8];
void* jvmtiEnv = jvmtiTable;
void* jniEnv = jniTable;
void* vm = vmTable;
int notifyMode = -1, notifyEvent = -1;
void (*vmInit)(void*, void*, void*) = nullptr;
std::map<std::string, std::string> props;  // System.setProperty calls
bool haveProperty = false;

int GetEnv(void* vm, void** penv, int version) {
    *penv = &jvmtiEnv;
    return version == 0x30010200 ? 0 : -3;
}
int SetEventNotificationMode(void* env, int mode, int event, void* thread, ...) {
    notifyMode = mode;
    notifyEvent = event;
    return 0;
}
int SetEventCallbacks(void* env, const void* callbacks, int size) {
    vmInit = ((void (**)(void*, void*, void*))callbacks)[0];
    return size >= (int)sizeof(void*) ? 0 : 103;
}
void* FindClass(void* env, const char* name) { return strcmp(name, "java/lang/System") == 0 ? (void*)0x5157 : nullptr; }
void* GetStaticMethodID(void* env, void* cls, const char* name, const char* sig) {
    if (!strcmp(name, "getProperty") && !strcmp(sig, "(Ljava/lang/String;)Ljava/lang/String;")) return (void*)1;
    if (!strcmp(name, "setProperty") && !strcmp(sig, "(Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;")) return (void*)2;
    return nullptr;
}
void* CallStaticObjectMethod(void* env, void* cls, void* mid, ...) {
    va_list ap;
    va_start(ap, mid);
    void* r = nullptr;
    if (mid == (void*)1) {
        const char* key = (const char*)va_arg(ap, void*);
        if (haveProperty && key && !strcmp(key, "org.lwjgl.opengl.libname")) r = strdup("/elsewhere/libGL.dylib");
    } else if (mid == (void*)2) {
        std::string key = (const char*)va_arg(ap, void*);
        props[key] = (const char*)va_arg(ap, void*);
    }
    va_end(ap);
    return r;
}
void* NewStringUTF(void* env, const char* s) { return strdup(s); }
const char* GetStringUTFChars(void* env, void* str, unsigned char* isCopy) { return (const char*)str; }
void ReleaseStringUTFChars(void* env, void* str, const char* chars) {}
void DeleteLocalRef(void* env, void* obj) {}
unsigned char ExceptionCheck(void* env) { return 0; }
void ExceptionClear(void* env) {}
int PushLocalFrame(void* env, int capacity) { return 0; }
void* PopLocalFrame(void* env, void* result) { return result; }

void init() {
    vmTable[6] = (void*)&GetEnv;
    jvmtiTable[1] = (void*)&SetEventNotificationMode;
    jvmtiTable[121] = (void*)&SetEventCallbacks;
    jniTable[6] = (void*)&FindClass;
    jniTable[17] = (void*)&ExceptionClear;
    jniTable[19] = (void*)&PushLocalFrame;
    jniTable[20] = (void*)&PopLocalFrame;
    jniTable[23] = (void*)&DeleteLocalRef;
    jniTable[113] = (void*)&GetStaticMethodID;
    jniTable[114] = (void*)&CallStaticObjectMethod;
    jniTable[167] = (void*)&NewStringUTF;
    jniTable[169] = (void*)&GetStringUTFChars;
    jniTable[170] = (void*)&ReleaseStringUTFChars;
    jniTable[228] = (void*)&ExceptionCheck;
}
}  // namespace fakejvm

static void testAgent(void* drv) {
    using namespace fakejvm;
    init();
    auto onLoad = (int (*)(void*, char*, void*))dlsym(drv, "Agent_OnLoad");
    CHECK(onLoad != nullptr);
    if (!onLoad) return;
    CHECK(onLoad(&vm, nullptr, nullptr) == 0);
    CHECK(notifyMode == 1 && notifyEvent == 50);  // JVMTI_ENABLE, JVMTI_EVENT_VM_INIT
    CHECK(vmInit != nullptr);
    if (!vmInit) return;
    vmInit(&jvmtiEnv, &jniEnv, nullptr);
    std::string& libname = props["org.lwjgl.opengl.libname"];
    printf("  agent: org.lwjgl.opengl.libname=%s, metalgldrv.activation=%s\n", libname.c_str(), props["metalgldrv.activation"].c_str());
    CHECK(libname.find("libmetalgldrv.dylib") != std::string::npos);
    CHECK(props["metalgldrv.activation"] == "agent");
    CHECK(!props["metalgldrv.version"].empty());
    // An existing value is left alone.
    props.clear();
    haveProperty = true;
    vmInit(&jvmtiEnv, &jniEnv, nullptr);
    CHECK(props.count("org.lwjgl.opengl.libname") == 0);
    // JNI_OnLoad (System.load) with replacement mode already on: nothing new, JNI 1.8.
    haveProperty = false;
    props.clear();
    vmTable[6] = (void*)+[](void* vm, void** penv, int version) -> int {
        *penv = &jniEnv;
        return version == 0x00010008 ? 0 : -3;
    };
    auto jniOnLoad = (int (*)(void*, void*))dlsym(drv, "JNI_OnLoad");
    CHECK(jniOnLoad != nullptr);
    if (!jniOnLoad) return;
    CHECK(jniOnLoad(&vm, nullptr) == 0x00010008);
    CHECK(props["org.lwjgl.opengl.libname"].find("libmetalgldrv.dylib") != std::string::npos);
    CHECK(props.count("metalgldrv.activation") == 0);  // the agent's report stands
    props.clear();
    CHECK(jniOnLoad(&vm, nullptr) == 0x00010008);  // once per image
    CHECK(props.empty());
}

int main(int argc, char** argv) {
    @autoreleasepool {
        setenv("METALGLDRV_HOOK_NSOPENGL", "1", 1);
        void* drv = dlopen(MGL_DYLIB_PATH, RTLD_NOW | RTLD_LOCAL);
        if (!drv) {
            printf("cannot load %s: %s\n", MGL_DYLIB_PATH, dlerror());
            return 1;
        }
        pCGLGetCurrentContext = (CGLContextObj(*)(void))dlsym(drv, "CGLGetCurrentContext");
        pCGLSetParameter = (CGLError(*)(CGLContextObj, CGLContextParameter, const GLint*))dlsym(drv, "CGLSetParameter");
        pCGLEnable = (CGLError(*)(CGLContextObj, CGLContextEnable))dlsym(drv, "CGLEnable");
        pCGLDisable = (CGLError(*)(CGLContextObj, CGLContextEnable))dlsym(drv, "CGLDisable");
        // dlopen of OpenGL.framework from this (rebound) image returns the driver.
        void* gl1 = dlopen("/System/Library/Frameworks/OpenGL.framework/OpenGL", RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD);
        CHECK(gl1 == drv);
        CHECK(dlsym(gl1, "glDrawElementsInstancedBaseVertexBaseInstance") != nullptr);

        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyProhibited];
        NSWindow* win = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, 320, 200)
                                                    styleMask:NSWindowStyleMaskBorderless
                                                      backing:NSBackingStoreBuffered
                                                        defer:NO];
        NSView* view = win.contentView;

        // GLFW's pixel format request (nsgl_context.m), 4.1 core.
        NSOpenGLPixelFormatAttribute attrs[] = {NSOpenGLPFAAccelerated, NSOpenGLPFAClosestPolicy, NSOpenGLPFAOpenGLProfile,
                                                NSOpenGLProfileVersion4_1Core, NSOpenGLPFAColorSize, 24, NSOpenGLPFAAlphaSize, 8,
                                                NSOpenGLPFADepthSize, 24, NSOpenGLPFAStencilSize, 8, NSOpenGLPFADoubleBuffer,
                                                NSOpenGLPFASampleBuffers, 0, 0};
        NSOpenGLPixelFormat* pf = [[NSOpenGLPixelFormat alloc] initWithAttributes:attrs];
        CHECK(pf != nil);
        if (!pf) return 1;
        GLint v = -1;
        [pf getValues:&v forAttribute:NSOpenGLPFADepthSize forVirtualScreen:0];
        CHECK(v == 24);
        [pf getValues:&v forAttribute:NSOpenGLPFAOpenGLProfile forVirtualScreen:0];
        CHECK(v == NSOpenGLProfileVersion4_1Core);
        CHECK(pf.numberOfVirtualScreens == 1);
        CHECK(pf.CGLPixelFormatObj != nullptr);
        NSOpenGLPixelFormatAttribute bad[] = {NSOpenGLPFAStereo, 0};
        CHECK([[NSOpenGLPixelFormat alloc] initWithAttributes:bad] == nil);

        MGLTestContext* ctx = [[MGLTestContext alloc] initWithFormat:pf shareContext:nil];
        CHECK(ctx != nil);
        if (!ctx) return 1;
        CHECK(ctx.CGLContextObj != nullptr);
        CHECK(ctx.pixelFormat == pf);
        NSOpenGLContext* shared = [[NSOpenGLContext alloc] initWithFormat:pf shareContext:ctx];
        CHECK(shared != nil);

        view.wantsBestResolutionOpenGLSurface = YES;
        [ctx setView:view];
        CHECK(ctx.view == view);
        CAMetalLayer* layer = ourLayer(view);
        CHECK(layer != nil);
        CHECK(layer && CGRectEqualToRect(layer.frame, view.layer.bounds));
        // METALGLDRV_RENDER_SCALE (replacement mode): the layer renders at that fraction of the backing scale
        // and the view reports the same scale to the app (GLFW / SDL size their framebuffer from it).
        const char* rsEnv = getenv("METALGLDRV_RENDER_SCALE");
        double rs = rsEnv && *rsEnv && atof(rsEnv) > 0 ? std::clamp(atof(rsEnv), 0.25, 1.0) : 1.0;
        CHECK(layer && layer.contentsScale == win.backingScaleFactor * rs);
        NSSize backingSize = [view convertSizeToBacking:view.bounds.size];
        CHECK(lround(backingSize.width) == lround(view.bounds.size.width * win.backingScaleFactor * rs));
        CHECK(layer && layer.opaque);

        [ctx makeCurrentContext];
        CHECK([NSOpenGLContext currentContext] == ctx);
        CHECK(pCGLGetCurrentContext() == ctx.CGLContextObj);

        auto glGetString_ = gl<const GLubyte* (*)(GLenum)>("glGetString");
        auto glGetIntegerv_ = gl<void (*)(GLenum, GLint*)>("glGetIntegerv");
        auto glClearColor_ = gl<void (*)(GLfloat, GLfloat, GLfloat, GLfloat)>("glClearColor");
        auto glClear_ = gl<void (*)(GLbitfield)>("glClear");
        auto glReadPixels_ = gl<void (*)(GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void*)>("glReadPixels");
        auto glGetError_ = gl<GLenum (*)(void)>("glGetError");
        CHECK(glGetString_ && glGetIntegerv_ && glClearColor_ && glClear_ && glReadPixels_ && glGetError_);
        if (!glGetString_ || !glGetIntegerv_ || !glClearColor_ || !glClear_ || !glReadPixels_ || !glGetError_) return 1;
        CHECK(dlsym(drv, "glGetString") == (void*)glGetString_);
        const char* ver = (const char*)glGetString_(GL_VERSION);
        printf("  GL_VERSION %s\n  GL_RENDERER %s\n", ver ? ver : "(null)", (const char*)glGetString_(GL_RENDERER));
        CHECK(ver && strncmp(ver, "4.6", 3) == 0);
        GLint mask = 0;
        glGetIntegerv_(GL_CONTEXT_PROFILE_MASK, &mask);
        CHECK(mask & GL_CONTEXT_CORE_PROFILE_BIT);
        GLint vp[4] = {};
        glGetIntegerv_(GL_VIEWPORT, vp);
        int expectW = (int)lround(view.bounds.size.width * win.backingScaleFactor * rs);
        int expectH = (int)lround(view.bounds.size.height * win.backingScaleFactor * rs);
        printf("  viewport %dx%d (expect %dx%d)\n", vp[2], vp[3], expectW, expectH);
        CHECK(vp[2] == expectW && vp[3] == expectH);
        // METALGLDRV_USE_METALFX: the drawable stays at the display's native size (the image is upscaled into it)
        if (const char* fx = getenv("METALGLDRV_USE_METALFX"); fx && atoi(fx) > 0) {
            printf("  drawable %.0fx%.0f\n", layer.drawableSize.width, layer.drawableSize.height);
            CHECK(lround(layer.drawableSize.width) == lround(view.bounds.size.width * 2.0));
        }

        glClearColor_(0, 1, 0, 1);
        glClear_(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        unsigned char px[4] = {};
        glReadPixels_(4, 4, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
        CHECK(px[0] == 0 && px[1] == 255 && px[2] == 0 && px[3] == 255);
        CHECK(glGetError_() == GL_NO_ERROR);

        // The window is sRGB-capable but stores linear RGBA8 until GL_FRAMEBUFFER_SRGB is enabled; then the
        // contents move to sRGB storage and writes are encoded.
        auto glEnable_ = gl<void (*)(GLenum)>("glEnable");
        auto glDisable_ = gl<void (*)(GLenum)>("glDisable");
        auto glScissor_ = gl<void (*)(GLint, GLint, GLsizei, GLsizei)>("glScissor");
        CHECK(layer && layer.pixelFormat == MTLPixelFormatBGRA8Unorm);
        glEnable_(GL_FRAMEBUFFER_SRGB);
        CHECK(layer && layer.pixelFormat == MTLPixelFormatBGRA8Unorm_sRGB);
        glReadPixels_(4, 4, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
        CHECK(px[0] == 0 && px[1] == 255 && px[2] == 0);  // kept across the switch
        glEnable_(GL_SCISSOR_TEST);
        glScissor_(0, 0, 8, 8);
        glClearColor_(0.5f, 0.5f, 0.5f, 1);
        glClear_(GL_COLOR_BUFFER_BIT);  // encoded: 0.5 -> 188
        glDisable_(GL_FRAMEBUFFER_SRGB);
        glReadPixels_(4, 4, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
        CHECK(px[0] >= 186 && px[0] <= 189);
        glClear_(GL_COLOR_BUFFER_BIT);  // not encoded
        glReadPixels_(4, 4, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
        CHECK(px[0] >= 127 && px[0] <= 128);
        glDisable_(GL_SCISSOR_TEST);
        glReadPixels_(20, 20, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
        CHECK(px[0] == 0 && px[1] == 255 && px[2] == 0);
        glClearColor_(0, 1, 0, 1);
        CHECK(glGetError_() == GL_NO_ERROR);

        GLint one = 1, got = 0;
        unsetenv("METALGLDRV_SWAP_INTERVAL");
        [ctx setValues:&one forParameter:NSOpenGLContextParameterSwapInterval];
        [ctx getValues:&got forParameter:NSOpenGLContextParameterSwapInterval];
        CHECK(got == 1);
        CHECK(layer && layer.displaySyncEnabled);
        // Swap interval 0 still presents with display sync; only METALGLDRV_SWAP_INTERVAL=0 turns it off.
        GLint zero = 0;
        [ctx setValues:&zero forParameter:NSOpenGLContextParameterSwapInterval];
        CHECK(layer && layer.displaySyncEnabled);
        setenv("METALGLDRV_SWAP_INTERVAL", "0", 1);
        [ctx setValues:&one forParameter:NSOpenGLContextParameterSwapInterval];
        CHECK(layer && !layer.displaySyncEnabled);
        unsetenv("METALGLDRV_SWAP_INTERVAL");
        [ctx setValues:&zero forParameter:NSOpenGLContextParameterSwapInterval];
        [ctx flushBuffer];

        // Resize: -update re-lays the layer; the next flush resizes the surface.
        [win setContentSize:NSMakeSize(400, 300)];
        [ctx update];
        CHECK(layer && CGRectEqualToRect(layer.frame, view.layer.bounds));
        [ctx flushBuffer];
        glClear_(GL_COLOR_BUFFER_BIT);
        glReadPixels_(390 * win.backingScaleFactor, 290 * win.backingScaleFactor, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
        CHECK(px[1] == 255);

        // Wine: CGL backing size fixes the render size whatever the view's size.
        GLint backing[2] = {200, 100};
        pCGLSetParameter(ctx.CGLContextObj, kCGLCPSurfaceBackingSize, backing);
        pCGLEnable(ctx.CGLContextObj, kCGLCESurfaceBackingSize);
        [ctx makeCurrentContext];
        glClear_(GL_COLOR_BUFFER_BIT);
        CHECK(layer && (int)layer.drawableSize.width == 200 && (int)layer.drawableSize.height == 100);
        [ctx flushBuffer];
        pCGLDisable(ctx.CGLContextObj, kCGLCESurfaceBackingSize);

        // A second context on the same view shares its layer.
        [shared setView:view];
        CHECK(ourLayer(view) == layer);
        int layers = 0;
        for (CALayer* l in view.layer.sublayers)
            if ([l isKindOfClass:[CAMetalLayer class]]) layers++;
        CHECK(layers == 1);

        // GL_APPLE_container_object_shareable: framebuffers, vertex arrays and program pipelines are shared
        // between CGL contexts of a share group, as on Apple's OpenGL (queries aren't).
        {
            auto genFb = gl<void (*)(GLsizei, GLuint*)>("glGenFramebuffers");
            auto bindFb = gl<void (*)(GLenum, GLuint)>("glBindFramebuffer");
            auto isFb = gl<GLboolean (*)(GLuint)>("glIsFramebuffer");
            auto genVao = gl<void (*)(GLsizei, GLuint*)>("glGenVertexArrays");
            auto bindVao = gl<void (*)(GLuint)>("glBindVertexArray");
            auto isVao = gl<GLboolean (*)(GLuint)>("glIsVertexArray");
            auto getStringi = gl<const GLubyte* (*)(GLenum, GLuint)>("glGetStringi");
            [ctx makeCurrentContext];
            GLuint fb = 0, vao = 0;
            genFb(1, &fb);
            bindFb(GL_FRAMEBUFFER, fb);
            bindFb(GL_FRAMEBUFFER, 0);
            genVao(1, &vao);
            bindVao(vao);
            bindVao(0);
            [shared makeCurrentContext];
            CHECK(isFb(fb));
            CHECK(isVao(vao));
            GLuint fb2 = 0;
            genFb(1, &fb2);
            CHECK(fb2 != fb);
            GLint n = 0;
            bool listed = false;
            glGetIntegerv_(GL_NUM_EXTENSIONS, &n);
            for (GLint i = 0; i < n; i++)
                if (!strcmp((const char*)getStringi(GL_EXTENSIONS, i), "GL_APPLE_container_object_shareable")) listed = true;
            CHECK(listed);
            [ctx makeCurrentContext];
        }

        // Wine wraps CGL contexts: -initWithCGLContextObj: on one of ours.
        NSOpenGLContext* wrap = [[NSOpenGLContext alloc] initWithCGLContextObj:ctx.CGLContextObj];
        CHECK(wrap.CGLContextObj == ctx.CGLContextObj);
        CHECK(wrap.pixelFormat.CGLPixelFormatObj == pf.CGLPixelFormatObj);
        [wrap release];

        [NSOpenGLContext clearCurrentContext];
        CHECK([NSOpenGLContext currentContext] == nil);
        CHECK(pCGLGetCurrentContext() == nullptr);
        [ctx clearDrawable];
        CHECK(ctx.view == nil);
        CHECK(ourLayer(view) == layer);  // the view keeps its surface (Wine relies on that)

        [shared release];
        [ctx release];
        CHECK(g_deallocs == 1);
        [pf release];
        [win close];
        testAgent(drv);
    }
    printf("%s (%d failure%s)\n", g_failures ? "FAILED" : "ok", g_failures, g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
