// CGL (the context API of OpenGL.framework) over metalgldrv contexts and surfaces.
//
// Exported under Apple's names and signatures so that code linked against (or
// dlopen'ing) OpenGL.framework can be pointed at libmetalgldrv instead: Wine's
// winemac.so after scripts/crossover-install.sh, and NSOpenGLContext /
// NSOpenGLPixelFormat once nsgl.mm has replaced their methods. We never link
// OpenGL.framework; only its headers are used for the types.
//
// Model: one accelerated renderer (the Metal device) with RGB888 / ARGB8888 color,
// 0/16/24/32-bit depth, 0/8-bit stencil and 1/2/4x multisampling. Every context is
// a GL 4.6 context (core for kCGLOGLPVersion_3_2_Core / GL4_Core pixel formats,
// compatibility otherwise). A context's drawable is a window surface (attached
// through NSOpenGLContext -setView:) or a pbuffer (CGLSetPBuffer); it switches at
// the context's next make-current / flush / update on its own thread.
#include "macos/cgl_internal.h"
#import <AppKit/AppKit.h>
#import <CoreGraphics/CoreGraphics.h>
#import <IOSurface/IOSurface.h>
#include <atomic>
#include <cstring>
#include <strings.h>
#include <map>
#include <mutex>
#include <set>
#include <unordered_set>
#include "gl/app_profile.h"
#include "gl/texture.h"
#include "gl/winsys.h"
#include "util/log.h"

#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#define CGL_EXPORT extern "C" __attribute__((visibility("default")))

using namespace mgl;

namespace {

// Renderer ID we report (vendor nibble 0x00024xxx is "generic hardware"; inside
// kCGLRendererIDMatchingMask so masked comparisons work).
constexpr GLint kRendererID = 0x00024F00;
constexpr GLint kMaxSamples = 4;

struct PixelFormat {
    std::atomic<int> refs{1};
    bool alpha = true;
    int depth = 24, stencil = 8;
    int samples = 0;  // 0 = single sample
    bool doubleBuffer = false, backingStore = false, pbuffer = false;
    int profile = kCGLOGLPVersion_Legacy;
    bool core() const { return profile >= kCGLOGLPVersion_3_2_Core; }
    SurfaceFormat surfaceFormat() const {
        SurfaceFormat f;
        f.a = alpha ? 8 : 0;
        f.depth = depth;
        f.stencil = stencil;
        f.samples = samples;
        return f;
    }
};

struct PBuffer {
    std::atomic<int> refs{1};
    GLsizei width = 0, height = 0;
    GLenum target = GL_TEXTURE_2D, internalFormat = GL_RGBA;
    GLint maxLevel = 0;
    std::mutex m;
    Ref<Surface> surf;  // made by the first CGLSetPBuffer, laid out per that context
    GLenum face = 0;    // of the last CGLSetPBuffer (cube map pbuffers)
};

bool cglTrace() {
    static const bool t = envFlag("METALGLDRV_CGL_TRACE", false);
    return t;
}

struct RendererInfo {
    int count = 1;
};

struct ContextImpl {
    std::atomic<int> refs{1};
    Ref<Context> ctx;
    PixelFormat* pf = nullptr;
    std::recursive_mutex lock;  // CGLLockContext

    std::mutex m;  // guards the requested state below
    Ref<Surface> drawable;  // window or pbuffer surface the context should render to
    PBuffer* pbuffer = nullptr;
    GLenum pbufferFace = 0;
    GLint pbufferLevel = 0;
    int swapInterval = 0;
    int opacity = 1, order = 1;
    bool backingSizeEnabled = false;
    GLint backingSize[2] = {0, 0};
    std::set<int> enables;
    std::map<int, std::array<GLint, 4>> params;  // other parameters, stored for CGLGetParameter

    // Owned by the thread the context is current on.
    Ref<Surface> bound;
};

std::mutex g_mutex;  // guards the registries
std::unordered_set<ContextImpl*> g_contexts;
std::unordered_set<PixelFormat*> g_pixelFormats;
std::unordered_set<PBuffer*> g_pbuffers;
std::unordered_set<RendererInfo*> g_rendererInfos;
std::map<int, GLint> g_globalOptions;
thread_local ContextImpl* t_current = nullptr;

ContextImpl* lookup(CGLContextObj obj) {
    std::lock_guard<std::mutex> g(g_mutex);
    auto* c = (ContextImpl*)obj;
    return g_contexts.count(c) ? c : nullptr;
}
PixelFormat* lookup(CGLPixelFormatObj obj) {
    std::lock_guard<std::mutex> g(g_mutex);
    auto* p = (PixelFormat*)obj;
    return g_pixelFormats.count(p) ? p : nullptr;
}
PBuffer* lookup(CGLPBufferObj obj) {
    std::lock_guard<std::mutex> g(g_mutex);
    auto* p = (PBuffer*)obj;
    return g_pbuffers.count(p) ? p : nullptr;
}

GLint activeDisplayMask() {
    CGDirectDisplayID displays[32];
    uint32_t n = 0;
    if (CGGetActiveDisplayList(32, displays, &n) != kCGErrorSuccess || n == 0) {
        displays[0] = CGMainDisplayID();
        n = 1;
    }
    GLint mask = 0;
    for (uint32_t i = 0; i < n; i++) mask |= (GLint)CGDisplayIDToOpenGLDisplayMask(displays[i]);
    return mask ? mask : 1;
}

GLint videoMemoryMB() {
    Device* d = Device::get();
    uint64_t bytes = d ? d->mtl.recommendedMaxWorkingSetSize : 0;
    return (GLint)std::min<uint64_t>(bytes >> 20, 0x7fffffff);
}

void releasePixelFormat(PixelFormat* p) {
    if (!p || p->refs.fetch_sub(1) != 1) return;
    {
        std::lock_guard<std::mutex> g(g_mutex);
        g_pixelFormats.erase(p);
    }
    delete p;
}

void releasePBuffer(PBuffer* p) {
    if (!p || p->refs.fetch_sub(1) != 1) return;
    {
        std::lock_guard<std::mutex> g(g_mutex);
        g_pbuffers.erase(p);
    }
    @autoreleasepool {  // releasing Metal objects autoreleases (mtl/device.h)
        delete p;
    }
}

void destroyContext(ContextImpl* c) {
    {
        std::lock_guard<std::mutex> g(g_mutex);
        g_contexts.erase(c);
    }
    if (t_current == c) {
        c->ctx->flush();
        t_current = nullptr;
        setCurrentContext(nullptr);
    }
    MGL_INFO("CGL: destroyed context %p", c);
    @autoreleasepool {
        releasePBuffer(c->pbuffer);
        releasePixelFormat(c->pf);
        delete c;
    }
}

// Switches the context to its requested drawable and applies the presentation
// parameters. Call on the thread the context is current on.
void syncDrawable(ContextImpl* c) {
    Ref<Surface> want;
    int interval, fixedW = 0, fixedH = 0;
    {
        std::lock_guard<std::mutex> g(c->m);
        want = c->drawable;
        interval = c->swapInterval;
        if (c->backingSizeEnabled) {
            fixedW = c->backingSize[0];
            fixedH = c->backingSize[1];
        }
    }
    Surface* s = want.get();
    if (s && s->layer) {
        setSurfaceSwapInterval(s, interval);
        int ow = s->width, oh = s->height;
        s->fixedWidth = fixedW;
        s->fixedHeight = fixedH;
        followSuperlayer(s->layer);
        s->syncSizeFromLayer();
        if (s->width != ow || s->height != oh) c->ctx->passDirty = true;
    }
    if (c->bound.get() != s) {
        c->ctx->bindSurfaces(s, s);
        c->bound = want;
        if (s) MGL_INFO("CGL: context %p drawable %s %dx%d", c, s->layer ? "window" : "pbuffer", s->width, s->height);
    }
}

void applyOpacity(Surface* s, int opacity) {
    if (!s || !s->layer) return;
    CAMetalLayer* layer = s->layer;
    BOOL opaque = opacity != 0;
    if ([NSThread isMainThread]) layer.opaque = opaque;
    else dispatch_async(dispatch_get_main_queue(), ^{ layer.opaque = opaque; });
}

// Parses a CGL / NSOpenGL attribute list. Returns kCGLNoError with *out == nullptr when
// nothing matches (CGL's convention), or kCGLBadAttribute / kCGLBadValue.
CGLError choosePixelFormat(const CGLPixelFormatAttribute* attribs, PixelFormat** out) {
    *out = nullptr;
    int colorSize = 0, alphaSize = 0, depthSize = 0, stencilSize = 0, accumSize = 0, auxBuffers = 0;
    int sampleBuffers = 0, samples = 0, profile = kCGLOGLPVersion_Legacy;
    GLint rendererID = 0;
    bool minimum = false, doubleBuffer = false, stereo = false, colorFloat = false, offScreen = false, backingStore = false;
    bool pbuffer = false, multisample = false;
    for (const CGLPixelFormatAttribute* a = attribs; a && *a; a++) {
        int attr = *a;
        switch (attr) {
        // boolean attributes
        case kCGLPFAAllRenderers: case kCGLPFATripleBuffer: case kCGLPFAMaximumPolicy: case kCGLPFASupersample:
        case kCGLPFASampleAlpha: case kCGLPFANoRecovery: case kCGLPFAAccelerated: case kCGLPFAClosestPolicy:
        case kCGLPFABackingVolatile: case kCGLPFAAllowOfflineRenderers: case kCGLPFAAcceleratedCompute:
        case kCGLPFASupportsAutomaticGraphicsSwitching: case kCGLPFAAuxDepthStencil: case kCGLPFAWindow:
        case kCGLPFACompliant: case kCGLPFARemotePBuffer: case kCGLPFASingleRenderer: case kCGLPFARobust:
        case kCGLPFAMPSafe: case kCGLPFAMultiScreen: case kCGLPFAFullScreen:
            break;
        case kCGLPFAMinimumPolicy: minimum = true; break;
        case kCGLPFADoubleBuffer: doubleBuffer = true; break;
        case kCGLPFAStereo: stereo = true; break;
        case kCGLPFAColorFloat: colorFloat = true; break;
        case kCGLPFAOffScreen: offScreen = true; break;
        case kCGLPFABackingStore: backingStore = true; break;
        case kCGLPFAPBuffer: pbuffer = true; break;
        case kCGLPFAMultisample: multisample = true; break;
        // attributes with a value
        case kCGLPFAColorSize: case kCGLPFAAlphaSize: case kCGLPFADepthSize: case kCGLPFAStencilSize:
        case kCGLPFAAccumSize: case kCGLPFAAuxBuffers: case kCGLPFASampleBuffers: case kCGLPFASamples:
        case kCGLPFARendererID: case kCGLPFADisplayMask: case kCGLPFAOpenGLProfile: case kCGLPFAVirtualScreenCount: {
            int v = *++a;
            switch (attr) {
            case kCGLPFAColorSize: colorSize = v; break;
            case kCGLPFAAlphaSize: alphaSize = v; break;
            case kCGLPFADepthSize: depthSize = v; break;
            case kCGLPFAStencilSize: stencilSize = v; break;
            case kCGLPFAAccumSize: accumSize = v; break;
            case kCGLPFAAuxBuffers: auxBuffers = v; break;
            case kCGLPFASampleBuffers: sampleBuffers = v; break;
            case kCGLPFASamples: samples = v; break;
            case kCGLPFARendererID: rendererID = v; break;
            case kCGLPFAOpenGLProfile: profile = v; break;
            default: break;
            }
            break;
        }
        default:
            MGL_WARN("CGL: unknown pixel format attribute %d", attr);
            return kCGLBadAttribute;
        }
    }
    if (profile != kCGLOGLPVersion_Legacy && profile != kCGLOGLPVersion_3_2_Core && profile != kCGLOGLPVersion_GL4_Core)
        return kCGLBadValue;
    // Things this renderer doesn't have.
    if (rendererID && (rendererID & kCGLRendererIDMatchingMask) != (kRendererID & kCGLRendererIDMatchingMask)) return kCGLNoError;
    if (stereo || colorFloat || offScreen) return kCGLNoError;
    if (minimum && (colorSize > 32 || alphaSize > 8 || depthSize > 32 || stencilSize > 8 || accumSize > 0 || auxBuffers > 0))
        return kCGLNoError;
    if (sampleBuffers > 1 || (minimum && samples > kMaxSamples)) return kCGLNoError;
    auto* pf = new PixelFormat();
    pf->alpha = alphaSize > 0 || colorSize > 24;
    pf->depth = depthSize <= 0 ? 0 : depthSize <= 16 ? 16 : depthSize <= 24 ? 24 : 32;
    pf->stencil = stencilSize > 0 ? 8 : 0;
    if ((sampleBuffers > 0 || multisample) && samples > 1) pf->samples = samples <= 2 ? 2 : kMaxSamples;
    pf->doubleBuffer = doubleBuffer;
    pf->backingStore = backingStore;
    pf->pbuffer = pbuffer;
    pf->profile = profile;
    *out = pf;
    return kCGLNoError;
}

const char* profileName(int p) {
    return p == kCGLOGLPVersion_GL4_Core ? "GL4_Core" : p == kCGLOGLPVersion_3_2_Core ? "3_2_Core" : "Legacy";
}

}  // namespace

// ---------------------------------------------------------------------------
// internal interface (cgl_internal.h)

namespace mgl::cgl {

bool isOurContext(CGLContextObj ctx) { return ctx && lookup(ctx); }
bool isOurPixelFormat(CGLPixelFormatObj pix) { return pix && lookup(pix); }

Ref<Surface> newWindowSurface(CGLContextObj obj, CAMetalLayer* layer) {
    ContextImpl* c = lookup(obj);
    if (!c) return nullptr;
    // sRGB-capable like macOS's 8-bit default framebuffers (GL_FRAMEBUFFER_SRGB encodes).
    Ref<Surface> s = mgl::newWindowSurface(c->pf->surfaceFormat(), layer, true, true);
    applyOpacity(s.get(), surfaceOpacity(obj));
    return s;
}

bool surfaceCompatible(CGLContextObj obj, Surface* s) {
    ContextImpl* c = lookup(obj);
    if (!c || !s) return false;
    const PixelFormat* pf = c->pf;
    return s->samples == pf->samples && (s->hasDepth || pf->depth == 0) && (s->hasStencil || pf->stencil == 0);
}

void setWindowDrawable(CGLContextObj obj, Surface* s) {
    ContextImpl* c = lookup(obj);
    if (!c) return;
    PBuffer* oldPbuffer = nullptr;
    {
        std::lock_guard<std::mutex> g(c->m);
        c->drawable = s;
        oldPbuffer = c->pbuffer;
        c->pbuffer = nullptr;
    }
    releasePBuffer(oldPbuffer);
    if (t_current == c) syncDrawable(c);
}

int surfaceOpacity(CGLContextObj obj) {
    ContextImpl* c = lookup(obj);
    if (!c) return 1;
    std::lock_guard<std::mutex> g(c->m);
    return c->opacity;
}

}  // namespace mgl::cgl

// ---------------------------------------------------------------------------
// pixel formats

CGL_EXPORT CGLError CGLChoosePixelFormat(const CGLPixelFormatAttribute* attribs, CGLPixelFormatObj* pix, GLint* npix) {
    if (!pix) return kCGLBadAddress;
    *pix = nullptr;
    if (npix) *npix = 0;
    if (!Device::get()) return kCGLBadDisplay;
    PixelFormat* pf = nullptr;
    CGLError err = choosePixelFormat(attribs, &pf);
    static const bool trace = envFlag("METALGLDRV_CGL_TRACE", false);  // pixel-format negotiation (Wine)
    if (trace) {
        static std::atomic<uint64_t> calls{0};
        uint64_t n = ++calls;
        if (n <= 3000 || n % 1000 == 0) {
            std::string a;
            for (const CGLPixelFormatAttribute* p = attribs; p && *p && a.size() < 400; p++) {
                int at = (int)*p;
                a += " " + std::to_string(at);
                switch (at) {  // the attributes that take a value
                case kCGLPFAAuxBuffers: case kCGLPFAColorSize: case kCGLPFAAlphaSize: case kCGLPFADepthSize: case kCGLPFAStencilSize:
                case kCGLPFAAccumSize: case kCGLPFASampleBuffers: case kCGLPFASamples: case kCGLPFARendererID: case kCGLPFADisplayMask:
                case kCGLPFAOpenGLProfile: case kCGLPFAVirtualScreenCount:
                    a += "=" + std::to_string((int)*++p);
                    break;
                default: break;
                }
            }
            MGL_INFO("CGLChoosePixelFormat #%llu ->%s:%s", (unsigned long long)n, err == kCGLNoError ? " ok" : " fail", a.c_str());
        }
    }
    if (err != kCGLNoError || !pf) return err;
    {
        std::lock_guard<std::mutex> g(g_mutex);
        g_pixelFormats.insert(pf);
    }
    *pix = (CGLPixelFormatObj)pf;
    if (npix) *npix = 1;
    MGL_DBG("CGL: pixel format %p: alpha %d depth %d stencil %d samples %d %s%s", pf, pf->alpha, pf->depth, pf->stencil,
            pf->samples, profileName(pf->profile), pf->doubleBuffer ? " double" : "");
    return kCGLNoError;
}

CGL_EXPORT CGLError CGLDestroyPixelFormat(CGLPixelFormatObj pix) {
    PixelFormat* p = lookup(pix);
    if (!p) return kCGLBadPixelFormat;
    releasePixelFormat(p);
    return kCGLNoError;
}

CGL_EXPORT void CGLReleasePixelFormat(CGLPixelFormatObj pix) { releasePixelFormat(lookup(pix)); }

CGL_EXPORT CGLPixelFormatObj CGLRetainPixelFormat(CGLPixelFormatObj pix) {
    if (PixelFormat* p = lookup(pix)) p->refs.fetch_add(1);
    return pix;
}

CGL_EXPORT GLuint CGLGetPixelFormatRetainCount(CGLPixelFormatObj pix) {
    PixelFormat* p = lookup(pix);
    return p ? (GLuint)p->refs.load() : 0;
}

CGL_EXPORT CGLError CGLDescribePixelFormat(CGLPixelFormatObj pix, GLint pix_num, CGLPixelFormatAttribute attrib, GLint* value) {
    PixelFormat* p = lookup(pix);
    if (!p) return kCGLBadPixelFormat;
    if (pix_num != 0) return kCGLBadValue;
    if (!value) return kCGLBadAddress;
    GLint v = 0;
    switch ((int)attrib) {
    case kCGLPFAColorSize: v = p->alpha ? 32 : 24; break;
    case kCGLPFAAlphaSize: v = p->alpha ? 8 : 0; break;
    case kCGLPFADepthSize: v = p->depth; break;
    case kCGLPFAStencilSize: v = p->stencil; break;
    case kCGLPFASampleBuffers: v = p->samples ? 1 : 0; break;
    case kCGLPFASamples: v = p->samples; break;
    case kCGLPFAMultisample: v = p->samples ? 1 : 0; break;
    case kCGLPFADoubleBuffer: v = p->doubleBuffer; break;
    case kCGLPFABackingStore: v = p->backingStore; break;
    case kCGLPFAOpenGLProfile: v = p->profile; break;
    case kCGLPFARendererID: v = kRendererID; break;
    case kCGLPFADisplayMask: v = activeDisplayMask(); break;
    case kCGLPFAVirtualScreenCount: v = 1; break;
    case kCGLPFAAccelerated: case kCGLPFANoRecovery: case kCGLPFAWindow: case kCGLPFAPBuffer: case kCGLPFACompliant:
    case kCGLPFASingleRenderer: case kCGLPFAMPSafe: case kCGLPFAMultiScreen: case kCGLPFAAllowOfflineRenderers:
    case kCGLPFAAcceleratedCompute: case kCGLPFASupportsAutomaticGraphicsSwitching:
        v = 1;
        break;
    case kCGLPFAAllRenderers: case kCGLPFATripleBuffer: case kCGLPFAStereo: case kCGLPFAColorFloat: case kCGLPFASupersample:
    case kCGLPFASampleAlpha: case kCGLPFAOffScreen: case kCGLPFAFullScreen: case kCGLPFARemotePBuffer: case kCGLPFARobust:
    case kCGLPFAAuxBuffers: case kCGLPFAAccumSize: case kCGLPFAAuxDepthStencil: case kCGLPFABackingVolatile:
    case kCGLPFAMinimumPolicy: case kCGLPFAMaximumPolicy: case kCGLPFAClosestPolicy:
        v = 0;
        break;
    default:
        if (cglTrace()) MGL_INFO("CGLDescribePixelFormat(attr %d) -> kCGLBadAttribute", (int)attrib);
        return kCGLBadAttribute;
    }
    *value = v;
    return kCGLNoError;
}

// ---------------------------------------------------------------------------
// renderer info

CGL_EXPORT CGLError CGLQueryRendererInfo(GLuint display_mask, CGLRendererInfoObj* rend, GLint* nrend) {
    if (!rend) return kCGLBadAddress;
    *rend = nullptr;
    if (nrend) *nrend = 0;
    if (!Device::get()) return kCGLBadDisplay;
    auto* r = new RendererInfo();
    {
        std::lock_guard<std::mutex> g(g_mutex);
        g_rendererInfos.insert(r);
    }
    *rend = (CGLRendererInfoObj)r;
    if (nrend) *nrend = r->count;
    if (cglTrace()) MGL_INFO("CGLQueryRendererInfo(mask 0x%x) -> %d renderer(s)", display_mask, r->count);
    return kCGLNoError;
}

CGL_EXPORT CGLError CGLDestroyRendererInfo(CGLRendererInfoObj rend) {
    std::lock_guard<std::mutex> g(g_mutex);
    auto* r = (RendererInfo*)rend;
    if (!g_rendererInfos.erase(r)) return kCGLBadRendererInfo;
    delete r;
    return kCGLNoError;
}

CGL_EXPORT CGLError CGLDescribeRenderer(CGLRendererInfoObj rend, GLint rend_num, CGLRendererProperty prop, GLint* value) {
    {
        std::lock_guard<std::mutex> g(g_mutex);
        if (!g_rendererInfos.count((RendererInfo*)rend)) return kCGLBadRendererInfo;
    }
    if (prop == kCGLRPRendererCount) {
        if (value) *value = 1;
        return kCGLNoError;
    }
    if (rend_num != 0) return kCGLBadValue;
    if (!value) return kCGLBadAddress;
    GLint v = 0;
    switch ((int)prop) {
    case kCGLRPRendererID: v = kRendererID; break;
    case kCGLRPDisplayMask: v = activeDisplayMask(); break;
    case kCGLRPAccelerated: case kCGLRPBackingStore: case kCGLRPWindow: case kCGLRPCompliant: case kCGLRPOnline:
    case kCGLRPGPUVertProcCapable: case kCGLRPGPUFragProcCapable: case kCGLRPAcceleratedCompute: case kCGLRPMPSafe:
    case kCGLRPMultiScreen: case kCGLRPFullScreen:
        v = 1;
        break;
    case kCGLRPOffScreen: case kCGLRPRobust: case kCGLRPSampleAlpha: case kCGLRPRemovable:
    case kCGLRPAccumModes:
        v = 0;
        break;
    case kCGLRPMaxAuxBuffers:
        // Like Apple's renderers, report one. Wine's pixel-format enumeration (winemac.drv) steps
        // its aux loop by this value, so 0 never terminates and hangs explorer.exe. Requests for aux
        // buffers fail under the minimum policy (choosePixelFormat), so no format claims any.
        v = 1;
        break;
    case kCGLRPBufferModes: v = kCGLMonoscopicBit | kCGLSingleBufferBit | kCGLDoubleBufferBit; break;
    case kCGLRPColorModes: v = kCGLRGB888Bit | kCGLARGB8888Bit; break;
    case kCGLRPDepthModes: v = kCGL0Bit | kCGL16Bit | kCGL24Bit | kCGL32Bit; break;
    case kCGLRPStencilModes: v = kCGL0Bit | kCGL8Bit; break;
    case kCGLRPMaxSampleBuffers: v = 1; break;
    case kCGLRPMaxSamples: v = kMaxSamples; break;
    case kCGLRPSampleModes: v = kCGLMultisampleBit; break;
    case kCGLRPVideoMemoryMegabytes: case kCGLRPTextureMemoryMegabytes: v = videoMemoryMB(); break;
    case kCGLRPVideoMemory: case kCGLRPTextureMemory: v = (GLint)std::min<int64_t>((int64_t)videoMemoryMB() << 20, 0x7fffffff); break;
    case kCGLRPMajorGLVersion: v = 4; break;
    case kCGLRPRegistryIDLow: v = Device::get() ? (GLint)(Device::get()->mtl.registryID & 0xffffffffu) : 0; break;
    case kCGLRPRegistryIDHigh: v = Device::get() ? (GLint)(Device::get()->mtl.registryID >> 32) : 0; break;
    default:
        if (cglTrace()) MGL_INFO("CGLDescribeRenderer(prop %d) -> kCGLBadProperty", (int)prop);
        return kCGLBadProperty;
    }
    *value = v;
    if (cglTrace()) MGL_INFO("CGLDescribeRenderer(prop %d) -> %d", (int)prop, v);
    return kCGLNoError;
}

// ---------------------------------------------------------------------------
// contexts

CGL_EXPORT CGLError CGLCreateContext(CGLPixelFormatObj pix, CGLContextObj share, CGLContextObj* ctx) {
    if (!ctx) return kCGLBadAddress;
    *ctx = nullptr;
    PixelFormat* pf = lookup(pix);
    if (!pf) return kCGLBadPixelFormat;
    if (!Device::get()) return kCGLBadDisplay;
    Ref<ShareGroup> group;
    if (share) {
        ContextImpl* sc = lookup(share);
        if (!sc) return kCGLBadMatch;
        group = sc->ctx->share;
    } else {
        group = new ShareGroup();
        // GL_APPLE_container_object_shareable, like Apple's OpenGL. Not for Windows programs under Wine, which
        // expect Windows drivers' behaviour (app profiles opt into theirs: AppProfile::shareFramebuffers).
        const std::string& exe = appProfile().exe;
        bool windowsExe = exe.size() > 4 && strcasecmp(exe.c_str() + exe.size() - 4, ".exe") == 0;
        group->shareContainers = !windowsExe;
    }
    bool core = pf->core();
    if (const char* p = getenv("METALGLDRV_PROFILE")) core = strcmp(p, "core") == 0;
    auto* c = new ContextImpl();
    // macOS core profile contexts are forward-compatible.
    c->ctx = new Context(group, 4, 6, core, false, core, false, false);
    c->pf = pf;
    pf->refs.fetch_add(1);
    c->enables = {kCGLCERasterization, kCGLCEDisplayListOptimization};  // on by default
    {
        std::lock_guard<std::mutex> g(g_mutex);
        g_contexts.insert(c);
    }
    *ctx = (CGLContextObj)c;
    MGL_INFO("CGL: created GL 4.6 %s context %p (pixel format %s, share %p; GL context %p, share group %p)", core ? "core" : "compatibility",
             c, profileName(pf->profile), share, (void*)c->ctx.get(), (void*)group.get());
    return kCGLNoError;
}

CGL_EXPORT CGLError CGLDestroyContext(CGLContextObj ctx) {
    ContextImpl* c = lookup(ctx);
    if (!c) return kCGLBadContext;
    destroyContext(c);
    return kCGLNoError;
}

CGL_EXPORT CGLContextObj CGLRetainContext(CGLContextObj ctx) {
    if (ContextImpl* c = lookup(ctx)) c->refs.fetch_add(1);
    return ctx;
}

CGL_EXPORT void CGLReleaseContext(CGLContextObj ctx) {
    ContextImpl* c = lookup(ctx);
    if (c && c->refs.fetch_sub(1) == 1) destroyContext(c);
}

CGL_EXPORT GLuint CGLGetContextRetainCount(CGLContextObj ctx) {
    ContextImpl* c = lookup(ctx);
    return c ? (GLuint)c->refs.load() : 0;
}

CGL_EXPORT CGLPixelFormatObj CGLGetPixelFormat(CGLContextObj ctx) {
    ContextImpl* c = lookup(ctx);
    return c ? (CGLPixelFormatObj)c->pf : nullptr;
}

CGL_EXPORT CGLShareGroupObj CGLGetShareGroup(CGLContextObj ctx) {
    ContextImpl* c = lookup(ctx);
    return c ? (CGLShareGroupObj)c->ctx->share.get() : nullptr;
}

CGL_EXPORT CGLError CGLCopyContext(CGLContextObj src, CGLContextObj dst, GLbitfield mask) {
    if (!lookup(src) || !lookup(dst)) return kCGLBadContext;
    MGL_WARN("CGL: CGLCopyContext is not supported (ignored)");
    return kCGLNoError;
}

CGL_EXPORT CGLError CGLSetCurrentContext(CGLContextObj ctx) {
    if (!ctx) {
        if (t_current) {
            t_current->ctx->flush();
            t_current = nullptr;
            setCurrentContext(nullptr);
        }
        return kCGLNoError;
    }
    ContextImpl* c = lookup(ctx);
    if (!c) return kCGLBadContext;
    if (t_current && t_current != c) t_current->ctx->flush();
    t_current = c;
    setCurrentContext(c->ctx.get());
    syncDrawable(c);
    return kCGLNoError;
}

CGL_EXPORT CGLContextObj CGLGetCurrentContext(void) { return (CGLContextObj)t_current; }

CGL_EXPORT CGLError CGLLockContext(CGLContextObj ctx) {
    ContextImpl* c = lookup(ctx);
    if (!c) return kCGLBadContext;
    c->lock.lock();
    return kCGLNoError;
}

CGL_EXPORT CGLError CGLUnlockContext(CGLContextObj ctx) {
    ContextImpl* c = lookup(ctx);
    if (!c) return kCGLBadContext;
    c->lock.unlock();
    return kCGLNoError;
}

CGL_EXPORT CGLError CGLFlushDrawable(CGLContextObj ctx) {
    ContextImpl* c = lookup(ctx);
    if (!c) return kCGLBadContext;
    Context* prev = currentContext();
    bool borrowed = prev != c->ctx.get();
    if (borrowed) setCurrentContext(c->ctx.get());
    Surface* s = c->bound.get();
    if (s && s->layer) presentWindowSurface(c->ctx.get(), s);
    else c->ctx->flush();
    // Then pick up a drawable / size change requested since the last frame.
    if (t_current == c) syncDrawable(c);
    if (borrowed) setCurrentContext(prev);
    return kCGLNoError;
}

CGL_EXPORT CGLError CGLUpdateContext(CGLContextObj ctx) {
    ContextImpl* c = lookup(ctx);
    if (!c) return kCGLBadContext;
    if (t_current == c) syncDrawable(c);
    return kCGLNoError;
}

CGL_EXPORT CGLError CGLClearDrawable(CGLContextObj ctx) {
    if (!lookup(ctx)) return kCGLBadContext;
    mgl::cgl::setWindowDrawable(ctx, nullptr);
    return kCGLNoError;
}

CGL_EXPORT CGLError CGLEnable(CGLContextObj ctx, CGLContextEnable pname) {
    ContextImpl* c = lookup(ctx);
    if (!c) return kCGLBadContext;
    std::lock_guard<std::mutex> g(c->m);
    if (pname == kCGLCESurfaceBackingSize) c->backingSizeEnabled = true;
    else c->enables.insert(pname);
    return kCGLNoError;
}

CGL_EXPORT CGLError CGLDisable(CGLContextObj ctx, CGLContextEnable pname) {
    ContextImpl* c = lookup(ctx);
    if (!c) return kCGLBadContext;
    std::lock_guard<std::mutex> g(c->m);
    if (pname == kCGLCESurfaceBackingSize) c->backingSizeEnabled = false;
    else c->enables.erase(pname);
    return kCGLNoError;
}

CGL_EXPORT CGLError CGLIsEnabled(CGLContextObj ctx, CGLContextEnable pname, GLint* enable) {
    ContextImpl* c = lookup(ctx);
    if (!c) return kCGLBadContext;
    if (!enable) return kCGLBadAddress;
    std::lock_guard<std::mutex> g(c->m);
    if (pname == kCGLCESurfaceBackingSize) *enable = c->backingSizeEnabled;
    else *enable = c->enables.count(pname) ? 1 : 0;
    return kCGLNoError;
}

CGL_EXPORT CGLError CGLSetParameter(CGLContextObj ctx, CGLContextParameter pname, const GLint* params) {
    ContextImpl* c = lookup(ctx);
    if (!c) return kCGLBadContext;
    if (!params && pname != kCGLCPReclaimResources) return kCGLBadAddress;
    Ref<Surface> window;
    {
        std::lock_guard<std::mutex> g(c->m);
        switch ((int)pname) {
        case kCGLCPSwapInterval: c->swapInterval = params[0]; break;
        case kCGLCPSurfaceOpacity: c->opacity = params[0]; break;
        case kCGLCPSurfaceOrder: c->order = params[0]; break;  // sublayers always draw above the view's content
        case kCGLCPSurfaceBackingSize:
            if (params[0] <= 0 || params[1] <= 0) return kCGLBadValue;
            c->backingSize[0] = params[0];
            c->backingSize[1] = params[1];
            break;
        case kCGLCPReclaimResources: break;
        case kCGLCPCurrentRendererID: case kCGLCPGPUVertexProcessing: case kCGLCPGPUFragmentProcessing:
        case kCGLCPHasDrawable:
            return kCGLBadEnumeration;  // read-only
        case kCGLCPSwapRectangle: c->params[pname] = {params[0], params[1], params[2], params[3]}; break;
        default: c->params[pname] = {params[0], 0, 0, 0}; break;
        }
        if (c->drawable && c->drawable->layer) window = c->drawable;
    }
    if (window) {
        if (pname == kCGLCPSwapInterval) setSurfaceSwapInterval(window.get(), params[0]);
        if (pname == kCGLCPSurfaceOpacity) applyOpacity(window.get(), params[0]);
    }
    return kCGLNoError;
}

CGL_EXPORT CGLError CGLGetParameter(CGLContextObj ctx, CGLContextParameter pname, GLint* params) {
    ContextImpl* c = lookup(ctx);
    if (!c) return kCGLBadContext;
    if (!params) return kCGLBadAddress;
    std::lock_guard<std::mutex> g(c->m);
    switch ((int)pname) {
    case kCGLCPSwapInterval: params[0] = c->swapInterval; break;
    case kCGLCPSurfaceOpacity: params[0] = c->opacity; break;
    case kCGLCPSurfaceOrder: params[0] = c->order; break;
    case kCGLCPSurfaceBackingSize:
        if (c->backingSizeEnabled || !c->drawable) {
            params[0] = c->backingSize[0];
            params[1] = c->backingSize[1];
        } else {
            params[0] = c->drawable->width;
            params[1] = c->drawable->height;
        }
        break;
    case kCGLCPCurrentRendererID: params[0] = kRendererID; break;
    case kCGLCPGPUVertexProcessing: case kCGLCPGPUFragmentProcessing: params[0] = 1; break;
    case kCGLCPHasDrawable: params[0] = c->drawable ? 1 : 0; break;
    case kCGLCPMPSwapsInFlight: params[0] = c->params.count(pname) ? c->params[pname][0] : 1; break;
    case kCGLCPGPURestartStatus: params[0] = kCGLCPGPURestartStatusNone; break;
    case kCGLCPSupportGPURestart: case kCGLCPSupportSeparateAddressSpace: params[0] = 1; break;
    case kCGLCPSwapRectangle: {
        auto it = c->params.find(pname);
        for (int i = 0; i < 4; i++) params[i] = it != c->params.end() ? it->second[i] : 0;
        break;
    }
    default: {
        auto it = c->params.find(pname);
        if (it == c->params.end()) return kCGLBadEnumeration;
        params[0] = it->second[0];
        break;
    }
    }
    return kCGLNoError;
}

CGL_EXPORT CGLError CGLSetVirtualScreen(CGLContextObj ctx, GLint screen) {
    if (!lookup(ctx)) return kCGLBadContext;
    return screen == 0 ? kCGLNoError : kCGLBadValue;
}

CGL_EXPORT CGLError CGLGetVirtualScreen(CGLContextObj ctx, GLint* screen) {
    if (!lookup(ctx)) return kCGLBadContext;
    if (screen) *screen = 0;
    return kCGLNoError;
}

CGL_EXPORT CGLError CGLSetOffScreen(CGLContextObj ctx, GLsizei width, GLsizei height, GLint rowbytes, void* baseaddr) {
    return lookup(ctx) ? kCGLBadOffScreen : kCGLBadContext;
}

CGL_EXPORT CGLError CGLGetOffScreen(CGLContextObj ctx, GLsizei* width, GLsizei* height, GLint* rowbytes, void** baseaddr) {
    if (!lookup(ctx)) return kCGLBadContext;
    if (width) *width = 0;
    if (height) *height = 0;
    if (rowbytes) *rowbytes = 0;
    if (baseaddr) *baseaddr = nullptr;
    return kCGLNoError;
}

CGL_EXPORT CGLError CGLSetFullScreen(CGLContextObj ctx) { return lookup(ctx) ? kCGLBadFullScreen : kCGLBadContext; }
CGL_EXPORT CGLError CGLSetFullScreenOnDisplay(CGLContextObj ctx, GLuint display_mask) {
    return lookup(ctx) ? kCGLBadFullScreen : kCGLBadContext;
}

// ---------------------------------------------------------------------------
// pbuffers

CGL_EXPORT CGLError CGLCreatePBuffer(GLsizei width, GLsizei height, GLenum target, GLenum internalFormat, GLint max_level,
                                     CGLPBufferObj* pbuffer) {
    if (!pbuffer) return kCGLBadAddress;
    *pbuffer = nullptr;
    if (width <= 0 || height <= 0 || max_level < 0) return kCGLBadValue;
    if (target != GL_TEXTURE_2D && target != GL_TEXTURE_RECTANGLE && target != GL_TEXTURE_CUBE_MAP) return kCGLBadValue;
    if (internalFormat != GL_RGB && internalFormat != GL_RGBA && internalFormat != GL_DEPTH_COMPONENT) return kCGLBadValue;
    if (target == GL_TEXTURE_CUBE_MAP && width != height) return kCGLBadValue;
    auto* p = new PBuffer();
    p->width = width;
    p->height = height;
    p->target = target;
    p->internalFormat = internalFormat;
    p->maxLevel = max_level;
    {
        std::lock_guard<std::mutex> g(g_mutex);
        g_pbuffers.insert(p);
    }
    *pbuffer = (CGLPBufferObj)p;
    return kCGLNoError;
}

CGL_EXPORT CGLError CGLDestroyPBuffer(CGLPBufferObj pbuffer) {
    PBuffer* p = lookup(pbuffer);
    if (!p) return kCGLBadAddress;
    releasePBuffer(p);
    return kCGLNoError;
}

CGL_EXPORT CGLPBufferObj CGLRetainPBuffer(CGLPBufferObj pbuffer) {
    if (PBuffer* p = lookup(pbuffer)) p->refs.fetch_add(1);
    return pbuffer;
}

CGL_EXPORT void CGLReleasePBuffer(CGLPBufferObj pbuffer) { releasePBuffer(lookup(pbuffer)); }

CGL_EXPORT GLuint CGLGetPBufferRetainCount(CGLPBufferObj pbuffer) {
    PBuffer* p = lookup(pbuffer);
    return p ? (GLuint)p->refs.load() : 0;
}

CGL_EXPORT CGLError CGLDescribePBuffer(CGLPBufferObj obj, GLsizei* width, GLsizei* height, GLenum* target,
                                       GLenum* internalFormat, GLint* mipmap) {
    PBuffer* p = lookup(obj);
    if (!p) return kCGLBadAddress;
    if (width) *width = p->width;
    if (height) *height = p->height;
    if (target) *target = p->target;
    if (internalFormat) *internalFormat = p->internalFormat;
    if (mipmap) *mipmap = p->maxLevel;
    return kCGLNoError;
}

CGL_EXPORT CGLError CGLSetPBuffer(CGLContextObj ctx, CGLPBufferObj pbuffer, GLenum face, GLint level, GLint screen) {
    ContextImpl* c = lookup(ctx);
    if (!c) return kCGLBadContext;
    PBuffer* p = lookup(pbuffer);
    if (!p) return kCGLBadAddress;
    if (level < 0 || level > p->maxLevel) return kCGLBadValue;
    if (level > 0) MGL_WARN("CGL: rendering to pbuffer mip level %d is not supported; using level 0", level);
    Ref<Surface> surf;
    {
        std::lock_guard<std::mutex> g(p->m);
        if (!p->surf) {
            SurfaceFormat f = c->pf->surfaceFormat();
            f.a = p->internalFormat == GL_RGB ? 0 : 8;
            p->surf = newDefaultSurface(f);
            p->surf->resize(p->width, p->height);
        }
        p->face = face;
        surf = p->surf;
    }
    p->refs.fetch_add(1);
    PBuffer* old;
    {
        std::lock_guard<std::mutex> g(c->m);
        old = c->pbuffer;
        c->pbuffer = p;
        c->pbufferFace = face;
        c->pbufferLevel = level;
        c->drawable = surf;
    }
    releasePBuffer(old);
    if (t_current == c) syncDrawable(c);
    return kCGLNoError;
}

CGL_EXPORT CGLError CGLGetPBuffer(CGLContextObj ctx, CGLPBufferObj* pbuffer, GLenum* face, GLint* level, GLint* screen) {
    ContextImpl* c = lookup(ctx);
    if (!c) return kCGLBadContext;
    std::lock_guard<std::mutex> g(c->m);
    if (pbuffer) *pbuffer = (CGLPBufferObj)c->pbuffer;
    if (face) *face = c->pbufferFace;
    if (level) *level = c->pbufferLevel;
    if (screen) *screen = 0;
    return kCGLNoError;
}

// Copies the pbuffer's color buffer into level 0 of the texture bound to the
// pbuffer's target in `ctx` (CGL binds; a copy is equivalent until the pbuffer is
// rendered to again, which is when apps call this again).
CGL_EXPORT CGLError CGLTexImagePBuffer(CGLContextObj ctx, CGLPBufferObj pbuffer, GLenum source) {
    ContextImpl* c = lookup(ctx);
    if (!c) return kCGLBadContext;
    PBuffer* p = lookup(pbuffer);
    if (!p) return kCGLBadAddress;
    if (source != GL_FRONT && source != GL_BACK && source != GL_AUX0) return kCGLBadValue;
    Ref<Surface> ps;
    GLenum face;
    {
        std::lock_guard<std::mutex> g(p->m);
        ps = p->surf;
        face = p->face;
    }
    if (!ps) return kCGLNoError;  // never rendered: contents undefined
    if (p->internalFormat == GL_DEPTH_COMPONENT) {
        MGL_WARN("CGL: CGLTexImagePBuffer of a depth pbuffer is not supported");
        return kCGLNoError;
    }
    Context* gl = c->ctx.get();
    Context* prev = currentContext();
    if (prev != gl) setCurrentContext(gl);
    Ref<Surface> draw = gl->drawSurface, read = gl->readSurface;
    gl->bindSurfaces(draw.get(), ps.get());
    GLint readFb = 0;
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &readFb);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    GLenum target = p->target == GL_TEXTURE_CUBE_MAP
                        ? (face >= GL_TEXTURE_CUBE_MAP_POSITIVE_X && face <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z ? face
                                                                                                            : GL_TEXTURE_CUBE_MAP_POSITIVE_X)
                        : p->target;
    glCopyTexImage2D(target, 0, p->internalFormat, 0, 0, p->width, p->height, 0);
    if (p->maxLevel > 0 && p->target != GL_TEXTURE_RECTANGLE) glGenerateMipmap(p->target);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, readFb);
    gl->bindSurfaces(draw.get(), read.get());
    if (prev != gl) setCurrentContext(prev);
    return kCGLNoError;
}

// ---------------------------------------------------------------------------
// IOSurface-backed textures

// Makes the texture bound to `target` (GL_TEXTURE_RECTANGLE or GL_TEXTURE_2D) in `ctx` a single-level image whose
// storage is plane `plane` of the IOSurface: no copy, both sides see each other's writes (GL writes once flushed and
// finished, e.g. glFlush + a fence, as with Apple's OpenGL). (internal_format, format, type) are the combinations
// Apple's OpenGL documents, e.g. GL_RGBA / GL_BGRA / GL_UNSIGNED_INT_8_8_8_8_REV for a 'BGRA' surface, GL_R8 / GL_RED /
// GL_UNSIGNED_BYTE and GL_RG8 / GL_RG for the planes of a bi-planar YUV surface. Respecifying the texture detaches it.
CGL_EXPORT CGLError CGLTexImageIOSurface2D(CGLContextObj ctx, GLenum target, GLenum internal_format, GLsizei width, GLsizei height,
                                           GLenum format, GLenum type, IOSurfaceRef ioSurface, GLuint plane) {
    ContextImpl* c = lookup(ctx);
    if (!c) return kCGLBadContext;
    if (!ioSurface) return kCGLBadAddress;
    if (target != GL_TEXTURE_RECTANGLE && target != GL_TEXTURE_2D) return kCGLBadEnumeration;
    const FormatInfo* f = iosurfaceFormat(internal_format, format, type);
    if (!f) {
        MGL_WARN("CGL: CGLTexImageIOSurface2D: unsupported format 0x%x / 0x%x / 0x%x", internal_format, format, type);
        return kCGLBadValue;
    }
    size_t planes = IOSurfaceGetPlaneCount(ioSurface);
    if (planes ? plane >= planes : plane != 0) return kCGLBadValue;
    size_t pw = planes ? IOSurfaceGetWidthOfPlane(ioSurface, plane) : IOSurfaceGetWidth(ioSurface);
    size_t ph = planes ? IOSurfaceGetHeightOfPlane(ioSurface, plane) : IOSurfaceGetHeight(ioSurface);
    size_t bpe = planes ? IOSurfaceGetBytesPerElementOfPlane(ioSurface, plane) : IOSurfaceGetBytesPerElement(ioSurface);
    if (width <= 0 || height <= 0 || (size_t)width > pw || (size_t)height > ph || bpe != f->bytesPerPixel) {
        MGL_WARN("CGL: CGLTexImageIOSurface2D: %dx%d, %d bytes per texel doesn't fit plane %u (%zux%zu, %zu bytes per element)",
                 width, height, f->bytesPerPixel, plane, pw, ph, bpe);
        return kCGLBadValue;
    }
    Context* gl = c->ctx.get();
    Context* prev = currentContext();
    if (prev != gl) setCurrentContext(gl);
    CGLError err = kCGLNoError;
    Texture* t = gl->boundTexture(target);
    if (!t || t->immutable || t->viewParent) err = kCGLBadState;
    else if (!textureFromIOSurface(gl, t, internal_format, f, width, height, ioSurface, plane)) {
        MGL_WARN("CGL: CGLTexImageIOSurface2D: Metal can't make a %dx%d texture (format %d) over plane %u of the surface", width,
                 height, (int)f->mtl, plane);
        err = kCGLBadAlloc;
    }
    if (prev != gl) setCurrentContext(prev);
    return err;
}

// ---------------------------------------------------------------------------
// global options, version, errors

CGL_EXPORT CGLError CGLSetGlobalOption(CGLGlobalOption pname, const GLint* params) {
    std::lock_guard<std::mutex> g(g_mutex);
    g_globalOptions[pname] = params ? params[0] : 0;
    return kCGLNoError;
}

CGL_EXPORT CGLError CGLGetGlobalOption(CGLGlobalOption pname, GLint* params) {
    if (!params) return kCGLBadAddress;
    std::lock_guard<std::mutex> g(g_mutex);
    auto it = g_globalOptions.find(pname);
    *params = it != g_globalOptions.end() ? it->second : 0;
    return kCGLNoError;
}

CGL_EXPORT CGLError CGLSetOption(CGLGlobalOption pname, GLint param) { return CGLSetGlobalOption(pname, &param); }
CGL_EXPORT CGLError CGLGetOption(CGLGlobalOption pname, GLint* param) { return CGLGetGlobalOption(pname, param); }

CGL_EXPORT void CGLGetVersion(GLint* majorvers, GLint* minorvers) {
    if (majorvers) *majorvers = 1;
    if (minorvers) *minorvers = 2;
}

CGL_EXPORT const char* CGLErrorString(CGLError error) {
    switch (error) {
    case kCGLNoError: return "no error";
    case kCGLBadAttribute: return "invalid pixel format attribute";
    case kCGLBadProperty: return "invalid renderer property";
    case kCGLBadPixelFormat: return "invalid pixel format";
    case kCGLBadRendererInfo: return "invalid renderer info";
    case kCGLBadContext: return "invalid context";
    case kCGLBadDrawable: return "invalid drawable";
    case kCGLBadDisplay: return "invalid graphics device";
    case kCGLBadState: return "invalid context state";
    case kCGLBadValue: return "invalid numerical value";
    case kCGLBadMatch: return "invalid share context";
    case kCGLBadEnumeration: return "invalid enumerant";
    case kCGLBadOffScreen: return "invalid offscreen drawable";
    case kCGLBadFullScreen: return "invalid fullscreen drawable";
    case kCGLBadWindow: return "invalid window";
    case kCGLBadAddress: return "invalid pointer";
    case kCGLBadCodeModule: return "invalid code module";
    case kCGLBadAlloc: return "invalid memory allocation";
    case kCGLBadConnection: return "invalid CoreGraphics connection";
    default: return "unknown error";
    }
}

// GLU's extension-string helper, which OpenGL.framework also exports (winemac.so imports it).
CGL_EXPORT GLboolean gluCheckExtension(const GLubyte* extName, const GLubyte* extString) {
    if (!extName || !extString) return GL_FALSE;
    const char* name = (const char*)extName;
    size_t len = strlen(name);
    if (!len) return GL_FALSE;
    for (const char* p = (const char*)extString; (p = strstr(p, name)); p += len) {
        bool startOk = p == (const char*)extString || p[-1] == ' ';
        bool endOk = p[len] == ' ' || p[len] == '\0';
        if (startOk && endOk) return GL_TRUE;
    }
    return GL_FALSE;
}

// ---------------------------------------------------------------------------

void* mgl::cgl::exportedFunction(const char* name) {
#define E(f) {#f, (void*)&f}
    static const struct {
        const char* name;
        void* fn;
    } kExports[] = {
        E(CGLChoosePixelFormat), E(CGLDestroyPixelFormat), E(CGLDescribePixelFormat), E(CGLReleasePixelFormat),
        E(CGLRetainPixelFormat), E(CGLGetPixelFormatRetainCount), E(CGLQueryRendererInfo), E(CGLDestroyRendererInfo),
        E(CGLDescribeRenderer), E(CGLCreateContext), E(CGLDestroyContext), E(CGLCopyContext), E(CGLRetainContext),
        E(CGLReleaseContext), E(CGLGetContextRetainCount), E(CGLGetPixelFormat), E(CGLGetShareGroup), E(CGLSetCurrentContext),
        E(CGLGetCurrentContext), E(CGLLockContext), E(CGLUnlockContext), E(CGLFlushDrawable), E(CGLUpdateContext),
        E(CGLClearDrawable), E(CGLEnable), E(CGLDisable), E(CGLIsEnabled), E(CGLSetParameter), E(CGLGetParameter),
        E(CGLSetVirtualScreen), E(CGLGetVirtualScreen), E(CGLSetOffScreen), E(CGLGetOffScreen), E(CGLSetFullScreen),
        E(CGLSetFullScreenOnDisplay), E(CGLCreatePBuffer), E(CGLDestroyPBuffer), E(CGLRetainPBuffer), E(CGLReleasePBuffer),
        E(CGLGetPBufferRetainCount), E(CGLDescribePBuffer), E(CGLSetPBuffer), E(CGLGetPBuffer), E(CGLTexImagePBuffer),
        E(CGLTexImageIOSurface2D), E(CGLSetGlobalOption), E(CGLGetGlobalOption), E(CGLSetOption), E(CGLGetOption),
        E(CGLGetVersion), E(CGLErrorString),
        E(gluCheckExtension),
    };
#undef E
    for (auto& e : kExports)
        if (strcmp(e.name, name) == 0) return e.fn;
    return nullptr;
}
