// EGL 1.5 frontend over CAMetalLayer.
#import <Foundation/Foundation.h>
#import <QuartzCore/CAMetalLayer.h>
#if TARGET_OS_IPHONE
#import <UIKit/UIKit.h>
#else
#import <AppKit/AppKit.h>
#endif
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <cstring>
#include <mutex>
#include <set>
#include <thread>
#include "gl/context.h"
#include "gl/internal_shaders.h"
#include "gl/procs.h"
#include "util/log.h"

#define EGLAPI_EXPORT extern "C" __attribute__((visibility("default")))

using namespace mgl;

namespace {

struct Config {
    EGLint id, r, g, b, a, depth, stencil, samples;
};

const Config kConfigs[] = {
    {1, 8, 8, 8, 8, 24, 8, 0},
    {2, 8, 8, 8, 8, 0, 0, 0},
    {3, 8, 8, 8, 8, 24, 0, 0},
    {4, 8, 8, 8, 8, 24, 8, 4},
    {5, 8, 8, 8, 0, 24, 8, 0},
    {6, 8, 8, 8, 8, 32, 8, 0},
};
const int kNumConfigs = sizeof(kConfigs) / sizeof(kConfigs[0]);

struct EGLSurfaceImpl {
    Ref<Surface> surf;
    const Config* config;
    bool pbuffer;
    EGLint swapBehavior = EGL_BUFFER_DESTROYED;
};

struct EGLContextImpl {
    Ref<Context> ctx;
    const Config* config;
    EGLint clientVersion;
    EGLenum api;
    EGLint major, minor, profileMask;
    bool boundAnywhere = false;
    std::thread::id boundThread;
};

struct Display {
    bool initialized = false;
    std::mutex mutex;
    std::set<EGLSurfaceImpl*> surfaces;
    std::set<EGLContextImpl*> contexts;
};

Display g_display;
const EGLDisplay kDisplay = (EGLDisplay)&g_display;

struct EGLThreadState {
    EGLint error = EGL_SUCCESS;
    EGLenum api = EGL_OPENGL_ES_API;
    EGLContextImpl* ctx = nullptr;
    EGLSurfaceImpl* draw = nullptr;
    EGLSurfaceImpl* read = nullptr;
};
thread_local EGLThreadState t_egl;

template <typename T>
T fail(EGLint err, T ret) {
    t_egl.error = err;
    return ret;
}

bool validDisplay(EGLDisplay dpy) {
    if (dpy != kDisplay) return fail(EGL_BAD_DISPLAY, false);
    if (!g_display.initialized) return fail(EGL_NOT_INITIALIZED, false);
    return true;
}

const Config* configFromHandle(EGLConfig c) {
    for (auto& k : kConfigs)
        if ((EGLConfig)&k == c) return &k;
    return nullptr;
}

EGLSurfaceImpl* surfaceFromHandle(EGLSurface s) {
    auto* p = (EGLSurfaceImpl*)s;
    return g_display.surfaces.count(p) ? p : nullptr;
}

EGLContextImpl* contextFromHandle(EGLContext c) {
    auto* p = (EGLContextImpl*)c;
    return g_display.contexts.count(p) ? p : nullptr;
}

EGLint configAttrib(const Config* c, EGLint attr, bool* ok) {
    *ok = true;
    switch (attr) {
    case EGL_BUFFER_SIZE: return c->r + c->g + c->b + c->a;
    case EGL_RED_SIZE: return c->r;
    case EGL_GREEN_SIZE: return c->g;
    case EGL_BLUE_SIZE: return c->b;
    case EGL_ALPHA_SIZE: return c->a;
    case EGL_LUMINANCE_SIZE: return 0;
    case EGL_ALPHA_MASK_SIZE: return 0;
    case EGL_DEPTH_SIZE: return c->depth;
    case EGL_STENCIL_SIZE: return c->stencil;
    case EGL_SAMPLES: return c->samples;
    case EGL_SAMPLE_BUFFERS: return c->samples > 0 ? 1 : 0;
    case EGL_CONFIG_ID: return c->id;
    case EGL_CONFIG_CAVEAT: return EGL_NONE;
    case EGL_COLOR_BUFFER_TYPE: return EGL_RGB_BUFFER;
    case EGL_LEVEL: return 0;
    case EGL_MAX_PBUFFER_WIDTH: case EGL_MAX_PBUFFER_HEIGHT: return 16384;
    case EGL_MAX_PBUFFER_PIXELS: return 16384 * 16384;
    case EGL_MAX_SWAP_INTERVAL: return 4;
    case EGL_MIN_SWAP_INTERVAL: return 0;
    case EGL_NATIVE_RENDERABLE: return EGL_TRUE;
    case EGL_NATIVE_VISUAL_ID: return 0;
    case EGL_NATIVE_VISUAL_TYPE: return EGL_NONE;
    case EGL_RENDERABLE_TYPE:
    case EGL_CONFORMANT:
        return EGL_OPENGL_BIT | EGL_OPENGL_ES_BIT | EGL_OPENGL_ES2_BIT | EGL_OPENGL_ES3_BIT;
    case EGL_SURFACE_TYPE: return EGL_WINDOW_BIT | EGL_PBUFFER_BIT | EGL_SWAP_BEHAVIOR_PRESERVED_BIT;
    case EGL_TRANSPARENT_TYPE: return EGL_NONE;
    case EGL_TRANSPARENT_RED_VALUE: case EGL_TRANSPARENT_GREEN_VALUE: case EGL_TRANSPARENT_BLUE_VALUE: return 0;
    case EGL_BIND_TO_TEXTURE_RGB: case EGL_BIND_TO_TEXTURE_RGBA: return EGL_FALSE;
    default: *ok = false; return 0;
    }
}

CAMetalLayer* layerFromNative(void* win) {
    id obj = (__bridge id)win;
    if ([obj isKindOfClass:[CAMetalLayer class]]) return (CAMetalLayer*)obj;
#if TARGET_OS_IPHONE
    if ([obj isKindOfClass:[UIView class]]) {
        CALayer* l = ((UIView*)obj).layer;
        if ([l isKindOfClass:[CAMetalLayer class]]) return (CAMetalLayer*)l;
    }
#else
    if ([obj isKindOfClass:[NSView class]]) {
        NSView* v = (NSView*)obj;
        if (![v.layer isKindOfClass:[CAMetalLayer class]]) {
            v.wantsLayer = YES;
            v.layer = [CAMetalLayer layer];
        }
        return (CAMetalLayer*)v.layer;
    }
#endif
    if ([obj isKindOfClass:[CALayer class]]) {
        CALayer* parent = (CALayer*)obj;
        for (CALayer* sub in parent.sublayers)
            if ([sub isKindOfClass:[CAMetalLayer class]]) return (CAMetalLayer*)sub;
        CAMetalLayer* ml = [CAMetalLayer layer];
        ml.frame = parent.bounds;
        ml.contentsScale = parent.contentsScale;
        [parent addSublayer:ml];
        return ml;
    }
    return nil;
}

Surface* newSurface(const Config* c) {
    Surface* s = new Surface();
    s->samples = c->samples;
    s->hasDepth = c->depth > 0;
    s->hasStencil = c->stencil > 0;
    s->colorFormat = MTLPixelFormatRGBA8Unorm;
    Framebuffer* fb = new Framebuffer();
    fb->isDefault = true;
    fb->surface = s;
    fb->drawBuffers[0] = GL_BACK;
    fb->readBuffer = GL_BACK;
    s->fb = fb;
    return s;
}

}  // namespace

// ---------------------------------------------------------------------------
EGLAPI_EXPORT EGLint EGLAPIENTRY eglGetError(void) {
    EGLint e = t_egl.error;
    t_egl.error = EGL_SUCCESS;
    return e;
}

EGLAPI_EXPORT EGLDisplay EGLAPIENTRY eglGetDisplay(EGLNativeDisplayType display_id) { return kDisplay; }

EGLAPI_EXPORT EGLDisplay EGLAPIENTRY eglGetPlatformDisplay(EGLenum platform, void* native_display, const EGLAttrib* attrib_list) {
    return kDisplay;
}

EGLAPI_EXPORT EGLDisplay EGLAPIENTRY eglGetPlatformDisplayEXT(EGLenum platform, void* native_display, const EGLint* attrib_list) {
    return kDisplay;
}

EGLAPI_EXPORT EGLBoolean EGLAPIENTRY eglInitialize(EGLDisplay dpy, EGLint* major, EGLint* minor) {
    if (dpy != kDisplay) return fail(EGL_BAD_DISPLAY, EGL_FALSE);
    if (!Device::get()) return fail(EGL_NOT_INITIALIZED, EGL_FALSE);
    g_display.initialized = true;
    if (major) *major = 1;
    if (minor) *minor = 5;
    t_egl.error = EGL_SUCCESS;
    return EGL_TRUE;
}

EGLAPI_EXPORT EGLBoolean EGLAPIENTRY eglTerminate(EGLDisplay dpy) {
    if (dpy != kDisplay) return fail(EGL_BAD_DISPLAY, EGL_FALSE);
    // Resources stay alive while current; we keep things simple and leave them.
    g_display.initialized = false;
    return EGL_TRUE;
}

EGLAPI_EXPORT const char* EGLAPIENTRY eglQueryString(EGLDisplay dpy, EGLint name) {
    if (dpy == EGL_NO_DISPLAY && name == EGL_EXTENSIONS)
        return "EGL_EXT_client_extensions EGL_EXT_platform_base EGL_KHR_client_get_all_proc_addresses EGL_KHR_debug";
    if (!validDisplay(dpy)) return nullptr;
    switch (name) {
    case EGL_VENDOR: return "metalgldrv";
    case EGL_VERSION: return "1.5 metalgldrv";
    case EGL_CLIENT_APIS: return "OpenGL OpenGL_ES";
    case EGL_EXTENSIONS:
        return "EGL_KHR_create_context EGL_KHR_create_context_no_error EGL_KHR_surfaceless_context "
               "EGL_KHR_get_all_proc_addresses EGL_KHR_no_config_context EGL_KHR_fence_sync "
               "EGL_KHR_gl_colorspace EGL_EXT_create_context_robustness";
    default: return fail(EGL_BAD_PARAMETER, (const char*)nullptr);
    }
}

EGLAPI_EXPORT EGLBoolean EGLAPIENTRY eglGetConfigs(EGLDisplay dpy, EGLConfig* configs, EGLint config_size, EGLint* num_config) {
    if (!validDisplay(dpy)) return EGL_FALSE;
    if (!num_config) return fail(EGL_BAD_PARAMETER, EGL_FALSE);
    if (!configs) {
        *num_config = kNumConfigs;
        return EGL_TRUE;
    }
    int n = std::min(config_size, (EGLint)kNumConfigs);
    for (int i = 0; i < n; i++) configs[i] = (EGLConfig)&kConfigs[i];
    *num_config = n;
    return EGL_TRUE;
}

EGLAPI_EXPORT EGLBoolean EGLAPIENTRY eglChooseConfig(EGLDisplay dpy, const EGLint* attrib_list, EGLConfig* configs, EGLint config_size,
                                                     EGLint* num_config) {
    if (!validDisplay(dpy)) return EGL_FALSE;
    if (!num_config) return fail(EGL_BAD_PARAMETER, EGL_FALSE);
    std::vector<const Config*> matches;
    for (auto& c : kConfigs) {
        bool ok = true;
        for (const EGLint* a = attrib_list; a && *a != EGL_NONE && ok; a += 2) {
            EGLint attr = a[0], val = a[1];
            if (val == EGL_DONT_CARE) continue;
            bool known;
            EGLint have = configAttrib(&c, attr, &known);
            switch (attr) {
            case EGL_RED_SIZE: case EGL_GREEN_SIZE: case EGL_BLUE_SIZE: case EGL_ALPHA_SIZE: case EGL_DEPTH_SIZE:
            case EGL_STENCIL_SIZE: case EGL_BUFFER_SIZE: case EGL_SAMPLES: case EGL_SAMPLE_BUFFERS:
                ok = have >= val;
                break;
            case EGL_RENDERABLE_TYPE: case EGL_SURFACE_TYPE: case EGL_CONFORMANT:
                ok = (have & val) == val;
                break;
            case EGL_CONFIG_ID: ok = have == val; break;
            case EGL_NATIVE_VISUAL_TYPE: case EGL_NATIVE_RENDERABLE: case EGL_LEVEL: case EGL_TRANSPARENT_TYPE:
            case EGL_COLOR_BUFFER_TYPE: case EGL_CONFIG_CAVEAT: case EGL_MIN_SWAP_INTERVAL: case EGL_MAX_SWAP_INTERVAL:
                break;
            default: break;
            }
        }
        if (ok) matches.push_back(&c);
    }
    // EGL sort: fewer samples first, then smaller depth/stencil when equal color.
    std::stable_sort(matches.begin(), matches.end(), [](const Config* x, const Config* y) {
        if (x->samples != y->samples) return x->samples < y->samples;
        return false;
    });
    if (!configs) {
        *num_config = (EGLint)matches.size();
        return EGL_TRUE;
    }
    int n = std::min((int)matches.size(), (int)config_size);
    for (int i = 0; i < n; i++) configs[i] = (EGLConfig)matches[i];
    *num_config = n;
    return EGL_TRUE;
}

EGLAPI_EXPORT EGLBoolean EGLAPIENTRY eglGetConfigAttrib(EGLDisplay dpy, EGLConfig config, EGLint attribute, EGLint* value) {
    if (!validDisplay(dpy)) return EGL_FALSE;
    const Config* c = configFromHandle(config);
    if (!c) return fail(EGL_BAD_CONFIG, EGL_FALSE);
    bool ok;
    EGLint v = configAttrib(c, attribute, &ok);
    if (!ok) return fail(EGL_BAD_ATTRIBUTE, EGL_FALSE);
    if (value) *value = v;
    return EGL_TRUE;
}

static EGLSurface createWindowSurface(EGLDisplay dpy, EGLConfig config, void* win, bool srgb) {
    if (!validDisplay(dpy)) return EGL_NO_SURFACE;
    const Config* c = configFromHandle(config);
    if (!c) return fail(EGL_BAD_CONFIG, EGL_NO_SURFACE);
    __block CAMetalLayer* layer = nil;
    if ([NSThread isMainThread]) layer = layerFromNative(win);
    else dispatch_sync(dispatch_get_main_queue(), ^{ layer = layerFromNative(win); });
    if (!layer) return fail(EGL_BAD_NATIVE_WINDOW, EGL_NO_SURFACE);
    Device* d = Device::get();
    layer.device = d->mtl;
    layer.pixelFormat = srgb ? MTLPixelFormatBGRA8Unorm_sRGB : MTLPixelFormatBGRA8Unorm;
    layer.framebufferOnly = YES;
    layer.opaque = c->a == 0 ? YES : layer.opaque;
    if (layer.drawableSize.width < 1 || layer.drawableSize.height < 1) {
        CGSize b = layer.bounds.size;
        CGFloat sc = layer.contentsScale;
        layer.drawableSize = CGSizeMake(std::max<CGFloat>(1, b.width * sc), std::max<CGFloat>(1, b.height * sc));
    }
    auto* s = new EGLSurfaceImpl();
    s->surf = newSurface(c);
    s->surf->layer = layer;
    s->surf->srgbCapable = srgb;
    if (srgb) s->surf->colorFormat = MTLPixelFormatRGBA8Unorm_sRGB;
    s->surf->syncSizeFromLayer();
    s->config = c;
    s->pbuffer = false;
    std::lock_guard<std::mutex> g(g_display.mutex);
    g_display.surfaces.insert(s);
    MGL_INFO("window surface %dx%d", s->surf->width, s->surf->height);
    return (EGLSurface)s;
}

static bool attribSRGB(const EGLint* attrib_list) {
    for (const EGLint* a = attrib_list; a && *a != EGL_NONE; a += 2)
        if (a[0] == EGL_GL_COLORSPACE && a[1] == EGL_GL_COLORSPACE_SRGB) return true;
    return false;
}

EGLAPI_EXPORT EGLSurface EGLAPIENTRY eglCreateWindowSurface(EGLDisplay dpy, EGLConfig config, EGLNativeWindowType win,
                                                            const EGLint* attrib_list) {
    return createWindowSurface(dpy, config, (void*)win, attribSRGB(attrib_list));
}

EGLAPI_EXPORT EGLSurface EGLAPIENTRY eglCreatePlatformWindowSurface(EGLDisplay dpy, EGLConfig config, void* native_window,
                                                                    const EGLAttrib* attrib_list) {
    bool srgb = false;
    for (const EGLAttrib* a = attrib_list; a && *a != EGL_NONE; a += 2)
        if (a[0] == EGL_GL_COLORSPACE && a[1] == EGL_GL_COLORSPACE_SRGB) srgb = true;
    return createWindowSurface(dpy, config, native_window, srgb);
}

EGLAPI_EXPORT EGLSurface EGLAPIENTRY eglCreatePbufferSurface(EGLDisplay dpy, EGLConfig config, const EGLint* attrib_list) {
    if (!validDisplay(dpy)) return EGL_NO_SURFACE;
    const Config* c = configFromHandle(config);
    if (!c) return fail(EGL_BAD_CONFIG, EGL_NO_SURFACE);
    int w = 0, h = 0;
    for (const EGLint* a = attrib_list; a && *a != EGL_NONE; a += 2) {
        if (a[0] == EGL_WIDTH) w = a[1];
        if (a[0] == EGL_HEIGHT) h = a[1];
    }
    if (w < 0 || h < 0) return fail(EGL_BAD_PARAMETER, EGL_NO_SURFACE);
    auto* s = new EGLSurfaceImpl();
    s->surf = newSurface(c);
    if (attribSRGB(attrib_list)) {
        s->surf->srgbCapable = true;
        s->surf->colorFormat = MTLPixelFormatRGBA8Unorm_sRGB;
    }
    s->surf->resize(std::max(w, 1), std::max(h, 1));
    s->surf->width = w;  // a 0-size pbuffer still needs storage, but report requested size
    s->surf->height = h;
    if (w == 0 || h == 0) s->surf->resize(1, 1);
    s->config = c;
    s->pbuffer = true;
    std::lock_guard<std::mutex> g(g_display.mutex);
    g_display.surfaces.insert(s);
    return (EGLSurface)s;
}

EGLAPI_EXPORT EGLSurface EGLAPIENTRY eglCreatePixmapSurface(EGLDisplay dpy, EGLConfig config, EGLNativePixmapType pixmap,
                                                            const EGLint* attrib_list) {
    return fail(EGL_BAD_MATCH, EGL_NO_SURFACE);
}

EGLAPI_EXPORT EGLBoolean EGLAPIENTRY eglDestroySurface(EGLDisplay dpy, EGLSurface surface) {
    if (!validDisplay(dpy)) return EGL_FALSE;
    std::lock_guard<std::mutex> g(g_display.mutex);
    EGLSurfaceImpl* s = surfaceFromHandle(surface);
    if (!s) return fail(EGL_BAD_SURFACE, EGL_FALSE);
    g_display.surfaces.erase(s);
    delete s;  // Surface itself is refcounted; contexts still bound keep it alive
    return EGL_TRUE;
}

EGLAPI_EXPORT EGLBoolean EGLAPIENTRY eglQuerySurface(EGLDisplay dpy, EGLSurface surface, EGLint attribute, EGLint* value) {
    if (!validDisplay(dpy)) return EGL_FALSE;
    EGLSurfaceImpl* s = surfaceFromHandle(surface);
    if (!s) return fail(EGL_BAD_SURFACE, EGL_FALSE);
    if (!s->pbuffer) s->surf->syncSizeFromLayer();
    switch (attribute) {
    case EGL_WIDTH: *value = s->surf->width; break;
    case EGL_HEIGHT: *value = s->surf->height; break;
    case EGL_CONFIG_ID: *value = s->config->id; break;
    case EGL_RENDER_BUFFER: *value = s->pbuffer ? EGL_SINGLE_BUFFER : EGL_BACK_BUFFER; break;
    case EGL_SWAP_BEHAVIOR: *value = s->swapBehavior; break;
    case EGL_MULTISAMPLE_RESOLVE: *value = EGL_MULTISAMPLE_RESOLVE_DEFAULT; break;
    case EGL_HORIZONTAL_RESOLUTION: case EGL_VERTICAL_RESOLUTION: case EGL_PIXEL_ASPECT_RATIO: *value = EGL_UNKNOWN; break;
    case EGL_LARGEST_PBUFFER: case EGL_MIPMAP_TEXTURE: case EGL_MIPMAP_LEVEL: *value = 0; break;
    case EGL_TEXTURE_FORMAT: case EGL_TEXTURE_TARGET: *value = EGL_NO_TEXTURE; break;
    case EGL_GL_COLORSPACE: *value = s->surf->srgbCapable ? EGL_GL_COLORSPACE_SRGB : EGL_GL_COLORSPACE_LINEAR; break;
    default: return fail(EGL_BAD_ATTRIBUTE, EGL_FALSE);
    }
    return EGL_TRUE;
}

EGLAPI_EXPORT EGLBoolean EGLAPIENTRY eglSurfaceAttrib(EGLDisplay dpy, EGLSurface surface, EGLint attribute, EGLint value) {
    if (!validDisplay(dpy)) return EGL_FALSE;
    EGLSurfaceImpl* s = surfaceFromHandle(surface);
    if (!s) return fail(EGL_BAD_SURFACE, EGL_FALSE);
    if (attribute == EGL_SWAP_BEHAVIOR) {
        s->swapBehavior = value;
        return EGL_TRUE;
    }
    return EGL_TRUE;
}

EGLAPI_EXPORT EGLBoolean EGLAPIENTRY eglBindAPI(EGLenum api) {
    if (api != EGL_OPENGL_API && api != EGL_OPENGL_ES_API) return fail(EGL_BAD_PARAMETER, EGL_FALSE);
    t_egl.api = api;
    return EGL_TRUE;
}

EGLAPI_EXPORT EGLenum EGLAPIENTRY eglQueryAPI(void) { return t_egl.api; }

EGLAPI_EXPORT EGLBoolean EGLAPIENTRY eglWaitClient(void) {
    if (Context* c = currentContext()) c->finish();
    return EGL_TRUE;
}

EGLAPI_EXPORT EGLBoolean EGLAPIENTRY eglWaitGL(void) { return eglWaitClient(); }
EGLAPI_EXPORT EGLBoolean EGLAPIENTRY eglWaitNative(EGLint engine) { return EGL_TRUE; }

EGLAPI_EXPORT EGLContext EGLAPIENTRY eglCreateContext(EGLDisplay dpy, EGLConfig config, EGLContext share_context,
                                                      const EGLint* attrib_list) {
    if (!validDisplay(dpy)) return EGL_NO_CONTEXT;
    const Config* c = configFromHandle(config);
    if (!c && config != EGL_NO_CONFIG_KHR) return fail(EGL_BAD_CONFIG, EGL_NO_CONTEXT);
    EGLint major = 0, minor = 0, profile = 0, flags = 0, clientVersion = 0;
    bool debug = false, fwd = false, robust = false, noError = false;
    for (const EGLint* a = attrib_list; a && *a != EGL_NONE; a += 2) {
        switch (a[0]) {
        case EGL_CONTEXT_MAJOR_VERSION: major = a[1]; break;  // == EGL_CONTEXT_CLIENT_VERSION
        case EGL_CONTEXT_MINOR_VERSION: minor = a[1]; break;
        case EGL_CONTEXT_OPENGL_PROFILE_MASK: profile = a[1]; break;
        case EGL_CONTEXT_OPENGL_DEBUG: debug = a[1]; break;
        case EGL_CONTEXT_OPENGL_FORWARD_COMPATIBLE: fwd = a[1]; break;
        case EGL_CONTEXT_OPENGL_ROBUST_ACCESS: robust = a[1]; break;
        case EGL_CONTEXT_FLAGS_KHR: flags = a[1]; break;
        case EGL_CONTEXT_OPENGL_NO_ERROR_KHR: noError = a[1]; break;
        case EGL_CONTEXT_OPENGL_RESET_NOTIFICATION_STRATEGY: break;
        default: return fail(EGL_BAD_ATTRIBUTE, EGL_NO_CONTEXT);
        }
    }
    if (flags & EGL_CONTEXT_OPENGL_DEBUG_BIT_KHR) debug = true;
    if (flags & EGL_CONTEXT_OPENGL_FORWARD_COMPATIBLE_BIT_KHR) fwd = true;
    if (flags & EGL_CONTEXT_OPENGL_ROBUST_ACCESS_BIT_KHR) robust = true;
    clientVersion = major;
    // We always implement desktop GL 4.6. Profile selection:
    //  - explicit core request for >= 3.2 -> core profile
    //  - otherwise compatibility (superset) unless METALGLDRV_PROFILE=core
    bool core = false;
    if (t_egl.api == EGL_OPENGL_API && (major > 3 || (major == 3 && minor >= 2)) && profile != EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT &&
        (profile == EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT))
        core = true;
    if (const char* p = getenv("METALGLDRV_PROFILE")) core = strcmp(p, "core") == 0;
    if (t_egl.api == EGL_OPENGL_API && major > 4) return fail(EGL_BAD_MATCH, EGL_NO_CONTEXT);
    if (t_egl.api == EGL_OPENGL_API && major == 4 && minor > 6) return fail(EGL_BAD_MATCH, EGL_NO_CONTEXT);

    Ref<ShareGroup> share;
    if (share_context != EGL_NO_CONTEXT) {
        EGLContextImpl* sc = contextFromHandle(share_context);
        if (!sc) return fail(EGL_BAD_CONTEXT, EGL_NO_CONTEXT);
        share = sc->ctx->share;
    } else {
        share = new ShareGroup();
    }
    auto* impl = new EGLContextImpl();
    impl->ctx = new Context(share, 4, 6, core, debug, fwd, robust, noError);
    impl->config = c;
    impl->clientVersion = clientVersion;
    impl->api = t_egl.api;
    impl->major = major;
    impl->minor = minor;
    impl->profileMask = core ? EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT : EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT;
    std::lock_guard<std::mutex> g(g_display.mutex);
    g_display.contexts.insert(impl);
    MGL_INFO("created GL 4.6 %s context (requested %d.%d)", core ? "core" : "compatibility", major, minor);
    return (EGLContext)impl;
}

EGLAPI_EXPORT EGLBoolean EGLAPIENTRY eglDestroyContext(EGLDisplay dpy, EGLContext ctx) {
    if (!validDisplay(dpy)) return EGL_FALSE;
    std::lock_guard<std::mutex> g(g_display.mutex);
    EGLContextImpl* c = contextFromHandle(ctx);
    if (!c) return fail(EGL_BAD_CONTEXT, EGL_FALSE);
    g_display.contexts.erase(c);
    if (t_egl.ctx == c) {
        // Stays alive until released by eglMakeCurrent.
        return EGL_TRUE;
    }
    delete c;
    return EGL_TRUE;
}

EGLAPI_EXPORT EGLBoolean EGLAPIENTRY eglMakeCurrent(EGLDisplay dpy, EGLSurface draw, EGLSurface read, EGLContext ctx) {
    if (dpy != kDisplay) return fail(EGL_BAD_DISPLAY, EGL_FALSE);
    if (ctx == EGL_NO_CONTEXT && (draw != EGL_NO_SURFACE || read != EGL_NO_SURFACE)) return fail(EGL_BAD_MATCH, EGL_FALSE);
    EGLContextImpl* c = nullptr;
    EGLSurfaceImpl *d = nullptr, *r = nullptr;
    {
        std::lock_guard<std::mutex> g(g_display.mutex);
        if (ctx != EGL_NO_CONTEXT) {
            c = contextFromHandle(ctx);
            if (!c && (EGLContextImpl*)ctx != t_egl.ctx) return fail(EGL_BAD_CONTEXT, EGL_FALSE);
            if (!c) c = t_egl.ctx;
        }
        if (draw != EGL_NO_SURFACE) {
            d = surfaceFromHandle(draw);
            if (!d) return fail(EGL_BAD_SURFACE, EGL_FALSE);
        }
        if (read != EGL_NO_SURFACE) {
            r = surfaceFromHandle(read);
            if (!r) return fail(EGL_BAD_SURFACE, EGL_FALSE);
        }
        if (c && c->boundAnywhere && c->boundThread != std::this_thread::get_id()) return fail(EGL_BAD_ACCESS, EGL_FALSE);
    }
    EGLContextImpl* old = t_egl.ctx;
    if (old && old != c) {
        old->ctx->flush();
        old->ctx->bindSurfaces(nullptr, nullptr);
        old->boundAnywhere = false;
        bool orphaned;
        {
            std::lock_guard<std::mutex> g(g_display.mutex);
            orphaned = !g_display.contexts.count(old);
        }
        if (orphaned) delete old;
    }
    t_egl.ctx = c;
    t_egl.draw = d;
    t_egl.read = r;
    if (c) {
        c->boundAnywhere = true;
        c->boundThread = std::this_thread::get_id();
        if (d && !d->pbuffer) d->surf->syncSizeFromLayer();
        if (r && !r->pbuffer && r != d) r->surf->syncSizeFromLayer();
        c->ctx->bindSurfaces(d ? d->surf.get() : nullptr, r ? r->surf.get() : nullptr);
        setCurrentContext(c->ctx.get());
    } else {
        setCurrentContext(nullptr);
    }
    return EGL_TRUE;
}

EGLAPI_EXPORT EGLContext EGLAPIENTRY eglGetCurrentContext(void) { return t_egl.ctx ? (EGLContext)t_egl.ctx : EGL_NO_CONTEXT; }

EGLAPI_EXPORT EGLSurface EGLAPIENTRY eglGetCurrentSurface(EGLint readdraw) {
    if (readdraw == EGL_DRAW) return t_egl.draw ? (EGLSurface)t_egl.draw : EGL_NO_SURFACE;
    if (readdraw == EGL_READ) return t_egl.read ? (EGLSurface)t_egl.read : EGL_NO_SURFACE;
    return fail(EGL_BAD_PARAMETER, EGL_NO_SURFACE);
}

EGLAPI_EXPORT EGLDisplay EGLAPIENTRY eglGetCurrentDisplay(void) { return t_egl.ctx ? kDisplay : EGL_NO_DISPLAY; }

EGLAPI_EXPORT EGLBoolean EGLAPIENTRY eglQueryContext(EGLDisplay dpy, EGLContext ctx, EGLint attribute, EGLint* value) {
    if (!validDisplay(dpy)) return EGL_FALSE;
    EGLContextImpl* c = contextFromHandle(ctx);
    if (!c) return fail(EGL_BAD_CONTEXT, EGL_FALSE);
    switch (attribute) {
    case EGL_CONFIG_ID: *value = c->config ? c->config->id : 0; break;
    case EGL_CONTEXT_CLIENT_TYPE: *value = c->api; break;
    case EGL_CONTEXT_CLIENT_VERSION: *value = c->clientVersion ? c->clientVersion : 4; break;
    case EGL_RENDER_BUFFER: *value = t_egl.draw && t_egl.ctx == c ? (t_egl.draw->pbuffer ? EGL_SINGLE_BUFFER : EGL_BACK_BUFFER) : EGL_NONE; break;
    default: return fail(EGL_BAD_ATTRIBUTE, EGL_FALSE);
    }
    return EGL_TRUE;
}

EGLAPI_EXPORT EGLBoolean EGLAPIENTRY eglSwapBuffers(EGLDisplay dpy, EGLSurface surface) {
    if (!validDisplay(dpy)) return EGL_FALSE;
    EGLSurfaceImpl* s = surfaceFromHandle(surface);
    if (!s) return fail(EGL_BAD_SURFACE, EGL_FALSE);
    if (!t_egl.ctx || t_egl.draw != s) return fail(EGL_BAD_SURFACE, EGL_FALSE);
    Context* ctx = t_egl.ctx->ctx.get();
    if (s->pbuffer) {
        ctx->flush();
        return EGL_TRUE;
    }
    // Don't touch the drawable / GPU while backgrounded.
    Device::get()->waitForeground();
    @autoreleasepool {
        Surface* surf = s->surf.get();
        ctx->applyClearNow();
        id<CAMetalDrawable> drawable = [surf->layer nextDrawable];
        if (drawable) {
            presentSurface(ctx, surf, drawable);
            ctx->flush(drawable);
        } else {
            ctx->flush();
        }
        // Pick up layer size / scale changes for the next frame.
        if (surf->layer) {
            int ow = surf->width, oh = surf->height;
            surf->syncSizeFromLayer();
            if (surf->width != ow || surf->height != oh) ctx->passDirty = true;
        }
    }
    return EGL_TRUE;
}

EGLAPI_EXPORT EGLBoolean EGLAPIENTRY eglSwapInterval(EGLDisplay dpy, EGLint interval) {
    if (!validDisplay(dpy)) return EGL_FALSE;
    if (!t_egl.ctx || !t_egl.draw) return fail(EGL_BAD_SURFACE, EGL_FALSE);
    Surface* s = t_egl.draw->surf.get();
    s->swapInterval = interval;
#if !TARGET_OS_IPHONE
    if (s->layer) s->layer.displaySyncEnabled = interval > 0;
#endif
    return EGL_TRUE;
}

EGLAPI_EXPORT EGLBoolean EGLAPIENTRY eglReleaseThread(void) {
    eglMakeCurrent(kDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    t_egl = EGLThreadState();
    return EGL_TRUE;
}

EGLAPI_EXPORT EGLBoolean EGLAPIENTRY eglBindTexImage(EGLDisplay dpy, EGLSurface surface, EGLint buffer) { return fail(EGL_BAD_MATCH, EGL_FALSE); }
EGLAPI_EXPORT EGLBoolean EGLAPIENTRY eglReleaseTexImage(EGLDisplay dpy, EGLSurface surface, EGLint buffer) { return fail(EGL_BAD_MATCH, EGL_FALSE); }
EGLAPI_EXPORT EGLBoolean EGLAPIENTRY eglCopyBuffers(EGLDisplay dpy, EGLSurface surface, EGLNativePixmapType target) { return fail(EGL_BAD_NATIVE_PIXMAP, EGL_FALSE); }
EGLAPI_EXPORT EGLSurface EGLAPIENTRY eglCreatePbufferFromClientBuffer(EGLDisplay dpy, EGLenum buftype, EGLClientBuffer buffer, EGLConfig config, const EGLint* attrib_list) {
    return fail(EGL_BAD_PARAMETER, EGL_NO_SURFACE);
}

// --- sync (EGL 1.5 / KHR_fence_sync) ---
struct EGLSyncImpl {
    Ref<SyncObj> sync;
};

EGLAPI_EXPORT EGLSync EGLAPIENTRY eglCreateSync(EGLDisplay dpy, EGLenum type, const EGLAttrib* attrib_list) {
    if (!validDisplay(dpy)) return EGL_NO_SYNC;
    if (type != EGL_SYNC_FENCE) return fail(EGL_BAD_ATTRIBUTE, EGL_NO_SYNC);
    Context* ctx = currentContext();
    if (!ctx) return fail(EGL_BAD_MATCH, EGL_NO_SYNC);
    auto* s = new EGLSyncImpl();
    s->sync = new SyncObj();
    s->sync->serial = ctx->serial();
    ctx->flush();
    return (EGLSync)s;
}

EGLAPI_EXPORT EGLBoolean EGLAPIENTRY eglDestroySync(EGLDisplay dpy, EGLSync sync) {
    delete (EGLSyncImpl*)sync;
    return EGL_TRUE;
}

EGLAPI_EXPORT EGLint EGLAPIENTRY eglClientWaitSync(EGLDisplay dpy, EGLSync sync, EGLint flags, EGLTime timeout) {
    auto* s = (EGLSyncImpl*)sync;
    if (!s) return fail(EGL_BAD_PARAMETER, EGL_FALSE);
    if (s->sync->signaled()) return EGL_CONDITION_SATISFIED;
    if (timeout == 0) return EGL_TIMEOUT_EXPIRED;
    Device::get()->waitFor(s->sync->serial);
    return EGL_CONDITION_SATISFIED;
}

EGLAPI_EXPORT EGLBoolean EGLAPIENTRY eglGetSyncAttrib(EGLDisplay dpy, EGLSync sync, EGLint attribute, EGLAttrib* value) {
    auto* s = (EGLSyncImpl*)sync;
    if (!s) return fail(EGL_BAD_PARAMETER, EGL_FALSE);
    switch (attribute) {
    case EGL_SYNC_TYPE: *value = EGL_SYNC_FENCE; return EGL_TRUE;
    case EGL_SYNC_STATUS: *value = s->sync->signaled() ? EGL_SIGNALED : EGL_UNSIGNALED; return EGL_TRUE;
    case EGL_SYNC_CONDITION: *value = EGL_SYNC_PRIOR_COMMANDS_COMPLETE; return EGL_TRUE;
    default: return fail(EGL_BAD_ATTRIBUTE, EGL_FALSE);
    }
}

EGLAPI_EXPORT EGLBoolean EGLAPIENTRY eglWaitSync(EGLDisplay dpy, EGLSync sync, EGLint flags) { return EGL_TRUE; }

EGLAPI_EXPORT EGLImage EGLAPIENTRY eglCreateImage(EGLDisplay dpy, EGLContext ctx, EGLenum target, EGLClientBuffer buffer, const EGLAttrib* attrib_list) {
    return fail(EGL_BAD_PARAMETER, EGL_NO_IMAGE);
}
EGLAPI_EXPORT EGLBoolean EGLAPIENTRY eglDestroyImage(EGLDisplay dpy, EGLImage image) { return fail(EGL_BAD_PARAMETER, EGL_FALSE); }

// KHR aliases
EGLAPI_EXPORT EGLSyncKHR EGLAPIENTRY eglCreateSyncKHR(EGLDisplay dpy, EGLenum type, const EGLint* attrib_list) {
    return (EGLSyncKHR)eglCreateSync(dpy, type, nullptr);
}
EGLAPI_EXPORT EGLBoolean EGLAPIENTRY eglDestroySyncKHR(EGLDisplay dpy, EGLSyncKHR sync) { return eglDestroySync(dpy, (EGLSync)sync); }
EGLAPI_EXPORT EGLint EGLAPIENTRY eglClientWaitSyncKHR(EGLDisplay dpy, EGLSyncKHR sync, EGLint flags, EGLTimeKHR timeout) {
    return eglClientWaitSync(dpy, (EGLSync)sync, flags, timeout);
}

extern "C" __eglMustCastToProperFunctionPointerType eglGetProcAddress(const char* procname) __attribute__((visibility("default")));

namespace {
struct EGLProc {
    const char* name;
    void* fn;
};
#define P(n) {#n, (void*)&n}
const EGLProc kEGLProcs[] = {
    P(eglGetError), P(eglGetDisplay), P(eglGetPlatformDisplay), P(eglGetPlatformDisplayEXT), P(eglInitialize), P(eglTerminate),
    P(eglQueryString), P(eglGetConfigs), P(eglChooseConfig), P(eglGetConfigAttrib), P(eglCreateWindowSurface),
    P(eglCreatePlatformWindowSurface), P(eglCreatePbufferSurface), P(eglCreatePixmapSurface), P(eglDestroySurface),
    P(eglQuerySurface), P(eglSurfaceAttrib), P(eglBindAPI), P(eglQueryAPI), P(eglWaitClient), P(eglWaitGL), P(eglWaitNative),
    P(eglCreateContext), P(eglDestroyContext), P(eglMakeCurrent), P(eglGetCurrentContext), P(eglGetCurrentSurface),
    P(eglGetCurrentDisplay), P(eglQueryContext), P(eglSwapBuffers), P(eglSwapInterval), P(eglReleaseThread),
    P(eglBindTexImage), P(eglReleaseTexImage), P(eglCopyBuffers), P(eglCreatePbufferFromClientBuffer), P(eglCreateSync),
    P(eglDestroySync), P(eglClientWaitSync), P(eglGetSyncAttrib), P(eglWaitSync), P(eglCreateImage), P(eglDestroyImage),
    P(eglCreateSyncKHR), P(eglDestroySyncKHR), P(eglClientWaitSyncKHR), P(eglGetProcAddress),
};
#undef P
}  // namespace

EGLAPI_EXPORT __eglMustCastToProperFunctionPointerType EGLAPIENTRY eglGetProcAddress(const char* procname) {
    if (!procname) return nullptr;
    if (procname[0] == 'g' && procname[1] == 'l') return (__eglMustCastToProperFunctionPointerType)mgl_lookup_proc(procname);
    for (auto& p : kEGLProcs)
        if (strcmp(p.name, procname) == 0) return (__eglMustCastToProperFunctionPointerType)p.fn;
    return nullptr;
}
