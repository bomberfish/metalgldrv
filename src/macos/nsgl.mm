// NSOpenGLContext / NSOpenGLPixelFormat on top of our CGL (cgl.mm).
//
// installNSOpenGLHooks() replaces the methods that apps and Wine use (stock GLFW's
// nsgl_context.m, LWJGL 2, NSOpenGLView, Wine's cocoa_opengl.m) so that AppKit never
// hands our CGL objects to Apple's CGL. Objects built from Apple's CGL handles (an
// app that still links OpenGL.framework and calls -initWithCGLContextObj: with
// Apple's context) keep Apple's implementation: every replacement looks the object
// up in our side tables first and calls the original method if it isn't ours.
//
// Attaching a view adds a CAMetalLayer on top of the view's layer (kept per view and
// shared by the contexts attached to it, like an NSOpenGL view surface), sized to
// the view's bounds x backingScaleFactor when wantsBestResolutionOpenGLSurface is
// set (else 1x), re-synced in -update and at each flush.
//
// Built without ARC: the replacements run inside -dealloc and init paths where
// ARC's implicit retains of self are not wanted.
#include "macos/cgl_internal.h"
#import <AppKit/AppKit.h>
#import <QuartzCore/QuartzCore.h>
#import <objc/message.h>
#import <objc/runtime.h>
#include <mutex>
#include <unordered_map>
#include "gl/winsys.h"
#include "macos/hooks.h"
#include "util/log.h"

#if __has_feature(objc_arc)
#error "nsgl.mm manages memory by hand: build it with -fno-objc-arc"
#endif

#pragma clang diagnostic ignored "-Wdeprecated-declarations"

// libobjc's weak-reference primitives (what ARC emits for __weak).
extern "C" id objc_initWeak(id* location, id value);
extern "C" id objc_storeWeak(id* location, id value);
extern "C" id objc_loadWeakRetained(id* location);
extern "C" void objc_destroyWeak(id* location);

using namespace mgl;

// One per view that has had one of our contexts attached: the CAMetalLayer we
// present into and the window surface the attached contexts share.
@interface MGLViewSurface : NSObject {
@public
    CAMetalLayer* layer;
    Ref<Surface> surface;
}
@end

@implementation MGLViewSurface
- (void)dealloc {
    CAMetalLayer* l = layer;
    if (l) {
        if ([NSThread isMainThread]) {
            [l removeFromSuperlayer];
            [l release];
        } else {
            dispatch_async(dispatch_get_main_queue(), ^{
                [l removeFromSuperlayer];
                [l release];
            });
        }
    }
    [super dealloc];
}
@end

namespace {

struct ContextState {
    CGLContextObj cgl = nullptr;           // retained
    NSOpenGLPixelFormat* format = nil;     // retained
    id view = nil;                         // weak (objc_storeWeak)
};

std::mutex g_mutex;
std::unordered_map<void*, CGLPixelFormatObj> g_formats;  // NSOpenGLPixelFormat -> retained CGL pixel format
std::unordered_map<void*, ContextState*> g_contexts;     // NSOpenGLContext -> state
thread_local NSOpenGLContext* t_current = nil;           // retained, ours only
char kViewSurfaceKey;

CGLPixelFormatObj formatOf(id obj) {
    std::lock_guard<std::mutex> g(g_mutex);
    auto it = g_formats.find((void*)obj);
    return it != g_formats.end() ? it->second : nullptr;
}

ContextState* stateOf(id obj) {
    std::lock_guard<std::mutex> g(g_mutex);
    auto it = g_contexts.find((void*)obj);
    return it != g_contexts.end() ? it->second : nullptr;
}

void onMain(void (^block)(void)) {
    if ([NSThread isMainThread]) block();
    else dispatch_sync(dispatch_get_main_queue(), block);
}

// NSObject's -dealloc, skipping Apple's NSOpenGL* -dealloc (our objects never ran Apple's init).
void deallocAsNSObject(id self) {
    struct objc_super sup = {self, [NSObject class]};
    ((void (*)(struct objc_super*, SEL))objc_msgSendSuper)(&sup, sel_registerName("dealloc"));
}

// Replaces cls's implementation of sel (adding an override when cls only inherits it)
// and returns the implementation to call for objects that aren't ours.
IMP replaceMethod(Class cls, SEL sel, IMP imp) {
    Method m = class_getInstanceMethod(cls, sel);
    if (!m) {
        MGL_WARN("NSOpenGL hooks: %s has no -%s", class_getName(cls), sel_getName(sel));
        return nullptr;
    }
    bool own = false;
    unsigned n = 0;
    Method* list = class_copyMethodList(cls, &n);
    for (unsigned i = 0; i < n; i++)
        if (list[i] == m) own = true;
    free(list);
    if (own) return method_setImplementation(m, imp);
    IMP inherited = method_getImplementation(m);
    class_addMethod(cls, sel, imp, method_getTypeEncoding(m));
    return inherited;
}

// --------------------------------------------------------------------------
// NSOpenGLPixelFormat

IMP o_pf_initWithAttributes, o_pf_initWithCGLPixelFormatObj, o_pf_CGLPixelFormatObj, o_pf_getValues, o_pf_numberOfVirtualScreens,
    o_pf_dealloc;

id pf_adopt(id self, CGLPixelFormatObj pix) {
    std::lock_guard<std::mutex> g(g_mutex);
    g_formats[(void*)self] = pix;
    return self;
}

id pf_initWithAttributes(id self, SEL _cmd, const NSOpenGLPixelFormatAttribute* attribs) {
    CGLPixelFormatObj pix = nullptr;
    GLint n = 0;
    CGLError err = CGLChoosePixelFormat((const CGLPixelFormatAttribute*)attribs, &pix, &n);
    if (err != kCGLNoError || !pix) {
        MGL_WARN("NSOpenGLPixelFormat: no matching pixel format (%s)", CGLErrorString(err));
        pf_adopt(self, nullptr);  // so -dealloc takes our path
        [self release];
        return nil;
    }
    return pf_adopt(self, pix);
}

id pf_initWithCGLPixelFormatObj(id self, SEL _cmd, CGLPixelFormatObj pix) {
    if (!cgl::isOurPixelFormat(pix))
        return ((id(*)(id, SEL, CGLPixelFormatObj))o_pf_initWithCGLPixelFormatObj)(self, _cmd, pix);
    return pf_adopt(self, CGLRetainPixelFormat(pix));
}

CGLPixelFormatObj pf_CGLPixelFormatObj(id self, SEL _cmd) {
    {
        std::lock_guard<std::mutex> g(g_mutex);
        auto it = g_formats.find((void*)self);
        if (it != g_formats.end()) return it->second;
    }
    return ((CGLPixelFormatObj(*)(id, SEL))o_pf_CGLPixelFormatObj)(self, _cmd);
}

void pf_getValues(id self, SEL _cmd, GLint* vals, NSOpenGLPixelFormatAttribute attrib, GLint screen) {
    if (CGLPixelFormatObj pix = formatOf(self)) {
        GLint v = 0;
        CGLDescribePixelFormat(pix, screen, (CGLPixelFormatAttribute)attrib, &v);
        if (vals) *vals = v;
        return;
    }
    ((void (*)(id, SEL, GLint*, NSOpenGLPixelFormatAttribute, GLint))o_pf_getValues)(self, _cmd, vals, attrib, screen);
}

GLint pf_numberOfVirtualScreens(id self, SEL _cmd) {
    if (formatOf(self)) return 1;
    return ((GLint(*)(id, SEL))o_pf_numberOfVirtualScreens)(self, _cmd);
}

void pf_dealloc(id self, SEL _cmd) {
    bool ours;
    CGLPixelFormatObj pix = nullptr;
    {
        std::lock_guard<std::mutex> g(g_mutex);
        auto it = g_formats.find((void*)self);
        ours = it != g_formats.end();
        if (ours) {
            pix = it->second;
            g_formats.erase(it);
        }
    }
    if (!ours) return ((void (*)(id, SEL))o_pf_dealloc)(self, _cmd);
    if (pix) CGLReleasePixelFormat(pix);
    deallocAsNSObject(self);
}

// --------------------------------------------------------------------------
// NSOpenGLContext

IMP o_ctx_initWithFormat, o_ctx_initWithCGLContextObj, o_ctx_dealloc, o_ctx_CGLContextObj, o_ctx_pixelFormat, o_ctx_setView,
    o_ctx_view, o_ctx_update, o_ctx_flushBuffer, o_ctx_makeCurrentContext, o_ctx_clearDrawable, o_ctx_setValues,
    o_ctx_getValues, o_ctx_setCurrentVirtualScreen, o_ctx_currentVirtualScreen, o_ctx_copyAttributes, o_ctx_setFullScreen,
    o_ctx_setOffScreen, o_ctx_createTexture, o_cls_clearCurrentContext, o_cls_currentContext;

id ctx_adopt(id self, CGLContextObj cgl, NSOpenGLPixelFormat* format) {
    auto* st = new ContextState();
    st->cgl = cgl;
    st->format = [format retain];
    objc_initWeak(&st->view, nil);
    std::lock_guard<std::mutex> g(g_mutex);
    g_contexts[(void*)self] = st;
    return self;
}

id ctx_initWithFormat(id self, SEL _cmd, NSOpenGLPixelFormat* format, NSOpenGLContext* share) {
    CGLPixelFormatObj pix = formatOf(format);
    if (!pix) return ((id(*)(id, SEL, id, id))o_ctx_initWithFormat)(self, _cmd, format, share);
    CGLContextObj shareCgl = nullptr;
    if (share) {
        ContextState* ss = stateOf(share);
        if (!ss) {
            MGL_ERR("NSOpenGLContext: cannot share with a context that isn't metalgldrv's");
            ctx_adopt(self, nullptr, nil);
            [self release];
            return nil;
        }
        shareCgl = ss->cgl;
    }
    CGLContextObj cgl = nullptr;
    CGLError err = CGLCreateContext(pix, shareCgl, &cgl);
    if (err != kCGLNoError || !cgl) {
        MGL_ERR("NSOpenGLContext: CGLCreateContext failed (%s)", CGLErrorString(err));
        ctx_adopt(self, nullptr, nil);
        [self release];
        return nil;
    }
    return ctx_adopt(self, cgl, format);
}

id ctx_initWithCGLContextObj(id self, SEL _cmd, CGLContextObj cgl) {
    if (!cgl::isOurContext(cgl)) return ((id(*)(id, SEL, CGLContextObj))o_ctx_initWithCGLContextObj)(self, _cmd, cgl);
    CGLRetainContext(cgl);
    NSOpenGLPixelFormat* format = [[NSOpenGLPixelFormat alloc] initWithCGLPixelFormatObj:CGLGetPixelFormat(cgl)];
    ctx_adopt(self, cgl, format);
    [format release];
    return self;
}

void ctx_dealloc(id self, SEL _cmd) {
    ContextState* st;
    {
        std::lock_guard<std::mutex> g(g_mutex);
        auto it = g_contexts.find((void*)self);
        if (it == g_contexts.end()) {
            st = nullptr;
        } else {
            st = it->second;
            g_contexts.erase(it);
        }
    }
    if (!st) return ((void (*)(id, SEL))o_ctx_dealloc)(self, _cmd);
    objc_destroyWeak(&st->view);
    if (st->cgl) {
        cgl::setWindowDrawable(st->cgl, nullptr);
        CGLReleaseContext(st->cgl);
    }
    [st->format release];
    delete st;
    deallocAsNSObject(self);
}

CGLContextObj ctx_CGLContextObj(id self, SEL _cmd) {
    if (ContextState* st = stateOf(self)) return st->cgl;
    return ((CGLContextObj(*)(id, SEL))o_ctx_CGLContextObj)(self, _cmd);
}

NSOpenGLPixelFormat* ctx_pixelFormat(id self, SEL _cmd) {
    if (ContextState* st = stateOf(self)) return st->format;
    return ((id(*)(id, SEL))o_ctx_pixelFormat)(self, _cmd);
}

// NSView backing conversions. Apple's NSOpenGL backs a view that doesn't want a best-resolution
// surface with a 1x layer, so AppKit's convert*ToBacking: report points for it, and GLFW/SDL size
// their framebuffer from that. Our views keep AppKit's own (2x on Retina) layer, so report 1x
// ourselves for views carrying one of our surfaces without wantsBestResolutionOpenGLSurface;
// otherwise the app renders a 2x framebuffer into the 1x drawable.
IMP o_view_rectToBacking, o_view_rectFromBacking, o_view_sizeToBacking, o_view_sizeFromBacking, o_view_pointToBacking,
    o_view_pointFromBacking;

// METALGLDRV_RENDER_SCALE multiplies the backing scale of views carrying our surfaces (macos::renderScale).
//
// The backing scale our layer renders at for the view: 0 for views that aren't ours (AppKit's own
// conversions apply), else 1 (no best-resolution surface) or the window's backing scale, times the render
// scale.
CGFloat ourBackingScale(NSView* view) {
    if (!objc_getAssociatedObject(view, &kViewSurfaceKey)) return 0;
    CGFloat s = 1.0;
    if (view.wantsBestResolutionOpenGLSurface) {
        NSWindow* w = view.window;
        s = w ? w.backingScaleFactor : (NSScreen.mainScreen ? NSScreen.mainScreen.backingScaleFactor : 1.0);
    }
    return s * macos::renderScale();
}

NSRect scaled(NSRect r, CGFloat k) { return NSMakeRect(r.origin.x * k, r.origin.y * k, r.size.width * k, r.size.height * k); }
NSSize scaled(NSSize v, CGFloat k) { return NSMakeSize(v.width * k, v.height * k); }
NSPoint scaled(NSPoint p, CGFloat k) { return NSMakePoint(p.x * k, p.y * k); }

#define BACKING_CONV(fn, T, TO)                                                    \
    T fn(NSView* self, SEL _cmd, T v) {                                            \
        CGFloat k = ourBackingScale(self);                                         \
        /* not ours, or a best-resolution view at render scale 1: AppKit's answer */ \
        if (k == 0 || (self.wantsBestResolutionOpenGLSurface && macos::renderScale() == 1.0)) \
            return ((T(*)(id, SEL, T))o_##fn)(self, _cmd, v);                      \
        return scaled(v, TO ? k : 1.0 / k);                                        \
    }
BACKING_CONV(view_rectToBacking, NSRect, true)
BACKING_CONV(view_rectFromBacking, NSRect, false)
BACKING_CONV(view_sizeToBacking, NSSize, true)
BACKING_CONV(view_sizeFromBacking, NSSize, false)
BACKING_CONV(view_pointToBacking, NSPoint, true)
BACKING_CONV(view_pointFromBacking, NSPoint, false)
#undef BACKING_CONV

// The color space the layer's pixels are tagged with. Apple's NSOpenGL drawables aren't
// color matched: their pixels go to the display as they are, and games rely on it (e.g. Iris
// shaders' "Display P3" output for wide-gamut Macs). Tag the layer with the screen's own space
// to match; an untagged layer would be matched from sRGB and look washed out.
// METALGLDRV_COLORSPACE=srgb|displayp3|none overrides ("display", the default, follows the screen).
CGColorSpaceRef layerColorSpace(NSView* view) {
    static const char* mode = [] {
        const char* v = getenv("METALGLDRV_COLORSPACE");
        return v && *v ? v : "display";
    }();
    if (!strcmp(mode, "none")) return nullptr;
    if (!strcmp(mode, "srgb")) return CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    if (!strcmp(mode, "displayp3")) return CGColorSpaceCreateWithName(kCGColorSpaceDisplayP3);
    NSScreen* screen = view.window.screen ?: NSScreen.mainScreen;
    CGColorSpaceRef cs = screen.colorSpace.CGColorSpace;
    return cs ? CGColorSpaceRetain(cs) : nullptr;
}

// Lays the view's CAMetalLayer over the view: its bounds, at the backing scale when the
// view wants a best-resolution surface, tagged with the screen's color space. Main thread.
void layoutViewLayer(NSView* view, CAMetalLayer* layer) {
    CALayer* parent = view.layer;
    if (!parent) return;
    if (layer.superlayer != parent) [parent addSublayer:layer];
    CGFloat scale = 1.0;
    if (view.wantsBestResolutionOpenGLSurface) {
        NSWindow* w = view.window;
        scale = w ? w.backingScaleFactor : (NSScreen.mainScreen ? NSScreen.mainScreen.backingScaleFactor : 1.0);
    }
    scale *= macos::renderScale();  // METALGLDRV_RENDER_SCALE (1 unless set, and under Wine)
    [CATransaction begin];
    [CATransaction setDisableActions:YES];
    if (!CGRectEqualToRect(layer.frame, parent.bounds)) layer.frame = parent.bounds;
    if (layer.contentsScale != scale) layer.contentsScale = scale;
    CGColorSpaceRef cs = layerColorSpace(view);
    if (!(cs && layer.colorspace && CFEqual(cs, layer.colorspace)) && cs != layer.colorspace) layer.colorspace = cs;
    if (cs) CGColorSpaceRelease(cs);
    [CATransaction commit];
}

// METALGLDRV_WINDOW_SCREEN=<index>|builtin: moves the app's window to that screen (its top-left corner), for
// testing on a particular display (apps like Unigine Heaven open on the main screen).
static void moveWindowToScreen(NSWindow* w) {
    const char* e = getenv("METALGLDRV_WINDOW_SCREEN");
    if (!w || !e || !*e) return;
    NSArray<NSScreen*>* screens = [NSScreen screens];
    NSScreen* target = nil;
    if (!strcmp(e, "builtin")) {
        for (NSScreen* sc in screens) {
            CGDirectDisplayID d = [sc.deviceDescription[@"NSScreenNumber"] unsignedIntValue];
            if (CGDisplayIsBuiltin(d)) target = sc;
        }
    } else if (atoi(e) >= 0 && (NSUInteger)atoi(e) < screens.count) {
        target = screens[(NSUInteger)atoi(e)];
    }
    if (!target) return MGL_WARN("METALGLDRV_WINDOW_SCREEN=%s: no such screen", e);
    NSRect f = target.visibleFrame;
    [w setFrameTopLeftPoint:NSMakePoint(NSMinX(f), NSMaxY(f))];
    MGL_INFO("moved the window to screen %s (%.0fx%.0f)", e, NSWidth(target.frame), NSHeight(target.frame));
}

// Attaches the context to the view's surface (creating the view's layer / surface on
// first use). Main thread.
void attachView(ContextState* st, NSView* view) {
    MGLViewSurface* vs = objc_getAssociatedObject(view, &kViewSurfaceKey);
    if (!vs) {
        view.wantsLayer = YES;
        CALayer* parent = view.layer;
        if (!parent) {
            MGL_ERR("NSOpenGLContext -setView: view %p has no layer", view);
            return;
        }
        vs = [[MGLViewSurface alloc] init];
        vs->layer = [newSublayer(parent, false) retain];
        objc_setAssociatedObject(view, &kViewSurfaceKey, vs, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        [vs release];
        moveWindowToScreen(view.window);
    }
    layoutViewLayer(view, vs->layer);
    if (!vs->surface || !cgl::surfaceCompatible(st->cgl, vs->surface.get())) {
        vs->surface = cgl::newWindowSurface(st->cgl, vs->layer);
        MGL_INFO("NSOpenGL: view %p (%s) surface %dx%d, scale %.1f", view, class_getName([view class]), vs->surface->width,
                 vs->surface->height, vs->layer.contentsScale);
    } else {
        vs->layer.opaque = cgl::surfaceOpacity(st->cgl) != 0;
    }
    objc_storeWeak(&st->view, view);
    cgl::setWindowDrawable(st->cgl, vs->surface.get());
}

void ctx_setView(id self, SEL _cmd, NSView* view) {
    ContextState* st = stateOf(self);
    if (!st) return ((void (*)(id, SEL, id))o_ctx_setView)(self, _cmd, view);
    if (!st->cgl) return;
    if (!view) {
        objc_storeWeak(&st->view, nil);
        cgl::setWindowDrawable(st->cgl, nullptr);
        return;
    }
    [view retain];
    onMain(^{ attachView(st, view); });
    [view release];
}

NSView* ctx_view(id self, SEL _cmd) {
    if (ContextState* st = stateOf(self)) return [objc_loadWeakRetained(&st->view) autorelease];
    return ((id(*)(id, SEL))o_ctx_view)(self, _cmd);
}

void ctx_clearDrawable(id self, SEL _cmd) {
    ContextState* st = stateOf(self);
    if (!st) return ((void (*)(id, SEL))o_ctx_clearDrawable)(self, _cmd);
    objc_storeWeak(&st->view, nil);
    if (st->cgl) CGLClearDrawable(st->cgl);
}

void ctx_update(id self, SEL _cmd) {
    ContextState* st = stateOf(self);
    if (!st) return ((void (*)(id, SEL))o_ctx_update)(self, _cmd);
    if (!st->cgl) return;
    NSView* view = objc_loadWeakRetained(&st->view);
    if (view) {
        onMain(^{
            if (MGLViewSurface* vs = objc_getAssociatedObject(view, &kViewSurfaceKey)) layoutViewLayer(view, vs->layer);
        });
        [view release];
    }
    CGLUpdateContext(st->cgl);
}

void ctx_flushBuffer(id self, SEL _cmd) {
    ContextState* st = stateOf(self);
    if (!st) return ((void (*)(id, SEL))o_ctx_flushBuffer)(self, _cmd);
    if (st->cgl) CGLFlushDrawable(st->cgl);
}

void ctx_makeCurrentContext(id self, SEL _cmd) {
    ContextState* st = stateOf(self);
    if (!st) return ((void (*)(id, SEL))o_ctx_makeCurrentContext)(self, _cmd);
    if (!st->cgl) return;
    CGLSetCurrentContext(st->cgl);
    if (t_current != self) {
        [self retain];
        [t_current release];
        t_current = self;
    }
}

void cls_clearCurrentContext(id cls, SEL _cmd) {
    if (t_current) {
        CGLSetCurrentContext(nullptr);
        [t_current release];
        t_current = nil;
        return;
    }
    ((void (*)(id, SEL))o_cls_clearCurrentContext)(cls, _cmd);
}

NSOpenGLContext* cls_currentContext(id cls, SEL _cmd) {
    if (t_current) {
        ContextState* st = stateOf(t_current);
        if (st && st->cgl && CGLGetCurrentContext() == st->cgl) return t_current;
        return nil;
    }
    return ((id(*)(id, SEL))o_cls_currentContext)(cls, _cmd);
}

// NSOpenGLContextParameter values that are CGLEnable switches rather than parameters.
bool isEnableParameter(NSInteger p) {
    return p == kCGLCESwapRectangle || p == kCGLCERasterization || p == kCGLCEStateValidation || p == kCGLCESwapLimit ||
           p == kCGLCEDisplayListOptimization || p == kCGLCEMPEngine;
}

void ctx_setValues(id self, SEL _cmd, const GLint* vals, NSInteger param) {
    ContextState* st = stateOf(self);
    if (!st) return ((void (*)(id, SEL, const GLint*, NSInteger))o_ctx_setValues)(self, _cmd, vals, param);
    if (!st->cgl || !vals) return;
    if (isEnableParameter(param)) {
        if (vals[0]) CGLEnable(st->cgl, (CGLContextEnable)param);
        else CGLDisable(st->cgl, (CGLContextEnable)param);
        return;
    }
    CGLSetParameter(st->cgl, (CGLContextParameter)param, vals);
}

void ctx_getValues(id self, SEL _cmd, GLint* vals, NSInteger param) {
    ContextState* st = stateOf(self);
    if (!st) return ((void (*)(id, SEL, GLint*, NSInteger))o_ctx_getValues)(self, _cmd, vals, param);
    if (!st->cgl || !vals) return;
    if (isEnableParameter(param)) CGLIsEnabled(st->cgl, (CGLContextEnable)param, vals);
    else if (CGLGetParameter(st->cgl, (CGLContextParameter)param, vals) != kCGLNoError) vals[0] = 0;
}

void ctx_setCurrentVirtualScreen(id self, SEL _cmd, GLint screen) {
    if (stateOf(self)) return;
    ((void (*)(id, SEL, GLint))o_ctx_setCurrentVirtualScreen)(self, _cmd, screen);
}

GLint ctx_currentVirtualScreen(id self, SEL _cmd) {
    if (stateOf(self)) return 0;
    return ((GLint(*)(id, SEL))o_ctx_currentVirtualScreen)(self, _cmd);
}

void ctx_copyAttributes(id self, SEL _cmd, NSOpenGLContext* other, GLbitfield mask) {
    if (stateOf(self)) return;
    ((void (*)(id, SEL, id, GLbitfield))o_ctx_copyAttributes)(self, _cmd, other, mask);
}

void ctx_setFullScreen(id self, SEL _cmd) {
    if (stateOf(self)) return;
    ((void (*)(id, SEL))o_ctx_setFullScreen)(self, _cmd);
}

void ctx_setOffScreen(id self, SEL _cmd, void* base, GLsizei w, GLsizei h, GLint rowbytes) {
    if (stateOf(self)) {
        MGL_WARN("NSOpenGLContext -setOffScreen: is not supported");
        return;
    }
    ((void (*)(id, SEL, void*, GLsizei, GLsizei, GLint))o_ctx_setOffScreen)(self, _cmd, base, w, h, rowbytes);
}

void ctx_createTexture(id self, SEL _cmd, GLenum target, NSView* view, GLenum format) {
    if (stateOf(self)) {
        MGL_WARN("NSOpenGLContext -createTexture:fromView:internalFormat: is not supported");
        return;
    }
    ((void (*)(id, SEL, GLenum, id, GLenum))o_ctx_createTexture)(self, _cmd, target, view, format);
}

}  // namespace

namespace mgl::macos {

CGDirectDisplayID displayOfContext(CGLContextObj cgl) {
    if (![NSThread isMainThread]) return 0;  // NSWindow.screen is main-thread only
    NSView* view = nil;
    {
        std::lock_guard<std::mutex> g(g_mutex);
        for (auto& [obj, st] : g_contexts) {
            if (st->cgl != cgl) continue;
            view = objc_loadWeakRetained(&st->view);
            break;
        }
    }
    NSNumber* n = view.window.screen.deviceDescription[@"NSScreenNumber"];
    [view release];
    return n ? (CGDirectDisplayID)n.unsignedIntValue : 0;
}

void installNSOpenGLHooks() {
    static std::once_flag once;
    std::call_once(once, [] {
        Class pf = objc_getClass("NSOpenGLPixelFormat");
        Class ctx = objc_getClass("NSOpenGLContext");
        if (!pf || !ctx) {
            MGL_ERR("NSOpenGL hooks: AppKit's NSOpenGL classes are missing");
            return;
        }
#define HOOK(cls, sel, fn) o_##fn = replaceMethod(cls, @selector(sel), (IMP)fn)
        HOOK(pf, initWithAttributes:, pf_initWithAttributes);
        HOOK(pf, initWithCGLPixelFormatObj:, pf_initWithCGLPixelFormatObj);
        HOOK(pf, CGLPixelFormatObj, pf_CGLPixelFormatObj);
        HOOK(pf, getValues:forAttribute:forVirtualScreen:, pf_getValues);
        HOOK(pf, numberOfVirtualScreens, pf_numberOfVirtualScreens);
        o_pf_dealloc = replaceMethod(pf, sel_registerName("dealloc"), (IMP)pf_dealloc);

        HOOK(ctx, initWithFormat:shareContext:, ctx_initWithFormat);
        HOOK(ctx, initWithCGLContextObj:, ctx_initWithCGLContextObj);
        o_ctx_dealloc = replaceMethod(ctx, sel_registerName("dealloc"), (IMP)ctx_dealloc);
        HOOK(ctx, CGLContextObj, ctx_CGLContextObj);
        HOOK(ctx, pixelFormat, ctx_pixelFormat);
        HOOK(ctx, setView:, ctx_setView);
        HOOK(ctx, view, ctx_view);
        HOOK(ctx, update, ctx_update);
        HOOK(ctx, flushBuffer, ctx_flushBuffer);
        HOOK(ctx, makeCurrentContext, ctx_makeCurrentContext);
        HOOK(ctx, clearDrawable, ctx_clearDrawable);
        HOOK(ctx, setValues:forParameter:, ctx_setValues);
        HOOK(ctx, getValues:forParameter:, ctx_getValues);
        HOOK(ctx, setCurrentVirtualScreen:, ctx_setCurrentVirtualScreen);
        HOOK(ctx, currentVirtualScreen, ctx_currentVirtualScreen);
        HOOK(ctx, copyAttributesFromContext:withMask:, ctx_copyAttributes);
        HOOK(ctx, setFullScreen, ctx_setFullScreen);
        HOOK(ctx, setOffScreen:width:height:rowbytes:, ctx_setOffScreen);
        HOOK(ctx, createTexture:fromView:internalFormat:, ctx_createTexture);
        Class meta = object_getClass(ctx);
        HOOK(meta, clearCurrentContext, cls_clearCurrentContext);
        HOOK(meta, currentContext, cls_currentContext);
        Class view = objc_getClass("NSView");
        HOOK(view, convertRectToBacking:, view_rectToBacking);
        HOOK(view, convertRectFromBacking:, view_rectFromBacking);
        HOOK(view, convertSizeToBacking:, view_sizeToBacking);
        HOOK(view, convertSizeFromBacking:, view_sizeFromBacking);
        HOOK(view, convertPointToBacking:, view_pointToBacking);
        HOOK(view, convertPointFromBacking:, view_pointFromBacking);
#undef HOOK
        MGL_INFO("NSOpenGLContext / NSOpenGLPixelFormat now run on metalgldrv");
    });
}

}  // namespace mgl::macos
