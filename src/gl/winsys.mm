// Window-system glue shared by the EGL and CGL frontends (see winsys.h).
#import <Foundation/Foundation.h>
#import <QuartzCore/QuartzCore.h>
#if TARGET_OS_IPHONE
#import <UIKit/UIKit.h>
#else
#import <AppKit/AppKit.h>
#endif
#import <ImageIO/ImageIO.h>
#include "gl/winsys.h"
#include "gl/internal_shaders.h"
#include "util/log.h"
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

namespace mgl {

// Sublayers we add to someone else's layer carry one of these names.
static NSString* const kSublayerFollowScale = @"metalgldrv";
static NSString* const kSublayerOwnScale = @"metalgldrv-nsgl";

Surface* newDefaultSurface(const SurfaceFormat& f) {
    Surface* s = new Surface();
    s->samples = f.samples;
    s->hasDepth = f.depth > 0;
    s->hasStencil = f.stencil > 0;
    s->hasAlpha = f.a > 0;
    s->colorFormat = MTLPixelFormatRGBA8Unorm;
    Framebuffer* fb = new Framebuffer();
    fb->isDefault = true;
    fb->surface = s;
    fb->drawBuffers[0] = GL_BACK;
    fb->readBuffer = GL_BACK;
    s->fb = fb;
    return s;
}

Surface* newWindowSurface(const SurfaceFormat& f, CAMetalLayer* layer, bool srgb) {
    Device* d = Device::get();
    layer.device = d->mtl;
    layer.pixelFormat = srgb ? MTLPixelFormatBGRA8Unorm_sRGB : MTLPixelFormatBGRA8Unorm;
    layer.framebufferOnly = YES;
    if (f.a == 0) layer.opaque = YES;
    if (layer.drawableSize.width < 1 || layer.drawableSize.height < 1) {
        CGSize b = layer.bounds.size;
        CGFloat sc = layer.contentsScale;
        layer.drawableSize = CGSizeMake(std::max<CGFloat>(1, b.width * sc), std::max<CGFloat>(1, b.height * sc));
    }
    Surface* s = newDefaultSurface(f);
    s->layer = layer;
    s->srgbCapable = srgb;
    if (srgb) s->colorFormat = MTLPixelFormatRGBA8Unorm_sRGB;
    s->syncSizeFromLayer();
    return s;
}

CAMetalLayer* newSublayer(CALayer* parent, bool followScale) {
    CAMetalLayer* ml = [CAMetalLayer layer];
    ml.name = followScale ? kSublayerFollowScale : kSublayerOwnScale;
    // No implicit animations: a resize would otherwise stretch the old frame for 0.25s.
    NSNull* none = [NSNull null];
    ml.actions = @{@"bounds" : none, @"position" : none, @"frame" : none, @"contentsScale" : none, @"contents" : none,
                   @"hidden" : none, @"opaque" : none};
    ml.anchorPoint = CGPointMake(0, 0);
    ml.frame = parent.bounds;
    ml.contentsScale = parent.contentsScale;
#if !TARGET_OS_IPHONE
    ml.autoresizingMask = kCALayerWidthSizable | kCALayerHeightSizable;
#endif
    ml.zPosition = 1;
    [parent addSublayer:ml];
    return ml;
}

void followSuperlayer(CAMetalLayer* layer) {
#if !TARGET_OS_IPHONE
    NSString* name = layer.name;
    if (!name) return;
    bool followScale = [name isEqualToString:kSublayerFollowScale];
    if (!followScale && ![name isEqualToString:kSublayerOwnScale]) return;
    CALayer* parent = layer.superlayer;
    if (!parent) return;
    CGRect pb = parent.bounds, f = layer.frame;
    CGFloat sc = parent.contentsScale;
    bool frameChanged = !CGRectEqualToRect(pb, f);
    bool scaleChanged = followScale && sc > 0 && layer.contentsScale != sc;
    if (!frameChanged && !scaleChanged) return;
    [CATransaction begin];
    [CATransaction setDisableActions:YES];
    if (frameChanged) layer.frame = pb;
    if (scaleChanged) layer.contentsScale = sc;
    [CATransaction commit];
#endif
}

#if TARGET_OS_IPHONE
CAMetalLayer* layerFromNative(void* win) {
    id obj = (__bridge id)win;
    if ([obj isKindOfClass:[CAMetalLayer class]]) return (CAMetalLayer*)obj;
    if ([obj isKindOfClass:[UIView class]]) {
        CALayer* l = ((UIView*)obj).layer;
        if ([l isKindOfClass:[CAMetalLayer class]]) return (CAMetalLayer*)l;
    }
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
#else
CAMetalLayer* layerFromNative(void* win) {
    id obj = (__bridge id)win;
    if ([obj isKindOfClass:[CAMetalLayer class]]) return (CAMetalLayer*)obj;
    if ([obj isKindOfClass:[NSView class]]) {
        NSView* v = (NSView*)obj;
        if (![v.layer isKindOfClass:[CAMetalLayer class]]) {
            v.wantsLayer = YES;
            v.layer = [CAMetalLayer layer];
        }
        return (CAMetalLayer*)v.layer;
    }
    if ([obj isKindOfClass:[CALayer class]]) {
        // A view's backing layer (GLFW's EGL path passes [view layer]): present through
        // a sublayer that follows the view's size and backing scale.
        CALayer* parent = (CALayer*)obj;
        for (CALayer* sub in parent.sublayers)
            if ([sub isKindOfClass:[CAMetalLayer class]]) return (CAMetalLayer*)sub;
        CAMetalLayer* ml = newSublayer(parent, true);
        // GL windows are opaque unless the app asks otherwise (CGL's default surface
        // opacity); alpha written by the app must not show the window behind.
        ml.opaque = YES;
        return ml;
    }
    return nil;
}
#endif

// METALGLDRV_CAPTURE=<seconds>: periodically saves the default framebuffer to
// $HOME/Documents/metalgldrv-frame.png (field debugging without screen access).
static void maybeCaptureFrame(Surface* surf) {
    static const double every = [] { const char* e = getenv("METALGLDRV_CAPTURE"); return e ? atof(e) : 0.0; }();
    if (every <= 0) return;
    static auto last = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration<double>(now - last).count() < every) return;
    last = now;
    int w = surf->width, h = surf->height;
    if (w <= 0 || h <= 0) return;
    GLint readFb = 0, packBuf = 0, align = 4, rowLen = 0, skipP = 0, skipR = 0;
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &readFb);
    glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &packBuf);
    glGetIntegerv(GL_PACK_ALIGNMENT, &align);
    glGetIntegerv(GL_PACK_ROW_LENGTH, &rowLen);
    glGetIntegerv(GL_PACK_SKIP_PIXELS, &skipP);
    glGetIntegerv(GL_PACK_SKIP_ROWS, &skipR);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glPixelStorei(GL_PACK_ROW_LENGTH, 0);
    glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
    glPixelStorei(GL_PACK_SKIP_ROWS, 0);
    std::vector<uint8_t> px((size_t)w * h * 4);
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
    glBindFramebuffer(GL_READ_FRAMEBUFFER, readFb);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, packBuf);
    glPixelStorei(GL_PACK_ALIGNMENT, align);
    glPixelStorei(GL_PACK_ROW_LENGTH, rowLen);
    glPixelStorei(GL_PACK_SKIP_PIXELS, skipP);
    glPixelStorei(GL_PACK_SKIP_ROWS, skipR);
    // GL rows are bottom-up; flip and force opaque.
    std::vector<uint8_t> img(px.size());
    for (int y = 0; y < h; y++) {
        uint8_t* d = &img[(size_t)(h - 1 - y) * w * 4];
        memcpy(d, &px[(size_t)y * w * 4], (size_t)w * 4);
        for (int x = 0; x < w; x++) d[x * 4 + 3] = 255;
    }
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGContextRef bc = CGBitmapContextCreate(img.data(), w, h, 8, (size_t)w * 4, cs, kCGImageAlphaPremultipliedLast);
    CGImageRef im = bc ? CGBitmapContextCreateImage(bc) : nullptr;
    std::string path = std::string(getenv("HOME") ? getenv("HOME") : ".") + "/Documents/metalgldrv-frame.png";
    NSURL* url = [NSURL fileURLWithPath:@(path.c_str())];
    CGImageDestinationRef dst = im ? CGImageDestinationCreateWithURL((__bridge CFURLRef)url, CFSTR("public.png"), 1, nullptr) : nullptr;
    if (dst) {
        CGImageDestinationAddImage(dst, im, nullptr);
        CGImageDestinationFinalize(dst);
        CFRelease(dst);
        MGL_INFO("captured frame %dx%d -> %s", w, h, path.c_str());
    }
    if (im) CGImageRelease(im);
    if (bc) CGContextRelease(bc);
    CGColorSpaceRelease(cs);
}

void presentWindowSurface(Context* ctx, Surface* surf) {
    // Don't touch the drawable / GPU while backgrounded.
    Device::get()->waitForeground();
    @autoreleasepool {
        if (currentContext() == ctx) maybeCaptureFrame(surf);
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
            followSuperlayer(surf->layer);
            int ow = surf->width, oh = surf->height;
            surf->syncSizeFromLayer();
            if (surf->width != ow || surf->height != oh) ctx->passDirty = true;
        }
    }
}

void setSurfaceSwapInterval(Surface* s, int interval) {
    s->swapInterval = interval;
#if !TARGET_OS_IPHONE
    if (s->layer) s->layer.displaySyncEnabled = interval > 0;
#endif
}

}  // namespace mgl
