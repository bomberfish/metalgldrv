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
#include "util/stats.h"
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

Surface* newWindowSurface(const SurfaceFormat& f, CAMetalLayer* layer, bool srgb, bool lazySRGB) {
    Device* d = Device::get();
    layer.device = d->mtl;
    bool srgbNow = srgb && !lazySRGB;
    layer.pixelFormat = srgbNow ? MTLPixelFormatBGRA8Unorm_sRGB : MTLPixelFormatBGRA8Unorm;
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
    if (srgbNow) s->colorFormat = MTLPixelFormatRGBA8Unorm_sRGB;
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
// $METALGLDRV_DUMP_DIR/metalgldrv-frame.png, else $HOME/Documents/metalgldrv-frame.png (field debugging
// without screen access; Documents is privacy-protected for apps like Wine, a dump dir is not).
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
    const char* dir = getenv("METALGLDRV_DUMP_DIR");
    std::string path = dir && *dir ? std::string(dir) + "/metalgldrv-frame.png"
                                   : std::string(getenv("HOME") ? getenv("HOME") : ".") + "/Documents/metalgldrv-frame.png";
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

// METALGLDRV_CAPTURE_SEQ=<file>: appends every presented frame, box-filtered to 480x270 RGB (top row first), to the
// file (finding frames that are out of order or broken in a sequence). Frames are read back through a ring of pixel
// pack buffers and written 4 frames later, so the GPU keeps several frames in flight as without the capture.
static void maybeCaptureSequence(Surface* surf) {
    static FILE* f = [] {
        const char* p = getenv("METALGLDRV_CAPTURE_SEQ");
        return p && *p ? fopen(p, "wb") : nullptr;
    }();
    if (!f) return;
    int w = surf->width, h = surf->height;
    const int ow = 480, oh = 270, kRing = 4;
    if (w < ow || h < oh) return;
    static GLuint pbo[kRing] = {};
    static int pw[kRing] = {}, ph[kRing] = {};
    static uint64_t frame = 0;
    GLint readFb = 0, packBuf = 0, align = 4;
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &readFb);
    glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &packBuf);
    glGetIntegerv(GL_PACK_ALIGNMENT, &align);
    if (!pbo[0]) glGenBuffers(kRing, pbo);
    int slot = (int)(frame % kRing);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[slot]);
    if (frame >= kRing) {  // the frame read into this slot kRing frames ago
        int sw = pw[slot], sh = ph[slot];
        if (const uint8_t* px = (const uint8_t*)glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, (GLsizeiptr)sw * sh * 4, GL_MAP_READ_BIT)) {
            std::vector<uint8_t> out((size_t)ow * oh * 3);
            for (int y = 0; y < oh; y++)
                for (int x = 0; x < ow; x++) {
                    int x0 = x * sw / ow, x1 = (x + 1) * sw / ow, y0 = y * sh / oh, y1 = (y + 1) * sh / oh;
                    uint32_t sum[3] = {}, n = 0;
                    for (int yy = y0; yy < y1; yy++)
                        for (int xx = x0; xx < x1; xx++, n++)
                            for (int c = 0; c < 3; c++) sum[c] += px[((size_t)yy * sw + xx) * 4 + c];
                    for (int c = 0; c < 3; c++) out[((size_t)(oh - 1 - y) * ow + x) * 3 + c] = (uint8_t)(sum[c] / std::max(1u, n));
                }
            glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
            fwrite(out.data(), 1, out.size(), f);
            fflush(f);
        }
    }
    glBufferData(GL_PIXEL_PACK_BUFFER, (GLsizeiptr)w * h * 4, nullptr, GL_STREAM_READ);
    pw[slot] = w;
    ph[slot] = h;
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, readFb);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, packBuf);
    glPixelStorei(GL_PACK_ALIGNMENT, align);
    frame++;
}

void presentWindowSurface(Context* ctx, Surface* surf) {
    // Don't touch the drawable / GPU while backgrounded.
    Device::get()->waitForeground();
    @autoreleasepool {
        if (currentContext() == ctx) maybeCaptureFrame(surf);
        if (currentContext() == ctx) maybeCaptureSequence(surf);
        hitchFrameEnd();
        ctx->applyClearNow();
        id<CAMetalDrawable> drawable = [surf->layer nextDrawable];
        if (drawable) {
            presentSurface(ctx, surf, drawable);
            ctx->flush(drawable);
        } else {
            ctx->flush();
        }
        // At most kMaxFramesInFlight frames queued on the GPU. nextDrawable doesn't always hold the CPU back: when
        // the display sleeps or the window is hidden, drawables come back at once, and a GPU-bound app (Unigine
        // Heaven) then queued command buffers, and the memory they hold, without limit.
        constexpr size_t kMaxFramesInFlight = 3;
        ctx->frameSerials.push_back(ctx->lastSubmittedSerial());
        while (ctx->frameSerials.size() > kMaxFramesInFlight) {
            ctx->dev->waitFor(ctx->frameSerials.front());
            ctx->frameSerials.pop_front();
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
    // Presenting without display sync (CAMetalLayer.displaySyncEnabled = NO) shows frames out of turn: the image
    // jumps back and forth and tears, also in windows (Unigine Heaven, 2026-10-11). Apple's GL composites its windows
    // and never shows that, so swap interval 0 presents with display sync too: frame rates above the display's
    // refresh rate are capped. METALGLDRV_SWAP_INTERVAL=<n> overrides the app's interval, and only an explicit 0 there
    // presents without display sync.
    const char* e = getenv("METALGLDRV_SWAP_INTERVAL");
    if (e && *e) interval = atoi(e);
    bool displaySync = interval > 0 || !(e && *e);
    s->swapInterval = interval;
#if !TARGET_OS_IPHONE
    if (s->layer) {
        s->layer.displaySyncEnabled = displaySync;
        // Without display sync, a window the display scans out directly (full screen, Game Mode: the Metal HUD
        // says "Direct") tears, and with framebuffer-only (losslessly compressed) drawables the tear shows
        // half-written compressed tiles: flickering bands of colored noise. Shader-readable drawables aren't
        // compressed that way, so the tear is a plain line.
        s->layer.framebufferOnly = displaySync;
    }
#endif
}

}  // namespace mgl
