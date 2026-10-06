// Field debugging: METALGLDRV_CAPTURE_PROG=<program>[,<program>...] saves every texture
// bound to that program's samplers as $HOME/Documents/capture_p<prog>_<sampler>.png at its
// next draw after each METALGLDRV_CAPTURE interval (default 30 s). Values are clamped to 8 bits.
#import <ImageIO/ImageIO.h>
#include <chrono>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "gl/context.h"
#include "gl/internal_shaders.h"
#include "gl/objects.h"
#include "util/log.h"

namespace mgl {

static bool writePng(const std::string& path, const uint8_t* rgba, int w, int h) {
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGContextRef bc = CGBitmapContextCreate((void*)rgba, w, h, 8, (size_t)w * 4, cs, kCGImageAlphaNoneSkipLast);
    CGImageRef im = bc ? CGBitmapContextCreateImage(bc) : nullptr;
    NSURL* url = [NSURL fileURLWithPath:@(path.c_str())];
    CGImageDestinationRef dst = im ? CGImageDestinationCreateWithURL((__bridge CFURLRef)url, CFSTR("public.png"), 1, nullptr) : nullptr;
    bool ok = false;
    if (dst) {
        CGImageDestinationAddImage(dst, im, nullptr);
        ok = CGImageDestinationFinalize(dst);
        CFRelease(dst);
    }
    if (im) CGImageRelease(im);
    if (bc) CGContextRelease(bc);
    CGColorSpaceRelease(cs);
    return ok;
}

// Converts `src` (level 0, slice 0) to RGBA8 on the GPU and writes it out (rows flipped to image order).
static void dumpTexture(Context* ctx, id<MTLTexture> src, const std::string& path, int level = 0) {
    if (!src || src.textureType == MTLTextureType3D || src.sampleCount > 1) return;
    int w = std::max(1, (int)src.width >> level), h = std::max(1, (int)src.height >> level);
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:w height:h mipmapped:NO];
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    id<MTLTexture> tmp = [ctx->dev->mtl newTextureWithDescriptor:td];
    if (!tmp) return;
    ctx->dev->makeResident(tmp);
    BlitParams bp;
    bp.src = src;
    bp.srcLevel = level;
    bp.srcRect[0] = bp.dstRect[0] = 0;
    bp.srcRect[1] = bp.dstRect[1] = 0;
    bp.srcRect[2] = bp.dstRect[2] = (float)w;
    bp.srcRect[3] = bp.dstRect[3] = (float)h;
    bp.what = BlitParams::Color;
    blitToTexture(ctx, bp, tmp, 0, 0);
    id<MTLBuffer> buf = ctx->dev->newBuffer((size_t)w * h * 4);
    id<MTL4ComputeCommandEncoder> e = ctx->computeEncoder();
    [e copyFromTexture:tmp sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(w, h, 1) toBuffer:buf
           destinationOffset:0 destinationBytesPerRow:(NSUInteger)w * 4 destinationBytesPerImage:(NSUInteger)w * h * 4];
    ctx->endEncoder();
    uint64_t serial = ctx->serial();
    ctx->waitForResource(serial);
    std::vector<uint8_t> img((size_t)w * h * 4);
    const uint8_t* p = (const uint8_t*)buf.contents;
    for (int y = 0; y < h; y++) memcpy(&img[(size_t)(h - 1 - y) * w * 4], p + (size_t)y * w * 4, (size_t)w * 4);
    if (writePng(path + ".png", img.data(), w, h)) MGL_INFO("captured %dx%d (fmt %d) -> %s.png", w, h, (int)src.pixelFormat, path.c_str());
    for (size_t i = 0; i < img.size(); i += 4) img[i] = img[i + 1] = img[i + 2] = img[i + 3];
    writePng(path + "_alpha.png", img.data(), w, h);
    ctx->dev->releaseAfter(tmp, serial);
    ctx->dev->releaseAfter(buf, serial);
}

void debugCaptureBeforeDraw(Context* ctx, ProgramObj* prog) {
    static const std::set<GLuint> progs = [] {
        std::set<GLuint> s;
        if (const char* e = getenv("METALGLDRV_CAPTURE_PROG"))
            for (const char* q = e; *q;) {
                s.insert((GLuint)strtoul(q, (char**)&q, 10));
                while (*q == ',') q++;
            }
        return s;
    }();
    if (progs.empty() || !progs.count(prog->name)) return;
    static const double every = [] { const char* e = getenv("METALGLDRV_CAPTURE"); return e ? atof(e) : 30.0; }();
    static std::map<GLuint, std::chrono::steady_clock::time_point> last;
    auto now = std::chrono::steady_clock::now();
    auto it = last.find(prog->name);
    if (it != last.end() && std::chrono::duration<double>(now - it->second).count() < every) return;
    if (it == last.end()) {  // first sighting: start the clock so the scene has time to load
        last[prog->name] = now;
        return;
    }
    last[prog->name] = now;
    std::string dir = std::string(getenv("HOME") ? getenv("HOME") : ".") + "/Documents/";
    const glsl::Reflection& r = prog->refl;
    for (size_t i = 0; i < r.slots.size(); i++) {
        const glsl::Slot& s = r.slots[i];
        if (s.kind != glsl::SlotKind::Texture) continue;
        int unit = prog->samplerUnits[i];
        int ti = texTargetIndex(s.textureTarget);
        if (unit < 0 || unit >= kMaxTextureUnits || ti < 0) continue;
        Texture* t = ctx->textures[unit][ti].get();
        if (!t || !t->mtl) continue;
        std::string name = r.uniforms[s.index].name;
        for (auto& c : name)
            if (c == '[' || c == ']' || c == '.') c = '_';
        Sampler* so = ctx->samplers[unit].get();
        const SamplerState& ss = so ? so->state : t->sampler;
        MGL_INFO("capture prog %u sampler %s: unit %d texture %u fmt %x %dx%d, %d metal levels, base %d max %d, min %x mag %x lod [%g,%g] bias %g%s",
                 prog->name, name.c_str(), unit, t->name, t->levels[0][0].internalFormat, (int)t->mtl.width, (int)t->mtl.height,
                 (int)t->mtl.mipmapLevelCount, t->baseLevel, t->maxLevel, ss.minFilter, ss.magFilter, ss.minLod, ss.maxLod, ss.lodBias,
                 so ? " (sampler object)" : "");
        {
            bool complete = false;
            SamplerState saved = t->sampler;
            if (so) t->sampler = so->state;
            id<MTLTexture> view = textureSamplingView(ctx, t, s.textureTarget, false, &complete);
            t->sampler = saved;
            int defined = 0;
            while (defined < Texture::kMaxLevels && t->levels[0][defined].width > 0) defined++;
            MTLTextureSwizzleChannels sw = view ? view.swizzle : MTLTextureSwizzleChannels{};
            MGL_INFO("  sampling view: %s, %d levels (parent level %d), GL levels defined %d, immutable %d, swizzle %d%d%d%d (GL %x %x %x %x), fmt %d",
                     view ? "ok" : "nil", view ? (int)view.mipmapLevelCount : 0, view ? (int)view.parentRelativeLevel : -1, defined,
                     t->immutableLevels, sw.red, sw.green, sw.blue, sw.alpha, t->swizzle[0], t->swizzle[1], t->swizzle[2], t->swizzle[3],
                     view ? (int)view.pixelFormat : 0);
            if (view && view != t->mtl)
                dumpTexture(ctx, view, dir + "capture_p" + std::to_string(prog->name) + "_" + name + "_view", 0);
        }
        int nl = getenv("METALGLDRV_CAPTURE_MIPS") ? (int)t->mtl.mipmapLevelCount : 1;
        for (int l = 0; l < nl; l++)
            dumpTexture(ctx, t->mtl, dir + "capture_p" + std::to_string(prog->name) + "_" + name + (l ? "_mip" + std::to_string(l) : ""), l);
    }
}
}  // namespace mgl
