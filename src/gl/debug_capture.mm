// Field debugging: METALGLDRV_CAPTURE_PROG=<program>[,<program>...] saves every texture
// bound to that program's samplers as $METALGLDRV_DUMP_DIR (else $HOME/Documents)/capture_p<prog>_<sampler>.png at its
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
    MTLTextureDescriptor* td = texture2DDescriptor(MTLPixelFormatRGBA8Unorm, w, h);
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
    const uint8_t* p = (const uint8_t*)bufferContents(buf);
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
    // Compatibility contexts draw with a derived program; match the one the app bound.
    if (progs.empty() || !(progs.count(prog->name) || (ctx->program && progs.count(ctx->program->name)))) return;
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
    const char* dd = getenv("METALGLDRV_DUMP_DIR");
    std::string dir = dd && *dd ? std::string(dd) + "/" : std::string(getenv("HOME") ? getenv("HOME") : ".") + "/Documents/";
    {
        Framebuffer* fb = ctx->fbForTarget(GL_DRAW_FRAMEBUFFER);
        const BlendState& b = ctx->blend[0];
        const StencilFace& sf = ctx->stencil[0];
        MGL_INFO("capture prog %u state: fb %u (%dx%d pass), depth test %d func %x write %d, stencil test %d func %x ref %d mask %x "
                 "ops %x/%x/%x, blend %d %x %x %x %x eq %x, color mask %x, cull %d %x front %x, draw buffers %x %x %x %x",
                 prog->name, fb ? fb->name : 0, ctx->curPass.width, ctx->curPass.height, ctx->depthTest, ctx->depthFunc,
                 ctx->depthWrite, ctx->stencilTest, sf.func, sf.ref, sf.valueMask, sf.sfail, sf.dpfail, sf.dppass, b.enabled,
                 b.srcRGB, b.dstRGB, b.srcA, b.dstA, b.eqRGB, b.colorMask, ctx->cullFace, ctx->cullMode, ctx->frontFace,
                 fb ? fb->drawBuffers[0] : 0, fb ? fb->drawBuffers[1] : 0, fb ? fb->drawBuffers[2] : 0, fb ? fb->drawBuffers[3] : 0);
        if (fb) {
            std::string atts;
            for (int i = 0; i < kNumAttachments; i++) {
                const Attachment& a = fb->att[i];
                if (a.type == GL_NONE) continue;
                atts += " " + std::to_string(i) + "=" + (a.tex ? "tex " + std::to_string(a.tex->name) : a.rb ? "rb " + std::to_string(a.rb->name) : "default");
            }
            MGL_INFO("capture prog %u attachments:%s", prog->name, atts.c_str());
        }
    }
    // METALGLDRV_CAPTURE_TARGET=1: also save color attachment 0 as it is before this draw.
    if (getenv("METALGLDRV_CAPTURE_TARGET")) {
        Framebuffer* fb = ctx->drawFb.get();
        if (fb && fb->att[0].tex && fb->att[0].tex->mtl)
            dumpTexture(ctx, fb->att[0].tex->mtl, dir + "capture_p" + std::to_string(prog->name) + "_target", fb->att[0].level);
    }
    const glsl::Reflection& r = prog->refl;
    for (size_t i = 0; i < r.slots.size(); i++) {
        const glsl::Slot& s = r.slots[i];
        if (s.kind != glsl::SlotKind::Texture) continue;
        int unit = prog->samplerUnits[i];
        int ti = texTargetIndex(s.textureTarget);
        if (unit < 0 || unit >= kMaxTextureUnits || ti < 0) continue;
        Texture* t = ctx->textures[unit][ti].get();
        if (!t || !t->mtl) {
            MGL_INFO("capture prog %u sampler %s: unit %d has %s %u", prog->name, r.uniforms[s.index].name.c_str(), unit,
                     !t ? "no texture" : "a texture without storage", t ? t->name : 0);
            continue;
        }
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
