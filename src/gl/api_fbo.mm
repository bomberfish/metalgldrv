// Framebuffers, renderbuffers, read pixels and blits.
#import "gl/texture.h"
#include "gl/internal_shaders.h"
#include "util/log.h"

using namespace mgl;
#define API extern "C" __attribute__((visibility("default")))

// ---------------------------------------------------------------------------
// completeness
// ---------------------------------------------------------------------------
static GLenum attachmentStatus(Context* ctx, const Attachment& a, bool isColor, bool isDepth, bool isStencil, int* w, int* h, int* samples,
                               int* layers, bool* layered) {
    *layered = false;
    *layers = 1;
    if (a.type == GL_RENDERBUFFER) {
        Renderbuffer* rb = a.rb.get();
        if (!rb || !rb->fmt || rb->width == 0 || rb->height == 0) return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        const FormatInfo* f = rb->fmt;
        if (isColor && (isDepthOrStencil(f) || !f->renderable)) return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        if (isDepth && !(f->kind == FmtKind::Depth || f->kind == FmtKind::DepthStencil)) return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        if (isStencil && !(f->kind == FmtKind::Stencil || f->kind == FmtKind::DepthStencil)) return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        *w = rb->width;
        *h = rb->height;
        *samples = rb->samples;
        return GL_FRAMEBUFFER_COMPLETE;
    }
    if (a.type == GL_TEXTURE) {
        Texture* t = a.tex.get();
        if (!t) return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        int face = t->target == GL_TEXTURE_CUBE_MAP && !a.layered ? a.layer : 0;
        if (a.level < 0 || a.level >= Texture::kMaxLevels) return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        const TexImageLevel& L = t->levels[face][a.level];
        if (!L.defined() || L.width == 0 || L.height == 0) return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        const FormatInfo* f = L.fmt;
        if (isColor && (isDepthOrStencil(f) || !f->renderable)) return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        if (isDepth && !(f->kind == FmtKind::Depth || f->kind == FmtKind::DepthStencil)) return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        if (isStencil && !(f->kind == FmtKind::Stencil || f->kind == FmtKind::DepthStencil)) return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        if (t->immutable && a.level >= t->immutableLevels) return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        if (t->target == GL_TEXTURE_3D && !a.layered && a.layer >= L.depth) return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        if ((t->target == GL_TEXTURE_2D_ARRAY || t->target == GL_TEXTURE_CUBE_MAP_ARRAY) && !a.layered && a.layer >= L.depth)
            return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        *w = L.width;
        *h = t->target == GL_TEXTURE_1D || t->target == GL_TEXTURE_1D_ARRAY ? 1 : L.height;
        *samples = t->samples;
        if (a.layered) {
            *layered = true;
            *layers = t->target == GL_TEXTURE_CUBE_MAP ? 6 : t->target == GL_TEXTURE_1D_ARRAY ? L.height : L.depth;
        }
        return GL_FRAMEBUFFER_COMPLETE;
    }
    return GL_FRAMEBUFFER_COMPLETE;
}

GLenum framebufferStatus(Context* ctx, Framebuffer* fb) {
    if (fb->isDefault) return fb->surface ? GL_FRAMEBUFFER_COMPLETE : GL_FRAMEBUFFER_UNDEFINED;
    bool any = false;
    int samples = -1;
    int layeredState = -1;
    GLenum layerTarget = 0;
    for (int i = 0; i < kNumAttachments; i++) {
        const Attachment& a = fb->att[i];
        if (a.type == GL_NONE) continue;
        int w, h, s, layers;
        bool layered;
        GLenum st = attachmentStatus(ctx, a, i < kMaxDrawBuffers, i == kAttDepth, i == kAttStencil, &w, &h, &s, &layers, &layered);
        if (st != GL_FRAMEBUFFER_COMPLETE) return st;
        any = true;
        if (samples < 0) samples = s;
        else if (samples != s) return GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE;
        if (layeredState < 0) {
            layeredState = layered;
            layerTarget = a.type == GL_TEXTURE ? a.tex->target : 0;
        } else if (layeredState != (int)layered) {
            return GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS;
        } else if (layered && a.type == GL_TEXTURE && a.tex->target != layerTarget) {
            return GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS;
        }
    }
    // depth & stencil must be the same image if both are combined formats from different images
    const Attachment& d = fb->att[kAttDepth];
    const Attachment& s = fb->att[kAttStencil];
    if (d.type != GL_NONE && s.type != GL_NONE && !(d == s)) {
        // Metal requires depth & stencil to come from one texture when the format is combined.
        const FormatInfo* df;
        const FormatInfo* sf;
        int w, h;
        attachmentTexture(ctx, d, &df, &w, &h);
        attachmentTexture(ctx, s, &sf, &w, &h);
        if (df && sf && (df->kind == FmtKind::DepthStencil || sf->kind == FmtKind::DepthStencil)) return GL_FRAMEBUFFER_UNSUPPORTED;
    }
    if (!any) {
        if (fb->defaultWidth == 0 || fb->defaultHeight == 0) return GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT;
    }
    return GL_FRAMEBUFFER_COMPLETE;
}

bool mglFramebufferComplete(Context* ctx, Framebuffer* fb, GLenum* status) {
    *status = framebufferStatus(ctx, fb);
    return *status == GL_FRAMEBUFFER_COMPLETE;
}

// ---------------------------------------------------------------------------
// renderbuffers
// ---------------------------------------------------------------------------
API void APIENTRY glGenRenderbuffers(GLsizei n, GLuint* rbs) {
    MGL_CTX();
    if (n < 0) return ctx->setError(GL_INVALID_VALUE);
    for (int i = 0; i < n; i++) rbs[i] = ctx->share->renderbuffers.genName();
}

API void APIENTRY glCreateRenderbuffers(GLsizei n, GLuint* rbs) {
    MGL_CTX();
    if (n < 0) return ctx->setError(GL_INVALID_VALUE);
    for (int i = 0; i < n; i++) {
        rbs[i] = ctx->share->renderbuffers.genName();
        ctx->share->renderbuffers.create(rbs[i], new Renderbuffer());
    }
}

API void APIENTRY glDeleteRenderbuffers(GLsizei n, const GLuint* rbs) {
    MGL_CTX();
    if (n < 0) return ctx->setError(GL_INVALID_VALUE);
    for (int i = 0; i < n; i++) {
        if (Renderbuffer* rb = ctx->share->renderbuffers.get(rbs[i])) ctx->unbindDeletedRenderbuffer(rb);
        ctx->share->renderbuffers.remove(rbs[i]);
    }
}

API GLboolean APIENTRY glIsRenderbuffer(GLuint rb) {
    MGL_CTX_RET(GL_FALSE);
    return rb && ctx->share->renderbuffers.get(rb) != nullptr;
}

API void APIENTRY glBindRenderbuffer(GLenum target, GLuint rb) {
    MGL_CTX();
    if (target != GL_RENDERBUFFER) return ctx->setError(GL_INVALID_ENUM);
    if (!rb) {
        ctx->renderbuffer = nullptr;
        return;
    }
    Renderbuffer* r = ctx->share->renderbuffers.get(rb);
    if (!r) {
        if (!ctx->share->renderbuffers.isName(rb) && ctx->coreProfile) return ctx->setError(GL_INVALID_OPERATION);
        r = ctx->share->renderbuffers.create(rb, new Renderbuffer());
    }
    ctx->renderbuffer = r;
}

static void renderbufferStorage(Context* ctx, Renderbuffer* rb, GLsizei samples, GLenum internalformat, GLsizei w, GLsizei h) {
    const FormatInfo* f = resolveInternalFormat(internalformat, GL_RGBA, GL_UNSIGNED_BYTE);
    if (internalformat == GL_DEPTH_COMPONENT) f = formatInfo(GL_DEPTH_COMPONENT24);
    if (internalformat == GL_DEPTH_STENCIL) f = formatInfo(GL_DEPTH24_STENCIL8);
    if (internalformat == GL_STENCIL_INDEX || internalformat == GL_STENCIL_INDEX1 || internalformat == GL_STENCIL_INDEX4 ||
        internalformat == GL_STENCIL_INDEX16)
        f = formatInfo(GL_STENCIL_INDEX8);
    // RGB9_E5 is accepted (as Mesa does on desktop GL) but leaves the framebuffer incomplete.
    if (!f || (!f->renderable && f->internalFormat != GL_RGB9_E5))
        return ctx->setError(GL_INVALID_ENUM, "renderbuffer format 0x%x", internalformat);
    if (w < 0 || h < 0 || samples < 0) return ctx->setError(GL_INVALID_VALUE);
    if (w > 16384 || h > 16384) return ctx->setError(GL_INVALID_VALUE);
    if (samples > 4) return ctx->setError(GL_INVALID_OPERATION);
    int s = samples == 0 ? 0 : samples <= 1 ? 1 : samples <= 2 ? 2 : 4;
    if (s == 1) s = 0;
    if (rb->mtl) ctx->dev->releaseAfter(rb->mtl, rb->lastUse);
    if (rb->linearView) ctx->dev->retainUntil(rb->linearView, rb->lastUse);
    rb->mtl = nil;
    rb->linearView = nil;
    rb->internalFormat = internalformat;
    rb->fmt = f;
    rb->width = w;
    rb->height = h;
    rb->samples = s;
    if (w > 0 && h > 0) {
        MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f->mtl width:w height:h mipmapped:NO];
        if (s > 1) {
            td.textureType = MTLTextureType2DMultisample;
            td.sampleCount = s;
        }
        td.storageMode = MTLStorageModePrivate;
        td.hazardTrackingMode = MTLHazardTrackingModeUntracked;
        td.usage = (f->renderable ? MTLTextureUsageRenderTarget : 0) | MTLTextureUsageShaderRead |
                   (f->kind == FmtKind::DepthStencil || f->srgb ? MTLTextureUsagePixelFormatView : 0);
        if (!losslessCompressionSafe(f->mtl)) td.allowGPUOptimizedContents = NO;
        rb->mtl = [ctx->dev->mtl newTextureWithDescriptor:td];
        if (!rb->mtl) return ctx->setError(GL_OUT_OF_MEMORY);
        ctx->dev->makeResident(rb->mtl);
        // Undefined contents in GL, but initialize to avoid garbage.
    }
    // Attachments referencing this renderbuffer changed.
    if (ctx->drawFb) ctx->drawFb->version++;
    if (ctx->readFb) ctx->readFb->version++;
    ctx->passDirty = true;
}

API void APIENTRY glRenderbufferStorage(GLenum target, GLenum internalformat, GLsizei w, GLsizei h) {
    MGL_CTX();
    if (target != GL_RENDERBUFFER) return ctx->setError(GL_INVALID_ENUM);
    if (!ctx->renderbuffer) return ctx->setError(GL_INVALID_OPERATION);
    renderbufferStorage(ctx, ctx->renderbuffer.get(), 0, internalformat, w, h);
}

API void APIENTRY glRenderbufferStorageMultisample(GLenum target, GLsizei samples, GLenum internalformat, GLsizei w, GLsizei h) {
    MGL_CTX();
    if (target != GL_RENDERBUFFER) return ctx->setError(GL_INVALID_ENUM);
    if (!ctx->renderbuffer) return ctx->setError(GL_INVALID_OPERATION);
    renderbufferStorage(ctx, ctx->renderbuffer.get(), samples, internalformat, w, h);
}

API void APIENTRY glNamedRenderbufferStorage(GLuint rb, GLenum internalformat, GLsizei w, GLsizei h) {
    MGL_CTX();
    Renderbuffer* r = ctx->share->renderbuffers.get(rb);
    if (!r) return ctx->setError(GL_INVALID_OPERATION);
    renderbufferStorage(ctx, r, 0, internalformat, w, h);
}

API void APIENTRY glNamedRenderbufferStorageMultisample(GLuint rb, GLsizei samples, GLenum internalformat, GLsizei w, GLsizei h) {
    MGL_CTX();
    Renderbuffer* r = ctx->share->renderbuffers.get(rb);
    if (!r) return ctx->setError(GL_INVALID_OPERATION);
    renderbufferStorage(ctx, r, samples, internalformat, w, h);
}

static void rbParam(Context* ctx, Renderbuffer* rb, GLenum pname, GLint* p) {
    const FormatInfo* f = rb->fmt;
    switch (pname) {
    case GL_RENDERBUFFER_WIDTH: *p = rb->width; break;
    case GL_RENDERBUFFER_HEIGHT: *p = rb->height; break;
    case GL_RENDERBUFFER_INTERNAL_FORMAT: *p = rb->internalFormat; break;
    case GL_RENDERBUFFER_SAMPLES: *p = rb->samples; break;
    case GL_RENDERBUFFER_RED_SIZE: *p = f ? f->rBits : 0; break;
    case GL_RENDERBUFFER_GREEN_SIZE: *p = f ? f->gBits : 0; break;
    case GL_RENDERBUFFER_BLUE_SIZE: *p = f ? f->bBits : 0; break;
    case GL_RENDERBUFFER_ALPHA_SIZE: *p = f ? f->aBits : 0; break;
    case GL_RENDERBUFFER_DEPTH_SIZE: *p = f ? f->dBits : 0; break;
    case GL_RENDERBUFFER_STENCIL_SIZE: *p = f ? f->sBits : 0; break;
    default: ctx->setError(GL_INVALID_ENUM);
    }
}

API void APIENTRY glGetRenderbufferParameteriv(GLenum target, GLenum pname, GLint* params) {
    MGL_CTX();
    if (target != GL_RENDERBUFFER) return ctx->setError(GL_INVALID_ENUM);
    if (!ctx->renderbuffer) return ctx->setError(GL_INVALID_OPERATION);
    rbParam(ctx, ctx->renderbuffer.get(), pname, params);
}

API void APIENTRY glGetNamedRenderbufferParameteriv(GLuint rb, GLenum pname, GLint* params) {
    MGL_CTX();
    Renderbuffer* r = ctx->share->renderbuffers.get(rb);
    if (!r) return ctx->setError(GL_INVALID_OPERATION);
    rbParam(ctx, r, pname, params);
}

// ---------------------------------------------------------------------------
// framebuffers
// ---------------------------------------------------------------------------
API void APIENTRY glGenFramebuffers(GLsizei n, GLuint* fbs) {
    MGL_CTX();
    if (n < 0) return ctx->setError(GL_INVALID_VALUE);
    for (int i = 0; i < n; i++) fbs[i] = ctx->framebuffers.genName();
}

API void APIENTRY glCreateFramebuffers(GLsizei n, GLuint* fbs) {
    MGL_CTX();
    if (n < 0) return ctx->setError(GL_INVALID_VALUE);
    for (int i = 0; i < n; i++) {
        fbs[i] = ctx->framebuffers.genName();
        ctx->framebuffers.create(fbs[i], new Framebuffer());
    }
}

API void APIENTRY glDeleteFramebuffers(GLsizei n, const GLuint* fbs) {
    MGL_CTX();
    if (n < 0) return ctx->setError(GL_INVALID_VALUE);
    for (int i = 0; i < n; i++) {
        Framebuffer* fb = ctx->framebuffers.get(fbs[i]);
        if (!fb) {
            ctx->framebuffers.remove(fbs[i]);
            continue;
        }
        if (ctx->drawFb.get() == fb) {
            ctx->applyClearNow();
            ctx->drawFb = ctx->defaultFramebuffer();
            ctx->passDirty = true;
        }
        if (ctx->readFb.get() == fb) ctx->readFb = ctx->readSurface ? ctx->readSurface->fb : nullptr;
        ctx->framebuffers.remove(fbs[i]);
    }
}

API GLboolean APIENTRY glIsFramebuffer(GLuint fb) {
    MGL_CTX_RET(GL_FALSE);
    return fb && ctx->framebuffers.get(fb) != nullptr;
}

API void APIENTRY glBindFramebuffer(GLenum target, GLuint framebuffer) {
    MGL_CTX();
    if (target != GL_FRAMEBUFFER && target != GL_DRAW_FRAMEBUFFER && target != GL_READ_FRAMEBUFFER) return ctx->setError(GL_INVALID_ENUM);
    Framebuffer* fb;
    if (!framebuffer) {
        fb = nullptr;
    } else {
        fb = ctx->framebuffers.get(framebuffer);
        if (!fb) {
            if (!ctx->framebuffers.isName(framebuffer) && ctx->coreProfile) return ctx->setError(GL_INVALID_OPERATION);
            fb = ctx->framebuffers.create(framebuffer, new Framebuffer());
        }
    }
    if (target == GL_FRAMEBUFFER || target == GL_DRAW_FRAMEBUFFER) {
        Framebuffer* nfb = fb ? fb : ctx->defaultFramebuffer();
        if (ctx->drawFb.get() != nfb) {
            ctx->applyClearNow();
            ctx->drawFb = nfb;
            ctx->passDirty = true;
        }
    }
    if (target == GL_FRAMEBUFFER || target == GL_READ_FRAMEBUFFER) {
        ctx->readFb = fb ? fb : (ctx->readSurface ? ctx->readSurface->fb.get() : nullptr);
    }
}

static int attachmentIndex(Context* ctx, GLenum attachment, bool* depthStencil) {
    *depthStencil = false;
    if (attachment >= GL_COLOR_ATTACHMENT0 && attachment < GL_COLOR_ATTACHMENT0 + 32) {
        int i = attachment - GL_COLOR_ATTACHMENT0;
        return i < kMaxDrawBuffers ? i : -2;
    }
    switch (attachment) {
    case GL_DEPTH_ATTACHMENT: return kAttDepth;
    case GL_STENCIL_ATTACHMENT: return kAttStencil;
    case GL_DEPTH_STENCIL_ATTACHMENT: *depthStencil = true; return kAttDepth;
    default: return -1;
    }
}

static Framebuffer* fbTarget(Context* ctx, GLenum target) {
    Framebuffer* fb = ctx->fbForTarget(target);
    if (!fb && target != GL_FRAMEBUFFER && target != GL_DRAW_FRAMEBUFFER && target != GL_READ_FRAMEBUFFER) {
        ctx->setError(GL_INVALID_ENUM);
        return nullptr;
    }
    return fb;
}

static void setAttachment(Context* ctx, Framebuffer* fb, GLenum attachment, const Attachment& a) {
    if (!fb || fb->isDefault) return ctx->setError(GL_INVALID_OPERATION, "cannot attach to the default framebuffer");
    bool ds;
    int idx = attachmentIndex(ctx, attachment, &ds);
    if (idx == -2) return ctx->setError(GL_INVALID_OPERATION);
    if (idx < 0) return ctx->setError(GL_INVALID_ENUM);
    if (fb == ctx->drawFb.get()) ctx->applyClearNow();
    fb->att[idx] = a;
    if (ds) fb->att[kAttStencil] = a;
    fb->version++;
    if (fb == ctx->drawFb.get()) ctx->passDirty = true;
}

enum class FbTexCall { Texture, TextureND, TextureLayer };

static void framebufferTexture(Context* ctx, Framebuffer* fb, GLenum attachment, GLenum textarget, GLuint texture, GLint level, GLint layer,
                               bool layered, FbTexCall call = FbTexCall::TextureND) {
    Attachment a;
    if (texture) {
        Texture* t = ctx->share->textures.get(texture);
        // Missing textures: INVALID_VALUE for (Named)FramebufferTexture, INVALID_OPERATION
        // for the others (GL 4.6 9.2.8; Mesa does the same).
        if (!t || !t->target)
            return ctx->setError(call == FbTexCall::Texture ? GL_INVALID_VALUE : GL_INVALID_OPERATION, "texture %u does not exist", texture);
        if (t->target == GL_TEXTURE_BUFFER) return ctx->setError(GL_INVALID_OPERATION);
        if (call == FbTexCall::TextureLayer) {
            switch (t->target) {
            case GL_TEXTURE_3D: case GL_TEXTURE_2D_ARRAY: case GL_TEXTURE_1D_ARRAY: case GL_TEXTURE_CUBE_MAP_ARRAY:
            case GL_TEXTURE_2D_MULTISAMPLE_ARRAY: case GL_TEXTURE_CUBE_MAP: break;
            default: return ctx->setError(GL_INVALID_OPERATION, "texture target 0x%x is not layered", t->target);
            }
            int maxLayer = t->target == GL_TEXTURE_3D ? 2048 : t->target == GL_TEXTURE_CUBE_MAP ? 6 : 2048;
            if (layer < 0 || layer >= maxLayer) return ctx->setError(GL_INVALID_VALUE, "layer %d out of range", layer);
        }
        int maxLevel = t->target == GL_TEXTURE_3D ? 11 : 14;
        if (level < 0 || level > maxLevel) return ctx->setError(GL_INVALID_VALUE);
        if ((t->target == GL_TEXTURE_RECTANGLE || t->target == GL_TEXTURE_2D_MULTISAMPLE || t->target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY) &&
            level != 0)
            return ctx->setError(GL_INVALID_VALUE);
        if (t->immutable && level >= t->immutableLevels) return ctx->setError(GL_INVALID_VALUE, "level %d beyond the texture's storage", level);
        if (textarget && textarget != t->target &&
            !(t->target == GL_TEXTURE_CUBE_MAP && textarget >= GL_TEXTURE_CUBE_MAP_POSITIVE_X && textarget <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z))
            return ctx->setError(GL_INVALID_OPERATION);
        a.type = GL_TEXTURE;
        a.tex = t;
        a.level = level;
        a.layered = layered && (t->target == GL_TEXTURE_3D || t->target == GL_TEXTURE_2D_ARRAY || t->target == GL_TEXTURE_CUBE_MAP ||
                                t->target == GL_TEXTURE_CUBE_MAP_ARRAY || t->target == GL_TEXTURE_1D_ARRAY ||
                                t->target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY);
        a.layer = layer;
        if (textarget >= GL_TEXTURE_CUBE_MAP_POSITIVE_X && textarget <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z) a.layer = textarget - GL_TEXTURE_CUBE_MAP_POSITIVE_X;
    }
    setAttachment(ctx, fb, attachment, a);
}

API void APIENTRY glFramebufferTexture2D(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level) {
    MGL_CTX();
    Framebuffer* fb = fbTarget(ctx, target);
    if (!fb && ctx->error) return;
    framebufferTexture(ctx, fb, attachment, texture ? textarget : 0, texture, level, 0, false);
}

API void APIENTRY glFramebufferTexture1D(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level) {
    MGL_CTX();
    Framebuffer* fb = fbTarget(ctx, target);
    if (!fb && ctx->error) return;
    framebufferTexture(ctx, fb, attachment, texture ? textarget : 0, texture, level, 0, false);
}

API void APIENTRY glFramebufferTexture3D(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level, GLint zoffset) {
    MGL_CTX();
    Framebuffer* fb = fbTarget(ctx, target);
    if (!fb && ctx->error) return;
    framebufferTexture(ctx, fb, attachment, texture ? textarget : 0, texture, level, zoffset, false);
}

API void APIENTRY glFramebufferTextureLayer(GLenum target, GLenum attachment, GLuint texture, GLint level, GLint layer) {
    MGL_CTX();
    Framebuffer* fb = fbTarget(ctx, target);
    if (!fb && ctx->error) return;
    framebufferTexture(ctx, fb, attachment, 0, texture, level, layer, false, FbTexCall::TextureLayer);
}

API void APIENTRY glFramebufferTexture(GLenum target, GLenum attachment, GLuint texture, GLint level) {
    MGL_CTX();
    Framebuffer* fb = fbTarget(ctx, target);
    if (!fb && ctx->error) return;
    framebufferTexture(ctx, fb, attachment, 0, texture, level, 0, true, FbTexCall::Texture);
}

API void APIENTRY glNamedFramebufferTexture(GLuint framebuffer, GLenum attachment, GLuint texture, GLint level) {
    MGL_CTX();
    Framebuffer* fb = ctx->framebuffers.get(framebuffer);
    if (!fb) return ctx->setError(GL_INVALID_OPERATION);
    framebufferTexture(ctx, fb, attachment, 0, texture, level, 0, true, FbTexCall::Texture);
}

API void APIENTRY glNamedFramebufferTextureLayer(GLuint framebuffer, GLenum attachment, GLuint texture, GLint level, GLint layer) {
    MGL_CTX();
    Framebuffer* fb = ctx->framebuffers.get(framebuffer);
    if (!fb) return ctx->setError(GL_INVALID_OPERATION);
    framebufferTexture(ctx, fb, attachment, 0, texture, level, layer, false, FbTexCall::TextureLayer);
}

static void framebufferRenderbuffer(Context* ctx, Framebuffer* fb, GLenum attachment, GLenum rbtarget, GLuint rb) {
    Attachment a;
    if (rb) {
        if (rbtarget != GL_RENDERBUFFER) return ctx->setError(GL_INVALID_ENUM);
        Renderbuffer* r = ctx->share->renderbuffers.get(rb);
        if (!r) return ctx->setError(GL_INVALID_OPERATION);
        a.type = GL_RENDERBUFFER;
        a.rb = r;
    }
    setAttachment(ctx, fb, attachment, a);
}

API void APIENTRY glFramebufferRenderbuffer(GLenum target, GLenum attachment, GLenum rbtarget, GLuint rb) {
    MGL_CTX();
    Framebuffer* fb = fbTarget(ctx, target);
    if (!fb && ctx->error) return;
    framebufferRenderbuffer(ctx, fb, attachment, rbtarget, rb);
}

API void APIENTRY glNamedFramebufferRenderbuffer(GLuint framebuffer, GLenum attachment, GLenum rbtarget, GLuint rb) {
    MGL_CTX();
    Framebuffer* fb = ctx->framebuffers.get(framebuffer);
    if (!fb) return ctx->setError(GL_INVALID_OPERATION);
    framebufferRenderbuffer(ctx, fb, attachment, rbtarget, rb);
}

API GLenum APIENTRY glCheckFramebufferStatus(GLenum target) {
    MGL_CTX_RET(0);
    if (target != GL_FRAMEBUFFER && target != GL_DRAW_FRAMEBUFFER && target != GL_READ_FRAMEBUFFER) {
        ctx->setError(GL_INVALID_ENUM);
        return 0;
    }
    Framebuffer* fb = ctx->fbForTarget(target);
    if (!fb) return GL_FRAMEBUFFER_UNDEFINED;
    return framebufferStatus(ctx, fb);
}

API GLenum APIENTRY glCheckNamedFramebufferStatus(GLuint framebuffer, GLenum target) {
    MGL_CTX_RET(0);
    if (target != GL_FRAMEBUFFER && target != GL_DRAW_FRAMEBUFFER && target != GL_READ_FRAMEBUFFER) {
        ctx->setError(GL_INVALID_ENUM);
        return 0;
    }
    Framebuffer* fb = framebuffer ? ctx->framebuffers.get(framebuffer) : ctx->defaultFramebuffer();
    if (!fb) {
        ctx->setError(GL_INVALID_OPERATION);
        return 0;
    }
    return framebufferStatus(ctx, fb);
}

static void fbAttachmentParam(Context* ctx, Framebuffer* fb, GLenum attachment, GLenum pname, GLint* params) {
    if (!fb) return ctx->setError(GL_INVALID_OPERATION);
    if (fb->isDefault) {
        Surface* s = fb->surface;
        bool isColor = attachment == GL_BACK || attachment == GL_FRONT || attachment == GL_BACK_LEFT || attachment == GL_FRONT_LEFT;
        bool isRight = attachment == GL_FRONT_RIGHT || attachment == GL_BACK_RIGHT;  // no stereo: never present
        bool isDepth = attachment == GL_DEPTH;
        bool isStencil = attachment == GL_STENCIL;
        if (!isColor && !isRight && !isDepth && !isStencil) return ctx->setError(GL_INVALID_ENUM);
        bool present = s && (isColor || (isDepth && s->hasDepth) || (isStencil && s->hasStencil));
        if (!present && pname != GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE) {
            if (pname == GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME) *params = 0;
            else ctx->setError(GL_INVALID_OPERATION);
            return;
        }
        switch (pname) {
        case GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE: *params = present ? GL_FRAMEBUFFER_DEFAULT : GL_NONE; return;
        case GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE: case GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE: case GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE:
        case GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE: *params = isColor ? 8 : 0; return;
        case GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE: *params = isDepth && present ? 24 : 0; return;
        case GL_FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE: *params = isStencil && present ? 8 : 0; return;
        case GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE: *params = isColor ? GL_UNSIGNED_NORMALIZED : GL_UNSIGNED_NORMALIZED; return;
        case GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING: *params = s && s->srgbCapable ? GL_SRGB : GL_LINEAR; return;
        default: return ctx->setError(GL_INVALID_ENUM);
        }
    }
    bool ds;
    int idx = attachmentIndex(ctx, attachment, &ds);
    if (idx == -2) return ctx->setError(GL_INVALID_OPERATION);
    if (idx < 0) return ctx->setError(GL_INVALID_ENUM);
    const Attachment& a = fb->att[idx];
    if (ds && !(fb->att[kAttDepth] == fb->att[kAttStencil])) return ctx->setError(GL_INVALID_OPERATION);
    if (pname == GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE) {
        *params = a.type;
        return;
    }
    if (a.type == GL_NONE) {
        if (pname == GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME) *params = 0;
        else ctx->setError(GL_INVALID_OPERATION);
        return;
    }
    const FormatInfo* f = nullptr;
    if (a.type == GL_RENDERBUFFER) f = a.rb->fmt;
    else {
        int face = a.tex->target == GL_TEXTURE_CUBE_MAP && !a.layered ? a.layer : 0;
        f = a.tex->levels[face][a.level].fmt;
    }
    switch (pname) {
    case GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME: *params = a.type == GL_RENDERBUFFER ? a.rb->name : a.tex->name; return;
    case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL: if (a.type != GL_TEXTURE) break; *params = a.level; return;
    case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE:
        if (a.type != GL_TEXTURE) break;
        *params = a.tex->target == GL_TEXTURE_CUBE_MAP && !a.layered ? GL_TEXTURE_CUBE_MAP_POSITIVE_X + a.layer : 0;
        return;
    case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER: if (a.type != GL_TEXTURE) break; *params = a.layer; return;
    case GL_FRAMEBUFFER_ATTACHMENT_LAYERED: if (a.type != GL_TEXTURE) break; *params = a.layered; return;
    case GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE: *params = f ? f->rBits : 0; return;
    case GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE: *params = f ? f->gBits : 0; return;
    case GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE: *params = f ? f->bBits : 0; return;
    case GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE: *params = f ? f->aBits : 0; return;
    case GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE: *params = f ? f->dBits : 0; return;
    case GL_FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE: *params = f ? f->sBits : 0; return;
    case GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE:
        if (ds) return ctx->setError(GL_INVALID_OPERATION);
        *params = f ? (idx == kAttStencil ? GL_UNSIGNED_INT : f->glType) : GL_NONE;
        return;
    case GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING: *params = f && f->srgb ? GL_SRGB : GL_LINEAR; return;
    default: break;
    }
    ctx->setError(a.type == GL_RENDERBUFFER ? GL_INVALID_ENUM : GL_INVALID_ENUM);
}

API void APIENTRY glGetFramebufferAttachmentParameteriv(GLenum target, GLenum attachment, GLenum pname, GLint* params) {
    MGL_CTX();
    Framebuffer* fb = fbTarget(ctx, target);
    if (!fb && ctx->error) return;
    fbAttachmentParam(ctx, fb, attachment, pname, params);
}

API void APIENTRY glGetNamedFramebufferAttachmentParameteriv(GLuint framebuffer, GLenum attachment, GLenum pname, GLint* params) {
    MGL_CTX();
    Framebuffer* fb = framebuffer ? ctx->framebuffers.get(framebuffer) : ctx->defaultFramebuffer();
    fbAttachmentParam(ctx, fb, attachment, pname, params);
}

static void fbParameter(Context* ctx, Framebuffer* fb, GLenum pname, GLint param) {
    if (!fb || fb->isDefault) return ctx->setError(GL_INVALID_OPERATION);
    switch (pname) {
    case GL_FRAMEBUFFER_DEFAULT_WIDTH: if (param < 0 || param > 16384) return ctx->setError(GL_INVALID_VALUE); fb->defaultWidth = param; break;
    case GL_FRAMEBUFFER_DEFAULT_HEIGHT: if (param < 0 || param > 16384) return ctx->setError(GL_INVALID_VALUE); fb->defaultHeight = param; break;
    case GL_FRAMEBUFFER_DEFAULT_LAYERS: if (param < 0 || param > 2048) return ctx->setError(GL_INVALID_VALUE); fb->defaultLayers = param; break;
    case GL_FRAMEBUFFER_DEFAULT_SAMPLES: if (param < 0 || param > 4) return ctx->setError(GL_INVALID_VALUE); fb->defaultSamples = param; break;
    case GL_FRAMEBUFFER_DEFAULT_FIXED_SAMPLE_LOCATIONS: fb->defaultFixedSampleLocations = param; break;
    default: return ctx->setError(GL_INVALID_ENUM);
    }
    fb->version++;
}

API void APIENTRY glFramebufferParameteri(GLenum target, GLenum pname, GLint param) {
    MGL_CTX();
    fbParameter(ctx, fbTarget(ctx, target), pname, param);
}

API void APIENTRY glNamedFramebufferParameteri(GLuint framebuffer, GLenum pname, GLint param) {
    MGL_CTX();
    fbParameter(ctx, ctx->framebuffers.get(framebuffer), pname, param);
}

static int fbSamples(Framebuffer* fb) {
    if (fb->isDefault) return fb->surface && fb->surface->samples > 1 ? fb->surface->samples : 0;
    for (auto& a : fb->att) {
        if (a.type == GL_RENDERBUFFER && a.rb) return a.rb->samples > 1 ? a.rb->samples : 0;
        if (a.type == GL_TEXTURE && a.tex) return a.tex->samples > 1 ? a.tex->samples : 0;
    }
    return fb->defaultSamples;
}

static void getFbParameter(Context* ctx, Framebuffer* fb, GLenum pname, GLint* params) {
    if (!fb) return ctx->setError(GL_INVALID_OPERATION);
    switch (pname) {
    case GL_FRAMEBUFFER_DEFAULT_WIDTH: case GL_FRAMEBUFFER_DEFAULT_HEIGHT: case GL_FRAMEBUFFER_DEFAULT_LAYERS:
    case GL_FRAMEBUFFER_DEFAULT_SAMPLES: case GL_FRAMEBUFFER_DEFAULT_FIXED_SAMPLE_LOCATIONS:
        // Only framebuffer objects have these (GL 4.6 9.2.3).
        if (fb->isDefault) return ctx->setError(GL_INVALID_OPERATION);
        break;
    default: break;
    }
    switch (pname) {
    case GL_FRAMEBUFFER_DEFAULT_WIDTH: *params = fb->defaultWidth; break;
    case GL_FRAMEBUFFER_DEFAULT_HEIGHT: *params = fb->defaultHeight; break;
    case GL_FRAMEBUFFER_DEFAULT_LAYERS: *params = fb->defaultLayers; break;
    case GL_FRAMEBUFFER_DEFAULT_SAMPLES: *params = fb->defaultSamples; break;
    case GL_FRAMEBUFFER_DEFAULT_FIXED_SAMPLE_LOCATIONS: *params = fb->defaultFixedSampleLocations; break;
    case GL_DOUBLEBUFFER: *params = fb->isDefault; break;
    case GL_STEREO: *params = 0; break;
    case GL_SAMPLES: *params = fbSamples(fb); break;
    case GL_SAMPLE_BUFFERS: *params = fbSamples(fb) > 1 ? 1 : 0; break;
    case GL_IMPLEMENTATION_COLOR_READ_FORMAT: *params = GL_RGBA; break;
    case GL_IMPLEMENTATION_COLOR_READ_TYPE: *params = GL_UNSIGNED_BYTE; break;
    default: ctx->setError(GL_INVALID_ENUM);
    }
}

API void APIENTRY glGetFramebufferParameteriv(GLenum target, GLenum pname, GLint* params) {
    MGL_CTX();
    getFbParameter(ctx, fbTarget(ctx, target), pname, params);
}

API void APIENTRY glGetNamedFramebufferParameteriv(GLuint framebuffer, GLenum pname, GLint* params) {
    MGL_CTX();
    getFbParameter(ctx, framebuffer ? ctx->framebuffers.get(framebuffer) : ctx->defaultFramebuffer(), pname, params);
}

API void APIENTRY glGetMultisamplefv(GLenum pname, GLuint index, GLfloat* val) {
    MGL_CTX();
    if (pname != GL_SAMPLE_POSITION) return ctx->setError(GL_INVALID_ENUM);
    PassDesc p;
    int samples = ctx->resolvePass(p) ? p.samples : 1;
    if (index >= (GLuint)std::max(1, samples)) return ctx->setError(GL_INVALID_VALUE);
    if (samples <= 1) {
        val[0] = val[1] = 0.5f;
        return;
    }
    // Metal's standard sample locations. Rendering keeps GL's row order (window y up =
    // texture row up), so the within-pixel offsets carry over unchanged.
    MTLSamplePosition pos[16];
    [ctx->dev->mtl getDefaultSamplePositions:pos count:(NSUInteger)samples];
    val[0] = pos[index].x;
    val[1] = pos[index].y;
}

// --- draw / read buffers ---
static bool validDefaultBuffer(GLenum b) {
    return b == GL_NONE || b == GL_BACK || b == GL_FRONT || b == GL_BACK_LEFT || b == GL_FRONT_LEFT || b == GL_LEFT ||
           b == GL_FRONT_AND_BACK;
}

static bool isColorAttachmentEnum(GLenum b) { return b >= GL_COLOR_ATTACHMENT0 && b < GL_COLOR_ATTACHMENT0 + 32; }

// glDrawBuffers / glNamedFramebufferDrawBuffers validation (GL 4.6 17.4.1, Mesa order).
static void drawBuffers(Context* ctx, Framebuffer* fb, GLsizei n, const GLenum* bufs) {
    if (!fb) return ctx->setError(GL_INVALID_OPERATION);
    if (n < 0 || n > kMaxDrawBuffers) return ctx->setError(GL_INVALID_VALUE);
    for (int i = 0; i < n; i++) {
        GLenum b = bufs[i];
        bool known = b == GL_NONE || b == GL_FRONT_LEFT || b == GL_FRONT_RIGHT || b == GL_BACK_LEFT || b == GL_BACK_RIGHT || b == GL_BACK ||
                     isColorAttachmentEnum(b);
        // FRONT, LEFT, RIGHT and FRONT_AND_BACK may name several buffers: never accepted here.
        if (!known) return ctx->setError(GL_INVALID_ENUM, "draw buffer 0x%x", b);
        if (b == GL_BACK && n != 1) return ctx->setError(GL_INVALID_OPERATION);
        if (fb->isDefault) {
            if (isColorAttachmentEnum(b)) return ctx->setError(GL_INVALID_OPERATION);
            if (b == GL_FRONT_RIGHT || b == GL_BACK_RIGHT) return ctx->setError(GL_INVALID_OPERATION);  // not stereo
        } else {
            if (b != GL_NONE && !isColorAttachmentEnum(b)) return ctx->setError(GL_INVALID_OPERATION);
            if (b != GL_NONE && b >= GL_COLOR_ATTACHMENT0 + kMaxDrawBuffers) return ctx->setError(GL_INVALID_OPERATION);
        }
        for (int j = 0; j < i; j++)
            if (b != GL_NONE && bufs[j] == b) return ctx->setError(GL_INVALID_OPERATION);
    }
    if (fb == ctx->drawFb.get()) ctx->applyClearNow();
    for (int i = 0; i < kMaxDrawBuffers; i++) fb->drawBuffers[i] = i < n ? bufs[i] : GL_NONE;
    fb->version++;
    if (fb == ctx->drawFb.get()) ctx->passDirty = true;
}

API void APIENTRY glDrawBuffers(GLsizei n, const GLenum* bufs) {
    MGL_CTX();
    drawBuffers(ctx, ctx->drawFb.get(), n, bufs);
}

API void APIENTRY glNamedFramebufferDrawBuffers(GLuint framebuffer, GLsizei n, const GLenum* bufs) {
    MGL_CTX();
    drawBuffers(ctx, framebuffer ? ctx->framebuffers.get(framebuffer) : ctx->defaultFramebuffer(), n, bufs);
}

static void drawBuffer(Context* ctx, Framebuffer* fb, GLenum buf) {
    if (!fb) return ctx->setError(GL_INVALID_OPERATION);
    if (fb->isDefault) {
        if (!validDefaultBuffer(buf)) {
            if (buf >= GL_COLOR_ATTACHMENT0 && buf < GL_COLOR_ATTACHMENT0 + 32) return ctx->setError(GL_INVALID_OPERATION);
            return ctx->setError(GL_INVALID_ENUM);
        }
        if (fb == ctx->drawFb.get()) ctx->applyClearNow();
        fb->drawBuffers[0] = buf;
        for (int i = 1; i < kMaxDrawBuffers; i++) fb->drawBuffers[i] = GL_NONE;
        fb->version++;
        if (fb == ctx->drawFb.get()) ctx->passDirty = true;
        return;
    }
    if (buf != GL_NONE && !isColorAttachmentEnum(buf)) {
        bool known = buf == GL_FRONT_LEFT || buf == GL_FRONT_RIGHT || buf == GL_BACK_LEFT || buf == GL_BACK_RIGHT || buf == GL_BACK ||
                     buf == GL_FRONT || buf == GL_LEFT || buf == GL_RIGHT || buf == GL_FRONT_AND_BACK;
        return ctx->setError(known ? GL_INVALID_OPERATION : GL_INVALID_ENUM);
    }
    drawBuffers(ctx, fb, 1, &buf);
}

API void APIENTRY glDrawBuffer(GLenum buf) {
    MGL_CTX();
    drawBuffer(ctx, ctx->drawFb.get(), buf);
}

API void APIENTRY glNamedFramebufferDrawBuffer(GLuint framebuffer, GLenum buf) {
    MGL_CTX();
    drawBuffer(ctx, framebuffer ? ctx->framebuffers.get(framebuffer) : ctx->defaultFramebuffer(), buf);
}

static void readBuffer(Context* ctx, Framebuffer* fb, GLenum src) {
    if (!fb) return ctx->setError(GL_INVALID_OPERATION);
    if (fb->isDefault) {
        if (!validDefaultBuffer(src)) {
            if (src >= GL_COLOR_ATTACHMENT0 && src < GL_COLOR_ATTACHMENT0 + 32) return ctx->setError(GL_INVALID_OPERATION);
            return ctx->setError(GL_INVALID_ENUM);
        }
    } else {
        if (src != GL_NONE && (src < GL_COLOR_ATTACHMENT0 || src >= GL_COLOR_ATTACHMENT0 + 32)) {
            if (validDefaultBuffer(src)) return ctx->setError(GL_INVALID_OPERATION);
            return ctx->setError(GL_INVALID_ENUM);
        }
        if (src >= GL_COLOR_ATTACHMENT0 + kMaxDrawBuffers) return ctx->setError(GL_INVALID_OPERATION);
    }
    fb->readBuffer = src;
}

API void APIENTRY glReadBuffer(GLenum src) {
    MGL_CTX();
    readBuffer(ctx, ctx->readFb.get(), src);
}

API void APIENTRY glNamedFramebufferReadBuffer(GLuint framebuffer, GLenum src) {
    MGL_CTX();
    readBuffer(ctx, framebuffer ? ctx->framebuffers.get(framebuffer) : ctx->readSurface ? ctx->readSurface->fb.get() : nullptr, src);
}

// Invalidation is only a hint (we keep contents); validate the arguments (GL 4.6 §17.4.4).
static void invalidateFramebuffer(Context* ctx, Framebuffer* fb, GLsizei n, const GLenum* attachments, GLsizei w, GLsizei h) {
    if (n < 0 || w < 0 || h < 0) return ctx->setError(GL_INVALID_VALUE);
    for (int i = 0; i < n; i++) {
        GLenum a = attachments[i];
        if (fb->isDefault) {
            if (a != GL_COLOR && a != GL_DEPTH && a != GL_STENCIL && a != GL_FRONT_LEFT && a != GL_FRONT_RIGHT && a != GL_BACK_LEFT &&
                a != GL_BACK_RIGHT)
                return ctx->setError(GL_INVALID_ENUM);
        } else if (a >= GL_COLOR_ATTACHMENT0 && a <= GL_COLOR_ATTACHMENT31) {
            if (a >= GL_COLOR_ATTACHMENT0 + kMaxDrawBuffers) return ctx->setError(GL_INVALID_OPERATION);
        } else if (a != GL_DEPTH_ATTACHMENT && a != GL_STENCIL_ATTACHMENT && a != GL_DEPTH_STENCIL_ATTACHMENT) {
            return ctx->setError(GL_INVALID_ENUM);
        }
    }
}
static Framebuffer* invalidateTarget(Context* ctx, GLenum target) {
    if (target == GL_FRAMEBUFFER || target == GL_DRAW_FRAMEBUFFER) return ctx->drawFb.get();
    if (target == GL_READ_FRAMEBUFFER) return ctx->readFb.get();
    ctx->setError(GL_INVALID_ENUM);
    return nullptr;
}
static Framebuffer* invalidateNamed(Context* ctx, GLuint name) {
    if (!name) {
        Framebuffer* d = ctx->defaultFramebuffer();
        if (!d) ctx->setError(GL_INVALID_OPERATION);
        return d;
    }
    Framebuffer* fb = ctx->framebuffers.get(name);
    if (!fb) ctx->setError(GL_INVALID_OPERATION);
    return fb;
}
API void APIENTRY glInvalidateFramebuffer(GLenum target, GLsizei n, const GLenum* attachments) {
    MGL_CTX();
    if (Framebuffer* fb = invalidateTarget(ctx, target)) invalidateFramebuffer(ctx, fb, n, attachments, 0, 0);
}
API void APIENTRY glInvalidateSubFramebuffer(GLenum target, GLsizei n, const GLenum* attachments, GLint, GLint, GLsizei w, GLsizei h) {
    MGL_CTX();
    if (Framebuffer* fb = invalidateTarget(ctx, target)) invalidateFramebuffer(ctx, fb, n, attachments, w, h);
}
API void APIENTRY glInvalidateNamedFramebufferData(GLuint fb, GLsizei n, const GLenum* attachments) {
    MGL_CTX();
    if (Framebuffer* f = invalidateNamed(ctx, fb)) invalidateFramebuffer(ctx, f, n, attachments, 0, 0);
}
API void APIENTRY glInvalidateNamedFramebufferSubData(GLuint fb, GLsizei n, const GLenum* attachments, GLint, GLint, GLsizei w, GLsizei h) {
    MGL_CTX();
    if (Framebuffer* f = invalidateNamed(ctx, fb)) invalidateFramebuffer(ctx, f, n, attachments, w, h);
}

// ---------------------------------------------------------------------------
// reads
// ---------------------------------------------------------------------------
// Resolves the read framebuffer's color read buffer to a Metal texture.
id<MTLTexture> readAttachmentTexture(Context* ctx, Framebuffer* fb, const FormatInfo** f, int* level, int* slice) {
    *level = 0;
    *slice = 0;
    if (!fb) return nil;
    if (fb->isDefault) {
        Surface* s = fb->surface;
        if (!s || fb->readBuffer == GL_NONE) return nil;
        *f = s->colorFmt ? s->colorFmt : formatInfo(GL_RGBA8);
        return s->color;
    }
    GLenum rb = fb->readBuffer;
    if (rb < GL_COLOR_ATTACHMENT0 || rb >= GL_COLOR_ATTACHMENT0 + kMaxDrawBuffers) return nil;
    const Attachment& a = fb->att[rb - GL_COLOR_ATTACHMENT0];
    int w, h;
    id<MTLTexture> t = attachmentTexture(ctx, a, f, &w, &h);
    if (a.type == GL_TEXTURE) {
        *level = a.level;
        *slice = a.layer;
    }
    return t;
}

id<MTLTexture> readDepthStencilTexture(Context* ctx, Framebuffer* fb, bool stencil, const FormatInfo** f, int* level, int* slice) {
    *level = 0;
    *slice = 0;
    if (!fb) return nil;
    if (fb->isDefault) {
        Surface* s = fb->surface;
        if (!s || !s->depthStencil) return nil;
        *f = s->dsFmt;
        return s->depthStencil;
    }
    const Attachment& a = fb->att[stencil ? kAttStencil : kAttDepth];
    int w, h;
    id<MTLTexture> t = attachmentTexture(ctx, a, f, &w, &h);
    if (a.type == GL_TEXTURE) {
        *level = a.level;
        *slice = a.layer;
    }
    return t;
}

// Resolves a multisampled texture region into a temporary single-sample texture.
static id<MTLTexture> resolveForRead(Context* ctx, id<MTLTexture> src) {
    if (src.sampleCount <= 1) return src;
    const FormatInfo* f = formatInfoForMTL(src.pixelFormat);
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:src.pixelFormat width:src.width height:src.height mipmapped:NO];
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead |
               (f && (f->srgb || f->kind == FmtKind::DepthStencil) ? MTLTextureUsagePixelFormatView : 0);
    id<MTLTexture> dst = [ctx->dev->mtl newTextureWithDescriptor:td];
    ctx->dev->makeResident(dst);
    ctx->endEncoder();
    if (f && (f->kind == FmtKind::UInt || f->kind == FmtKind::SInt)) {
        // Integer samples can't be averaged: GL takes a single sample (we use sample 0).
        resolveSample0(ctx, src, dst, f->kind == FmtKind::SInt);
        ctx->dev->releaseAfter(dst, ctx->serial());
        return dst;
    }
    MTL4RenderPassDescriptor* rp = [MTL4RenderPassDescriptor new];
    bool hasDepth = f && (f->kind == FmtKind::Depth || f->kind == FmtKind::DepthStencil);
    bool hasStencil = f && (f->kind == FmtKind::Stencil || f->kind == FmtKind::DepthStencil);
    if (hasDepth) {
        rp.depthAttachment.texture = src;
        rp.depthAttachment.loadAction = MTLLoadActionLoad;
        rp.depthAttachment.storeAction = MTLStoreActionMultisampleResolve;
        rp.depthAttachment.resolveTexture = dst;
        rp.depthAttachment.depthResolveFilter = MTLMultisampleDepthResolveFilterSample0;
    }
    if (hasStencil) {
        rp.stencilAttachment.texture = src;
        rp.stencilAttachment.loadAction = MTLLoadActionLoad;
        rp.stencilAttachment.storeAction = MTLStoreActionMultisampleResolve;
        rp.stencilAttachment.resolveTexture = dst;
        rp.stencilAttachment.stencilResolveFilter = MTLMultisampleStencilResolveFilterSample0;
    }
    if (!hasDepth && !hasStencil) {
        rp.colorAttachments[0].texture = src;
        rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
        rp.colorAttachments[0].storeAction = MTLStoreActionMultisampleResolve;
        rp.colorAttachments[0].resolveTexture = dst;
    }
    id<MTL4RenderCommandEncoder> e = [ctx->commandBuffer() renderCommandEncoderWithDescriptor:rp];
    [e barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageFragment visibilityOptions:MTL4VisibilityOptionDevice];
    [e endEncoding];
    ctx->dev->releaseAfter(dst, ctx->serial());
    return dst;
}

bool readFramebufferPixels(Context* ctx, Framebuffer* fb, int x, int y, int w, int h, GLenum format, GLenum type, void* pixels,
                           const PixelStore& ps3d, GLsizei bufSize) {
    // ReadPixels writes a single image: PACK_SKIP_IMAGES / PACK_IMAGE_HEIGHT don't apply.
    PixelStore ps = ps3d;
    ps.skipImages = 0;
    ps.imageHeight = 0;
    bool isDepth = format == GL_DEPTH_COMPONENT, isStencil = format == GL_STENCIL_INDEX, isDS = format == GL_DEPTH_STENCIL;
    const FormatInfo* f;
    int level, slice;
    id<MTLTexture> src = (isDepth || isStencil || isDS) ? readDepthStencilTexture(ctx, fb, isStencil, &f, &level, &slice)
                                                        : readAttachmentTexture(ctx, fb, &f, &level, &slice);
    if (!src) {
        ctx->setError(GL_INVALID_OPERATION, "no readable buffer");
        return false;
    }
    // DEPTH_STENCIL needs a packed depth+stencil buffer attached as both depth and stencil.
    if (isDS && (f->baseFormat != GL_DEPTH_STENCIL ||
                 (!fb->isDefault && (fb->att[kAttDepth].type == GL_NONE || fb->att[kAttStencil].type == GL_NONE)))) {
        ctx->setError(GL_INVALID_OPERATION, "DEPTH_STENCIL read without depth+stencil buffer");
        return false;
    }
    if (!isDepth && !isStencil && !isDS) {
        bool intFmt = f->kind == FmtKind::UInt || f->kind == FmtKind::SInt;
        bool intReq = format == GL_RED_INTEGER || format == GL_RG_INTEGER || format == GL_RGB_INTEGER || format == GL_RGBA_INTEGER ||
                      format == GL_BGRA_INTEGER || format == GL_BGR_INTEGER || format == GL_GREEN_INTEGER || format == GL_BLUE_INTEGER ||
                      format == GL_ALPHA_INTEGER;
        if (intFmt != intReq) {
            ctx->setError(GL_INVALID_OPERATION, "integer read mismatch");
            return false;
        }
    }
    ctx->applyClearNow();
    src = resolveForRead(ctx, src);
    int lw = std::max(1, (int)src.width >> level), lh = std::max(1, (int)src.height >> level);
    ClientLayout cl = clientLayout(format, type, w, h, ps);
    size_t need = clientImageSize(format, type, w, h, 1, ps);
    uint8_t* dst;
    if (Buffer* pbo = ctx->pixelPackBuffer.get()) {
        if ((size_t)pixels + need > (size_t)pbo->size) {
            ctx->setError(GL_INVALID_OPERATION);
            return false;
        }
        if (pbo->mapped && !(pbo->mapAccess & GL_MAP_PERSISTENT_BIT)) {
            ctx->setError(GL_INVALID_OPERATION);
            return false;
        }
        if (pbo->lastUse && !ctx->dev->isComplete(pbo->lastUse)) ctx->waitForResource(pbo->lastUse);
        dst = pbo->cpu + (size_t)pixels;
    } else {
        if (bufSize >= 0 && need > (size_t)bufSize) {
            ctx->setError(GL_INVALID_OPERATION);
            return false;
        }
        dst = (uint8_t*)pixels;
    }
    if (!dst) return false;
    dst += cl.skipBytes;
    // Clip the read rect; pixels outside the framebuffer are left untouched.
    int x0 = std::max(0, x), y0 = std::max(0, y), x1 = std::min(lw, x + w), y1 = std::min(lh, y + h);
    if (x1 <= x0 || y1 <= y0) return true;
    int rw = x1 - x0, rh = y1 - y0;
    std::vector<uint8_t> data;
    size_t rowBytes;
    if (f->comp == Comp::D32F_S8) {
        std::vector<uint8_t> dd, sd;
        size_t r1, r2;
        if (!isStencil) readTextureRegion(ctx, src, level, slice, x0, y0, rw, rh, dd, r1, f, MTLBlitOptionDepthFromDepthStencil);
        if (isStencil || isDS) readTextureRegion(ctx, src, level, slice, x0, y0, rw, rh, sd, r2, f, MTLBlitOptionStencilFromDepthStencil);
        data.assign((size_t)rw * rh * 8, 0);
        for (int p = 0; p < rw * rh; p++) {
            if (!dd.empty()) memcpy(&data[p * 8], &dd[p * 4], 4);
            if (!sd.empty()) data[p * 8 + 4] = sd[p];
        }
        rowBytes = (size_t)rw * 8;
    } else {
        readTextureRegion(ctx, src, level, slice, x0, y0, rw, rh, data, rowBytes, f, MTLBlitOptionNone);
    }
    int ps_ = clientPixelSize(format, type);
    uint8_t* out = dst + (size_t)(y0 - y) * cl.rowStride + (size_t)(x0 - x) * ps_;
    // GL row 0 is the bottom row == Metal row 0: no flip.
    const FormatInfo* rf = f;
    if (fb->isDefault && ctx->framebufferSRGB == false) rf = f;
    // GL_CLAMP_READ_COLOR: TRUE clamps to [0,1]; FIXED_ONLY (default) clamps fixed-point buffers
    bool isInt = rf->kind == FmtKind::UInt || rf->kind == FmtKind::SInt;
    bool fixedPoint = rf->kind == FmtKind::Unorm || rf->kind == FmtKind::Snorm;
    bool clamp = !isInt && rf->comp != Comp::Compressed &&
                 (ctx->clampReadColor == GL_TRUE || (ctx->clampReadColor == GL_FIXED_ONLY && fixedPoint));
    convertFromStorage(rf, format, type, data.data(), rowBytes, out, cl.rowStride, rw, rh, ps.swapBytes, true, clamp);
    return true;
}

// §18.2.2: ReadPixels with DEPTH_STENCIL takes the packed depth/stencil types only (INVALID_ENUM).
static GLenum readPixelsFormatTypeError(GLenum format, GLenum type) {
    if (format == GL_DEPTH_STENCIL && type != GL_UNSIGNED_INT_24_8 && type != GL_FLOAT_32_UNSIGNED_INT_24_8_REV) return GL_INVALID_ENUM;
    return pixelFormatTypeError(format, type);
}

API void APIENTRY glReadPixels(GLint x, GLint y, GLsizei w, GLsizei h, GLenum format, GLenum type, void* pixels) {
    MGL_CTX();
    if (w < 0 || h < 0) return ctx->setError(GL_INVALID_VALUE);
    if (GLenum e = readPixelsFormatTypeError(format, type)) return ctx->setError(e, "glReadPixels format/type");
    Framebuffer* fb = ctx->readFb.get();
    if (!fb) return ctx->setError(GL_INVALID_FRAMEBUFFER_OPERATION);
    GLenum status;
    if (!mglFramebufferComplete(ctx, fb, &status)) return ctx->setError(GL_INVALID_FRAMEBUFFER_OPERATION);
    if (!fb->isDefault) {
        for (auto& a : fb->att)
            if (a.type == GL_RENDERBUFFER && a.rb && a.rb->samples > 1) return ctx->setError(GL_INVALID_OPERATION);
            else if (a.type == GL_TEXTURE && a.tex && a.tex->samples > 1) return ctx->setError(GL_INVALID_OPERATION);
    } else if (fb->surface && fb->surface->samples > 1) {
        return ctx->setError(GL_INVALID_OPERATION);
    }
    if (w == 0 || h == 0) return;
    readFramebufferPixels(ctx, fb, x, y, w, h, format, type, pixels, ctx->pack, -1);
}

API void APIENTRY glReadnPixels(GLint x, GLint y, GLsizei w, GLsizei h, GLenum format, GLenum type, GLsizei bufSize, void* data) {
    MGL_CTX();
    if (w < 0 || h < 0) return ctx->setError(GL_INVALID_VALUE);
    if (GLenum e = readPixelsFormatTypeError(format, type)) return ctx->setError(e);
    Framebuffer* fb = ctx->readFb.get();
    if (!fb) return ctx->setError(GL_INVALID_FRAMEBUFFER_OPERATION);
    GLenum status;
    if (!mglFramebufferComplete(ctx, fb, &status)) return ctx->setError(GL_INVALID_FRAMEBUFFER_OPERATION);
    readFramebufferPixels(ctx, fb, x, y, w, h, format, type, data, ctx->pack, bufSize);
}

// ---------------------------------------------------------------------------
// blit
// ---------------------------------------------------------------------------
API void APIENTRY glBlitFramebuffer(GLint sx0, GLint sy0, GLint sx1, GLint sy1, GLint dx0, GLint dy0, GLint dx1, GLint dy1, GLbitfield mask,
                                    GLenum filter) {
    MGL_CTX();
    if (mask & ~(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)) return ctx->setError(GL_INVALID_VALUE);
    if (filter != GL_NEAREST && filter != GL_LINEAR) return ctx->setError(GL_INVALID_ENUM);
    if (filter == GL_LINEAR && (mask & (GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT))) return ctx->setError(GL_INVALID_OPERATION);
    Framebuffer* rfb = ctx->readFb.get();
    Framebuffer* dfb = ctx->drawFb.get();
    if (!rfb || !dfb) return ctx->setError(GL_INVALID_FRAMEBUFFER_OPERATION);
    GLenum st;
    if (!mglFramebufferComplete(ctx, rfb, &st) || !mglFramebufferComplete(ctx, dfb, &st)) return ctx->setError(GL_INVALID_FRAMEBUFFER_OPERATION);
    if (ctx->rasterizerDiscard) return;
    if (dfb->isDefault && dfb->surface) {
        static int logged = 0;
        static int last[8] = {};
        int cur[8] = {sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1};
        if (logged < 30 && memcmp(cur, last, sizeof cur)) {
            MGL_INFO("blit to default fb: src (%d,%d)-(%d,%d) dst (%d,%d)-(%d,%d) surface %dx%d", sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1,
                     dfb->surface->width, dfb->surface->height);
            memcpy(last, cur, sizeof cur);
            logged++;
        }
    }
    ctx->applyClearNow();
    ctx->endEncoder();
    auto blitOne = [&](id<MTLTexture> src, int slevel, int sslice, id<MTLTexture> dst, int dlevel, int dslice, BlitParams::What what, bool linear) {
        if (!src || !dst) return;
        src = resolveForRead(ctx, src);
        if (!ctx->framebufferSRGB && what == BlitParams::Color && (src.usage & MTLTextureUsagePixelFormatView) &&
            (src.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB || src.pixelFormat == MTLPixelFormatBGRA8Unorm_sRGB)) {
            // FRAMEBUFFER_SRGB disabled: copy encoded values as they are (no sRGB decode on read).
            src = [src newTextureViewWithPixelFormat:src.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB ? MTLPixelFormatRGBA8Unorm
                                                                                                       : MTLPixelFormatBGRA8Unorm];
            ctx->dev->retainUntil(src, ctx->serial());
        }
        BlitParams bp;
        bp.src = src;
        bp.srcLevel = slevel;
        bp.srcSlice = sslice;
        bp.srcRect[0] = sx0;
        bp.srcRect[1] = sy0;
        bp.srcRect[2] = sx1;
        bp.srcRect[3] = sy1;
        bp.dstRect[0] = dx0;
        bp.dstRect[1] = dy0;
        bp.dstRect[2] = dx1;
        bp.dstRect[3] = dy1;
        bp.linear = linear;
        bp.what = what;
        if (ctx->scissorEnabled & 1) {
            int dw = std::max(1, (int)dst.width >> dlevel), dh = std::max(1, (int)dst.height >> dlevel);
            int x0 = std::clamp(ctx->scissor[0][0], 0, dw), y0 = std::clamp(ctx->scissor[0][1], 0, dh);
            int x1 = std::clamp(ctx->scissor[0][0] + ctx->scissor[0][2], 0, dw), y1 = std::clamp(ctx->scissor[0][1] + ctx->scissor[0][3], 0, dh);
            if (x1 <= x0 || y1 <= y0) return;
            bp.scissored = true;
            bp.scissor = {(NSUInteger)x0, (NSUInteger)y0, (NSUInteger)(x1 - x0), (NSUInteger)(y1 - y0)};
        }
        // Fast path: identical rects/formats -> straight copy.
        bool sameSize = sx1 - sx0 == dx1 - dx0 && sy1 - sy0 == dy1 - dy0 && sx1 > sx0 && sy1 > sy0;
        if (sameSize && !bp.scissored && src.pixelFormat == dst.pixelFormat && dst.sampleCount == 1 && what == BlitParams::Color &&
            sx0 >= 0 && sy0 >= 0 && dx0 >= 0 && dy0 >= 0 && sx1 <= (int)(src.width >> slevel) && sy1 <= (int)(src.height >> slevel) &&
            dx1 <= (int)(dst.width >> dlevel) && dy1 <= (int)(dst.height >> dlevel)) {
            id<MTL4ComputeCommandEncoder> e = ctx->computeEncoder();
            [e copyFromTexture:src sourceSlice:sslice sourceLevel:slevel sourceOrigin:MTLOriginMake(sx0, sy0, 0)
                    sourceSize:MTLSizeMake(sx1 - sx0, sy1 - sy0, 1) toTexture:dst destinationSlice:dslice destinationLevel:dlevel
             destinationOrigin:MTLOriginMake(dx0, dy0, 0)];
            ctx->endEncoder();
            return;
        }
        blitToTexture(ctx, bp, dst, dlevel, dslice);
    };
    PassDesc dp;
    if (!ctx->resolvePass(dp)) return;
    if (mask & GL_COLOR_BUFFER_BIT) {
        const FormatInfo* sf;
        int sl, ss;
        id<MTLTexture> src = readAttachmentTexture(ctx, rfb, &sf, &sl, &ss);
        if (src) {
            for (int i = 0; i < kMaxDrawBuffers; i++) {
                if (!dp.color[i].tex) continue;
                const FormatInfo* df = formatInfoForMTL(dp.color[i].tex.pixelFormat);
                bool si = sf->kind == FmtKind::UInt || sf->kind == FmtKind::SInt;
                bool di = df && (df->kind == FmtKind::UInt || df->kind == FmtKind::SInt);
                if (si != di) return ctx->setError(GL_INVALID_OPERATION);
                if (si && filter == GL_LINEAR) return ctx->setError(GL_INVALID_OPERATION);
                blitOne(src, sl, ss, dp.color[i].tex, dp.color[i].level, dp.color[i].slice, BlitParams::Color, filter == GL_LINEAR);
            }
        }
    }
    if (mask & GL_DEPTH_BUFFER_BIT) {
        const FormatInfo* sf;
        int sl, ss;
        id<MTLTexture> src = readDepthStencilTexture(ctx, rfb, false, &sf, &sl, &ss);
        if (src && dp.depth.tex) blitOne(src, sl, ss, dp.depth.tex, dp.depth.level, dp.depth.slice, BlitParams::Depth, false);
    }
    if (mask & GL_STENCIL_BUFFER_BIT) {
        const FormatInfo* sf;
        int sl, ss;
        id<MTLTexture> src = readDepthStencilTexture(ctx, rfb, true, &sf, &sl, &ss);
        if (src && dp.stencil.tex) blitOne(src, sl, ss, dp.stencil.tex, dp.stencil.level, dp.stencil.slice, BlitParams::Stencil, false);
    }
    ctx->passDirty = true;
}

API void APIENTRY glBlitNamedFramebuffer(GLuint readFb, GLuint drawFb, GLint sx0, GLint sy0, GLint sx1, GLint sy1, GLint dx0, GLint dy0,
                                         GLint dx1, GLint dy1, GLbitfield mask, GLenum filter) {
    MGL_CTX();
    Ref<Framebuffer> sr = ctx->readFb, sd = ctx->drawFb;
    Framebuffer* r = readFb ? ctx->framebuffers.get(readFb) : (ctx->readSurface ? ctx->readSurface->fb.get() : nullptr);
    Framebuffer* d = drawFb ? ctx->framebuffers.get(drawFb) : ctx->defaultFramebuffer();
    if (!r || !d) return ctx->setError(GL_INVALID_OPERATION);
    ctx->applyClearNow();
    ctx->readFb = r;
    ctx->drawFb = d;
    ctx->passDirty = true;
    glBlitFramebuffer(sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1, mask, filter);
    ctx->readFb = sr;
    ctx->drawFb = sd;
    ctx->passDirty = true;
}
