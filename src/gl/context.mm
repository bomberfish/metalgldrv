#import "gl/context.h"
#include <cstdarg>
#include "gl/internal_shaders.h"
#include "gl/pipeline.h"
#include "util/log.h"

namespace mgl {

const GLenum kTexTargets[kNumTexTargets] = {
    GL_TEXTURE_1D, GL_TEXTURE_2D, GL_TEXTURE_3D, GL_TEXTURE_1D_ARRAY, GL_TEXTURE_2D_ARRAY, GL_TEXTURE_RECTANGLE,
    GL_TEXTURE_CUBE_MAP, GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_BUFFER, GL_TEXTURE_2D_MULTISAMPLE,
    GL_TEXTURE_2D_MULTISAMPLE_ARRAY,
};

int texTargetIndex(GLenum target) {
    for (int i = 0; i < kNumTexTargets; i++)
        if (kTexTargets[i] == target) return i;
    return -1;
}

static thread_local Context* t_current = nullptr;
Context* currentContext() { return t_current; }
void setCurrentContext(Context* c) { t_current = c; }

bool PassDesc::operator==(const PassDesc& o) const {
    for (int i = 0; i < kMaxDrawBuffers; i++)
        if (!(color[i] == o.color[i])) return false;
    return depth == o.depth && stencil == o.stencil && width == o.width && height == o.height && layers == o.layers &&
           samples == o.samples;
}

Context::Context(Ref<ShareGroup> sh, int major, int minor, bool core, bool debug, bool fwd, bool robust, bool noErr)
    : dev(Device::get()), share(sh), majorVersion(major), minorVersion(minor), coreProfile(core), debugContext(debug),
      forwardCompatible(fwd), robustAccess(robust), noErrorMode(noErr) {
    debugOutput = debug;
    for (int i = 0; i < kMaxViewports; i++) {
        viewport[i][0] = viewport[i][1] = viewport[i][2] = viewport[i][3] = 0;
        depthRange[i][0] = 0;
        depthRange[i][1] = 1;
        scissor[i][0] = scissor[i][1] = scissor[i][2] = scissor[i][3] = 0;
    }
    for (int i = 0; i < kMaxVertexAttribs; i++) {
        currentAttrib[i][0] = currentAttrib[i][1] = currentAttrib[i][2] = 0;
        currentAttrib[i][3] = 1;
        currentAttribType[i] = GL_FLOAT;
    }
    for (int i = 0; i < kNumTexTargets; i++) {
        Texture* t = new Texture();
        t->target = kTexTargets[i];
        t->name = 0;
        if (t->target == GL_TEXTURE_RECTANGLE) {
            t->sampler.minFilter = GL_LINEAR;
            t->sampler.wrapS = t->sampler.wrapT = t->sampler.wrapR = GL_CLAMP_TO_EDGE;
        }
        defaultTextures[i] = t;
    }
    for (int u = 0; u < kMaxTextureUnits; u++)
        for (int i = 0; i < kNumTexTargets; i++) textures[u][i] = defaultTextures[i];
    defaultVao = new VertexArray();
    for (int i = 0; i < kMaxVertexBindings; i++) defaultVao->attribs[i].binding = i;
    if (!coreProfile) vao = defaultVao;
    defaultXfb = new TransformFeedbackObj();
    xfb = defaultXfb;

    MTL4ArgumentTableDescriptor* ad = [MTL4ArgumentTableDescriptor new];
    ad.maxBufferBindCount = 31;
    ad.maxTextureBindCount = 0;
    ad.maxSamplerStateBindCount = 0;
    ad.supportAttributeStrides = YES;
    ad.initializeBindings = NO;
    NSError* err = nil;
    renderArgTable = [dev->mtl newArgumentTableWithDescriptor:ad error:&err];
    if (!renderArgTable) MGL_ERR("render argument table: %s", err.localizedDescription.UTF8String);
    ad.supportAttributeStrides = NO;
    computeArgTable = [dev->mtl newArgumentTableWithDescriptor:ad error:&err];
    pipelines_ = new PipelineCache(this);
}

Context::~Context() {
    if (cb_) flush();
    delete pipelines_;
    for (auto& [s, b] : uploadRetired_) dev->releaseAfter(b, s);
    if (uploadChunk_) dev->releaseAfter(uploadChunk_, serial_);
}

void Context::setError(GLenum e, const char* fmt, ...) {
    if (error == GL_NO_ERROR) error = e;
    char buf[512];
    buf[0] = 0;
    if (fmt) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(buf, sizeof buf, fmt, ap);
        va_end(ap);
    }
    MGL_DBG("GL error 0x%04x: %s", e, buf);
    if (debugOutput) debugMessage(GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_ERROR, e, GL_DEBUG_SEVERITY_HIGH, buf[0] ? buf : "GL error");
}

static uint64_t debugKey(GLenum source, GLenum type, GLenum severity) {
    return (uint64_t)source << 40 | (uint64_t)type << 20 | severity;
}

void Context::debugMessage(GLenum source, GLenum type, GLuint id, GLenum severity, const std::string& msg) {
    if (!debugOutput) return;
    auto it = debugIdControl.find((uint64_t)source << 40 | (uint64_t)type << 32 | id);
    if (it != debugIdControl.end()) {
        if (!it->second) return;
    } else {
        // GL 4.6 §20.4: all messages start enabled except those of severity LOW.
        bool enabled = severity != GL_DEBUG_SEVERITY_LOW;
        for (auto& [k, v] : debugControl) {
            GLenum ks = (GLenum)(k >> 40), kt = (GLenum)((k >> 20) & 0xfffff), kv = (GLenum)(k & 0xfffff);
            if ((ks == GL_DONT_CARE || ks == source) && (kt == GL_DONT_CARE || kt == type) && (kv == GL_DONT_CARE || kv == severity))
                enabled = v;
        }
        if (!enabled) return;
    }
    if (debugCallback) {
        debugCallback(source, type, id, severity, (GLsizei)msg.size(), msg.c_str(), debugUserParam);
    } else if (debugLog.size() < 64) {
        debugLog.push_back({source, type, severity, id, msg});
    }
}

// ---------------------------------------------------------------------------
// bindings
// ---------------------------------------------------------------------------
Texture* Context::boundTexture(GLenum target) {
    int idx = texTargetIndex(target);
    if (idx < 0) return nullptr;
    return textures[activeTexture][idx].get();
}

Texture* Context::texForUnit(int unit, int targetIdx) { return textures[unit][targetIdx].get(); }

Ref<Buffer>* Context::bufferBindingRef(GLenum target) {
    switch (target) {
    case GL_ARRAY_BUFFER: return &arrayBuffer;
    case GL_ELEMENT_ARRAY_BUFFER: return vao ? &vao->elementBuffer : nullptr;
    case GL_COPY_READ_BUFFER: return &copyReadBuffer;
    case GL_COPY_WRITE_BUFFER: return &copyWriteBuffer;
    case GL_PIXEL_PACK_BUFFER: return &pixelPackBuffer;
    case GL_PIXEL_UNPACK_BUFFER: return &pixelUnpackBuffer;
    case GL_DRAW_INDIRECT_BUFFER: return &drawIndirectBuffer;
    case GL_DISPATCH_INDIRECT_BUFFER: return &dispatchIndirectBuffer;
    case GL_QUERY_BUFFER: return &queryBuffer;
    case GL_TEXTURE_BUFFER: return &textureBufferBinding;
    case GL_PARAMETER_BUFFER: return &parameterBuffer;
    case GL_UNIFORM_BUFFER: return &uniformBufferGeneric;
    case GL_SHADER_STORAGE_BUFFER: return &ssboGeneric;
    case GL_ATOMIC_COUNTER_BUFFER: return &atomicCounterGeneric;
    case GL_TRANSFORM_FEEDBACK_BUFFER: return &xfbGeneric;
    default: return nullptr;
    }
}

Framebuffer* Context::fbForTarget(GLenum target) {
    switch (target) {
    case GL_FRAMEBUFFER:
    case GL_DRAW_FRAMEBUFFER: return drawFb.get();
    case GL_READ_FRAMEBUFFER: return readFb.get();
    default: return nullptr;
    }
}

ProgramObj* Context::activeProgram() {
    if (program) return program.get();
    if (pipeline) return pipeline->activeProgram.get();
    return nullptr;
}

ProgramObj* Context::activeProgramForStage(glsl::Stage s) {
    if (program) return program.get();
    if (pipeline) return pipeline->stages[(int)s].get();
    return nullptr;
}

void Context::unbindDeletedBuffer(Buffer* b) {
    Ref<Buffer>* refs[] = {&arrayBuffer, &copyReadBuffer, &copyWriteBuffer, &pixelPackBuffer, &pixelUnpackBuffer,
                           &drawIndirectBuffer, &dispatchIndirectBuffer, &queryBuffer, &textureBufferBinding,
                           &parameterBuffer, &uniformBufferGeneric, &ssboGeneric, &atomicCounterGeneric, &xfbGeneric};
    for (auto r : refs)
        if (r->get() == b) *r = nullptr;
    for (auto& u : uniformBuffers)
        if (u.buffer.get() == b) u = IndexedBufferBinding();
    for (auto& u : storageBuffers)
        if (u.buffer.get() == b) u = IndexedBufferBinding();
    for (auto& u : atomicCounterBuffers)
        if (u.buffer.get() == b) u = IndexedBufferBinding();
    if (vao) {
        if (vao->elementBuffer.get() == b) vao->elementBuffer = nullptr;
        for (auto& bd : vao->bindings)
            if (bd.buffer.get() == b) bd.buffer = nullptr;
        vao->version++;
    }
    if (xfb && !xfb->active)
        for (auto& bb : xfb->buffers)
            if (bb.buffer.get() == b) bb = TransformFeedbackObj::BufBinding();
}

void Context::unbindDeletedTexture(Texture* t) {
    int idx = texTargetIndex(t->target);
    if (idx >= 0)
        for (int u = 0; u < kMaxTextureUnits; u++)
            if (textures[u][idx].get() == t) textures[u][idx] = defaultTextures[idx];
    for (auto& iu : imageUnits)
        if (iu.tex.get() == t) iu = ImageUnit();
    Framebuffer* fbs[2] = {drawFb.get(), readFb.get()};
    for (Framebuffer* fb : fbs) {
        if (!fb || fb->isDefault) continue;
        for (auto& a : fb->att)
            if (a.tex.get() == t) {
                a = Attachment();
                fb->version++;
                passDirty = true;
            }
    }
}

void Context::unbindDeletedRenderbuffer(Renderbuffer* rb) {
    if (renderbuffer.get() == rb) renderbuffer = nullptr;
    Framebuffer* fbs[2] = {drawFb.get(), readFb.get()};
    for (Framebuffer* fb : fbs) {
        if (!fb || fb->isDefault) continue;
        for (auto& a : fb->att)
            if (a.rb.get() == rb) {
                a = Attachment();
                fb->version++;
                passDirty = true;
            }
    }
}

void Context::bindSurfaces(Surface* draw, Surface* read) {
    bool drawWasDefault = !drawFb || drawFb->isDefault;
    bool readWasDefault = !readFb || readFb->isDefault;
    if (drawSurface.get() != draw) endEncoder();
    drawSurface = draw;
    readSurface = read;
    if (drawWasDefault) drawFb = draw ? draw->fb : nullptr;
    if (readWasDefault) readFb = read ? read->fb : nullptr;
    passDirty = true;
    if (draw && !everCurrent) {
        everCurrent = true;
        viewport[0][2] = scissor[0][2] = draw->width;
        viewport[0][3] = scissor[0][3] = draw->height;
        for (int i = 1; i < kMaxViewports; i++) {
            viewport[i][2] = scissor[i][2] = draw->width;
            viewport[i][3] = scissor[i][3] = draw->height;
        }
    }
}

// ---------------------------------------------------------------------------
// command recording
// ---------------------------------------------------------------------------
uint64_t Context::serial() {
    if (!cb_) {
        serial_ = dev->reserveSerial();
        alloc_ = dev->acquireAllocator();
        cb_ = dev->newCommandBuffer();
        [cb_ beginCommandBufferWithAllocator:alloc_];
    }
    return serial_;
}

id<MTL4CommandBuffer> Context::commandBuffer() {
    serial();
    return cb_;
}

void Context::endEncoder() {
    if (renderEnc) endPass();
    if (computeEnc) {
        [computeEnc endEncoding];
        computeEnc = nil;
    }
}

id<MTL4ComputeCommandEncoder> Context::computeEncoder() {
    if (pendingClear.any()) applyClearNow();
    if (renderEnc) endPass();
    if (!computeEnc) {
        computeEnc = [commandBuffer() computeCommandEncoder];
        // Conservative GL-style ordering against all prior work.
        [computeEnc barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];
        [computeEnc setArgumentTable:computeArgTable];
    }
    return computeEnc;
}

UploadAlloc Context::upload(size_t size, size_t align) {
    const size_t kChunk = 4 << 20;
    serial();
    uploadOffset_ = (uploadOffset_ + align - 1) & ~(align - 1);
    if (!uploadChunk_ || uploadOffset_ + size > uploadChunk_.length) {
        if (uploadChunk_) uploadUsed_.push_back(uploadChunk_);
        uploadChunk_ = nil;
        size_t want = std::max(kChunk, size);
        // reuse a retired chunk
        for (auto it = uploadRetired_.begin(); it != uploadRetired_.end(); ++it) {
            if (it->second.length >= want && dev->isComplete(it->first)) {
                uploadChunk_ = it->second;
                uploadRetired_.erase(it);
                break;
            }
        }
        if (!uploadChunk_) uploadChunk_ = dev->newBuffer(want);
        uploadOffset_ = 0;
    }
    UploadAlloc a{uploadChunk_, (uint8_t*)uploadChunk_.contents + uploadOffset_, uploadChunk_.gpuAddress + uploadOffset_,
                  uploadOffset_};
    uploadOffset_ += size;
    return a;
}

void Context::flush(id<CAMetalDrawable> present) {
    if (pendingClear.any()) applyClearNow();
    if (!cb_) {
        if (present) {
            // Nothing recorded but a present was requested: record an empty buffer.
            serial();
        } else {
            return;
        }
    }
    endEncoder();
    [cb_ endCommandBuffer];
    dev->submit(cb_, alloc_, serial_, present);
    if (uploadChunk_) uploadUsed_.push_back(uploadChunk_);
    uploadChunk_ = nil;
    uploadOffset_ = 0;
    for (auto& b : uploadUsed_) uploadRetired_.push_back({serial_, b});
    uploadUsed_.clear();
    // Trim the retired pool.
    while (uploadRetired_.size() > 16) {
        dev->releaseAfter(uploadRetired_.front().second, uploadRetired_.front().first);
        uploadRetired_.erase(uploadRetired_.begin());
    }
    cb_ = nil;
    alloc_ = nil;
    passOpen = false;
    passDirty = true;
    dev->collect();
}

void Context::finish() {
    uint64_t s = cb_ ? serial_ : 0;
    flush();
    if (s) dev->waitFor(s);
    else dev->waitIdle();
}

void Context::waitForResource(uint64_t lastUse) {
    if (lastUse == 0) return;
    if (cb_ && lastUse == serial_) flush();
    dev->waitFor(lastUse);
}

// ---------------------------------------------------------------------------
// render passes
// ---------------------------------------------------------------------------
bool Context::resolvePass(PassDesc& p) {
    p = PassDesc();
    Framebuffer* fb = drawFb.get();
    if (!fb) return false;
    int w = INT32_MAX, h = INT32_MAX;
    bool any = false;
    int samples = 0;
    bool layered = false;
    int layers = INT32_MAX;
    if (fb->isDefault) {
        Surface* s = fb->surface;
        if (!s || !s->color) return false;
        for (int i = 0; i < kMaxDrawBuffers; i++) {
            GLenum db = fb->drawBuffers[i];
            if (db == GL_BACK || db == GL_BACK_LEFT || db == GL_FRONT || db == GL_FRONT_LEFT || db == GL_FRONT_AND_BACK || db == GL_LEFT)
                p.color[i].tex = s->msColor ? s->msColor : s->color;
        }
        if (s->depthStencil) {
            if (s->hasDepth) p.depth.tex = s->depthStencil;
            if (s->hasStencil) p.stencil.tex = s->depthStencil;
        }
        p.width = s->width;
        p.height = s->height;
        p.samples = std::max(1, s->samples);
        return true;
    }
    for (int i = 0; i < kMaxDrawBuffers; i++) {
        GLenum db = fb->drawBuffers[i];
        if (db < GL_COLOR_ATTACHMENT0 || db >= GL_COLOR_ATTACHMENT0 + kMaxDrawBuffers) continue;
        Attachment& a = fb->att[db - GL_COLOR_ATTACHMENT0];
        if (a.type == GL_NONE) continue;
        const FormatInfo* f;
        int aw, ah;
        id<MTLTexture> t = attachmentTexture(this, a, &f, &aw, &ah);
        if (!t) continue;
        p.color[i].tex = t;
        p.color[i].level = a.type == GL_TEXTURE ? a.level : 0;
        if (t.textureType == MTLTextureType3D) p.color[i].depthPlane = a.layered ? 0 : a.layer;
        else p.color[i].slice = a.layered ? 0 : a.layer;
        if (a.layered) {
            layered = true;
            int l = t.textureType == MTLTextureType3D ? std::max(1, (int)(t.depth >> a.level)) : (int)t.arrayLength * (t.textureType == MTLTextureTypeCube || t.textureType == MTLTextureTypeCubeArray ? 6 : 1);
            layers = std::min(layers, l);
        }
        w = std::min(w, aw);
        h = std::min(h, ah);
        samples = std::max(samples, (int)t.sampleCount);
        any = true;
    }
    for (int k = 0; k < 2; k++) {
        Attachment& a = fb->att[k == 0 ? kAttDepth : kAttStencil];
        if (a.type == GL_NONE) continue;
        const FormatInfo* f;
        int aw, ah;
        id<MTLTexture> t = attachmentTexture(this, a, &f, &aw, &ah);
        if (!t) continue;
        bool hasD = f->kind == FmtKind::Depth || f->kind == FmtKind::DepthStencil;
        bool hasS = f->kind == FmtKind::Stencil || f->kind == FmtKind::DepthStencil;
        PassAttachment pa;
        pa.tex = t;
        pa.level = a.type == GL_TEXTURE ? a.level : 0;
        pa.slice = a.layered ? 0 : a.layer;
        if (k == 0 && hasD) p.depth = pa;
        if (k == 1 && hasS) p.stencil = pa;
        if (a.layered) {
            layered = true;
            int l = (int)t.arrayLength * (t.textureType == MTLTextureTypeCube || t.textureType == MTLTextureTypeCubeArray ? 6 : 1);
            layers = std::min(layers, l);
        }
        w = std::min(w, aw);
        h = std::min(h, ah);
        samples = std::max(samples, (int)t.sampleCount);
        any = true;
    }
    if (!any) {
        if (fb->defaultWidth <= 0 || fb->defaultHeight <= 0) return false;
        w = fb->defaultWidth;
        h = fb->defaultHeight;
        samples = fb->defaultSamples;
        layers = std::max(1, fb->defaultLayers);
        layered = fb->defaultLayers > 0;
    }
    p.width = w;
    p.height = h;
    p.samples = std::max(1, samples);
    p.layers = layered ? layers : 1;
    return true;
}

void Context::beginPass() {
    PassDesc p;
    if (!resolvePass(p)) {
        passOpen = false;
        return;
    }
    MTL4RenderPassDescriptor* rp = [MTL4RenderPassDescriptor new];
    for (int i = 0; i < kMaxDrawBuffers; i++) {
        if (!p.color[i].tex) continue;
        MTLRenderPassColorAttachmentDescriptor* c = rp.colorAttachments[i];
        c.texture = p.color[i].tex;
        c.level = p.color[i].level;
        c.slice = p.color[i].slice;
        c.depthPlane = p.color[i].depthPlane;
        c.storeAction = MTLStoreActionStore;
        if (pendingClear.colorMask & (1u << i)) {
            c.loadAction = MTLLoadActionClear;
            const FormatInfo* f = formatInfoForMTL(p.color[i].tex.pixelFormat);
            if (f && (f->kind == FmtKind::UInt || f->kind == FmtKind::SInt))
                c.clearColor = MTLClearColorMake((double)(int32_t)pendingClear.colorInt[i][0], (double)(int32_t)pendingClear.colorInt[i][1],
                                                 (double)(int32_t)pendingClear.colorInt[i][2], (double)(int32_t)pendingClear.colorInt[i][3]);
            else
                c.clearColor = MTLClearColorMake(pendingClear.color[i][0], pendingClear.color[i][1], pendingClear.color[i][2],
                                                 pendingClear.color[i][3]);
            if (f && f->kind == FmtKind::UInt)
                c.clearColor = MTLClearColorMake((double)pendingClear.colorInt[i][0], (double)pendingClear.colorInt[i][1],
                                                 (double)pendingClear.colorInt[i][2], (double)pendingClear.colorInt[i][3]);
        } else {
            c.loadAction = MTLLoadActionLoad;
        }
    }
    if (p.depth.tex) {
        MTLRenderPassDepthAttachmentDescriptor* d = rp.depthAttachment;
        d.texture = p.depth.tex;
        d.level = p.depth.level;
        d.slice = p.depth.slice;
        d.storeAction = MTLStoreActionStore;
        d.loadAction = pendingClear.depth ? MTLLoadActionClear : MTLLoadActionLoad;
        d.clearDepth = pendingClear.depthValue;
    }
    if (p.stencil.tex) {
        MTLRenderPassStencilAttachmentDescriptor* s = rp.stencilAttachment;
        s.texture = p.stencil.tex;
        s.level = p.stencil.level;
        s.slice = p.stencil.slice;
        s.storeAction = MTLStoreActionStore;
        s.loadAction = pendingClear.stencil ? MTLLoadActionClear : MTLLoadActionLoad;
        s.clearStencil = pendingClear.stencilValue;
    }
    rp.renderTargetWidth = p.width;
    rp.renderTargetHeight = p.height;
    if (p.layers > 1) rp.renderTargetArrayLength = p.layers;
    rp.defaultRasterSampleCount = p.samples;
    if (visibilityBuffer) {
        rp.visibilityResultBuffer = visibilityBuffer;
        rp.visibilityResultType = MTLVisibilityResultTypeAccumulate;
    }
    passVisibilityBuffer = visibilityBuffer;
    encVisibilityOffset = -2;
    if (computeEnc) {
        [computeEnc endEncoding];
        computeEnc = nil;
    }
    renderEnc = [commandBuffer() renderCommandEncoderWithDescriptor:rp];
    if (!renderEnc) {
        MGL_ERR("failed to create render encoder (%dx%d)", p.width, p.height);
        passOpen = false;
        return;
    }
    [renderEnc barrierAfterQueueStages:MTLStageAll
                          beforeStages:MTLStageVertex | MTLStageFragment | MTLStageObject | MTLStageMesh
                     visibilityOptions:MTL4VisibilityOptionDevice];
    [renderEnc setArgumentTable:renderArgTable
                       atStages:MTLRenderStageVertex | MTLRenderStageFragment | MTLRenderStageObject | MTLRenderStageMesh];
    pendingClear.reset();
    curPass = p;
    passOpen = true;
    passDirty = false;
    enc.reset();
    // Record usage for the attachment owners lazily via Texture::lastUse (done in attachmentTexture).
}

void Context::endPass() {
    if (renderEnc) {
        [renderEnc endEncoding];
        renderEnc = nil;
    }
    passOpen = false;
}

id<MTL4RenderCommandEncoder> Context::renderEncoder() {
    if (drawFb && drawFb->version != drawFbVersionSeen) {
        passDirty = true;
        drawFbVersionSeen = drawFb->version;
    }
    if (passOpen && passDirty) {
        PassDesc p;
        if (resolvePass(p) && p == curPass && !pendingClear.any()) {
            passDirty = false;
        } else {
            endPass();
        }
    }
    if (passOpen && pendingClear.any()) endPass();
    if (!passOpen) beginPass();
    return renderEnc;
}

void Context::applyClearNow() {
    if (!pendingClear.any()) return;
    renderEncoder();
}

// ---------------------------------------------------------------------------
// Surface
// ---------------------------------------------------------------------------
Surface::~Surface() {
    Device* d = Device::get();
    if (color) d->releaseAfter(color, lastUse);
    if (msColor) d->releaseAfter(msColor, lastUse);
    if (depthStencil) d->releaseAfter(depthStencil, lastUse);
}

// Window surfaces follow the layer's bounds * contentsScale (like ANGLE): apps such
// as Amethyst express render resolution through contentsScale.
void Surface::syncSizeFromLayer() {
    if (!layer) return;
    CGSize b = layer.bounds.size;
    CGFloat sc = layer.contentsScale;
    int w = (int)lround(b.width * sc), h = (int)lround(b.height * sc);
    if (w <= 0 || h <= 0) {
        CGSize d = layer.drawableSize;
        w = (int)d.width;
        h = (int)d.height;
    } else if ((int)layer.drawableSize.width != w || (int)layer.drawableSize.height != h) {
        layer.drawableSize = CGSizeMake(w, h);
    }
    resize(w, h);
}

bool Surface::resize(int w, int h) {
    if (w <= 0) w = 1;
    if (h <= 0) h = 1;
    if (color && w == width && h == height) return false;
    Device* d = Device::get();
    if (color) d->releaseAfter(color, lastUse);
    if (msColor) d->releaseAfter(msColor, lastUse);
    if (depthStencil) d->releaseAfter(depthStencil, lastUse);
    width = w;
    height = h;
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:colorFormat width:w height:h mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    td.storageMode = MTLStorageModePrivate;
    td.hazardTrackingMode = MTLHazardTrackingModeUntracked;
    color = [d->mtl newTextureWithDescriptor:td];
    color.label = @"default fb color";
    d->makeResident(color);
    msColor = nil;
    if (samples > 1) {
        td.textureType = MTLTextureType2DMultisample;
        td.sampleCount = samples;
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        msColor = [d->mtl newTextureWithDescriptor:td];
        d->makeResident(msColor);
    }
    depthStencil = nil;
    if (hasDepth || hasStencil) {
        MTLPixelFormat df = hasStencil ? MTLPixelFormatDepth32Float_Stencil8 : MTLPixelFormatDepth32Float;
        MTLTextureDescriptor* dd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:df width:w height:h mipmapped:NO];
        dd.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        dd.storageMode = MTLStorageModePrivate;
        dd.hazardTrackingMode = MTLHazardTrackingModeUntracked;
        if (samples > 1) {
            dd.textureType = MTLTextureType2DMultisample;
            dd.sampleCount = samples;
        }
        depthStencil = [d->mtl newTextureWithDescriptor:dd];
        depthStencil.label = @"default fb depth/stencil";
        d->makeResident(depthStencil);
        dsFmt = formatInfo(hasStencil ? GL_DEPTH24_STENCIL8 : GL_DEPTH_COMPONENT24);
    }
    colorFmt = formatInfoForMTL(colorFormat);
    if (fb) fb->version++;
    return true;
}

}  // namespace mgl
