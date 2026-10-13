#import "gl/context.h"
#include "gl/app_profile.h"
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#import <UIKit/UIKit.h>
#endif
#include <cstdarg>
#include <execinfo.h>
#include "gl/internal_shaders.h"
#include "gl/pipeline.h"
#include "util/log.h"
#include "util/stats.h"

void mglCompatDestroy(mgl::Context* ctx);  // compat.mm

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

thread_local ThreadGL t_gl;
Context* currentContext() { return t_gl.current; }

NameSpace<Framebuffer>& framebufferNames(ShareGroup* share, NameSpace<Framebuffer>& own) {
    return share && (appProfile().shareFramebuffers || share->shareContainers) ? share->framebuffers : own;
}
void setCurrentContext(Context* c) { t_gl.current = c; }

bool PassDesc::operator==(const PassDesc& o) const {
    for (int i = 0; i < kMaxDrawBuffers; i++)
        if (!(color[i] == o.color[i])) return false;
    return depth == o.depth && stencil == o.stencil && width == o.width && height == o.height && layers == o.layers &&
           samples == o.samples;
}

Context::Context(Ref<ShareGroup> sh, int major, int minor, bool core, bool debug, bool fwd, bool robust, bool noErr)
    : dev(Device::get()), share(sh), majorVersion(major), minorVersion(minor), coreProfile(core), debugContext(debug),
      forwardCompatible(fwd), robustAccess(robust), noErrorMode(noErr), strictLevel(dev ? dev->caps.strictLevel : 0) {
    debugOutput = debug;
    (void)appProfile();  // resolves the application profile and logs its summary once per process
    for (int i = 0; i < kMaxViewports; i++) {
        viewport[i][0] = viewport[i][1] = viewport[i][2] = viewport[i][3] = 0;
        depthRange[i][0] = 0;
        depthRange[i][1] = 1;
        scissor[i][0] = scissor[i][1] = scissor[i][2] = scissor[i][3] = 0;
    }
    for (int i = 0; i < kMaxVertexAttribs; i++) {
        currentAttrib[i][0] = currentAttrib[i][1] = currentAttrib[i][2] = 0;
        currentAttrib[i][3] = 1;
        currentAttribD[i][0] = currentAttribD[i][1] = currentAttribD[i][2] = 0;
        currentAttribD[i][3] = 1;
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
    vao = defaultVao;  // hidden in core profiles (see api_vertex.mm specVao)
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
    renderArgTableFS = [dev->mtl newArgumentTableWithDescriptor:ad error:&err];
    if (!renderArgTable) MGL_ERR("render argument table: %s", err.localizedDescription.UTF8String);
    ad.supportAttributeStrides = NO;
    computeArgTable = [dev->mtl newArgumentTableWithDescriptor:ad error:&err];
    pipelines_ = new PipelineCache(this);
}

Context::~Context() {
    if (cb_ || tessSeg.open) flush();
    releaseDrawCaches(this);
    mglCompatDestroy(this);
    delete pipelines_;
    for (auto& [s, b] : uploadRetired_) dev->releaseAfter(b, s);
    if (uploadChunk_.buffer) dev->releaseAfter(uploadChunk_.buffer, lastWorkSerial_);
}


static const char* errorName(GLenum e) {
    switch (e) {
    case GL_INVALID_ENUM: return "GL_INVALID_ENUM in";
    case GL_INVALID_VALUE: return "GL_INVALID_VALUE in";
    case GL_INVALID_OPERATION: return "GL_INVALID_OPERATION in";
    case GL_INVALID_FRAMEBUFFER_OPERATION: return "GL_INVALID_FRAMEBUFFER_OPERATION in";
    case GL_OUT_OF_MEMORY: return "GL_OUT_OF_MEMORY in";
    case GL_STACK_OVERFLOW: return "GL_STACK_OVERFLOW in";
    case GL_STACK_UNDERFLOW: return "GL_STACK_UNDERFLOW in";
    default: return "GL error in";
    }
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
    MGL_DBG("GL error 0x%04x in %s: %s", e, t_apiName, buf);
    if (g_logUnsupported) {
        // once per entry point and message (the messages of most INVALID_ENUM errors name the enum)
        char what[640];
        if (e == GL_INVALID_ENUM) snprintf(what, sizeof what, "unsupported enum in %s%s%s", t_apiName, buf[0] ? ": " : "", buf);
        else snprintf(what, sizeof what, "GL error 0x%04x in %s%s%s", e, t_apiName, buf[0] ? ": " : "", buf);
        logUnsupportedOnce(what);
    }
    if (::mgl::g_logLevel >= ::mgl::LOG_TRACE) {  // METALGLDRV_LOG=4: who raised it
        void* frames[6];
        int n = backtrace(frames, 6);
        if (n > 1) backtrace_symbols_fd(frames + 1, n - 1, 2);
    }
    if (debugOutput) {
        // Name the entry point: apps (Minecraft/Iris) log these messages, often as the only trace.
        char msg[640];
        snprintf(msg, sizeof msg, "%s %s%s%s", errorName(e), t_apiName[0] ? t_apiName : "GL call", buf[0] ? ": " : "", buf);
        debugMessage(GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_ERROR, e, GL_DEBUG_SEVERITY_HIGH, msg);
    }
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
    // core profile without a VAO: the binding lives in the (otherwise unusable) default VAO, as in Mesa
    case GL_ELEMENT_ARRAY_BUFFER: return vao ? &vao->elementBuffer : &defaultVao->elementBuffer;
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
    if (defaultVao && defaultVao->elementBuffer.get() == b) defaultVao->elementBuffer = nullptr;
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
        if (fb->shadingRateTex.get() == t) fb->shadingRateTex = nullptr;
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
    syncSurfaceSRGB();
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
    if (tessSeg.open) return tessSeg.serial;
    if (!cb_) {
        serial_ = dev->reserveSerial();
        alloc_ = dev->acquireAllocator();
        cb_ = dev->newCommandBuffer();
        [cb_ beginCommandBufferWithAllocator:alloc_];
    }
    return serial_;
}

id<MTL4CommandBuffer> Context::commandBuffer() {
    if (tessSeg.open) endTessSegment();  // Metal 4 work after the segment's
    serial();
    cbUsed_ = true;
    return cb_;
}

id<MTL4RenderCommandEncoder> Context::newRenderEncoder(MTL4RenderPassDescriptor* rp) {
    @autoreleasepool {
        return [commandBuffer() renderCommandEncoderWithDescriptor:rp];
    }
}

id<MTL4ComputeCommandEncoder> Context::newComputeEncoder() {
    @autoreleasepool {
        return [commandBuffer() computeCommandEncoder];
    }
}

void Context::endEncoder() {
    if (renderEnc) endPass();
    if (computeEnc) {
        [computeEnc endEncoding];
        computeEnc = nil;
    }
}

id<MTL4ComputeCommandEncoder> Context::computeEncoder() {
    if (renderEnc) MGL_STAT_CALLER("pass-split:compute");
    if (pendingClear.any()) applyClearNow();
    if (renderEnc) endPass();
    if (!computeEnc) {
        if (g_profilePasses) {
            char label[64];
            snprintf(label, sizeof label, "blit/compute from %s", *t_apiName ? t_apiName : "?");
            profileTag(serial(), label);
        }
        computeEnc = newComputeEncoder();
        // Conservative GL-style ordering against all prior work.
        [computeEnc barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];
        [computeEnc setArgumentTable:computeArgTable];
    }
    return computeEnc;
}

// A retired upload buffer of at least `want` bytes the GPU is done with (the smallest), or nil.
id<MTLBuffer> Context::takeRetiredUpload(size_t want) {
    auto best = uploadRetired_.end();
    for (auto it = uploadRetired_.begin(); it != uploadRetired_.end(); ++it)
        if (it->second.length >= want && (best == uploadRetired_.end() || it->second.length < best->second.length) && dev->isComplete(it->first))
            best = it;
    if (best == uploadRetired_.end()) return nil;
    id<MTLBuffer> b = best->second;
    uploadRetired_.erase(best);
    return b;
}

// Bytes the retired upload pool keeps: 1/16 of the GPU's working set, 128 MB to 768 MB.
size_t Context::uploadPoolBytes() const {
    static const size_t bytes = std::clamp<size_t>((size_t)dev->mtl.recommendedMaxWorkingSetSize / 16, 128u << 20, 768u << 20);
    return bytes;
}

// An upload buffer of at least `want` bytes: a retired one the GPU is done with, else a new one. While more than
// 3/4 of the pool is retired but still in use by the GPU, waits for the oldest that fits instead: the CPU stays at
// most that far ahead, and buffers aren't created (and made resident) and freed again every frame.
id<MTLBuffer> Context::newUpload(size_t want) {
    const size_t kUploadInFlight = uploadPoolBytes() * 3 / 4;
    if (id<MTLBuffer> b = takeRetiredUpload(want)) return b;
    size_t pending = 0;
    auto oldest = uploadRetired_.end();
    for (auto it = uploadRetired_.begin(); it != uploadRetired_.end(); ++it) {
        pending += it->second.length;
        if (it->second.length >= want && (oldest == uploadRetired_.end() || it->first < oldest->first)) oldest = it;
    }
    if (pending > kUploadInFlight && oldest != uploadRetired_.end()) {
        MGL_STAT("upload-wait");
        dev->waitFor(oldest->first);
        id<MTLBuffer> b = oldest->second;
        uploadRetired_.erase(oldest);
        return b;
    }
    MGL_STAT("upload-new");
    return dev->newBuffer(want);
}

UploadAlloc Context::upload(size_t size, size_t align) {
    const size_t kChunk = 4 << 20;
    serial();
    if (g_statsEnabled) statAddCaller(size > kChunk / 2 ? "upload KB (large)" : "upload KB", __builtin_return_address(0), size >> 10);
    if (size > kChunk / 2) {
        // Large (tessellation records of a big draw): a buffer of its own, a power of two so that the next frame's
        // finds it again; the current chunk stays.
        size_t want = kChunk;
        while (want < size) want *= 2;
        id<MTLBuffer> b = newUpload(want);
        uploadUsed_.push_back({b, (uint8_t*)bufferContents(b), b.gpuAddress, b.length});
        return UploadAlloc{b, uploadUsed_.back().cpu, uploadUsed_.back().gpu, 0};
    }
    uploadOffset_ = (uploadOffset_ + align - 1) & ~(align - 1);
    if (!uploadChunk_.buffer || uploadOffset_ + size > uploadChunk_.length) {
        if (uploadChunk_.buffer) uploadUsed_.push_back(uploadChunk_);
        id<MTLBuffer> b = newUpload(kChunk);
        uploadChunk_ = {b, (uint8_t*)bufferContents(b), b.gpuAddress, b.length};
        uploadOffset_ = 0;
    }
    UploadAlloc a{uploadChunk_.buffer, uploadChunk_.cpu + uploadOffset_, uploadChunk_.gpu + uploadOffset_, uploadOffset_};
    uploadOffset_ += size;
    return a;
}

// Filled upload chunks retire with the work submitted last, which completes after the others (Context::TessSegment).
// The current chunk continues: it retires with the last work that used it. Retiring it at every flush would need a
// new chunk per command buffer.
void Context::retireUploads() {
    for (auto& b : uploadUsed_) uploadRetired_.push_back({lastWorkSerial_, b.buffer});
    uploadUsed_.clear();
    // Trim the retired pool to uploadPoolBytes() and 256 buffers: a frame of a tessellation-heavy game uploads ~100 MB
    // (Unigine Heaven), and with 3 frames in flight a smaller pool frees buffers the next frame allocates again.
    size_t retiredBytes = 0;
    for (auto& r : uploadRetired_) retiredBytes += r.second.length;
    while (uploadRetired_.size() > 256 || (retiredBytes > uploadPoolBytes() && uploadRetired_.size() > 1)) {
        retiredBytes -= uploadRetired_.front().second.length;
        dev->releaseAfter(uploadRetired_.front().second, uploadRetired_.front().first);
        uploadRetired_.erase(uploadRetired_.begin());
    }
}

void Context::flush(id<CAMetalDrawable> present, bool keepUploads) {
    MGL_STAT_CALLER("flush");
    struct InFlush {
        bool& f, saved;
        explicit InFlush(bool& b) : f(b), saved(b) { f = true; }
        ~InFlush() { f = saved; }
    } inFlush(inFlush_);
    if (tessSeg.open) endTessSegment();
    statMaybeReport();
    if (pendingClear.any()) applyClearNow();
    if (!cb_) {
        if (present) {
            // Nothing recorded but a present was requested: record an empty buffer.
            serial();
        } else {
            // only segments ran since the last flush
            if (!keepUploads) retireUploads();
            return;
        }
    }
    endEncoder();
    [cb_ endCommandBuffer];
    queueAfterTessSegments();
    dev->submit(cb_, alloc_, serial_, present);
    lastWorkSerial_ = serial_;
    if (cbUsed_) tessMtlWork = true;
    cbUsed_ = false;
    if (!keepUploads) retireUploads();
    cb_ = nil;
    alloc_ = nil;
    passOpen = false;
    passDirty = true;
    dev->collect();
}

bool Context::uploadBufferOf(uint64_t gpu, NSUInteger* offset, __unsafe_unretained id<MTLBuffer>* out) {
    auto in = [&](const UploadBuf& b) { return b.buffer && gpu >= b.gpu && gpu < b.gpu + b.length; };
    const UploadBuf* found = in(uploadChunk_) ? &uploadChunk_ : nullptr;
    for (size_t i = uploadUsed_.size(); !found && i-- > 0;)
        if (in(uploadUsed_[i])) found = &uploadUsed_[i];
    if (!found) return false;
    *offset = (NSUInteger)(gpu - found->gpu);
    *out = found->buffer;
    return true;
}

id<MTLBuffer> Context::uploadBuffer(uint64_t gpu, NSUInteger* offset) {
    __unsafe_unretained id<MTLBuffer> b = nil;
    return uploadBufferOf(gpu, offset, &b) ? b : nil;
}

// Fills a render pass descriptor (Metal 4 or classic) for pass `p`, with the pending clears as load actions.
template <class RP>
static void fillPassDescriptor(RP* rp, const PassDesc& p, const PendingClear& clear) {
    for (int i = 0; i < kMaxDrawBuffers; i++) {
        if (!p.color[i].tex) continue;
        MTLRenderPassColorAttachmentDescriptor* c = rp.colorAttachments[i];
        c.texture = p.color[i].tex;
        c.level = p.color[i].level;
        c.slice = p.color[i].slice;
        c.depthPlane = p.color[i].depthPlane;
        c.storeAction = MTLStoreActionStore;
        if (clear.colorMask & (1u << i)) {
            c.loadAction = MTLLoadActionClear;
            const FormatInfo* f = formatInfoForMTL(p.color[i].tex.pixelFormat);
            if (f && (f->kind == FmtKind::UInt || f->kind == FmtKind::SInt))
                c.clearColor = MTLClearColorMake((double)(int32_t)clear.colorInt[i][0], (double)(int32_t)clear.colorInt[i][1],
                                                 (double)(int32_t)clear.colorInt[i][2], (double)(int32_t)clear.colorInt[i][3]);
            else
                c.clearColor = MTLClearColorMake(clear.color[i][0], clear.color[i][1], clear.color[i][2], p.color[i].noAlpha ? 1.0 : clear.color[i][3]);
            if (f && f->kind == FmtKind::UInt)
                c.clearColor = MTLClearColorMake((double)clear.colorInt[i][0], (double)clear.colorInt[i][1], (double)clear.colorInt[i][2],
                                                 (double)clear.colorInt[i][3]);
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
        d.loadAction = clear.depth ? MTLLoadActionClear : MTLLoadActionLoad;
        d.clearDepth = clear.depthValue;
    }
    if (p.stencil.tex) {
        MTLRenderPassStencilAttachmentDescriptor* s = rp.stencilAttachment;
        s.texture = p.stencil.tex;
        s.level = p.stencil.level;
        s.slice = p.stencil.slice;
        s.storeAction = MTLStoreActionStore;
        s.loadAction = clear.stencil ? MTLLoadActionClear : MTLLoadActionLoad;
        s.clearStencil = clear.stencilValue;
    }
    rp.renderTargetWidth = p.width;
    rp.renderTargetHeight = p.height;
    if (p.layers > 1) rp.renderTargetArrayLength = p.layers;
    rp.defaultRasterSampleCount = p.samples;
}

bool Context::openTessSegment(const PassDesc& p, bool prevWrites) {
    id<MTLCommandQueue> q = dev->classicQueue();
    if (!q) return false;
    if (tessSeg.open) endTessSegment();
    MGL_STAT("tess-segment");
    // Pending clears become the segment pass's load actions. The upload memory used so far may be read by the
    // segment: it retires with later work.
    // A Metal 4 command buffer with nothing recorded (opened for a serial: resources tagged as used by what comes
    // next) stays open: it runs after the segment.
    PendingClear clear = pendingClear;
    pendingClear.reset();
    if (cb_ && cbUsed_) flush(nil, true);
    if (!tessMtlEvent) {
        tessMtlEvent = [dev->mtl newEvent];
        tessPrepEvent = [dev->mtl newEvent];
        tessMainEvent = [dev->mtl newEvent];
    }
    tessSeg.mtl = 0;
    if (tessMtlWork) {  // the Metal 4 work so far; else the previous pre-pass (which waited for it), in queue order
        tessSeg.mtl = ++tessMtlValue;
        dev->queueSignal(tessMtlEvent, tessSeg.mtl);
        tessMtlWork = false;
    }
    tessSeg.prepChain = ++tessPrepValue;
    tessSeg.prepDone = ++tessPrepValue;
    tessSeg.mainDone = ++tessMainValue;
    bool waitMain = (prevWrites || tessBarrier) && tessLastMain;
    if (waitMain) tessMainsWaited();
    tessBarrier = false;
    @autoreleasepool {
        tessSeg.prep = [q commandBuffer];
        tessSeg.main = [q commandBuffer];
        if (tessSeg.mtl) [tessSeg.prep encodeWaitForEvent:tessMtlEvent value:tessSeg.mtl];
        else if (tessLastPrep) [tessSeg.prep encodeWaitForEvent:tessPrepEvent value:tessLastPrep];
        if (waitMain) [tessSeg.prep encodeWaitForEvent:tessMainEvent value:tessLastMain];
        [tessSeg.main encodeWaitForEvent:tessPrepEvent value:tessSeg.prepDone];
        if (tessLastMain) [tessSeg.main encodeWaitForEvent:tessMainEvent value:tessLastMain];
        MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor new];
        fillPassDescriptor(rp, p, clear);
        tessSeg.compute = [tessSeg.prep computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];  // TCS dispatches are independent
        tessSeg.render = [tessSeg.main renderCommandEncoderWithDescriptor:rp];
    }
    tessSeg.serial = dev->reserveSerial();
    tessSeg.prepWaitsMain = waitMain || !tessLastMain;
    tessSeg.pass = p;
    tessSeg.open = true;
    tessSeg.count++;
    tessSeg.tessDraws = 0;
    if (!tessSeg.render || !tessSeg.compute) {
        MGL_ERR("failed to begin a tessellation pass (%dx%d)", p.width, p.height);
        endTessSegment();
        return false;
    }
    // the attachments are in use until the segment completes
    PassDesc tagged;
    resolvePass(tagged);
    tessSegmentSnapshot();
    // the segment's pass is the current one for the draw state (applyDynamicState, fillColorKey)
    curPass = p;
    enc.reset();
    return tessSeg.open;
}

void Context::tessSegmentSnapshot() {
    tessSeg.fb = drawFb.get();
    tessSeg.fbVersion = tessSeg.fb ? tessSeg.fb->version : 0;
    tessSeg.fbSRGB = framebufferSRGB;
    tessSeg.depthFormat = tessSeg.pass.depth.tex ? (uint32_t)tessSeg.pass.depth.tex.pixelFormat : 0;
    tessSeg.stencilFormat = tessSeg.pass.stencil.tex ? (uint32_t)tessSeg.pass.stencil.tex.pixelFormat : 0;
    for (int i = 0; i < kNumAttachments; i++) tessSeg.storage[i] = nil;
    if (Framebuffer* fb = tessSeg.fb) {
        if (fb->isDefault && fb->surface) {
            tessSeg.storage[0] = fb->surface->color;
            tessSeg.storage[1] = fb->surface->msColor;
            tessSeg.storage[2] = fb->surface->depthStencil;
            tessSeg.surfaceW = fb->surface->width;
            tessSeg.surfaceH = fb->surface->height;
        } else {
            for (int i = 0; i < kNumAttachments; i++) {
                const Attachment& a = fb->att[i];
                tessSeg.storage[i] = a.type == GL_TEXTURE && a.tex ? a.tex->mtl : a.type == GL_RENDERBUFFER && a.rb ? a.rb->mtl : nil;
            }
        }
    }
}

bool Context::tessSegmentPassCurrent() const {
    const TessSegment& s = tessSeg;
    Framebuffer* fb = drawFb.get();
    if (!s.open || !fb || fb != s.fb || fb->version != s.fbVersion || framebufferSRGB != s.fbSRGB || pendingClear.any()) return false;
    if (fb->isDefault) {
        Surface* sf = fb->surface;
        return sf && sf->color == s.storage[0] && sf->msColor == s.storage[1] && sf->depthStencil == s.storage[2] && sf->width == s.surfaceW &&
               sf->height == s.surfaceH;
    }
    for (int i = 0; i < kNumAttachments; i++) {
        const Attachment& a = fb->att[i];
        if (a.type == GL_TEXTURE && a.tex && (a.tex->mtl != s.storage[i] || a.tex->storageDirty)) return false;
        if (a.type == GL_RENDERBUFFER && a.rb && a.rb->mtl != s.storage[i]) return false;
    }
    return true;
}

void Context::tessPrepWaitForPrevious() {
    if (!tessSeg.open || tessSeg.prepWaitsMain) return;
    // the pre-pass so far goes first; the next one waits for it and for the previous segment
    @autoreleasepool {
        [tessSeg.compute endEncoding];
        [tessSeg.prep encodeSignalEvent:tessPrepEvent value:tessSeg.prepChain];
        dev->commitClassic(tessSeg.prep, false, 0);
        tessSeg.prep = [dev->classicQueue() commandBuffer];
        [tessSeg.prep encodeWaitForEvent:tessPrepEvent value:tessSeg.prepChain];
        [tessSeg.prep encodeWaitForEvent:tessMainEvent value:tessLastMain];
        tessSeg.compute = [tessSeg.prep computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    }
    tessSeg.prepWaitsMain = true;
    tessMainsWaited();
}

void Context::endTessSegment() {
    if (!tessSeg.open) return;
    tessSeg.open = false;
    auto unwaited = [&](id<MTLTexture> t) {  // each texture once: the list stays as small as the set of render targets
        if (!t) return;
        if (t.parentTexture) t = t.parentTexture;
        if (std::find(tessUnwaitedTargets.begin(), tessUnwaitedTargets.end(), t) == tessUnwaitedTargets.end())
            tessUnwaitedTargets.push_back(t);
    };
    unwaited(tessSeg.pass.depth.tex);
    unwaited(tessSeg.pass.stencil.tex);
    for (const PassAttachment& a : tessSeg.pass.color) unwaited(a.tex);
    if (g_profilePasses) {
        char label[96];
        snprintf(label, sizeof label, "tess segment fb %u %dx%d%s", tessSeg.fb ? tessSeg.fb->name : 0, tessSeg.pass.width, tessSeg.pass.height,
                 tessSeg.pass.depth.tex ? " +depth" : "");
        profileTag(tessSeg.serial, label);
    }
    if (!tessSeg.tessDraws && tessSeg.fb && tessSeg.fb == drawFb.get()) tessSeg.fb->tessHint = false;  // a guess that missed
    @autoreleasepool {
        [tessSeg.compute endEncoding];
        [tessSeg.prep encodeSignalEvent:tessPrepEvent value:tessSeg.prepDone];
        dev->commitClassic(tessSeg.prep, false, 0);
        [tessSeg.render endEncoding];
        [tessSeg.main encodeSignalEvent:tessMainEvent value:tessSeg.mainDone];
        dev->commitClassic(tessSeg.main, true, tessSeg.serial);
    }
    tessLastMain = tessSeg.mainDone;
    tessLastPrep = tessSeg.prepDone;
    lastWorkSerial_ = tessSeg.serial;
    tessSeg.prep = tessSeg.main = nil;
    tessSeg.compute = nil;
    tessSeg.render = nil;
    for (int i = 0; i < kNumAttachments; i++) tessSeg.storage[i] = nil;
    enc.reset();
    passDirty = true;
}

void Context::queueAfterTessSegments() {
    if (tessLastMain <= tessQueueWaited) return;
    dev->queueWait(tessMainEvent, tessLastMain);
    tessQueueWaited = tessLastMain;
}

void Context::finish() {
    MGL_STAT_CALLER("finish");
    uint64_t s = tessSeg.open ? tessSeg.serial : cb_ ? serial_ : 0;
    flush();
    if (s) dev->waitFor(s);
    else dev->waitIdle();
}

void Context::waitForResource(uint64_t lastUse) {
    if (lastUse == 0) return;
    MGL_STAT_CALLER("gpu-wait");
    if ((cb_ && lastUse == serial_) || (tessSeg.open && lastUse == tessSeg.serial)) flush();
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
            if (db == GL_BACK || db == GL_BACK_LEFT || db == GL_FRONT || db == GL_FRONT_LEFT || db == GL_FRONT_AND_BACK || db == GL_LEFT) {
                p.color[i].tex = s->msColor ? s->msColor : s->color;
                // GL_FRAMEBUFFER_SRGB disabled: the sRGB-capable default framebuffer is written
                // without encoding, as for FBO attachments (attachmentWriteView) and Apple's GL.
                if (!framebufferSRGB && s->srgbCapable) p.color[i].tex = s->linearWriteView(p.color[i].tex);
                p.color[i].noAlpha = !s->hasAlpha;
            }
        }
        if (s->depthStencil && s->dsUsed) {
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
        // GL_FRAMEBUFFER_SRGB disabled: sRGB attachments are written without encoding.
        t = attachmentWriteView(this, a, t);
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
    MGL_STAT("render-pass");
    passCounter++;
    PassDesc p;
    if (!resolvePass(p)) {
        passOpen = false;
        return;
    }
    MTL4RenderPassDescriptor* rp = [MTL4RenderPassDescriptor new];
    fillPassDescriptor(rp, p, pendingClear);
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
    renderEnc = newRenderEncoder(rp);
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
    if (g_profilePasses) {
        snprintf(passLabel_, sizeof passLabel_, "fb %s%u %dx%d%s%s from %s", drawFb && drawFb->isDefault ? "window " : "", drawFb ? drawFb->name : 0,
                 p.width, p.height, p.depth.tex ? " +depth" : "", p.layers > 1 ? " layered" : "", *t_apiName ? t_apiName : "?");
    }
    pendingClear.reset();
    passFb_ = drawFb.get();
    curPass = p;
    passOpen = true;
    passDirty = false;
    enc.reset();
    // Record usage for the attachment owners lazily via Texture::lastUse (done in attachmentTexture).
}

void Context::endPass() {
    if (renderEnc) MGL_STAT_CALLER("endPass");
    bool profile = g_profilePasses && renderEnc;
    if (renderEnc) {
        [renderEnc endEncoding];
        renderEnc = nil;
    }
    passOpen = false;
    if (profile) {
        char label[192];
        snprintf(label, sizeof label, "%s, %d draw%s, first program %u", passLabel_, passDraws_, passDraws_ == 1 ? "" : "s", passFirstProgram_);
        profileTag(serial_, label);
        passDraws_ = 0;
        passFirstProgram_ = 0;
        if (!inFlush_) flush();  // one pass per command buffer
    }
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
            if (pendingClear.any()) MGL_STAT("split:pending-clear");
            else if (drawFb.get() == passFb_) MGL_STAT("split:same-fb-new-attachments");
            else MGL_STAT("split:other-fb");
            endPass();
        }
    }
    if (passOpen && pendingClear.any()) {
        MGL_STAT("split:pending-clear-2");
        endPass();
    }
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
    }
    bool fixed = fixedWidth > 0 && fixedHeight > 0;
    if (fixed) {
        w = fixedWidth;
        h = fixedHeight;
    }
    // Drawable size: the render size, or the native size when upscaling with MetalFX.
    int ow = w, oh = h;
    static const bool upscale = envFlag("METALGLDRV_USE_METALFX", false);
    if (upscale && !fixed && b.width > 0) {
#if TARGET_OS_IPHONE
        CGFloat native = UIScreen.mainScreen.scale;
#else
        CGFloat native = 2.0;
#endif
        ow = std::max(w, (int)lround(b.width * native));
        oh = std::max(h, (int)lround(b.height * native));
    }
    if (ow > 0 && oh > 0 && ((int)layer.drawableSize.width != ow || (int)layer.drawableSize.height != oh))
        layer.drawableSize = CGSizeMake(ow, oh);
    outWidth = ow;
    outHeight = oh;
    resize(w, h);
}

id<MTLTexture> Surface::linearWriteView(id<MTLTexture> t) {
    MTLPixelFormat lf = t.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB   ? MTLPixelFormatRGBA8Unorm
                        : t.pixelFormat == MTLPixelFormatBGRA8Unorm_sRGB ? MTLPixelFormatBGRA8Unorm
                                                                         : MTLPixelFormatInvalid;
    if (lf == MTLPixelFormatInvalid) return t;
    id<MTLTexture> __strong& cached = t == msColor ? msColorLinear : colorLinear;
    if (!cached) cached = [t newTextureViewWithPixelFormat:lf];
    return cached;
}

void Surface::allocColor() {
    Device* d = Device::get();
    MTLTextureDescriptor* td = texture2DDescriptor(colorFormat, width, height);
    // Only sRGB storage needs other-format views (linearWriteView). Format-view usage turns off lossless
    // compression on Apple GPUs, which makes every load and store of the color buffer cost two to three
    // times as much (a 4K window: ~0.2 ms per pass).
    const MTLTextureUsage viewUsage = srgbStorage() ? MTLTextureUsagePixelFormatView : 0;
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | viewUsage;
    td.storageMode = MTLStorageModePrivate;
    td.hazardTrackingMode = MTLHazardTrackingModeUntracked;
    color = [d->mtl newTextureWithDescriptor:td];
    color.label = @"default fb color";
    d->makeResident(color);
    msColor = nil;
    colorLinear = msColorLinear = nil;
    if (samples > 1) {
        td.textureType = MTLTextureType2DMultisample;
        td.sampleCount = samples;
        msColor = [d->mtl newTextureWithDescriptor:td];
        d->makeResident(msColor);
    }
}

void Surface::useSRGBStorage(Context* ctx) {
    if (!srgbCapable || srgbStorage()) return;
    colorFormat = MTLPixelFormatRGBA8Unorm_sRGB;
    // the drawable stores what the color buffer holds: sRGB-encoded values either way
    if (layer) layer.pixelFormat = MTLPixelFormatBGRA8Unorm_sRGB;
    colorFmt = formatInfoForMTL(colorFormat);
    if (fb) fb->version++;
    ctx->passDirty = true;
    if (!color) return;
    MGL_DBG("surface %dx%d: sRGB storage (GL_FRAMEBUFFER_SRGB)", width, height);
    // Same bytes in the new storage: copy through its linear views.
    id<MTLTexture> oldColor = color, oldMs = msColor;
    allocColor();
    ctx->endEncoder();
    id<MTL4ComputeCommandEncoder> ce = ctx->newComputeEncoder();
    [ce barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
    [ce copyFromTexture:oldColor toTexture:linearWriteView(color)];
    if (oldMs && msColor) [ce copyFromTexture:oldMs toTexture:linearWriteView(msColor)];
    [ce endEncoding];
    lastUse = ctx->serial();
    Device* d = Device::get();
    d->releaseAfter(oldColor, lastUse);
    if (oldMs) d->releaseAfter(oldMs, lastUse);
}

void Context::syncSurfaceSRGB() {
    if (!framebufferSRGB) return;
    for (Framebuffer* f : {drawFb.get(), readFb.get()})
        if (f && f->isDefault && f->surface && f->surface->srgbCapable && !f->surface->srgbStorage()) f->surface->useSRGBStorage(this);
}

bool Surface::resize(int w, int h) {
    if (w <= 0) w = 1;
    if (h <= 0) h = 1;
    if (color && w == width && h == height) return false;
    MGL_INFO("surface resize %dx%d -> %dx%d (layer bounds %.0fx%.0f scale %.2f drawable %.0fx%.0f)", width, height, w, h,
             layer ? layer.bounds.size.width : 0.0, layer ? layer.bounds.size.height : 0.0, layer ? layer.contentsScale : 0.0,
             layer ? layer.drawableSize.width : 0.0, layer ? layer.drawableSize.height : 0.0);
    Device* d = Device::get();
    if (color) d->releaseAfter(color, lastUse);
    if (msColor) d->releaseAfter(msColor, lastUse);
    if (depthStencil) d->releaseAfter(depthStencil, lastUse);
    width = w;
    height = h;
    allocColor();
    depthStencil = nil;
    if (hasDepth || hasStencil) {
        MTLPixelFormat df = hasStencil ? MTLPixelFormatDepth32Float_Stencil8 : MTLPixelFormatDepth32Float;
        MTLTextureDescriptor* dd = texture2DDescriptor(df, w, h);
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

// The GLSL compiler validates bindings against its own copies of these limits.
static_assert(mgl::kMaxImageUnits == glsl::kMaxImageUnits, "image unit limits disagree");
static_assert(mgl::kMaxTextureUnits == glsl::kMaxCombinedTextureImageUnits, "texture unit limits disagree");
static_assert(mgl::kMaxAtomicCounterBindings == glsl::kMaxAtomicCounterBufferBindings, "atomic counter binding limits disagree");
