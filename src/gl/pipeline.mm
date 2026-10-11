#import "gl/pipeline.h"
#include <atomic>
#include <type_traits>
#include "gl/context.h"
#include "util/log.h"
#include "util/stats.h"

namespace mgl {

ProgramVariant::~ProgramVariant() {
    // Metal 4 command buffers don't retain pipeline states: keep the pipelines of a variant
    // dropped (program deleted or relinked) right after a draw or dispatch alive until that
    // work has run.
    Device* d = Device::get();
    uint64_t s = d->lastReservedSerial();
    d->retainUntil(compute, s);
    for (auto& [n, p] : computeByName) d->retainUntil(p, s);
    for (auto& [k, p] : renderPipelines) d->retainUntil(p, s);
}

uint64_t hashBytes(const void* data, size_t n, uint64_t h) {
    const uint8_t* p = (const uint8_t*)data;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

ProgramVariant* PipelineCache::variant(ProgramObj* prog, const glsl::VariantKey& key) {
    uint64_t h = hashBytes(&key.kind, sizeof key.kind);
    h = hashBytes(&key.primitive, sizeof key.primitive, h);
    h = hashBytes(&key.flatshadeLastVertex, 1, h);
    h = hashBytes(&key.rasterDiscard, 1, h);
    h = hashBytes(&key.attribSignFlip, sizeof key.attribSignFlip, h);
    h = hashBytes(&key.manualFetch, 1, h);
    h = hashBytes(&key.xfbInVertex, 1, h);
    h = hashBytes(&key.sampleShading, 1, h);
    h = hashBytes(&key.sampleMask, 1, h);
    if (key.sparseImageSlots) h = hashBytes(&key.sparseImageSlots, sizeof key.sparseImageSlots, h);
    if (key.msArraySlots) {
        h = hashBytes(&key.msArraySlots, sizeof key.msArraySlots, h);
        h = hashBytes(&key.msArray4, sizeof key.msArray4, h);
    }
    h = hashBytes(&key.advancedBlend, 1, h);
    if (key.source1FromLocation1) h = hashBytes("s1", 2, h);
    if (key.fsr) h = hashBytes(&key.fsr, 1, h);
    if (key.pdc) h = hashBytes(&key.pdc, 1, h);
    if (key.tessComputePass) h = hashBytes("tcp", 3, h);
    if (key.tessFused) h = hashBytes(&key.tessFused, 1, h ^ 0x7f);
    if (key.tessMetal) h = hashBytes("mt", 2, h ^ 0x3b);
    if (key.patchVertices) h = hashBytes(&key.patchVertices, 1, h ^ 0x5d);
    if ((key.kind != glsl::PipelineKind::Classic && key.kind != glsl::PipelineKind::Compute) || key.manualFetch || key.xfbInVertex) {
        // Field by field: VertexFetch has padding bytes with indeterminate contents.
        for (const glsl::VertexFetch& f : key.attribs) {
            uint64_t packed[2] = {(uint64_t)f.enabled | (uint64_t)f.normalized << 1 | (uint64_t)f.integer << 2 | (uint64_t)f.isDouble << 3 |
                                      (uint64_t)f.size << 8 | (uint64_t)f.binding << 16 | (uint64_t)f.glType << 32,
                                  f.relativeOffset};
            if (!f.enabled) packed[0] = packed[1] = 0;
            h = hashBytes(packed, sizeof packed, h);
        }
    }
    auto it = prog->variants.find(h);
    if (it != prog->variants.end()) return it->second.get();
    auto v = std::make_shared<ProgramVariant>();
    MGL_STAT("variant-compile");
    auto t0 = std::chrono::steady_clock::now();
    double genMs = 0;
    static std::atomic<uint64_t> nextUid{1};
    v->uid = nextUid++;
    v->manualFetch = key.manualFetch;
    if (!prog->program) {
        v->error = "program not linked";
    } else {
        if (prog->composite) {
            const glsl::LinkedProgram* st[(int)glsl::Stage::Count];
            for (int i = 0; i < (int)glsl::Stage::Count; i++) st[i] = prog->stageLinked[i].get();
            v->msl = glsl::generateMSL(st, key);
        } else {
            v->msl = glsl::generateMSL(*prog->program, key);
        }
        genMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (!v->msl.error.empty()) {
            v->error = v->msl.error;
        } else {
            // Debugging aid: METALGLDRV_MSL_OVERRIDE=<file> applies literal edits to the generated MSL
            // (read at every compile so tests can switch it between programs). The file holds blocks
            // "old\n----\nnew" separated by "\n====\n".
            if (const char* ov = getenv("METALGLDRV_MSL_OVERRIDE"); ov && *ov) {
                NSString* f = [NSString stringWithContentsOfFile:@(ov) encoding:NSUTF8StringEncoding error:nil];
                for (NSString* blk in [f componentsSeparatedByString:@"\n====\n"]) {
                    NSArray* parts = [blk componentsSeparatedByString:@"\n----\n"];
                    if (parts.count != 2) continue;
                    std::string from = [parts[0] UTF8String], to = [parts[1] UTF8String];
                    size_t at = v->msl.source.find(from);
                    if (at == std::string::npos) {
                        MGL_WARN("MSL override: pattern not found: %s", from.c_str());
                        continue;
                    }
                    for (; at != std::string::npos; at = v->msl.source.find(from, at + to.size()))
                        v->msl.source.replace(at, from.size(), to);
                }
            }
            v->lib = ctx_->dev->compileLibrary(v->msl.source, &v->error);
            if (!v->lib) {
                if (g_logLevel >= LOG_DEBUG || envFlag("METALGLDRV_DUMP_FAILED_MSL", true))
                    MGL_ERR("MSL compile failed for program %u:\n%s\n---- source ----\n%s", prog->name, v->error.c_str(),
                            v->msl.source.c_str());
            } else {
                v->ok = true;
                // METALGLDRV_DUMP_MSL=1 dumps every variant; any other value dumps variants whose MSL contains it.
                static const char* dumpMsl = getenv("METALGLDRV_DUMP_MSL");
                // METALGLDRV_DUMP_MSL=1 (all), <substring>, or #<n> (program n)
                if (dumpMsl && *dumpMsl && strcmp(dumpMsl, "0") &&
                    (!strcmp(dumpMsl, "1") || (dumpMsl[0] == '#' ? strtoul(dumpMsl + 1, nullptr, 10) == prog->name
                                                                  : v->msl.source.find(dumpMsl) != std::string::npos)))
                    MGL_INFO("program %u MSL:\n%s", prog->name, v->msl.source.c_str());
            }
        }
    }
    if (!v->ok) MGL_ERR("program %u variant failed: %s", prog->name, v->error.c_str());
    if (g_hitchMs > 0) {
        double total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        // Which key fields made a draw compile a variant the link-time one didn't cover.
        if (strcmp(t_apiName, "glLinkProgram") != 0) {
            std::string why;
            const glsl::VariantKey d;
            if (key.kind != d.kind) why += " kind=" + std::to_string((int)key.kind);
            if (key.primitive != d.primitive) why += " primitive=" + std::to_string(key.primitive);
            if (key.flatshadeLastVertex != d.flatshadeLastVertex) why += " flatshadeFirst";
            if (key.rasterDiscard) why += " rasterDiscard";
            if (key.attribSignFlip) why += " attribSignFlip";
            if (key.manualFetch) why += " manualFetch";
            if (key.xfbInVertex) why += " xfbInVertex";
            if (key.sampleShading) why += " sampleShading";
            if (key.sampleMask) why += " sampleMask";
            if (key.advancedBlend) why += " advancedBlend";
            if (key.source1FromLocation1) why += " source1FromLocation1";
            if (key.patchVertices) why += " patchVertices";
            if (key.tessMetal) why += " tessMetal";
            MGL_INFO("draw-time variant: program %u (%s) %.1f ms:%s", prog->name, t_apiName, total, why.empty() ? " (link-time key)" : why.c_str());
        }
        hitchCompile("MSL generation", genMs);
        hitchCompile("Metal library compile", total - genMs);
    }
    ProgramVariant* r = v.get();
    prog->variants[h] = std::move(v);
    return r;
}

static MTL4LibraryFunctionDescriptor* fn(id<MTLLibrary> lib, const std::string& name) {
    MTL4LibraryFunctionDescriptor* f = [MTL4LibraryFunctionDescriptor new];
    f.library = lib;
    f.name = @(name.c_str());
    return f;
}

static void setColorAttachment(MTLRenderPipelineColorAttachmentDescriptor* c, MTLPixelFormat pf, const BlendKey& b);
static void setColorAttachment(MTL4RenderPipelineColorAttachmentDescriptor* c, MTLPixelFormat pf, const BlendKey& b);

// Vertex descriptor of a pipeline key (nil without attributes); strides are set per draw.
static MTLVertexDescriptor* vertexDescriptor(const RenderPipelineKey& key) {
    MTLVertexDescriptor* vd = [MTLVertexDescriptor vertexDescriptor];
    bool anyAttr = false;
    for (int i = 0; i < 32; i++) {
        const VertexAttribKey& a = key.attribs[i];
        if (!a.format) continue;
        vd.attributes[i].format = (MTLVertexFormat)a.format;
        vd.attributes[i].bufferIndex = a.bufferIndex;
        vd.attributes[i].offset = a.offset;
        anyAttr = true;
    }
    for (int b = 0; b < 32; b++) {
        const VertexBufferKey& vb = key.buffers[b];
        if (!vb.used) continue;
        vd.layouts[b].stride = MTLBufferLayoutStrideDynamic;
        vd.layouts[b].stepFunction = (MTLVertexStepFunction)vb.stepFunction;
        vd.layouts[b].stepRate = vb.stepFunction == MTLVertexStepFunctionConstant ? 0 : std::max(1u, vb.stepRate);
    }
    return anyAttr ? vd : nil;
}

// RenderPipelineKey::classic: for classic render encoders (a tessellation segment). With tessMetal the vertex
// function is the TES (mgl_tes_vs) behind Metal's tessellator.
static id<MTLRenderPipelineState> newClassicPipeline(Device* dev, const RenderPipelineKey& key) {
    const ProgramVariant* v = key.variant;
    const glsl::EntryPoint* vs = v->entry(glsl::EntryPoint::Vertex);
    const glsl::EntryPoint* fs = v->entry(glsl::EntryPoint::Fragment);
    id<MTLLibrary> fsLib = v->lib;
    if (key.fsVariantUid) {  // program pipeline: fragment stage from another program (or none)
        fs = key.fsVariant ? key.fsVariant->entry(glsl::EntryPoint::Fragment) : nullptr;
        fsLib = key.fsVariant ? key.fsVariant->lib : nil;
    }
    if (!vs) return nil;
    MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = [v->lib newFunctionWithName:@(vs->name.c_str())];
    if (fs && key.rasterEnabled) d.fragmentFunction = [fsLib newFunctionWithName:@(fs->name.c_str())];
    d.rasterSampleCount = key.sampleCount;
    d.alphaToCoverageEnabled = key.alphaToCoverage;
    d.alphaToOneEnabled = key.alphaToOne;
    d.rasterizationEnabled = key.rasterEnabled;
    if (key.tessMetal) {
        d.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
        d.tessellationPartitionMode = (MTLTessellationPartitionMode)key.tessPartition;
        d.maxTessellationFactor = 64;
        d.tessellationFactorFormat = MTLTessellationFactorFormatHalf;
        d.tessellationControlPointIndexType = MTLTessellationControlPointIndexTypeNone;
        d.tessellationFactorStepFunction = MTLTessellationFactorStepFunctionPerPatchAndPerInstance;
        d.tessellationOutputWindingOrder = (MTLWinding)key.tessWinding;
        d.tessellationFactorScaleEnabled = NO;
    } else {
        d.vertexDescriptor = vertexDescriptor(key);
        d.inputPrimitiveTopology = (MTLPrimitiveTopologyClass)key.topologyClass;
    }
    for (int i = 0; i < 8; i++)
        if (key.colorFormats[i]) setColorAttachment(d.colorAttachments[i], (MTLPixelFormat)key.colorFormats[i], key.blend[i]);
    d.depthAttachmentPixelFormat = (MTLPixelFormat)key.depthFormat;
    d.stencilAttachmentPixelFormat = (MTLPixelFormat)key.stencilFormat;
    NSError* err = nil;
    MGL_STAT("render-pipeline-create");
    id<MTLRenderPipelineState> pso = [dev->mtl newRenderPipelineStateWithDescriptor:d error:&err];
    if (!pso) MGL_ERR("classic render pipeline creation failed: %s", err.localizedDescription.UTF8String);
    return pso;
}

static id<MTLRenderPipelineState> newRenderPipeline(Device* dev, const RenderPipelineKey& key) {
    if (key.classic) return newClassicPipeline(dev, key);
    const ProgramVariant* v = key.variant;
    MTL4PipelineDescriptor* desc = nil;
    const glsl::EntryPoint* vs = v->entry(glsl::EntryPoint::Vertex);
    const glsl::EntryPoint* fs = v->entry(glsl::EntryPoint::Fragment);
    id<MTLLibrary> fsLib = v->lib;
    if (key.fsVariantUid) {  // program pipeline: fragment stage from another program (or none)
        fs = key.fsVariant ? key.fsVariant->entry(glsl::EntryPoint::Fragment) : nullptr;
        fsLib = key.fsVariant ? key.fsVariant->lib : nil;
    }
    if (key.pdcPrepass) {
        const ProgramVariant* fv = key.fsVariantUid ? key.fsVariant : v;
        fs = fv ? fv->entry("mgl_fs_pdc") : nullptr;
        if (!fs) return nil;
    }
    const glsl::EntryPoint* ms = v->entry(glsl::EntryPoint::Mesh);
    const glsl::EntryPoint* os = v->entry(glsl::EntryPoint::Object);
    MTL4RenderPipelineColorAttachmentDescriptorArray* colors = nil;
    if (ms) {
        MTL4MeshRenderPipelineDescriptor* d = [MTL4MeshRenderPipelineDescriptor new];
        if (os) {
            d.objectFunctionDescriptor = fn(v->lib, os->name);
            d.payloadMemoryLength = os->payloadBytes;
            d.maxTotalThreadsPerObjectThreadgroup = os->threadsPerThreadgroup[0] * os->threadsPerThreadgroup[1] * os->threadsPerThreadgroup[2];
        }
        d.meshFunctionDescriptor = fn(v->lib, ms->name);
        d.maxTotalThreadsPerMeshThreadgroup = ms->threadsPerThreadgroup[0] * ms->threadsPerThreadgroup[1] * ms->threadsPerThreadgroup[2];
        if (fs && key.rasterEnabled) d.fragmentFunctionDescriptor = fn(fsLib, fs->name);
        d.rasterSampleCount = key.sampleCount;
        d.alphaToCoverageState = key.alphaToCoverage ? MTL4AlphaToCoverageStateEnabled : MTL4AlphaToCoverageStateDisabled;
        d.alphaToOneState = key.alphaToOne ? MTL4AlphaToOneStateEnabled : MTL4AlphaToOneStateDisabled;
        d.rasterizationEnabled = key.rasterEnabled;
        colors = d.colorAttachments;
        desc = d;
    } else {
        MTL4RenderPipelineDescriptor* d = [MTL4RenderPipelineDescriptor new];
        if (!vs) return nil;
        d.vertexFunctionDescriptor = fn(v->lib, vs->name);
        if (fs && key.rasterEnabled) d.fragmentFunctionDescriptor = fn(fsLib, fs->name);
        d.vertexDescriptor = vertexDescriptor(key);
        d.rasterSampleCount = key.sampleCount;
        d.alphaToCoverageState = key.alphaToCoverage ? MTL4AlphaToCoverageStateEnabled : MTL4AlphaToCoverageStateDisabled;
        d.alphaToOneState = key.alphaToOne ? MTL4AlphaToOneStateEnabled : MTL4AlphaToOneStateDisabled;
        d.rasterizationEnabled = key.rasterEnabled;
        d.inputPrimitiveTopology = (MTLPrimitiveTopologyClass)key.topologyClass;
        colors = d.colorAttachments;
        desc = d;
    }
    for (int i = 0; i < 8; i++) {
        MTLPixelFormat pf = (MTLPixelFormat)key.colorFormats[i];
        if (pf == MTLPixelFormatInvalid) continue;
        setColorAttachment(colors[i], pf, key.blend[i]);
    }
    NSError* err = nil;
    MGL_STAT("render-pipeline-create");
    id<MTLRenderPipelineState> pso = [dev->compiler newRenderPipelineStateWithDescriptor:desc compilerTaskOptions:nil error:&err];
    if (!pso) MGL_ERR("render pipeline creation failed: %s", err.localizedDescription.UTF8String);
    return pso;
}

// Color attachment state, for MTL4 and classic pipeline descriptors alike.
template <class C>
static void fillColorAttachment(C* c, MTLPixelFormat pf, const BlendKey& b) {
    c.pixelFormat = pf;
    if constexpr (std::is_same_v<C, MTL4RenderPipelineColorAttachmentDescriptor>) c.blendingState = b.enabled ? MTL4BlendStateEnabled : MTL4BlendStateDisabled;
    else c.blendingEnabled = b.enabled;
    if (b.enabled) {
        c.sourceRGBBlendFactor = (MTLBlendFactor)b.srcRGB;
        c.destinationRGBBlendFactor = (MTLBlendFactor)b.dstRGB;
        c.sourceAlphaBlendFactor = (MTLBlendFactor)b.srcA;
        c.destinationAlphaBlendFactor = (MTLBlendFactor)b.dstA;
        c.rgbBlendOperation = (MTLBlendOperation)b.eqRGB;
        c.alphaBlendOperation = (MTLBlendOperation)b.eqA;
    }
    MTLColorWriteMask m = MTLColorWriteMaskNone;
    if (b.writeMask & 1) m |= MTLColorWriteMaskRed;
    if (b.writeMask & 2) m |= MTLColorWriteMaskGreen;
    if (b.writeMask & 4) m |= MTLColorWriteMaskBlue;
    if (b.writeMask & 8) m |= MTLColorWriteMaskAlpha;
    c.writeMask = m;
}
static void setColorAttachment(MTL4RenderPipelineColorAttachmentDescriptor* c, MTLPixelFormat pf, const BlendKey& b) {
    fillColorAttachment(c, pf, b);
}
static void setColorAttachment(MTLRenderPipelineColorAttachmentDescriptor* c, MTLPixelFormat pf, const BlendKey& b) {
    fillColorAttachment(c, pf, b);
}

id<MTLRenderPipelineState> PipelineCache::renderPipeline(const RenderPipelineKey& keyIn) {
    RenderPipelineKey key = keyIn;
    key.variantUid = key.variant->uid;
    const ProgramVariant* vptr = key.variant;
    const ProgramVariant* fptr = key.fsVariant;
    key.variant = nullptr;
    key.fsVariant = nullptr;
    std::string k((const char*)&key, sizeof key);
    key.variant = vptr;
    key.fsVariant = fptr;
    auto& cache = vptr->renderPipelines;
    auto it = cache.find(k);
    if (it != cache.end()) return it->second;
    id<MTLRenderPipelineState> pso;
    auto t0 = std::chrono::steady_clock::now();
    @autoreleasepool {  // descriptor setup and pipeline compiles autorelease
        pso = newRenderPipeline(ctx_->dev, key);
    }
    if (g_hitchMs > 0) hitchCompile("render pipeline creation", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    cache[k] = pso;
    return pso;
}

static id<MTLComputePipelineState> newComputePipeline(Device* dev, ProgramVariant* v, const glsl::EntryPoint* cs) {
    @autoreleasepool {  // function descriptor names and compiler errors autorelease
        MTL4ComputePipelineDescriptor* d = [MTL4ComputePipelineDescriptor new];
        d.computeFunctionDescriptor = fn(v->lib, cs->name);
        MTLSize tg = MTLSizeMake(cs->threadsPerThreadgroup[0], cs->threadsPerThreadgroup[1], cs->threadsPerThreadgroup[2]);
        d.requiredThreadsPerThreadgroup = tg;
        NSError* err = nil;
        id<MTLComputePipelineState> p = [dev->compiler newComputePipelineStateWithDescriptor:d compilerTaskOptions:nil error:&err];
        if (!p) {
            MGL_ERR("compute pipeline %s creation failed: %s", cs->name.c_str(), err.localizedDescription.UTF8String);
            return nil;
        }
        // Metal's pipeline cache does not key on requiredThreadsPerThreadgroup: a kernel whose AIR
        // matches one compiled for a smaller threadgroup can come back limited to that size, and
        // dispatches above the limit are silently dropped. Kernels whose body does not depend on the
        // threadgroup size carry it as [[max_total_threads_per_threadgroup]] (glsl/msl.cpp).
        if (p.maxTotalThreadsPerThreadgroup < tg.width * tg.height * tg.depth) {
            MGL_ERR("compute pipeline %s allows %lu threads per threadgroup, needs %lux%lux%lu", cs->name.c_str(),
                    (unsigned long)p.maxTotalThreadsPerThreadgroup, (unsigned long)tg.width, (unsigned long)tg.height, (unsigned long)tg.depth);
            return nil;
        }
        return p;
    }
}

id<MTLComputePipelineState> PipelineCache::computePipeline(ProgramVariant* v, const std::string& name) {
    auto it = v->computeByName.find(name);
    if (it != v->computeByName.end()) return it->second;
    const glsl::EntryPoint* cs = v->entry(name);
    if (!cs || cs->kind != glsl::EntryPoint::Kernel) return nil;
    id<MTLComputePipelineState> p = newComputePipeline(ctx_->dev, v, cs);
    v->computeByName[name] = p;
    return p;
}

id<MTLComputePipelineState> PipelineCache::classicComputePipeline(ProgramVariant* v, const std::string& name) {
    std::string k = "classic:" + name;
    auto it = v->computeByName.find(k);
    if (it != v->computeByName.end()) return it->second;
    const glsl::EntryPoint* cs = v->entry(name);
    if (!cs || cs->kind != glsl::EntryPoint::Kernel) return nil;
    id<MTLComputePipelineState> p = nil;
    @autoreleasepool {
        MTLComputePipelineDescriptor* d = [MTLComputePipelineDescriptor new];
        d.computeFunction = [v->lib newFunctionWithName:@(name.c_str())];
        NSUInteger threads = (NSUInteger)cs->threadsPerThreadgroup[0] * cs->threadsPerThreadgroup[1] * cs->threadsPerThreadgroup[2];
        d.maxTotalThreadsPerThreadgroup = threads;
        NSError* err = nil;
        p = [ctx_->dev->mtl newComputePipelineStateWithDescriptor:d options:MTLPipelineOptionNone reflection:nil error:&err];
        if (!p) MGL_ERR("compute pipeline %s creation failed: %s", name.c_str(), err.localizedDescription.UTF8String);
        else if (p.maxTotalThreadsPerThreadgroup < threads) p = nil;
    }
    v->computeByName[k] = p;
    return p;
}

id<MTLComputePipelineState> PipelineCache::computePipeline(ProgramVariant* v) {
    if (v->compute) return v->compute;
    const glsl::EntryPoint* cs = v->entry(glsl::EntryPoint::Kernel);
    if (!cs) return nil;
    v->compute = newComputePipeline(ctx_->dev, v, cs);
    return v->compute;
}

id<MTLDepthStencilState> PipelineCache::depthStencil(const DepthStencilKey& key) {
    std::string k((const char*)&key, sizeof key);
    auto it = ds_.find(k);
    if (it != ds_.end()) return it->second;
    MTLDepthStencilDescriptor* d = [MTLDepthStencilDescriptor new];
    d.depthCompareFunction = (MTLCompareFunction)key.depthFunc;
    d.depthWriteEnabled = key.depthWrite;
    if (key.stencilEnabled) {
        for (int f = 0; f < 2; f++) {
            MTLStencilDescriptor* s = [MTLStencilDescriptor new];
            s.stencilCompareFunction = (MTLCompareFunction)key.face[f].func;
            s.stencilFailureOperation = (MTLStencilOperation)key.face[f].sfail;
            s.depthFailureOperation = (MTLStencilOperation)key.face[f].dpfail;
            s.depthStencilPassOperation = (MTLStencilOperation)key.face[f].dppass;
            s.readMask = key.face[f].readMask;
            s.writeMask = key.face[f].writeMask;
            if (f == 0) d.frontFaceStencil = s;
            else d.backFaceStencil = s;
        }
    }
    id<MTLDepthStencilState> st = [ctx_->dev->mtl newDepthStencilStateWithDescriptor:d];
    ds_[k] = st;
    return st;
}

MTLVertexFormat vertexFormatFor(GLenum type, int size, bool normalized, bool integer) {
    if (size == GL_BGRA) {
        if (type == GL_UNSIGNED_BYTE) return MTLVertexFormatUChar4Normalized_BGRA;
        if (type == GL_UNSIGNED_INT_2_10_10_10_REV) return MTLVertexFormatUInt1010102Normalized;  // BGRA swizzle not available
        if (type == GL_INT_2_10_10_10_REV) return MTLVertexFormatInt1010102Normalized;
        return MTLVertexFormatInvalid;
    }
    auto pick = [&](MTLVertexFormat f1, MTLVertexFormat f2, MTLVertexFormat f3, MTLVertexFormat f4) {
        switch (size) {
        case 1: return f1;
        case 2: return f2;
        case 3: return f3;
        default: return f4;
        }
    };
    switch (type) {
    case GL_BYTE:
        if (integer || !normalized) {
            if (integer) return pick(MTLVertexFormatChar, MTLVertexFormatChar2, MTLVertexFormatChar3, MTLVertexFormatChar4);
            // float input from unnormalized bytes: Metal converts Char -> float? It does not; use int->float in shader
            // isn't possible via stage_in, so treat as Char and let the shader declare float: Metal allows
            // non-normalized integer formats only for integer inputs. Fallback to normalized-free conversion path.
            return pick(MTLVertexFormatChar, MTLVertexFormatChar2, MTLVertexFormatChar3, MTLVertexFormatChar4);
        }
        return pick(MTLVertexFormatCharNormalized, MTLVertexFormatChar2Normalized, MTLVertexFormatChar3Normalized, MTLVertexFormatChar4Normalized);
    case GL_UNSIGNED_BYTE:
        if (integer || !normalized) return pick(MTLVertexFormatUChar, MTLVertexFormatUChar2, MTLVertexFormatUChar3, MTLVertexFormatUChar4);
        return pick(MTLVertexFormatUCharNormalized, MTLVertexFormatUChar2Normalized, MTLVertexFormatUChar3Normalized, MTLVertexFormatUChar4Normalized);
    case GL_SHORT:
        if (integer || !normalized) return pick(MTLVertexFormatShort, MTLVertexFormatShort2, MTLVertexFormatShort3, MTLVertexFormatShort4);
        return pick(MTLVertexFormatShortNormalized, MTLVertexFormatShort2Normalized, MTLVertexFormatShort3Normalized, MTLVertexFormatShort4Normalized);
    case GL_UNSIGNED_SHORT:
        if (integer || !normalized) return pick(MTLVertexFormatUShort, MTLVertexFormatUShort2, MTLVertexFormatUShort3, MTLVertexFormatUShort4);
        return pick(MTLVertexFormatUShortNormalized, MTLVertexFormatUShort2Normalized, MTLVertexFormatUShort3Normalized, MTLVertexFormatUShort4Normalized);
    case GL_INT: return pick(MTLVertexFormatInt, MTLVertexFormatInt2, MTLVertexFormatInt3, MTLVertexFormatInt4);
    case GL_UNSIGNED_INT: return pick(MTLVertexFormatUInt, MTLVertexFormatUInt2, MTLVertexFormatUInt3, MTLVertexFormatUInt4);
    case GL_FLOAT: return pick(MTLVertexFormatFloat, MTLVertexFormatFloat2, MTLVertexFormatFloat3, MTLVertexFormatFloat4);
    case GL_HALF_FLOAT: return pick(MTLVertexFormatHalf, MTLVertexFormatHalf2, MTLVertexFormatHalf3, MTLVertexFormatHalf4);
    case GL_INT_2_10_10_10_REV: return normalized ? MTLVertexFormatInt1010102Normalized : MTLVertexFormatInvalid;
    case GL_UNSIGNED_INT_2_10_10_10_REV: return normalized ? MTLVertexFormatUInt1010102Normalized : MTLVertexFormatInvalid;
    case GL_UNSIGNED_INT_10F_11F_11F_REV: return MTLVertexFormatFloatRG11B10;
    default: return MTLVertexFormatInvalid;
    }
}

MTLBlendFactor blendFactor(GLenum f) {
    switch (f) {
    case GL_ZERO: return MTLBlendFactorZero;
    case GL_ONE: return MTLBlendFactorOne;
    case GL_SRC_COLOR: return MTLBlendFactorSourceColor;
    case GL_ONE_MINUS_SRC_COLOR: return MTLBlendFactorOneMinusSourceColor;
    case GL_DST_COLOR: return MTLBlendFactorDestinationColor;
    case GL_ONE_MINUS_DST_COLOR: return MTLBlendFactorOneMinusDestinationColor;
    case GL_SRC_ALPHA: return MTLBlendFactorSourceAlpha;
    case GL_ONE_MINUS_SRC_ALPHA: return MTLBlendFactorOneMinusSourceAlpha;
    case GL_DST_ALPHA: return MTLBlendFactorDestinationAlpha;
    case GL_ONE_MINUS_DST_ALPHA: return MTLBlendFactorOneMinusDestinationAlpha;
    case GL_CONSTANT_COLOR: return MTLBlendFactorBlendColor;
    case GL_ONE_MINUS_CONSTANT_COLOR: return MTLBlendFactorOneMinusBlendColor;
    case GL_CONSTANT_ALPHA: return MTLBlendFactorBlendAlpha;
    case GL_ONE_MINUS_CONSTANT_ALPHA: return MTLBlendFactorOneMinusBlendAlpha;
    case GL_SRC_ALPHA_SATURATE: return MTLBlendFactorSourceAlphaSaturated;
    case GL_SRC1_COLOR: return MTLBlendFactorSource1Color;
    case GL_ONE_MINUS_SRC1_COLOR: return MTLBlendFactorOneMinusSource1Color;
    case GL_SRC1_ALPHA: return MTLBlendFactorSource1Alpha;
    case GL_ONE_MINUS_SRC1_ALPHA: return MTLBlendFactorOneMinusSource1Alpha;
    default: return MTLBlendFactorOne;
    }
}

MTLBlendOperation blendOp(GLenum e) {
    switch (e) {
    case GL_FUNC_ADD: return MTLBlendOperationAdd;
    case GL_FUNC_SUBTRACT: return MTLBlendOperationSubtract;
    case GL_FUNC_REVERSE_SUBTRACT: return MTLBlendOperationReverseSubtract;
    case GL_MIN: return MTLBlendOperationMin;
    case GL_MAX: return MTLBlendOperationMax;
    default: return MTLBlendOperationAdd;
    }
}

MTLCompareFunction compareFunc(GLenum f) {
    switch (f) {
    case GL_NEVER: return MTLCompareFunctionNever;
    case GL_LESS: return MTLCompareFunctionLess;
    case GL_EQUAL: return MTLCompareFunctionEqual;
    case GL_LEQUAL: return MTLCompareFunctionLessEqual;
    case GL_GREATER: return MTLCompareFunctionGreater;
    case GL_NOTEQUAL: return MTLCompareFunctionNotEqual;
    case GL_GEQUAL: return MTLCompareFunctionGreaterEqual;
    default: return MTLCompareFunctionAlways;
    }
}

MTLStencilOperation stencilOp(GLenum op) {
    switch (op) {
    case GL_ZERO: return MTLStencilOperationZero;
    case GL_REPLACE: return MTLStencilOperationReplace;
    case GL_INCR: return MTLStencilOperationIncrementClamp;
    case GL_DECR: return MTLStencilOperationDecrementClamp;
    case GL_INVERT: return MTLStencilOperationInvert;
    case GL_INCR_WRAP: return MTLStencilOperationIncrementWrap;
    case GL_DECR_WRAP: return MTLStencilOperationDecrementWrap;
    default: return MTLStencilOperationKeep;
    }
}

}  // namespace mgl
