// Shader variant + Metal pipeline state caching.
#pragma once
#import <Metal/Metal.h>
#include <string>
#include <unordered_map>
#include "gl/objects.h"
#include "glsl/glsl.h"

namespace mgl {

class Context;

// Extra argument-table index used by the runtime only (not referenced by MSL):
// one constant-step buffer holding current generic vertex attribute values.
enum : uint32_t { kBufCurrentAttribs = 30 };

struct ProgramVariant {
    uint64_t uid = 0;  // unique for the process lifetime (pipeline cache key; pointers get reused)
    glsl::MSLOutput msl;
    id<MTLLibrary> lib = nil;
    bool ok = false;
    bool manualFetch = false;  // classic VS pulls vertices from MGLDrawInfo (no vertex descriptor)
    std::string error;
    const glsl::EntryPoint* entry(glsl::EntryPoint::Kind k) const {
        for (auto& e : msl.entryPoints)
            if (e.kind == k) return &e;
        return nullptr;
    }
    id<MTLComputePipelineState> compute = nil;
};

#pragma pack(push, 1)
struct VertexAttribKey {
    uint8_t format = 0;       // MTLVertexFormat (0 = attribute not consumed)
    uint8_t bufferIndex = 0;  // argument table index
    uint16_t pad = 0;
    uint32_t offset = 0;
};
struct VertexBufferKey {
    uint8_t stepFunction = 0;  // MTLVertexStepFunction
    uint8_t used = 0;
    uint16_t pad = 0;
    uint32_t stepRate = 0;
};
struct BlendKey {
    uint8_t enabled = 0, srcRGB = 0, dstRGB = 0, srcA = 0, dstA = 0, eqRGB = 0, eqA = 0, writeMask = 0;
};
struct RenderPipelineKey {
    uint64_t variantUid = 0;
    const ProgramVariant* variant = nullptr;  // not part of the hashed identity
    // Program pipelines: fragment function from another program's variant (0 = same variant,
    // ~0 = no fragment shader).
    uint64_t fsVariantUid = 0;
    const ProgramVariant* fsVariant = nullptr;  // not hashed
    uint32_t colorFormats[8] = {};
    BlendKey blend[8];
    VertexAttribKey attribs[16];
    VertexBufferKey buffers[32];
    uint8_t sampleCount = 1, alphaToCoverage = 0, alphaToOne = 0, rasterEnabled = 1;
    uint8_t topologyClass = 0, logicOp = 0, pad[2] = {};
};
struct DepthStencilKey {
    uint8_t depthFunc = 0, depthWrite = 0, stencilEnabled = 0, pad = 0;
    struct Face { uint8_t func, sfail, dpfail, dppass; uint32_t readMask, writeMask; } face[2] = {};
};
#pragma pack(pop)

class PipelineCache {
public:
    explicit PipelineCache(Context* ctx) : ctx_(ctx) {}
    ProgramVariant* variant(ProgramObj* prog, const glsl::VariantKey& key);
    id<MTLRenderPipelineState> renderPipeline(const RenderPipelineKey& key);
    id<MTLComputePipelineState> computePipeline(ProgramVariant* v);
    id<MTLDepthStencilState> depthStencil(const DepthStencilKey& key);

private:
    Context* ctx_;
    std::unordered_map<std::string, id<MTLRenderPipelineState>> render_;
    std::unordered_map<std::string, id<MTLDepthStencilState>> ds_;
};

uint64_t hashBytes(const void* data, size_t n, uint64_t seed = 1469598103934665603ull);
MTLVertexFormat vertexFormatFor(GLenum type, int size, bool normalized, bool integer);
MTLBlendFactor blendFactor(GLenum f);
MTLBlendOperation blendOp(GLenum e);
MTLCompareFunction compareFunc(GLenum f);
MTLStencilOperation stencilOp(GLenum op);

}  // namespace mgl
