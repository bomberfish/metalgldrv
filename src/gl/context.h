// GL context: complete GL state + Metal command recording.
#pragma once
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include <array>
#include <functional>
#include <vector>
#include "gl/objects.h"
#include "mtl/device.h"

namespace mgl {

enum {
    kMaxTextureUnits = 96,      // combined units exposed (GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS)
    kMaxImageUnits = 8,
    kMaxUniformBufferBindings = 84,
    kMaxSSBOBindings = 16,
    kMaxAtomicCounterBindings = 8,
    kMaxXfbBuffers = 4,
    kMaxViewports = 16,
    kMaxClipDistances = 8,
    kNumTexTargets = 11,
};

// Texture target index for per-unit binding arrays.
int texTargetIndex(GLenum target);  // -1 if invalid
extern const GLenum kTexTargets[kNumTexTargets];

struct BlendState {
    bool enabled = false;
    GLenum srcRGB = GL_ONE, dstRGB = GL_ZERO, srcA = GL_ONE, dstA = GL_ZERO;
    GLenum eqRGB = GL_FUNC_ADD, eqA = GL_FUNC_ADD;
    uint8_t colorMask = 0xF;  // RGBA bits
};

struct StencilFace {
    GLenum func = GL_ALWAYS;
    GLint ref = 0;
    GLuint valueMask = ~0u, writeMask = ~0u;
    GLenum sfail = GL_KEEP, dpfail = GL_KEEP, dppass = GL_KEEP;
};

struct IndexedBufferBinding {
    Ref<Buffer> buffer;
    GLintptr offset = 0;
    GLsizeiptr size = 0;  // 0 = whole buffer (glBindBufferBase)
};

struct ImageUnit {
    Ref<Texture> tex;
    GLint level = 0;
    bool layered = false;
    GLint layer = 0;
    GLenum access = GL_READ_ONLY;
    GLenum format = GL_R8;
};

// A drawable surface (EGL window/pbuffer). Holds the default framebuffer images.
class Surface : public RefCounted {
public:
    ~Surface() override;
    CAMetalLayer* layer = nil;  // nil for pbuffers
    int width = 0, height = 0;
    int samples = 0;
    MTLPixelFormat colorFormat = MTLPixelFormatRGBA8Unorm;
    bool hasDepth = true, hasStencil = true;
    bool srgbCapable = false;
    int swapInterval = 1;
    id<MTLTexture> color = nil, depthStencil = nil, msColor = nil;
    const FormatInfo* colorFmt = nullptr;
    const FormatInfo* dsFmt = nullptr;
    Ref<Framebuffer> fb;  // the default framebuffer object (name 0)
    uint64_t lastUse = 0;
    // MGL_UPSCALER=metalfx: the drawable is kept at native resolution and the
    // rendered image is upscaled with MetalFX's spatial scaler at present time.
    id upscaler = nil;               // id<MTL4FXSpatialScaler>
    id<MTLTexture> upscaled = nil;
    id<MTLFence> upscaleFence = nil;
    int outWidth = 0, outHeight = 0; // drawable size
    bool resize(int w, int h);  // (re)allocate images; returns true if changed
    void syncSizeFromLayer();
};

// Transient GPU memory for per-draw data.
struct UploadAlloc {
    id<MTLBuffer> buffer;
    uint8_t* cpu;
    uint64_t gpu;
    size_t offset;
};

struct PassAttachment {
    id<MTLTexture> tex = nil;
    int level = 0, slice = 0, depthPlane = 0;
    bool operator==(const PassAttachment& o) const {
        return tex == o.tex && level == o.level && slice == o.slice && depthPlane == o.depthPlane;
    }
};

struct PassDesc {
    PassAttachment color[kMaxDrawBuffers];
    PassAttachment depth, stencil;
    int width = 0, height = 0, layers = 1, samples = 1;
    bool operator==(const PassDesc& o) const;
};

struct PendingClear {
    uint8_t colorMask = 0;  // which color attachments to clear
    float color[kMaxDrawBuffers][4];
    uint32_t colorInt[kMaxDrawBuffers][4];
    bool depth = false, stencil = false;
    float depthValue = 1;
    uint32_t stencilValue = 0;
    void reset() { colorMask = 0; depth = stencil = false; }
    bool any() const { return colorMask || depth || stencil; }
};

struct ProgramVariant;
class PipelineCache;

// Debug output (KHR_debug)
struct DebugMessage {
    GLenum source, type, severity;
    GLuint id;
    std::string text;
};

class Context : public RefCounted {
public:
    Context(Ref<ShareGroup> share, int major, int minor, bool core, bool debug, bool forwardCompat, bool robust, bool noError);
    ~Context() override;

    // ---- identity ----
    Device* dev;
    Ref<ShareGroup> share;
    int majorVersion, minorVersion;
    bool coreProfile, debugContext, forwardCompatible, robustAccess, noErrorMode;
    bool everCurrent = false;

    // ---- errors / debug ----
    GLenum error = GL_NO_ERROR;
    void setError(GLenum e, const char* fmt = nullptr, ...);
    bool debugOutput = false, debugOutputSync = false;
    GLDEBUGPROC debugCallback = nullptr;
    const void* debugUserParam = nullptr;
    std::vector<DebugMessage> debugLog;
    struct DebugGroup { GLenum source; GLuint id; std::string message; };
    std::vector<DebugGroup> debugGroups;
    // message control: key = (source,type,severity) enable flags; id overrides
    std::map<uint64_t, bool> debugControl;
    std::map<uint64_t, bool> debugIdControl;
    void debugMessage(GLenum source, GLenum type, GLuint id, GLenum severity, const std::string& msg);

    // ---- surfaces ----
    Ref<Surface> drawSurface, readSurface;
    void bindSurfaces(Surface* draw, Surface* read);

    // ---- object bindings ----
    Ref<Buffer> arrayBuffer, copyReadBuffer, copyWriteBuffer, pixelPackBuffer, pixelUnpackBuffer, drawIndirectBuffer,
        dispatchIndirectBuffer, queryBuffer, textureBufferBinding, parameterBuffer, uniformBufferGeneric,
        ssboGeneric, atomicCounterGeneric, xfbGeneric;
    IndexedBufferBinding uniformBuffers[kMaxUniformBufferBindings];
    IndexedBufferBinding storageBuffers[kMaxSSBOBindings];
    IndexedBufferBinding atomicCounterBuffers[kMaxAtomicCounterBindings];

    GLuint activeTexture = 0;
    Ref<Texture> textures[kMaxTextureUnits][kNumTexTargets];
    Ref<Sampler> samplers[kMaxTextureUnits];
    Ref<Texture> defaultTextures[kNumTexTargets];  // texture object 0 per target
    std::map<GLenum, Ref<Texture>> proxyTextures;    // GL_PROXY_TEXTURE_* level state
    ImageUnit imageUnits[kMaxImageUnits];

    Ref<ProgramObj> program;
    Ref<ProgramPipelineObj> pipeline;
    Ref<VertexArray> vao;
    Ref<VertexArray> defaultVao;  // compat profile VAO 0
    Ref<Framebuffer> drawFb, readFb;
    Ref<Renderbuffer> renderbuffer;
    Ref<TransformFeedbackObj> xfb;
    Ref<TransformFeedbackObj> defaultXfb;

    // per-context namespaces
    NameSpace<VertexArray> vaos;
    NameSpace<Framebuffer> framebuffers;
    NameSpace<Query> queries;
    NameSpace<TransformFeedbackObj> xfbs;
    NameSpace<ProgramPipelineObj> pipelines;

    // ---- fixed function state ----
    float viewport[kMaxViewports][4];
    float depthRange[kMaxViewports][2];
    int scissor[kMaxViewports][4];
    uint32_t scissorEnabled = 0;  // bit per viewport
    float clearColor[4] = {0, 0, 0, 0};
    double clearDepth = 1.0;
    GLint clearStencil = 0;
    BlendState blend[kMaxDrawBuffers];
    float blendColor[4] = {0, 0, 0, 0};
    bool depthTest = false, depthWrite = true;
    GLenum depthFunc = GL_LESS;
    bool stencilTest = false;
    StencilFace stencil[2];  // front, back
    bool cullFace = false;
    GLenum cullMode = GL_BACK, frontFace = GL_CCW;
    GLenum polygonMode = GL_FILL;
    bool polygonOffsetFill = false, polygonOffsetLine = false, polygonOffsetPoint = false;
    float polygonOffsetFactor = 0, polygonOffsetUnits = 0, polygonOffsetClamp = 0;
    float lineWidth = 1, pointSize = 1;
    bool programPointSize = false;
    GLenum pointSpriteOrigin = GL_UPPER_LEFT;
    float pointFadeThreshold = 1;
    bool primitiveRestart = false, primitiveRestartFixed = false;
    GLuint primitiveRestartIndex = 0;
    bool depthClamp = false, rasterizerDiscard = false;
    bool multisample = true, sampleAlphaToCoverage = false, sampleAlphaToOne = false, sampleCoverage = false;
    float sampleCoverageValue = 1;
    bool sampleCoverageInvert = false;
    bool sampleMaskEnabled = false;
    uint32_t sampleMaskValue = ~0u;
    bool sampleShading = false;
    float minSampleShading = 0;
    bool colorLogicOp = false;
    GLenum logicOp = GL_COPY;
    bool dither = true, framebufferSRGB = false;
    uint32_t clipDistanceMask = 0;
    GLenum clipOrigin = GL_LOWER_LEFT, clipDepthMode = GL_NEGATIVE_ONE_TO_ONE;
    GLenum provokingVertex = GL_LAST_VERTEX_CONVENTION;
    bool lineSmooth = false, polygonSmooth = false, textureCubeSeamless = false;
    GLint patchVertices = 3;
    float patchDefaultOuter[4] = {1, 1, 1, 1}, patchDefaultInner[2] = {1, 1};
    PixelStore pack, unpack;
    GLenum hints[8] = {GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE};
    float currentAttrib[kMaxVertexAttribs][4];
    uint32_t currentAttribType[kMaxVertexAttribs];  // GL_FLOAT / GL_INT / GL_UNSIGNED_INT
    Ref<Query> activeQueries[8][4];  // [query target index][index]
    GLenum conditionalRenderMode = 0;
    Ref<Query> conditionalQuery;

    struct CompatState* compat = nullptr;
    GLenum compatAlphaFunc = GL_ALWAYS;  // legacy alpha test, applied by compat fragment shaders
    float compatAlphaRef = 0;  // compatibility-profile fixed-function state (compat.mm)

    // ---- object helpers ----
    Texture* boundTexture(GLenum target);         // active unit
    Texture* texForUnit(int unit, int targetIdx);
    Buffer** bufferBindingPtr(GLenum target);     // nullptr on invalid target
    Ref<Buffer>* bufferBindingRef(GLenum target);
    Framebuffer* fbForTarget(GLenum target);
    Framebuffer* defaultFramebuffer() { return drawSurface ? drawSurface->fb.get() : nullptr; }
    ProgramObj* activeProgramForStage(glsl::Stage s);
    ProgramObj* activeProgram();  // program or pipeline's active program
    void unbindDeletedBuffer(Buffer* b);
    void unbindDeletedTexture(Texture* t);
    void unbindDeletedRenderbuffer(Renderbuffer* rb);

    // ---- Metal recording ----
    // Lazily begins a command buffer; returns its serial.
    uint64_t serial();
    id<MTL4CommandBuffer> commandBuffer();
    id<MTL4RenderCommandEncoder> renderEncoder();  // ensures a pass for the current draw framebuffer
    id<MTL4ComputeCommandEncoder> computeEncoder(); // ends any render pass
    void endEncoder();
    // Submits the command buffer. present: drawable to present after it.
    void flush(id<CAMetalDrawable> present = nil);
    void finish();
    // Submits the open command buffer if it is the one recorded under `serial`.
    void flushIfPending(uint64_t serial) {
        if (cb_ && serial_ == serial) flush();
    }
    UploadAlloc upload(size_t size, size_t align = 256);
    id<MTL4ArgumentTable> renderArgTable, computeArgTable;

    // render pass tracking
    PassDesc curPass;
    bool passOpen = false;
    Framebuffer* passFb_ = nullptr;  // framebuffer the open pass was started for (stats only)
    bool passDirty = true;           // framebuffer binding/attachments changed
    PendingClear pendingClear;       // clears not yet applied (folded into next pass load action)
    uint64_t drawFbVersionSeen = 0;
    id<MTL4RenderCommandEncoder> renderEnc = nil;
    id<MTL4ComputeCommandEncoder> computeEnc = nil;
    // dynamic state applied to the current render encoder
    struct EncState {
        id pipeline = nil;
        id depthStencil = nil;
        MTLViewport vp[kMaxViewports];
        int numViewports = 0;
        MTLScissorRect sc[kMaxViewports];
        int cull = -1, winding = -1, fill = -1, depthClip = -1;
        float depthBias[3] = {NAN, NAN, NAN};
        float blendColor[4] = {NAN, NAN, NAN, NAN};
        uint32_t stencilRef[2] = {~0u, ~0u};
        void reset() { *this = EncState(); }
    } enc;
    // Visibility (occlusion query) buffer for the current pass
    id<MTLBuffer> visibilityBuffer = nil;
    uint32_t visibilityNext = 0;
    id<MTLBuffer> passVisibilityBuffer = nil;  // buffer attached to the open pass
    int64_t encVisibilityOffset = -2;           // -1 disabled, -2 unknown
    bool conditionalPass = true;

    bool resolvePass(PassDesc& out);  // builds a pass description from the draw framebuffer
    void beginPass();
    void endPass();
    void applyClearNow();  // apply pending clears by opening/closing a pass

    // Synchronizes CPU access to a resource last used in `lastUse`.
    void waitForResource(uint64_t lastUse);

    // Per-context pipeline cache (shared state lives in Device later).
    PipelineCache* pipelines_ = nullptr;

    // Draw-time helpers implemented in draw.mm
    bool prepareDraw(GLenum mode, bool indexed);
    struct DrawInfo;

private:
    std::vector<std::pair<uint64_t, id<MTLBuffer>>> uploadRetired_;  // (serial, chunk)
    id<MTLBuffer> uploadChunk_ = nil;
    size_t uploadOffset_ = 0;
    std::vector<id<MTLBuffer>> uploadUsed_;  // chunks used by the open command buffer
    id<MTL4CommandBuffer> cb_ = nil;
    id<MTL4CommandAllocator> alloc_ = nil;
    uint64_t serial_ = 0;
};

// Thread-current context.
Context* currentContext();
void setCurrentContext(Context* c);

// Texture storage management (texture.mm)
bool ensureTextureStorage(Context* ctx, Texture* t);
id<MTLTexture> textureSamplingView(Context* ctx, Texture* t, GLenum samplerTarget, bool stencilSampling, bool* complete);
id<MTLSamplerState> samplerStateFor(const SamplerState& s, bool isInteger);
id<MTLTexture> attachmentTexture(Context* ctx, const Attachment& a, const FormatInfo** fmt, int* w, int* h);
void markTextureWrite(Context* ctx, Texture* t);

// Common entry-point prologue.
#define MGL_CTX_RET(r)                     \
    ::mgl::Context* ctx = ::mgl::currentContext(); \
    if (!ctx) return r
#define MGL_CTX() MGL_CTX_RET()

}  // namespace mgl
