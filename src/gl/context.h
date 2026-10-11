// GL context: complete GL state + Metal command recording.
#pragma once
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include <deque>
#include <array>
#include <functional>
#include <vector>
#include "gl/objects.h"
#include "mtl/device.h"

namespace mgl {

enum {
    kMaxTextureUnits = 96,      // combined units exposed (GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS)
    kMaxImageUnits = 16,
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
class Context;
class Surface : public RefCounted {
public:
    ~Surface() override;
    CAMetalLayer* layer = nil;  // nil for pbuffers
    int width = 0, height = 0;
    int samples = 0;
    MTLPixelFormat colorFormat = MTLPixelFormatRGBA8Unorm;
    bool hasDepth = true, hasStencil = true;
    bool hasAlpha = true;  // EGL config with alpha bits; without them alpha reads and blends as 1
    // Whether the app has used depthStencil (cleared it, drawn with depth or stencil testing, blitted into
    // it). Until then render passes on the default framebuffer leave it out, so apps that render offscreen
    // and only blit to the window don't load and store it every frame.
    bool dsUsed = false;
    bool srgbCapable = false;
    int swapInterval = 1;
    id<MTLTexture> color = nil, depthStencil = nil, msColor = nil;
    // Non-sRGB views of color/msColor: GL_FRAMEBUFFER_SRGB disabled writes an sRGB-capable
    // default framebuffer without encoding (linearWriteView). Rebuilt on resize.
    id<MTLTexture> colorLinear = nil, msColorLinear = nil;
    id<MTLTexture> linearWriteView(id<MTLTexture> t);
    // sRGB-capable window surfaces from CGL / NSOpenGL store linear RGBA8 until the app enables
    // GL_FRAMEBUFFER_SRGB with this surface bound (Context::syncSurfaceSRGB), and then switch to sRGB
    // storage with format views for the rest of their life: the views cost lossless compression.
    bool srgbStorage() const { return colorFormat == MTLPixelFormatRGBA8Unorm_sRGB; }
    void useSRGBStorage(Context* ctx);
    void allocColor();  // color / msColor at width x height in colorFormat
    const FormatInfo* colorFmt = nullptr;
    const FormatInfo* dsFmt = nullptr;
    Ref<Framebuffer> fb;  // the default framebuffer object (name 0)
    uint64_t lastUse = 0;
    // METALGLDRV_USE_METALFX=1: the drawable is kept at native resolution and the
    // rendered image is upscaled with MetalFX's spatial scaler at present time.
    id upscaler = nil;               // id<MTL4FXSpatialScaler>
    id<MTLTexture> upscaled = nil;
    id<MTLFence> upscaleFence = nil;
    int outWidth = 0, outHeight = 0; // drawable size
    // CGL kCGLCPSurfaceBackingSize: render at this size whatever the layer's size
    // (the layer scales the drawable to its bounds). 0 = follow the layer.
    int fixedWidth = 0, fixedHeight = 0;
    bool resize(int w, int h);  // (re)allocate images; returns true if changed
    void syncSizeFromLayer();
};

// Transient GPU memory for per-draw data.
struct UploadAlloc {
    // owned by the context's upload lists (or the GL buffer it points into) for as long as an UploadAlloc is used:
    // unretained saves a retain / release per upload
    __unsafe_unretained id<MTLBuffer> buffer;
    uint8_t* cpu;
    uint64_t gpu;
    size_t offset;
};

struct PassAttachment {
    id<MTLTexture> tex = nil;
    int level = 0, slice = 0, depthPlane = 0;
    bool noAlpha = false;  // default framebuffer without alpha bits: alpha stays 1 (never written, cleared to 1)
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

// The framebuffer namespace of a new context: `own`, or the share group's (AppProfile::shareFramebuffers).
NameSpace<Framebuffer>& framebufferNames(ShareGroup* share, NameSpace<Framebuffer>& own);

class Context : public RefCounted {
public:
    Context(Ref<ShareGroup> share, int major, int minor, bool core, bool debug, bool forwardCompat, bool robust, bool noError);
    ~Context() override;

    // ---- identity ----
    Device* dev;
    Ref<ShareGroup> share;
    int majorVersion, minorVersion;
    bool coreProfile, debugContext, forwardCompatible, robustAccess, noErrorMode;
    int strictLevel;  // dev->caps.strictLevel (METALGLDRV_STRICT)
    bool strict() const { return strictLevel >= 1; }
    bool superStrict() const { return strictLevel >= 2; }
    bool loseContextOnReset = false;
    GLenum clampReadColor = GL_FIXED_ONLY;  // glClampColor(GL_CLAMP_READ_COLOR)  // RESET_NOTIFICATION_STRATEGY requested at creation
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

    // per-context namespaces (vertex arrays and pipelines: the share group's with ShareGroup::shareContainers)
    NameSpace<VertexArray> ownVaos;
    NameSpace<VertexArray>& vaos = share && share->shareContainers ? share->vaos : ownVaos;
    // GL_APPLE_fence names: the serial SetFenceAPPLE recorded, 0 while generated but not set.
    std::unordered_map<GLuint, uint64_t> appleFences;
    std::unordered_map<GLuint, uint64_t> nvFences;  // GL_NV_fence, likewise
    int extMarkerDepth = 0;
    // GL_ARB_vertex_program / GL_ARB_fragment_program: enables, bound programs (program 0 is this context's own
    // default object), environment parameters, the last load's error
    bool arbVertexProgram = false, arbFragmentProgram = false;
    Ref<ArbProgram> arbVP, arbFP, arbDefaultVP, arbDefaultFP;
    float arbVPEnv[kArbMaxEnvParams][4] = {}, arbFPEnv[kArbMaxEnvParams][4] = {};
    std::string arbErrorString;
    GLint arbErrorPos = -1;
    // GL_HP_occlusion_test: an ANY_SAMPLES_PASSED_CONSERVATIVE query runs while it is enabled (api_query.mm)
    bool hpOcclusionTest = false, hpOcclusionResult = false;
    GLuint hpOcclusionQuery = 0;  // GL_EXT_debug_marker groups pushed (popping more is ignored)
    // Framebuffer names: this context's own, or the share group's when the app profile shares them
    // (AppProfile::shareFramebuffers).
    NameSpace<Framebuffer> ownFramebuffers;
    NameSpace<Framebuffer>& framebuffers = framebufferNames(share.get(), ownFramebuffers);
    NameSpace<Query> queries;
    NameSpace<TransformFeedbackObj> xfbs;
    NameSpace<ProgramPipelineObj> ownPipelines;
    NameSpace<ProgramPipelineObj>& pipelines = share && share->shareContainers ? share->pipelines : ownPipelines;
    // Held while using container object names that the share group owns (ShareGroup::shareContainers).
    std::unique_lock<std::recursive_mutex> containerLock() {
        return share && share->shareContainers ? std::unique_lock<std::recursive_mutex>(share->mutex) : std::unique_lock<std::recursive_mutex>();
    }

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
    // GL_EXT_stencil_two_side: glStencilFunc/Op/Mask set the active face. With two-side stencil off, back faces
    // render with the front state (stencil[1] mirrors stencil[0]) and the back state waits in extStencilBack.
    bool stencilTwoSideEXT = false;
    GLenum activeStencilFaceEXT = GL_FRONT;
    StencilFace extStencilBack;
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
    // EXT_fragment_shading_rate (METALGLDRV_STRICT): the draw call rate and combiner ops 0 and 1
    GLenum shadingRate = GL_SHADING_RATE_1X1_PIXELS_EXT;
    GLenum shadingRateCombiner[2] = {GL_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_EXT, GL_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_EXT};
    bool colorLogicOp = false;
    GLenum logicOp = GL_COPY;
    bool dither = true, framebufferSRGB = false;
    bool blendAdvancedCoherent = true;  // GL_BLEND_ADVANCED_COHERENT_KHR (framebuffer fetch is always coherent)
    bool advancedBlendUsed = false;     // an advanced blend equation was ever set: draws validate it
    uint32_t clipDistanceMask = 0;
    GLenum clipOrigin = GL_LOWER_LEFT, clipDepthMode = GL_NEGATIVE_ONE_TO_ONE;
    GLenum provokingVertex = GL_LAST_VERTEX_CONVENTION;
    bool lineSmooth = false, polygonSmooth = false, textureCubeSeamless = false;
    GLint patchVertices = 3;
    float patchDefaultOuter[4] = {1, 1, 1, 1}, patchDefaultInner[2] = {1, 1};
    PixelStore pack, unpack;
    GLenum hints[8] = {GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE};
    float currentAttrib[kMaxVertexAttribs][4];
    double currentAttribD[kMaxVertexAttribs][4];  // double view of the current values (glVertexAttribL*)
    uint32_t currentAttribType[kMaxVertexAttribs];  // GL_FLOAT / GL_INT / GL_UNSIGNED_INT
    Ref<Query> activeQueries[19][4];  // [query target index][index] (api_query.mm queryTargetIndex)
    int statsQueriesActive = 0;       // active ARB_pipeline_statistics_query queries (draw-time counting)
    id<MTLBuffer> statsSlots = nil;   // GPU-side statistics counters (2 x uint per draw slot)
    uint32_t statsNext = 0;
    GLenum conditionalRenderMode = 0;
    Ref<Query> conditionalQuery;
    // Emulated geometry per command buffer (draw.mm meshWork*): GPU work of the tessellation / geometry shader
    // emulation is estimated in items (patches x expected triangles) and command buffers are committed before
    // they hold more than a budget, so that none gets near the GPU watchdog. ns per item is learned from the
    // GPU times of completed command buffers.
    uint64_t meshWorkSerial = 0;
    double meshWorkItems = 0;                                 // recorded in command buffer meshWorkSerial
    double meshWorkNsPerItem = 200;                           // estimate (conservative until measured)
    std::vector<std::pair<uint64_t, double>> meshWorkPending; // submitted command buffers: (serial, items)

    std::vector<uint32_t> subroutineSelection[(int)glsl::Stage::Count];  // glUniformSubroutinesuiv

    struct CompatState* compat = nullptr;
    GLenum compatAlphaFunc = GL_ALWAYS;  // legacy alpha test, applied by compat fragment shaders
    float compatAlphaRef = 0;
    bool compatVertexPointSize = false;  // fixed-function program writing gl_PointSize (APPLE_vertex_point_size)  // compatibility-profile fixed-function state (compat.mm)

    // ---- object helpers ----
    Texture* boundTexture(GLenum target);         // active unit
    Texture* texForUnit(int unit, int targetIdx);
    Buffer** bufferBindingPtr(GLenum target);     // nullptr on invalid target
    Ref<Buffer>* bufferBindingRef(GLenum target);
    Framebuffer* fbForTarget(GLenum target);
    Framebuffer* defaultFramebuffer() { return drawSurface ? drawSurface->fb.get() : nullptr; }
    // Marks the default framebuffer's depth/stencil buffer as used (Surface::dsUsed) when fb is it.
    void useDefaultDepthStencil(Framebuffer* fb) {
        if (fb && fb->isDefault && fb->surface && !fb->surface->dsUsed) {
            fb->surface->dsUsed = true;
            passDirty = true;
        }
    }
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
    // New encoders on the command buffer, created inside an @autoreleasepool (see device.h).
    id<MTL4RenderCommandEncoder> newRenderEncoder(MTL4RenderPassDescriptor* rp) NS_RETURNS_RETAINED;
    id<MTL4ComputeCommandEncoder> newComputeEncoder() NS_RETURNS_RETAINED;
    void endEncoder();
    // Submits the command buffer. present: drawable to present after it. keepUploads: the upload memory used so far
    // stays with the next command buffer (retired with its serial; openTessSegment).
    void flush(id<CAMetalDrawable> present = nil, bool keepUploads = false);
    // The command buffer flush() submitted last (its serial).
    uint64_t lastSubmittedSerial() const { return serial_; }
    // Serials of the last frames presented to window surfaces (presentWindowSurface keeps the CPU at most
    // kMaxFramesInFlight frames ahead of the GPU).
    std::deque<uint64_t> frameSerials;
    void finish();
    // Submits the open command buffer if it is the one recorded under `serial`.
    void flushIfPending(uint64_t serial) {
        if ((cb_ && serial_ == serial) || (tessSeg.open && tessSeg.serial == serial)) flush();
    }
    UploadAlloc upload(size_t size, size_t align = 256);
    // The buffer and offset of upload memory at GPU address `gpu` used by the open command buffer (nil if none):
    // classic encoders bind buffers, not addresses.
    id<MTLBuffer> uploadBuffer(uint64_t gpu, NSUInteger* offset);

    // ---- Metal's tessellator (draw.mm executeTessMetal) ----
    // Metal 4 has no tessellation. Tessellated draws, and the draws after them into the same pass, run on the
    // device's classic queue in a segment: a pre-pass command buffer runs the VS + TCS kernels, and a render pass
    // over the same attachments draws (drawPatches). The segment has a serial of its own (Context::serial() while it
    // is open), complete when its render command buffer is. Anything recorded for Metal 4 (Context::commandBuffer)
    // ends it. Ordering, with one event:
    //  - openTessSegment submits the Metal 4 work so far and signals `mtl` after it; the pre-pass waits for it (or,
    //    without Metal 4 work since the last segment, for that one's pre-pass).
    //  - The render command buffer waits for the pre-pass (`prep`) and the previous segment's render command buffer
    //    (`main` of that one): same queue, but untracked resources.
    //  - The pre-pass waits for the previous segment only when it may read what that one wrote (prepWaitsMain:
    //    textures, images or storage in the VS / TCS, or a memory barrier since). Otherwise it runs while the
    //    previous segment draws, as compute next to the render pass.
    //  - The Metal 4 queue waits for the last segment before it runs the next Metal 4 command buffer (flush).
    // Completions therefore stay in submission order (serials, upload memory: Context::lastWorkSerial_).
    struct TessSegment {
        bool open = false;
        PassDesc pass;
        id<MTLCommandBuffer> prep = nil, main = nil;
        id<MTLComputeCommandEncoder> compute = nil;
        id<MTLRenderCommandEncoder> render = nil;
        uint64_t serial = 0;
        uint64_t mtl = 0, prepDone = 0, mainDone = 0;  // event values (above)
        bool prepWaitsMain = false;
        uint64_t count = 0;  // segments opened by this context
        // what the pass was resolved from (tessSegmentPassCurrent)
        Framebuffer* fb = nullptr;
        uint64_t fbVersion = 0;
        bool fbSRGB = false;
        int surfaceW = 0, surfaceH = 0;
        id<MTLTexture> storage[kNumAttachments] = {};
        uint32_t depthFormat = 0, stencilFormat = 0;  // MTLPixelFormat of the pass's depth / stencil attachments
    } tessSeg;
    // The draw framebuffer is still the open segment's pass (what resolvePass would give, without resolving it):
    // the same framebuffer, unchanged, its attachments on the same storage.
    bool tessSegmentPassCurrent() const;
    void tessSegmentSnapshot();  // records the draw framebuffer's state for tessSegmentPassCurrent
    id<MTLEvent> tessEvent = nil;
    uint64_t tessEventValue = 0;
    uint64_t tessLastMain = 0;    // event value of the last segment's render command buffer
    uint64_t tessLastPrep = 0;    // and of its (last) pre-pass
    bool tessMtlWork = false;     // Metal 4 work submitted since the last segment opened
    uint64_t tessQueueWaited = 0;  // the last such value the Metal 4 queue waits for
    bool tessBarrier = false;      // glMemoryBarrier / glTextureBarrier since the last segment
    // prevWrites: the first draw's VS / TCS read what the previous segment may have written (prepWaitsMain)
    bool openTessSegment(const PassDesc& p, bool prevWrites);  // false if there is no classic queue
    void endTessSegment();
    // The pre-pass of the open segment waits for the previous segment (a new pre-pass command buffer if it doesn't).
    void tessPrepWaitForPrevious();
    // Work submitted to the Metal 4 queue from now on (sparse mapping updates) waits for the segments so far.
    void queueAfterTessSegments();
    id<MTL4ArgumentTable> renderArgTable, computeArgTable;
    id<MTL4ArgumentTable> renderArgTableFS;  // fragment stage of a program pipeline whose FS comes from another program

    // render pass tracking
    PassDesc curPass;
    bool passOpen = false;
    uint64_t passCounter = 0;  // render passes begun (feedback-loop snapshots are per pass)
    Framebuffer* passFb_ = nullptr;  // framebuffer the open pass was started for (stats only)
    // METALGLDRV_PROFILE_PASSES: the open pass's label parts, and whether flush() is running.
    char passLabel_[128] = {};
    int passDraws_ = 0;
    GLuint passFirstProgram_ = 0;
    bool inFlush_ = false;
    bool passDirty = true;           // framebuffer binding/attachments changed
    PendingClear pendingClear;       // clears not yet applied (folded into next pass load action)
    // Non-layered image bindings of a 3D texture slice go through a 2D copy (Metal has no 2D
    // views of 3D textures); writable ones are copied back after the draw/dispatch.
    struct ImageSliceWriteback {
        id<MTLTexture> tmp, dst;
        NSUInteger level, z;
    };
    std::vector<ImageSliceWriteback> imageWritebacks;
    // Writable multisample image bindings (METALGLDRV_STRICT): per-sample shadows written back into
    // their multisample textures after the draw/dispatch (draw.mm, msImageShadow).
    struct MSImageWriteback {
        id<MTLTexture> shadow, ms;
        int kind;  // 0 float, 1 int, 2 uint
    };
    std::vector<MSImageWriteback> msImageWritebacks;
    uint64_t drawFbVersionSeen = 0;
    id<MTL4RenderCommandEncoder> renderEnc = nil;
    id<MTL4ComputeCommandEncoder> computeEnc = nil;
    // dynamic state applied to the current render encoder
    struct EncState {
        id pipeline = nil;
        id depthStencil = nil;
        MTLViewport vp[kMaxViewports];
        int numViewports = 0, numScissors = 0;
        MTLScissorRect sc[kMaxViewports];
        int cull = -1, winding = -1, fill = -1, depthClip = -1;
        float depthBias[3] = {NAN, NAN, NAN};
        float blendColor[4] = {NAN, NAN, NAN, NAN};
        uint32_t stencilRef[2] = {~0u, ~0u};
        float lineWidth = 1;  // spi::setLineWidth (encoders start at 1)
        bool fsTableSplit = false;  // fragment stage bound to renderArgTableFS (pipeline mixing programs)
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
    void syncSurfaceSRGB();  // GL_FRAMEBUFFER_SRGB on: bound default framebuffers get sRGB storage (Surface::srgbStorage)

    // Synchronizes CPU access to a resource last used in `lastUse`.
    void waitForResource(uint64_t lastUse);

    // Per-context pipeline cache (shared state lives in Device later).
    PipelineCache* pipelines_ = nullptr;

    // Draw-time helpers implemented in draw.mm
    bool prepareDraw(GLenum mode, bool indexed);
    struct DrawInfo;

private:
    std::vector<std::pair<uint64_t, id<MTLBuffer>>> uploadRetired_;  // (serial, chunk)
    // An upload buffer with its addresses (Objective-C calls per upload are slow under Rosetta)
    struct UploadBuf {
        id<MTLBuffer> buffer = nil;
        uint8_t* cpu = nullptr;
        uint64_t gpu = 0, length = 0;
    };
    UploadBuf uploadChunk_;
    id<MTLBuffer> takeRetiredUpload(size_t want);
    id<MTLBuffer> newUpload(size_t want);
    void retireUploads();
    size_t uploadPoolBytes() const;
    size_t uploadOffset_ = 0;
    std::vector<UploadBuf> uploadUsed_;  // chunks used by the open command buffer
    id<MTL4CommandBuffer> cb_ = nil;
    id<MTL4CommandAllocator> alloc_ = nil;
    uint64_t serial_ = 0;
    uint64_t lastWorkSerial_ = 0;
    bool cbUsed_ = false;  // something was recorded in cb_ (commandBuffer()), not just its serial taken  // the Metal 4 command buffer or segment submitted last (completes after the others)
};

// Thread-current context.
Context* currentContext();
void setCurrentContext(Context* c);

// Texture storage management (texture.mm)
bool ensureTextureStorage(Context* ctx, Texture* t);
// Before data is uploaded or copied into t: storage that lossless compression would alter goes uncompressed
// (Texture::bitExact).
void requireBitExact(Context* ctx, Texture* t);
id<MTLTexture> textureSamplingView(Context* ctx, Texture* t, GLenum samplerTarget, bool stencilSampling, bool* complete);
void textureSamplingViewInto(Context* ctx, Texture* t, GLenum samplerTarget, bool stencilSampling, bool* complete,
                             __unsafe_unretained id<MTLTexture>* out);
id<MTLSamplerState> samplerStateFor(const SamplerState& s, bool isInteger);
void samplerStateInto(const SamplerState& s, bool isInteger, __unsafe_unretained id<MTLSamplerState>* sampler, MTLResourceID* rid);
id<MTLTexture> attachmentTexture(Context* ctx, const Attachment& a, const FormatInfo** fmt, int* w, int* h);
// The texture a color attachment is rendered through (linear view of sRGB storage when
// GL_FRAMEBUFFER_SRGB is disabled).
id<MTLTexture> attachmentWriteView(Context* ctx, const Attachment& a, id<MTLTexture> t);
void markTextureWrite(Context* ctx, Texture* t);

// ARB_pipeline_statistics_query (api_query.mm): counts of one draw/dispatch, added to the
// active statistics queries. GPU-only counts go through mglStatsGpuSlot (returns the GPU
// address of uint[3] {GS primitives emitted, TES invocations, tessellated primitives}, or 0).
struct DrawStats {
    uint64_t vertices = 0, primitives = 0, vsInvocations = 0, patches = 0, tesInvocations = 0, gsInvocations = 0, gsPrimitives = 0,
             csInvocations = 0, clipPrimitives = 0, fsEstimate = 0;
};
void mglStatsCountDraw(Context* ctx, const DrawStats& st);
uint64_t mglStatsGpuSlot(Context* ctx, bool gs, bool tess);
void releaseDrawCaches(Context* ctx);  // draw.mm: per-context caches, from ~Context

// Extensions advertised only on GPUs that can do them (api_get.mm): GL_EXTENSIONS names, and
// the GLSL #extension names the shader compiler accepts on this device (CompileOptions).
std::vector<const char*> mglDeviceExtensions(const DeviceCaps& caps);
const std::vector<std::string>& mglDeviceGlslExtensions(const DeviceCaps& caps);

// Common entry-point prologue.
// The entry point's name is remembered for GL errors (log lines and KHR_debug messages): that of the outermost
// MGL_CTX on the stack, so an entry point that calls others reports its own name (plain wrappers without
// MGL_CTX, such as glDrawElements, report the entry point implementing them).
extern thread_local const char* t_apiName;
// METALGLDRV_TRACE_CALLS=1: per-entry-point call counts logged every 5 s from a background thread
// (stats.mm), also when the app makes no frames. METALGLDRV_HITCH_MS: time spent in each call, reported for
// slow frames (hitchFrameEnd). One branch per call when both are off.
extern bool g_traceCalls;
uint64_t traceCall(const char* name);  // returns the start time when hitches are timed, else 0
void traceCallEnd(const char* name, uint64_t start);
struct ApiName {
    const char* saved;
    uint64_t start = 0;
    explicit ApiName(const char* name) : saved(t_apiName) {
        if (!*saved) {
            t_apiName = name;
            if (g_traceCalls) start = traceCall(name);
        }
    }
    ~ApiName() {
        if (start) traceCallEnd(t_apiName, start);
        t_apiName = saved;
    }
};
#define MGL_CTX_RET(r)                                                         \
    ::mgl::ApiName mgl_apiName_(__func__);                                 \
    ::mgl::Context* ctx = ::mgl::currentContext();                         \
    if (!ctx) return r
#define MGL_CTX() MGL_CTX_RET()

}  // namespace mgl
