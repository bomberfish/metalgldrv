// GL object model.
#pragma once
#import <Metal/Metal.h>
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "glapi.h"
#include "glsl/glsl.h"
#include "gl/arb_program.h"
#include "gl/gltypes.h"
#include "mtl/device.h"
#include "mtl/formats.h"

namespace mgl {

class Context;

// ---------------------------------------------------------------------------
// Intrusive refcounting
// ---------------------------------------------------------------------------
class RefCounted {
public:
    virtual ~RefCounted() = default;
    void ref() { refs_.fetch_add(1, std::memory_order_relaxed); }
    void unref() {
        if (refs_.fetch_sub(1, std::memory_order_acq_rel) == 1) delete this;
    }

private:
    std::atomic<int> refs_{0};
};

template <typename T>
class Ref {
public:
    Ref() = default;
    Ref(T* p) : p_(p) { if (p_) p_->ref(); }
    Ref(const Ref& o) : p_(o.p_) { if (p_) p_->ref(); }
    Ref(Ref&& o) noexcept : p_(o.p_) { o.p_ = nullptr; }
    template <typename U>
    Ref(const Ref<U>& o) : p_(o.get()) { if (p_) p_->ref(); }
    ~Ref() { if (p_) p_->unref(); }
    Ref& operator=(const Ref& o) { Ref(o).swap(*this); return *this; }
    Ref& operator=(Ref&& o) noexcept { Ref(std::move(o)).swap(*this); return *this; }
    Ref& operator=(T* p) { Ref(p).swap(*this); return *this; }
    void swap(Ref& o) noexcept { std::swap(p_, o.p_); }
    T* get() const { return p_; }
    T* operator->() const { return p_; }
    T& operator*() const { return *p_; }
    explicit operator bool() const { return p_ != nullptr; }
    bool operator==(const T* p) const { return p_ == p; }
    bool operator!=(const T* p) const { return p_ != p; }
    bool operator==(const Ref& o) const { return p_ == o.p_; }
    bool operator!=(const Ref& o) const { return p_ != o.p_; }

private:
    T* p_ = nullptr;
};

// Base for named GL objects.
class Object : public RefCounted {
public:
    GLuint name = 0;
    std::string label;      // KHR_debug object label
    bool deleted = false;   // glDelete* called while still bound/attached
};

// ---------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------
class Buffer : public Object {
public:
    ~Buffer() override;
    id<MTLBuffer> mtl = nil;     // current storage (may be renamed on orphan/busy writes)
    uint8_t* cpu = nullptr;      // contents of mtl (shared storage)
    GLsizeiptr size = 0;
    GLenum usage = GL_STATIC_DRAW;
    GLbitfield storageFlags = 0;  // glBufferStorage flags (0 = mutable)
    bool immutable = false;
    // GL_APPLE_flush_buffer_range (glBufferParameteriAPPLE), for glMapBuffer: when off, mapping doesn't wait
    // for the GPU (serializedModify) / only ranges flushed with glFlushMappedBufferRangeAPPLE are written
    // (flushingUnmap).
    bool appleSerializedModify = true, appleFlushingUnmap = true;
    GLenum applePurgeable = 0;  // GL_APPLE_object_purgeable: VOLATILE_APPLE / RELEASED_APPLE while purgeable
    uint64_t lastUse = 0;         // serial of last command buffer referencing mtl
    uint64_t lastGpuWrite = 0;    // serial of last GPU write (SSBO/xfb/copy) for read-back sync
    // mapping
    bool mapped = false;
    GLbitfield mapAccess = 0;
    GLintptr mapOffset = 0;
    GLsizeiptr mapLength = 0;
    void* mapPointer = nullptr;
    uint8_t* mapShadow = nullptr;   // staging for non-persistent writes to busy buffers
    // texture buffer views referencing this buffer get invalidated on rename
    uint32_t generation = 0;
    uint64_t contentVersion = 0;  // bumped on every CPU/GPU write path (derived-data caches)
    // Flat-shading provoking-vertex emulation: the whole index buffer with every
    // triangle/line rotated (draw.mm, prepareRotatedIndices).
    id<MTLBuffer> rotated = nil;
    uint64_t rotatedVersion = 0, rotatedGpuWrite = 0, rotatedLastUse = 0;
    uint32_t rotatedLayout = 0;  // index size | vertices per primitive | phase
    // ARB_sparse_buffer (sparse.mm): `mtl` is a private placement sparse buffer (cpu == nullptr);
    // committed pages own a device tile each.
    bool sparse = false;
    std::vector<SparseTiles> sparsePages;
    double sparseUnmapTime = 0;  // CFAbsoluteTime of the last decommit

    // mtl.gpuAddress, cached: draws read it for every bound buffer, and an Objective-C call per read is slow under
    // Rosetta. The cache holds the buffer it belongs to, so a replaced `mtl` can't be mistaken for it.
    uint64_t gpuAddress() const {
        if (mtl != gpuAddressOf_) {
            gpuAddressOf_ = mtl;
            gpuAddress_ = mtl ? mtl.gpuAddress : 0;
        }
        return gpuAddress_;
    }
    mutable id<MTLBuffer> gpuAddressOf_ = nil;
    mutable uint64_t gpuAddress_ = 0;
};

// ---------------------------------------------------------------------------
// Samplers
// ---------------------------------------------------------------------------
struct SamplerState {
    GLenum minFilter = GL_NEAREST_MIPMAP_LINEAR, magFilter = GL_LINEAR;
    GLenum wrapS = GL_REPEAT, wrapT = GL_REPEAT, wrapR = GL_REPEAT;
    float minLod = -1000, maxLod = 1000, lodBias = 0;
    GLenum compareMode = GL_NONE, compareFunc = GL_LEQUAL;
    GLenum reductionMode = GL_WEIGHTED_AVERAGE_ARB;  // ARB_texture_filter_minmax
    float maxAnisotropy = 1;
    float borderColor[4] = {0, 0, 0, 0};
    bool borderIsInt = false;
    bool seamless = true;
    bool operator==(const SamplerState& o) const;
};

class Sampler : public Object {
public:
    SamplerState state;
};

// ---------------------------------------------------------------------------
// Textures
// ---------------------------------------------------------------------------
struct TexImageLevel {
    int width = 0, height = 0, depth = 0;  // depth = layers for arrays, 6 for cube faces stored per face
    GLenum internalFormat = 0;            // as specified by the app (for queries)
    const FormatInfo* fmt = nullptr;
    bool defined() const { return fmt != nullptr; }
};

class Texture : public Object {
public:
    ~Texture() override;
    GLenum target = 0;
    // Per-level image specification (faces of cube maps share the per-level
    // description; cube faces are slices 0..5 in Metal).
    static constexpr int kMaxLevels = 16;
    TexImageLevel levels[6][kMaxLevels];  // [face][level]
    bool immutable = false;
    int immutableLevels = 0;
    int samples = 0;
    bool fixedSampleLocations = true;

    // Bumped whenever a level's specification (levels[][]) changes: mipmapComplete remembers its last answer
    uint32_t specVersion = 1;
    struct MipCompleteMemo {
        uint32_t version = 0;
        int base = -1, maxLevel = -1, last = 0;
        bool complete = false;
    } mipMemo;

    // Texture parameters
    SamplerState sampler;
    int baseLevel = 0, maxLevel = 1000;
    GLenum swizzle[4] = {GL_RED, GL_GREEN, GL_BLUE, GL_ALPHA};
    GLenum depthStencilMode = GL_DEPTH_COMPONENT;
    GLenum srgbDecode = GL_DECODE_EXT;
    bool generateMipmapHint = false;  // GL_GENERATE_MIPMAP (compat)

    // Texture views (ARB_texture_view)
    Ref<Texture> viewParent;
    int viewMinLevel = 0, viewNumLevels = 0, viewMinLayer = 0, viewNumLayers = 0;
    int viewImmutableLevels = 0;  // TEXTURE_IMMUTABLE_LEVELS inherited from the original texture

    // Buffer textures
    Ref<Buffer> buffer;
    GLenum bufferFormat = 0;
    GLintptr bufferOffset = 0;
    GLsizeiptr bufferSize = -1;
    uint32_t bufferGeneration = 0;
    // GL_RGB32* buffer textures read an RGBA32 copy (Metal has no 96-bit format),
    // refreshed when the source buffer's contents may have changed.
    id<MTLBuffer> bufferExpanded = nil;
    uint64_t expandedVersion = ~0ull, expandedGpuWrite = ~0ull;

    // Metal storage. Allocated lazily when the texture is first used
    // (images may be specified level by level).
    id<MTLTexture> mtl = nil;
    const FormatInfo* storageFmt = nullptr;
    int storageLevels = 0, storageW = 0, storageH = 0, storageD = 0, storageLayers = 0;
    bool storageDirty = true;  // level specs changed -> (re)allocate and migrate
    bool needsShaderWrite = false;  // bound as an image -> storage needs ShaderWrite usage
    GLenum applePurgeable = 0;      // GL_APPLE_object_purgeable: VOLATILE_APPLE / RELEASED_APPLE while purgeable
    // GL_APPLE_texture_range: where the app would like the storage (a hint; Metal storage stays private) and the
    // client memory range it set aside for it.
    GLenum appleStorageHint = GL_STORAGE_PRIVATE_APPLE;
    GLsizei appleRangeLength = 0;
    void* appleRangePointer = nullptr;
    // Data was uploaded or copied in, which GL keeps bit-exact: storage of the formats lossless compression
    // alters (losslessCompressionSafe) opts out of it from then on (requireBitExact). Render targets never
    // uploaded to keep it (SuperTuxKart's R11F_G11F_B10F light buffers).
    bool bitExact = false;
    bool needsFormatView = false;   // reinterpreted by a texture view
    // CGLTexImageIOSurface2D: `mtl` is this IOSurface's memory (an IOSurfaceRef, retained; plane `iosurfacePlane`), shared
    // with whoever else uses the surface. It is never reallocated behind the app's back (it has every usage the
    // surface allows from the start); respecifying the texture (glTexImage*, glTexStorage*) detaches it.
    void* iosurface = nullptr;
    uint32_t iosurfacePlane = 0;
    uint64_t lastUse = 0;
    uint64_t lastGpuWrite = 0;

    // ARB_sparse_texture (sparse.mm). The parameters are settable until the storage is immutable;
    // sparse storage is a placement sparse texture allocated once with every usage it may need.
    bool sparse = false;       // TEXTURE_SPARSE_ARB
    int pageSizeIndex = 0;     // VIRTUAL_PAGE_SIZE_INDEX_ARB
    int sparseLevels = 0;      // NUM_SPARSE_LEVELS_ARB: levels before the Metal mip tail
    std::unordered_map<uint64_t, SparseTiles> sparseTiles;  // committed Metal tiles (sparseTileKey)
    // Read-only alias of `mtl` (same tiles) used for sampling: sparse residency reported for
    // shader-writable textures with non-square tiles is wrong (Metal), so lookups use this one.
    id<MTLTexture> sparseRead = nil;
    // Sparse multisample textures are stored as a 2D array with `samples` slices per layer (Metal
    // shaders can't write multisample textures); they can't be framebuffer attachments.
    bool msArray = false;
    double sparseUnmapTime = 0;  // CFAbsoluteTime of the last decommit

    // Properties of the Metal texture sampled last (textureSamplingView), cached like Buffer::gpuAddress().
    struct MtlProps {
        id<MTLTexture> of = nil;
        MTLPixelFormat pixelFormat = MTLPixelFormatInvalid;
        MTLTextureType type = MTLTextureType2D;
        NSUInteger levels = 0, arrayLength = 0;
    };
    MtlProps mtlProps;
    const MtlProps& propsOf(__unsafe_unretained id<MTLTexture> t) {
        if (mtlProps.of != t) mtlProps = {t, t.pixelFormat, t.textureType, t.mipmapLevelCount, t.arrayLength};
        return mtlProps;
    }
    // gpuResourceID of the view sampled last
    id<MTLTexture> resourceIDOf = nil;
    MTLResourceID resourceID = {};
    MTLResourceID resourceIDFor(__unsafe_unretained id<MTLTexture> v) {
        if (resourceIDOf != v) {
            resourceIDOf = v;
            resourceID = v.gpuResourceID;
        }
        return resourceID;
    }

    // Cached sampling views keyed by (base,max level, swizzle, format, ds mode)
    struct ViewKey {
        uint64_t a, b;
        bool operator==(const ViewKey& o) const { return a == o.a && b == o.b; }
    };
    struct ViewKeyHash {
        size_t operator()(const ViewKey& k) const { return std::hash<uint64_t>()(k.a * 31 + k.b); }
    };
    std::unordered_map<ViewKey, id<MTLTexture>, ViewKeyHash> views;
    // Feedback loops (sampled while attached to the draw framebuffer): sampling reads a copy taken
    // before the render pass, see feedbackSnapshot() in draw.mm.
    id<MTLTexture> fbSnapshot = nil;
    id<MTLTexture> fbSnapshotSrc = nil;
    uint64_t fbSnapshotPass = 0;
    std::unordered_map<ViewKey, id<MTLTexture>, ViewKeyHash> fbSnapshotViews;
    // Multisample textures bound as images (METALGLDRV_STRICT): per-sample 2D-array shadow, see
    // msImageShadow() in draw.mm.
    id<MTLTexture> msShadow = nil;

    // Drops cached views, keeping them alive until in-flight GPU work that may
    // reference them (by MTLResourceID) has completed.
    void dropViews();

    const TexImageLevel& level(int l, int face = 0) const { return levels[face][l]; }
    const TexImageLevel& baseImage() const { return levels[0][std::min(std::max(baseLevel, 0), kMaxLevels - 1)]; }
    bool isCube() const { return target == GL_TEXTURE_CUBE_MAP || target == GL_TEXTURE_CUBE_MAP_ARRAY; }
    int faces() const { return target == GL_TEXTURE_CUBE_MAP ? 6 : 1; }
};

// ---------------------------------------------------------------------------
// Renderbuffers
// ---------------------------------------------------------------------------
class Renderbuffer : public Object {
public:
    ~Renderbuffer() override;
    GLenum internalFormat = GL_RGBA4;
    const FormatInfo* fmt = nullptr;
    int width = 0, height = 0, samples = 0;
    id<MTLTexture> mtl = nil;
    id<MTLTexture> linearView = nil;  // non-sRGB view for rendering with FRAMEBUFFER_SRGB disabled
    uint64_t lastUse = 0;
};

// ---------------------------------------------------------------------------
// Framebuffers
// ---------------------------------------------------------------------------
struct Attachment {
    GLenum type = GL_NONE;  // GL_NONE, GL_TEXTURE, GL_RENDERBUFFER, GL_FRAMEBUFFER_DEFAULT
    Ref<Texture> tex;
    Ref<Renderbuffer> rb;
    int level = 0;
    int layer = 0;          // layer / cube face (cube: face index), -1 = layered
    bool layered = false;
    bool operator==(const Attachment& o) const {
        return type == o.type && tex.get() == o.tex.get() && rb.get() == o.rb.get() && level == o.level && layer == o.layer &&
               layered == o.layered;
    }
};

enum { kMaxDrawBuffers = 8, kAttDepth = 8, kAttStencil = 9, kNumAttachments = 10 };

class Framebuffer : public Object {
public:
    Attachment att[kNumAttachments];  // color 0..7, depth, stencil
    GLenum drawBuffers[kMaxDrawBuffers] = {GL_COLOR_ATTACHMENT0, GL_NONE, GL_NONE, GL_NONE, GL_NONE, GL_NONE, GL_NONE, GL_NONE};
    GLenum readBuffer = GL_COLOR_ATTACHMENT0;
    // default framebuffer parameters (ARB_framebuffer_no_attachments)
    int defaultWidth = 0, defaultHeight = 0, defaultLayers = 0, defaultSamples = 0;
    bool defaultFixedSampleLocations = false;
    uint32_t version = 1;  // bumped on any attachment change
    bool isDefault = false;
    // its last pass drew tessellation: the next one starts in a tessellation segment (Context::TessSegment), so a
    // tessellated draw after other draws doesn't end a Metal 4 pass (storing and reloading every attachment)
    bool tessHint = false;
    // framebufferStatus's last answer and what it depended on (every draw asks)
    struct StatusMemo {
        struct Att {
            const void* obj = nullptr;
            const void* fmt = nullptr;
            int x = 0, y = 0, z = 0;
            bool operator==(const Att& o) const { return obj == o.obj && fmt == o.fmt && x == o.x && y == o.y && z == o.z; }
        };
        bool valid = false;
        uint32_t version = 0;
        Att att[kNumAttachments];
        GLenum status = 0;
    } statusMemo;
    // EXT_fragment_shading_rate_attachment (METALGLDRV_STRICT): an R8UI texture, layer 0 of it (the
    // layer limit is 1), each texel covering 2^shadingRateTexelLog2 pixels.
    Ref<Texture> shadingRateTex;
    int shadingRateLayers = 0;
    int shadingRateTexelLog2[2] = {0, 0};
    class Surface* surface = nullptr;  // default framebuffer only
};

// ---------------------------------------------------------------------------
// Vertex arrays
// ---------------------------------------------------------------------------
struct VertexAttrib {
    bool enabled = false;
    GLint size = 4;
    GLenum type = GL_FLOAT;
    bool normalized = false, integer = false, isLong = false;
    GLuint relativeOffset = 0;
    GLuint binding = 0;
    const void* pointer = nullptr;  // VERTEX_ATTRIB_ARRAY_POINTER: as given to *Pointer
    GLsizei stride = 0;  // VERTEX_ATTRIB_ARRAY_STRIDE: as given to *Pointer (the binding holds the effective stride)
    const void* clientPointer = nullptr;  // compat: client-side arrays (no buffer bound)
};

struct VertexBinding {
    Ref<Buffer> buffer;
    GLintptr offset = 0;
    GLsizei stride = 16;
    GLuint divisor = 0;
};

enum { kMaxVertexAttribs = 16, kMaxVertexBindings = 16 };

class VertexArray : public Object {
public:
    VertexAttrib attribs[kMaxVertexAttribs];
    VertexBinding bindings[kMaxVertexBindings];
    Ref<Buffer> elementBuffer;
    uint32_t version = 1;
};

// ---------------------------------------------------------------------------
// Shaders / programs
// ---------------------------------------------------------------------------
class ShaderObj : public Object {
public:
    GLenum type = 0;
    std::vector<std::string> sources;
    bool compiled = false;
    std::string infoLog;
    std::shared_ptr<const glsl::Shader> shader;
    int attachCount = 0;  // programs this shader is attached to (deletion is deferred until 0)
    // ARB_gl_spirv: module from glShaderBinary (SPIR_V_BINARY state); compiled once specialized.
    bool spirvBinary = false;
    bool specialized = false;
    // A GLSL 1.20-style geometry shader of ARB/EXT_geometry_shader4 (no input layout): compiled again at link
    // time with the program's GEOMETRY_*_ARB parameters (api_shader.mm, gs4Source).
    bool gs4Legacy = false;
    std::shared_ptr<const glsl::SpirvModule> spirvModule;
};

struct PipelineCacheEntry;
struct ProgramVariant;

class ProgramObj : public Object {
public:
    std::vector<Ref<ShaderObj>> attached;
    std::map<std::string, int> attribBindings, fragDataBindings, fragDataIndices;
    std::vector<std::string> xfbVaryings;
    GLenum xfbBufferMode = GL_INTERLEAVED_ATTRIBS;
    bool separable = false, retrievableHint = false;
    // ARB/EXT_geometry_shader4 program parameters for legacy geometry shaders (ShaderObj::gs4Legacy)
    GLenum gs4InputType = GL_TRIANGLES, gs4OutputType = GL_TRIANGLE_STRIP;
    int gs4VerticesOut = 0;
    bool linkedSeparable = false;  // PROGRAM_SEPARABLE as of the last successful link (what queries report)
    bool linkedSpirv = false;      // linked from SPIR-V shaders: reflection carries no names

    // link results
    bool linked = false, validated = false;
    uint32_t linkGeneration = 0;
    std::string infoLog;
    std::shared_ptr<const glsl::LinkedProgram> program;
    glsl::Reflection refl;
    // refl.attributes' shapes and the locations holding a dvec3 / dvec4, for `program` (setupVertexInput, every draw)
    struct AttribShapes {
        std::shared_ptr<const glsl::LinkedProgram> of;
        std::vector<TypeShape> shapes;
        uint32_t wideMask = 0;
        bool wideAt15 = false;  // one reaches location 15 (wantManualFetch)
    } attribShapes;

    // default uniform block storage (MSL layout as reported by reflection)
    std::vector<uint8_t> uniformData;
    uint32_t uniformVersion = 1;
    // location -> (uniform index, array element)
    struct Loc { int uniform; int element; };
    std::vector<Loc> locations;
    // current values of sampler/image uniforms (texture unit per element)
    std::vector<int> samplerUnits;  // indexed by slot
    int8_t fragmentWritesMemory = -1;  // the fragment stage uses images, storage blocks or atomic counters (-1: not known yet)
    std::vector<int> blockBindings; // uniform block index -> binding point
    std::vector<int> storageBindings;

    // compiled pipeline variants
    std::unordered_map<uint64_t, std::shared_ptr<ProgramVariant>> variants;

    // Program pipeline composite (separable programs per stage feeding a multi-pass pipeline):
    // a draw-only program whose variants are generated from the stage programs, each stage
    // reading its own program's argument buffer.
    bool composite = false;
    Ref<ProgramObj> stagePrograms[(int)glsl::Stage::Count];
    std::shared_ptr<const glsl::LinkedProgram> stageLinked[(int)glsl::Stage::Count];
};

class ProgramPipelineObj : public Object {
public:
    Ref<ProgramObj> stages[(int)glsl::Stage::Count];
    Ref<ProgramObj> composite;          // see ProgramObj::composite
    std::vector<uint64_t> compositeKey;
    Ref<ProgramObj> activeProgram;
    bool validated = false;
    std::string infoLog;
};

// ---------------------------------------------------------------------------
// Queries / sync / transform feedback
// ---------------------------------------------------------------------------
class Query : public Object {
public:
    GLenum target = 0;
    bool active = false;
    bool everActive = false;
    uint64_t result = 0;
    bool resultAvailable = true;
    uint64_t serial = 0;       // command buffer whose completion makes result available
    uint64_t beginSerial = 0;  // timer queries: first command buffer measured
    uint32_t index = 0;        // indexed queries (stream / xfb buffer)
    id<MTLBuffer> resultBuffer = nil;  // occlusion: visibility result buffer
    uint32_t resultOffset = 0;
    uint64_t cpuCount = 0;     // primitive queries counted on the CPU
    // pipeline statistics counted on the GPU: uint counters (buffer, byte offset) summed at resolve
    std::vector<std::pair<id<MTLBuffer>, uint32_t>> gpuCounts;
};

class SyncObj : public RefCounted {
public:
    uint64_t serial = 0;
    bool signaled() const;
};

class TransformFeedbackObj : public Object {
public:
    struct BufBinding {
        Ref<Buffer> buffer;
        GLintptr offset = 0;
        GLsizeiptr size = 0;
    };
    BufBinding buffers[4];
    bool active = false, paused = false;
    GLenum primitiveMode = 0;
    Ref<ProgramObj> program;
    uint32_t written[4] = {};  // vertices captured per stream in the current session
    uint32_t drawCount[4] = {};  // vertices captured per stream by the last completed session (DrawTransformFeedback)
    bool ended = false;          // EndTransformFeedback was called at least once
    uint64_t verticesWritten = 0;
};

// ---------------------------------------------------------------------------
// Name spaces
// ---------------------------------------------------------------------------
template <typename T>
class NameSpace {
public:
    GLuint genName() {
        while (objects_.count(next_) || reserved_.count(next_)) next_++;
        reserved_.insert({next_, true});
        return next_++;
    }
    T* get(GLuint name) const {
        auto it = objects_.find(name);
        return it == objects_.end() ? nullptr : it->second.get();
    }
    bool isName(GLuint name) const { return objects_.count(name) || reserved_.count(name); }
    bool isReservedOnly(GLuint name) const { return reserved_.count(name) && !objects_.count(name); }
    T* create(GLuint name, T* obj) {
        obj->name = name;
        reserved_.erase(name);
        objects_[name] = Ref<T>(obj);
        return obj;
    }
    void reserve(GLuint name) {
        if (!objects_.count(name)) reserved_[name] = true;
    }
    Ref<T> remove(GLuint name) {
        reserved_.erase(name);
        auto it = objects_.find(name);
        if (it == objects_.end()) return Ref<T>();
        Ref<T> r = it->second;
        objects_.erase(it);
        if (r) r->deleted = true;
        return r;
    }
    template <typename F>
    void forEach(F f) {
        for (auto& [n, o] : objects_) f(o.get());
    }

private:
    std::unordered_map<GLuint, Ref<T>> objects_;
    std::unordered_map<GLuint, bool> reserved_;
    GLuint next_ = 1;
};

// GL_ARB_vertex_program / GL_ARB_fragment_program program objects (api_arb_program.mm): the source, its GLSL
// translation (arb_program.h) and the local parameters.
class ArbProgram : public Object {
public:
    GLenum target = 0;  // set by the first bind
    std::string source;
    ArbTranslation tr;  // tr.ok: a valid program has been loaded
    float local[kArbMaxLocalParams][4] = {};
    uint64_t version = 0;  // bumped by glProgramStringARB (the linked programs are cached by it)
};

// Objects shared between contexts.
class ShareGroup : public RefCounted {
public:
    std::recursive_mutex mutex;
    NameSpace<Buffer> buffers;
    NameSpace<Texture> textures;
    NameSpace<Renderbuffer> renderbuffers;
    NameSpace<Sampler> samplers;
    NameSpace<Object> shaderPrograms;  // ShaderObj or ProgramObj share one namespace
    std::unordered_map<SyncObj*, Ref<SyncObj>> syncs;
    // Framebuffers, when the app profile shares them (AppProfile::shareFramebuffers); per context otherwise.
    NameSpace<Framebuffer> framebuffers;
    // GL_APPLE_container_object_shareable (contexts made through CGL / NSOpenGL): framebuffers, vertex arrays
    // and program pipelines belong to the share group, as on Apple's OpenGL, instead of one context. Uses of
    // these names then hold `mutex` (Context::containerLock).
    bool shareContainers = false;
    NameSpace<VertexArray> vaos;
    NameSpace<ProgramPipelineObj> pipelines;
    NameSpace<ArbProgram> arbPrograms;  // ARB_vertex_program / ARB_fragment_program
    // Display lists (compat) live here too in the future.
    ShaderObj* shader(GLuint n) { return dynamic_cast<ShaderObj*>(shaderPrograms.get(n)); }
    ProgramObj* program(GLuint n) { return dynamic_cast<ProgramObj*>(shaderPrograms.get(n)); }
};

}  // namespace mgl
