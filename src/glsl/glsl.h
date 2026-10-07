// metalgldrv GLSL compiler: public interface used by the GL runtime.
//
// The compiler is split in two phases mirroring GL:
//   compile()  - glCompileShader: preprocess, parse, type-check one shader object.
//   link()     - glLinkProgram: cross-stage validation, interface matching,
//                resource layout assignment and reflection.
// MSL is generated per *pipeline variant* (generateMSL), because some stage
// combinations (GS -> mesh, tessellation -> compute + mesh) and some fixed
// function state (vertex fetch formats for the non-stage_in paths) change the
// generated code.
//
// Coordinate conventions (shared with the runtime, see docs/ARCHITECTURE.md):
//   * Every texture/renderbuffer stores GL row 0 in Metal row 0. Vertex-ish
//     stages emit position.y = -y so GL window y == Metal raster row; the
//     runtime flips once when presenting the default framebuffer.
//   * Clip-space z is remapped to [0,1] by the shader using MGLSys::depthMode.
//   * Front-face winding is inverted by the runtime to compensate for the flip.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace glsl {

enum class Stage : uint8_t { Vertex, TessControl, TessEval, Geometry, Fragment, Compute, Count };

// ---------------------------------------------------------------------------
// Shader objects
// ---------------------------------------------------------------------------
struct Shader;  // opaque, immutable after compile

struct CompileResult {
    std::shared_ptr<const Shader> shader;  // null on failure
    std::string infoLog;
};

// Implementation limits the compiler enforces (must match the runtime's GL_MAX_* queries, src/gl/context.h).
constexpr int kMaxCombinedTextureImageUnits = 96;
constexpr int kMaxImageUnits = 8;
constexpr int kMaxUniformBufferBindings = 84;
constexpr int kMaxShaderStorageBufferBindings = 16;
constexpr int kMaxAtomicCounterBufferBindings = 8;

struct CompileOptions {
    bool compatibilityProfile = true;  // accept GL compatibility builtins (gl_Vertex, ftransform, ...)
    int maxVersion = 460;
};

CompileResult compile(Stage stage, const std::vector<std::string>& sources, const CompileOptions& opts = {});

// ---------------------------------------------------------------------------
// Program reflection. All "type" fields are GL enums (GL_FLOAT_VEC4, GL_SAMPLER_2D...).
// ---------------------------------------------------------------------------

// Each program owns an argument buffer made of 8-byte slots, laid out by the
// linker. Slot contents are written by the runtime before every draw/dispatch:
//   Ubo/Ssbo/AtomicCounter/DefaultUniforms : GPU address (uint64)
//   Texture/Image                          : MTLResourceID of the bound view
//   Sampler                                : MTLResourceID of the sampler state
//   BufferSize                             : uint64 byte size of the bound SSBO range
//                                            (for runtime-sized arrays .length())
enum class SlotKind : uint8_t { DefaultUniforms, SystemValues, Ubo, Ssbo, AtomicCounter, Texture, Sampler, Image, BufferSize, XfbBuffer,
                                SubroutineIndices /* index = Stage: GPU address of uint[numLocations] selected subroutine indices */ };

struct Slot {
    SlotKind kind;
    uint32_t index;  // UBO/SSBO: block index; Texture/Sampler/Image: uniform index (+array element in `element`)
    uint32_t element = 0;
    uint32_t textureTarget = 0;  // for textures: GL_TEXTURE_2D etc expected by the sampler type
};

struct Attribute {  // active vertex shader inputs
    std::string name;
    uint32_t type;       // GL_FLOAT_VEC4 ...
    int arraySize;       // 1 for non-arrays
    int location;        // assigned (explicit, glBindAttribLocation, or by linker)
    int locationsUsed;   // matrices/arrays/dvec use several
};

struct FragOutput {
    std::string name;
    uint32_t type;
    int arraySize;
    int location;
    int index;  // dual source blend index
};

struct Uniform {  // GL "active uniform" (default block or block member)
    std::string name;       // full name, arrays get "[0]"
    uint32_t type;
    int arraySize;          // 1 for non-arrays
    int blockIndex = -1;    // -1: default block
    int offset = -1;        // byte offset in its block (default block uses MSL layout)
    int arrayStride = 0;
    int matrixStride = 0;
    bool rowMajor = false;
    int location = -1;      // default block only; arrays occupy [location, location+arraySize)
    int slot = -1;          // sampler/image uniforms: first argument slot (array elements follow)
    int binding = 0;        // initial texture unit / image unit / atomic counter binding
    int atomicCounterBufferIndex = -1;
    uint32_t stageMask = 0; // 1 << Stage referencing it
};

struct Block {  // uniform block or shader storage block
    std::string name;       // block name (arrays: "Name[0]" entries are separate blocks)
    int binding = 0;        // initial binding from layout(binding=)
    uint32_t dataSize = 0;  // GL_BUFFER_DATA_SIZE (std140/std430/shared)
    std::vector<int> activeUniforms;  // indices into LinkedProgram::uniforms (UBO) or bufferVariables (SSBO)
    int slot = -1;          // argument slot holding the buffer address
    int sizeSlot = -1;      // SSBO: slot receiving bound range size (or -1)
    uint32_t stageMask = 0;
};

struct BufferVariable {  // GL_BUFFER_VARIABLE program interface
    std::string name;
    uint32_t type;
    int arraySize, blockIndex, offset, arrayStride, matrixStride;
    bool rowMajor;
    int topLevelArraySize, topLevelArrayStride;
};

struct AtomicCounterBuffer {
    int binding;
    uint32_t dataSize;
    std::vector<int> activeUniforms;
    int slot;
    uint32_t stageMask;
};

struct XfbVarying {
    std::string name;
    uint32_t type;
    int arraySize;
    int buffer, offset;
};

struct SubroutineInfo;  // TODO: ARB_shader_subroutine

struct LinkInput {
    std::vector<std::shared_ptr<const Shader>> shaders;
    std::map<std::string, int> attribBindings;    // glBindAttribLocation
    std::map<std::string, int> fragDataBindings;  // glBindFragDataLocation(Indexed)
    std::map<std::string, int> fragDataIndices;
    std::vector<std::string> xfbVaryings;         // glTransformFeedbackVaryings
    bool xfbInterleaved = true;
    bool separable = false;
};

struct LinkedProgram;  // opaque compiled program state (holds the per-stage IR)

// Additional reflection for multi-pass pipelines (filled by the linker).
struct MultiPassInfo {
    uint32_t tcsVertexOutBytes = 0;   // bytes per control point record in MGLDrawInfo::tcsVertexOut
    uint32_t tcsPatchOutBytes = 0;    // bytes per patch record in MGLDrawInfo::tcsPatchOut (levels at 0: outer[4], inner[2])
    uint32_t tcsOutputVertices = 0;   // control points per output patch (patch input size when there is no TCS: 0 = use patchVertices)
    uint32_t xfbStride[4] = {0, 0, 0, 0};
    uint32_t xfbBufferMask = 0;       // buffers written by transform feedback
    uint32_t xfbBufferStream[4] = {0, 0, 0, 0};  // vertex stream captured into each buffer
    uint32_t tesVertexOutBytes = 0;   // bytes per TES output record (tessellation feeding a GS or transform feedback)
};

// ARB_shader_subroutine (per stage)
struct SubroutineUniform {
    std::string name;
    int location = -1;               // GL subroutine uniform location (per stage); arrays occupy [location, location+arraySize)
    int arraySize = 1;               // total elements (arrays of arrays are flattened)
    std::vector<int> dims;           // array dimensions, outermost first (empty: not an array)
    std::vector<int> compatible;     // indices into StageSubroutines::functions
};
struct StageSubroutines {
    std::vector<std::string> functions;      // GL subroutine index = position
    std::vector<SubroutineUniform> uniforms;
    int numLocations = 0;                    // GL_ACTIVE_SUBROUTINE_UNIFORM_LOCATIONS
    int slot = -1;                           // argument slot holding a GPU address of `uint index[numLocations]`
};

struct Reflection {
    uint32_t stageMask = 0;
    std::vector<Attribute> attributes;
    std::vector<FragOutput> fragOutputs;
    std::vector<Uniform> uniforms;
    std::vector<Block> uniformBlocks;
    std::vector<Block> storageBlocks;
    std::vector<BufferVariable> bufferVariables;
    std::vector<AtomicCounterBuffer> atomicCounterBuffers;
    std::vector<XfbVarying> xfbVaryings;
    std::vector<Slot> slots;              // argument buffer layout (slot i at byte offset 8*i)
    uint32_t defaultUniformSize = 0;      // bytes of the default uniform block (MSL layout)
    int localSize[3] = {1, 1, 1};         // compute
    // geometry
    uint32_t gsInputPrimitive = 0, gsOutputPrimitive = 0;  // GL_TRIANGLES...
    int gsMaxVertices = 0, gsInvocations = 1;
    // tessellation
    int tcsOutputVertices = 0;
    uint32_t tesPrimitiveMode = 0, tesSpacing = 0, tesOrder = 0;
    bool tesPointMode = false;
    bool usesFragDepth = false, earlyFragmentTests = false, usesDiscard = false;
    bool usesFp64 = false;
    bool usesFlatShading = false;           // any flat-qualified input reaching the fragment shader
    bool writesLayer = false, writesViewportIndex = false;  // by the last pre-rasterization stage
    MultiPassInfo multipass;
    std::vector<uint8_t> defaultUniformInit;  // initial contents of the default uniform block (defaultUniformSize bytes)
    StageSubroutines subroutines[(int)Stage::Count];
};

struct LinkResult {
    std::shared_ptr<const LinkedProgram> program;  // null on failure
    Reflection reflection;
    std::string infoLog;
};

LinkResult link(const LinkInput& input);

// ---------------------------------------------------------------------------
// MSL generation
// ---------------------------------------------------------------------------

// Metal vertex formats are expressed as GL terms; the generator chooses
// stage_in when possible and manual fetch otherwise.
struct VertexFetch {
    bool enabled = false;      // false -> attribute comes from MGLSys current attrib value
    uint32_t glType = 0;       // GL_FLOAT, GL_UNSIGNED_BYTE, GL_INT_2_10_10_10_REV ...
    uint8_t size = 4;          // 1..4 or GL_BGRA (stored as 5)
    bool normalized = false;
    bool integer = false;      // glVertexAttribIPointer
    bool isDouble = false;     // glVertexAttribLPointer
    uint8_t binding = 0;       // vertex buffer binding index
    uint32_t relativeOffset = 0;
};

enum class PipelineKind : uint8_t {
    Classic,        // vertex [[stage_in]] + fragment
    MeshGeometry,   // VS+GS emulated with object+mesh, + fragment
    Tessellation,   // VS+TCS as compute kernel(s), TES(+GS) as object/mesh, + fragment
    Compute,
    XfbCapture,     // vertex-processing stages as compute kernel writing xfb buffers
};

struct VariantKey {
    PipelineKind kind = PipelineKind::Classic;
    VertexFetch attribs[32];   // indexed by location; only used for manual fetch
    uint8_t primitive = 0;     // input GL primitive mode for GS/tess paths
    bool flatshadeLastVertex = true;
    bool rasterDiscard = false;
    // Classic path: locations whose int/uint input is fed integer data of the other
    // signedness (GL allows it, Metal attribute formats don't); declared flipped + cast.
    uint32_t attribSignFlip = 0;
    // Classic path: pull vertex attributes from MGLDrawInfo (buffer 22, formats in
    // `attribs`) instead of [[stage_in]]; used for layouts Metal's vertex fetch rejects.
    bool manualFetch = false;
    // Classic path: the vertex shader also writes transform feedback (MGLDrawInfo at buffer 22,
    // non-indexed draw over the decomposed primitive list); used when it has side effects.
    bool xfbInVertex = false;
    // Fragment: run at sample rate (GL_SAMPLE_SHADING with MIN_SAMPLE_SHADING_VALUE > 0).
    bool sampleShading = false;
    // Fragment: AND the coverage with MGLSys::sampleMask (GL_SAMPLE_MASK) and the
    // GL_SAMPLE_COVERAGE mask derived from MGLSys::sampleCoverage.
    bool sampleMask = false;
};

struct EntryPoint {
    std::string name;  // MSL function name
    Stage glStage;     // stage it mostly represents
    enum Kind : uint8_t { Vertex, Fragment, Kernel, Object, Mesh } kind;
    int threadsPerThreadgroup[3] = {1, 1, 1};
    uint32_t payloadBytes = 0;   // object -> mesh
    uint32_t maxMeshVertices = 0, maxMeshPrimitives = 0;
    uint32_t scratchWordsPerItem = 0;  // mgl_gs_copy: size of each item's MGLDrawInfo::gsRecord area
};

struct MSLOutput {
    std::string source;
    std::vector<EntryPoint> entryPoints;
    std::string error;  // nonempty on failure
};

MSLOutput generateMSL(const LinkedProgram& program, const VariantKey& key);
// Program pipelines: pre-rasterization and fragment stages from different separable programs
// (stages[s] = program providing stage s, or null). Each stage reads its own program's argument
// buffer; MGLArgs at buffer 0 is then a table of 6 GPU addresses indexed by Stage.
MSLOutput generateMSL(const LinkedProgram* const stages[(int)Stage::Count], const VariantKey& key);
MultiPassInfo multipassInfo(const LinkedProgram* const stages[(int)Stage::Count]);

// ---------------------------------------------------------------------------
// Runtime <-> shader ABI. Mirrored verbatim in the MSL prelude.
// ---------------------------------------------------------------------------
// Argument table buffer indices
enum : uint32_t {
    kBufArgs = 0,         // program argument buffer (slots)
    kBufSys = 1,          // MGLSys
    kBufVertex0 = 2,      // vertex buffer bindings 0..15 -> 2..17 (stage_in path)
    kMaxVertexBuffers = 16,
    kBufXfb0 = 18,        // transform feedback buffers 0..3
    kBufStageIO = 22,     // scratch buffers for multi-pass pipelines (GS/tess)
};

struct alignas(16) MGLSys {
    float depthRangeNear, depthRangeFar;   // glDepthRange
    uint32_t depthMode;                    // 0: GL [-1,1] -> [0,1] remap, 1: GL_ZERO_TO_ONE
    uint32_t clipDistanceMask;             // glEnable(GL_CLIP_DISTANCEi)
    float viewport[4];                     // x, y, w, h of viewport 0
    float pointSize;                       // glPointSize (when program doesn't write gl_PointSize)
    uint32_t flags;                        // bit0: program point size, bit1: point sprite origin lower-left, bit2: upper-left clip origin, bit3: invert sample coverage
    int32_t baseVertex, baseInstance;      // gl_BaseVertex / gl_BaseInstance / gl_DrawID support
    int32_t drawID, firstVertex;
    uint32_t sampleMask;
    float alphaRef;                        // compat alpha test
    uint32_t alphaFunc;                    // compat alpha test (GL_ALWAYS -> disabled)
    float minSampleShading;
    float renderTargetHeight;              // height of the bound framebuffer (layout(origin_upper_left) gl_FragCoord)
    float sampleCoverage;                  // GL_SAMPLE_COVERAGE value (1 when disabled); flags bit3: invert
    float tessDefaultOuter[4];             // GL_PATCH_DEFAULT_OUTER_LEVEL (used when there is no TCS)
    float tessDefaultInner[2];             // GL_PATCH_DEFAULT_INNER_LEVEL
    uint32_t pad2[2];
    float currentAttrib[32][4];            // glVertexAttrib* values for disabled arrays
    double currentAttribD[16][4];          // the same as doubles (glVertexAttribL*; dvec inputs)
};

// Per-draw description for pipelines without stage_in (PipelineKind::MeshGeometry,
// Tessellation, XfbCapture), bound at argument-table index kBufStageIO.
// Vertex attribute formats for manual fetch come from VariantKey::attribs.
struct alignas(16) MGLDrawInfo {
    uint64_t vertexBuffer[16];    // GPU address of vertex binding b (binding offset applied)
    uint32_t vertexStride[16];
    uint32_t vertexDivisor[16];   // 0 = per vertex
    uint64_t indexBuffer;         // 0 = non-indexed (vertex id = firstVertex + i)
    uint32_t indexType;           // 0 none, 2 = uint16, 4 = uint32 (u8 is widened by the runtime)
    uint32_t count;               // index/vertex count of the draw
    int32_t baseVertex;
    uint32_t baseInstance;
    uint32_t firstVertex;
    uint32_t instanceCount;
    uint32_t primitiveCount;      // input primitives per instance (after topology decomposition)
    uint32_t primitiveMode;       // GL mode of the draw (lists after runtime expansion of fans/loops)
    uint32_t verticesPerPrimitive;// 1,2,3,4 (lines adj),6 (tris adj), patch size for GL_PATCHES
    uint32_t patchVertices;
    // transform feedback
    uint64_t xfbBuffer[4];        // GPU address (binding offset applied)
    uint32_t xfbStride[4];        // bytes per captured vertex in each buffer
    uint32_t xfbCapacity[4];      // max vertices that fit in each buffer
    uint32_t xfbVertexBase;       // vertices already written by earlier draws in this xfb session
    uint32_t pad0;
    // tessellation scratch (allocated by the runtime, sized from Reflection)
    uint64_t tcsVertexOut;        // patchCount * tcsOutputVertices * tcsVertexOutBytes
    uint64_t tcsPatchOut;         // patchCount * tcsPatchOutBytes (includes tess levels first: outer[4], inner[2] floats)
    uint64_t counters;            // device atomic counters (primitives generated/written)
    // Counted multi-pass work (GS / tessellation output whose size only the GPU knows):
    // a count kernel fills itemCounts, the runtime reads them back, computes prefix sums
    // into itemOffsets and runs the write kernels (see docs/glsl-compiler.md).
    uint64_t itemCounts;          // uint4 per work item (written by mgl_tes_count / mgl_gs_count)
    uint64_t itemOffsets;         // uint4 per work item (first output primitive per stream / TES vertex+primitive)
    uint64_t tesVertexOut;        // TES output records (multipass.tesVertexOutBytes each)
    uint64_t tesPrimOut;          // TES primitive list: vertex indices into tesVertexOut, tesVerticesPerPrimitive each
    uint32_t itemCount;           // work items of the counted pass
    uint32_t tesPrimCount;        // primitives in tesPrimOut
    uint32_t xfbPrimLimit[4];     // per stream: output primitives that fit in the bound buffers (capture stops there)
    uint32_t xfbStreamBase[4];    // per stream: vertices already captured in this session
    uint32_t xfbBufferStream[4];  // stream feeding each buffer
    uint64_t gsRecord;            // optional: mgl_gs_count stores each item's captured primitives here
                                  // (EntryPoint::scratchWordsPerItem uint32 per item); mgl_gs_copy places them
};


}  // namespace glsl
