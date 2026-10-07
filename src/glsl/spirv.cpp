// SPIR-V front end (ARB_gl_spirv, GL 4.6 §7.2).
//
// glShaderBinary stores a module checked by parseSpirv(). glSpecializeShader calls specializeSpirv(),
// which applies the specialization constants and decompiles the selected entry point to GLSL 4.60;
// the regular GLSL compiler, linker and MSL emitter then handle it like any other shader.
//
// Translation scheme:
//  * Structured control flow maps onto GLSL: OpSelectionMerge + OpBranchConditional -> if/else,
//    OpLoopMerge -> for (;;) with the continue construct emitted at every branch to the continue
//    target, OpSwitch -> switch (cases ordered by block order, so fallthrough is preserved).
//  * Every SSA value becomes a variable declared at the top of its function and assigned where the
//    instruction is, so definitions dominate their uses whatever the GLSL scoping. OpPhi values are
//    copied on the incoming edges. Pointers (access chains) and opaque values (samplers, images) stay
//    inline expressions.
//  * Interfaces are keyed by Location/Binding, never by OpName: user inputs/outputs get explicit
//    locations (I/O blocks are flattened into located variables), default-block uniforms are named
//    after their location, uniform/storage blocks and atomic counters after their binding, so
//    separately compiled stages match. Reflection names of SPIR-V programs are empty (api_shader.mm).
//  * GLSL.std.450 extended instructions map onto the GLSL builtins they come from.
// METALGLDRV_DUMP_SPIRV=1 prints the generated GLSL to stderr.
#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <unordered_map>

#include "glsl/ast.h"
#include "glsl/glsl.h"
#include "glsl/spirv_defs.h"

namespace glsl {

struct SpirvModule {
    std::vector<uint32_t> words;  // host byte order
    struct EntryPoint {
        uint32_t model = 0, fn = 0;
        std::string name;
        std::vector<uint32_t> interface;
    };
    std::vector<EntryPoint> entryPoints;
    std::set<uint32_t> specIds;
};

const std::vector<std::string>& supportedSpirvExtensions() {
    static const std::vector<std::string> exts = {"SPV_KHR_shader_draw_parameters", "SPV_KHR_storage_buffer_storage_class",
                                                  "SPV_KHR_no_integer_wrap_decoration"};
    return exts;
}

using namespace spv;

namespace {

std::string literalString(const uint32_t* w, size_t maxWords, size_t* used) {
    std::string s;
    size_t i = 0;
    bool done = false;
    for (; i < maxWords && !done; i++) {
        for (int b = 0; b < 4; b++) {
            char c = (char)((w[i] >> (8 * b)) & 0xff);
            if (!c) { done = true; break; }
            s += c;
        }
    }
    if (used) *used = i;
    return s;
}

struct SpvError {
    std::string msg;
};
[[noreturn]] void fail(const std::string& m) { throw SpvError{m}; }

std::string num(int64_t v) { return std::to_string(v); }

struct Deco {
    int location = -1, component = -1, binding = -1, offset = -1, index = -1, arrayStride = -1, matrixStride = -1;
    int builtin = -1, specId = -1, xfbBuffer = -1, xfbStride = -1, stream = -1;
    bool block = false, bufferBlock = false, rowMajor = false, colMajor = false;
    bool flat = false, noPersp = false, centroid = false, sample = false, patch = false, invariant = false;
    bool nonWritable = false, nonReadable = false, coherent = false, volatile_ = false, restrict_ = false;
    void apply(uint32_t d, const uint32_t* a, size_t n) {
        int v = n ? (int)a[0] : 0;
        switch (d) {
            case DecorationLocation: location = v; break;
            case DecorationComponent: component = v; break;
            case DecorationBinding: binding = v; break;
            case DecorationOffset: offset = v; break;
            case DecorationIndex: index = v; break;
            case DecorationArrayStride: arrayStride = v; break;
            case DecorationMatrixStride: matrixStride = v; break;
            case DecorationBuiltIn: builtin = v; break;
            case DecorationSpecId: specId = v; break;
            case DecorationXfbBuffer: xfbBuffer = v; break;
            case DecorationXfbStride: xfbStride = v; break;
            case DecorationStream: stream = v; break;
            case DecorationBlock: block = true; break;
            case DecorationBufferBlock: bufferBlock = true; break;
            case DecorationRowMajor: rowMajor = true; break;
            case DecorationColMajor: colMajor = true; break;
            case DecorationFlat: flat = true; break;
            case DecorationNoPerspective: noPersp = true; break;
            case DecorationCentroid: centroid = true; break;
            case DecorationSample: sample = true; break;
            case DecorationPatch: patch = true; break;
            case DecorationInvariant: invariant = true; break;
            case DecorationNonWritable: nonWritable = true; break;
            case DecorationNonReadable: nonReadable = true; break;
            case DecorationCoherent: coherent = true; break;
            case DecorationVolatile: volatile_ = true; break;
            case DecorationRestrict: restrict_ = true; break;
            default: break;
        }
    }
};

// Pointer value: an lvalue expression plus the pointee type. Flattened I/O blocks have no GLSL
// variable of their own; `flat` names the block variable until a member is selected (an arrayed
// block's vertex index is kept in `pend` and applied to the member's variable).
struct Ptr {
    std::string expr;
    uint32_t type = 0;     // pointee type id
    uint32_t storage = 0;  // StorageClass of the root variable
    uint32_t flat = 0;
    std::string pend;
    bool flatArrayed = false;
};

struct TexelPtr {
    std::string image, coord, sample;
    uint32_t type = 0;  // pointee (scalar) type
};

struct Block {
    uint32_t label = 0;
    size_t first = 0;   // word offset of the first instruction after OpLabel
    size_t merge = 0;   // word offset of OpSelectionMerge/OpLoopMerge (0 = none)
    size_t term = 0;    // word offset of the terminator
    int order = 0;
    std::vector<size_t> phis;
};

enum class CK : uint8_t { Loop, Sel, Switch };
struct Ctx {
    CK k;
    uint32_t merge = 0, cont = 0, header = 0, fallthrough = 0;
};

class Decompiler {
public:
    Decompiler(const SpirvModule& m, Stage stage, const SpirvModule::EntryPoint& ep, const std::map<uint32_t, uint32_t>& spec)
        : W(m.words), stage_(stage), ep_(ep), spec_(spec) {}
    std::string run();

private:
    const std::vector<uint32_t>& W;
    Stage stage_;
    SpirvModule::EntryPoint ep_;
    std::map<uint32_t, uint32_t> spec_;
    std::vector<size_t> def_;  // id -> word offset of its defining instruction
    std::unordered_map<uint32_t, Deco> deco_;
    std::map<std::pair<uint32_t, uint32_t>, Deco> mdeco_;
    std::set<uint32_t> glslExt_, ignoredExt_;
    std::vector<size_t> globalsOrder_;  // word offsets of types/constants/global variables in module order
    std::map<uint32_t, size_t> fnBegin_;  // function id -> word offset of OpFunction
    std::vector<std::pair<uint32_t, std::vector<uint32_t>>> execModes_;
    std::set<uint32_t> usedGlobals_;
    std::unordered_map<uint32_t, Ptr> ptr_;
    std::unordered_map<uint32_t, TexelPtr> texel_;
    std::unordered_map<uint32_t, std::string> inl_;
    std::map<int, uint32_t> usedBuiltins_;  // builtin -> type (arrays: for redeclarations)
    std::set<uint32_t> neededConsts_;
    bool usesFp64_ = false;

    // per function
    std::unordered_map<uint32_t, Block> blk_;
    std::vector<Ctx> cs_;
    std::string out_;
    int indent_ = 1;

    // ---- module access ----
    uint32_t opAt(size_t o) const { return W[o] & 0xffff; }
    uint32_t lenAt(size_t o) const { return W[o] >> 16; }
    bool has(uint32_t id) const { return id < def_.size() && def_[id]; }
    uint32_t op(uint32_t id) const { return has(id) ? opAt(def_[id]) : 0; }
    const uint32_t* in(uint32_t id) const {
        if (!has(id)) fail("undefined id %" + num(id));
        return &W[def_[id]];
    }
    uint32_t len(uint32_t id) const { return lenAt(def_[id]); }
    uint32_t typeOf(uint32_t id) const { return in(id)[1]; }
    const Deco& deco(uint32_t id) {
        static const Deco none;
        auto it = deco_.find(id);
        return it == deco_.end() ? none : it->second;
    }
    const Deco& mdeco(uint32_t id, uint32_t m) {
        static const Deco none;
        auto it = mdeco_.find({id, m});
        return it == mdeco_.end() ? none : it->second;
    }

    // ---- types ----
    bool isStruct(uint32_t t) { return op(t) == OpTypeStruct; }
    bool isArrayT(uint32_t t) { return op(t) == OpTypeArray || op(t) == OpTypeRuntimeArray; }
    uint32_t elemOf(uint32_t t) {
        switch (op(t)) {
            case OpTypeArray:
            case OpTypeRuntimeArray:
            case OpTypeVector:
            case OpTypeMatrix: return in(t)[2];
            case OpTypePointer: return in(t)[3];
            default: fail("type %" + num(t) + " has no element type");
        }
    }
    uint32_t scalarOf(uint32_t t) {
        while (op(t) == OpTypeVector || op(t) == OpTypeMatrix) t = in(t)[2];
        return t;
    }
    int comps(uint32_t t) { return op(t) == OpTypeVector ? (int)in(t)[3] : 1; }
    bool isFloatT(uint32_t t) { return op(scalarOf(t)) == OpTypeFloat; }
    bool isIntT(uint32_t t) { return op(scalarOf(t)) == OpTypeInt; }
    bool isBoolT(uint32_t t) { return op(scalarOf(t)) == OpTypeBool; }
    bool isSignedT(uint32_t t) { t = scalarOf(t); return op(t) == OpTypeInt && in(t)[3]; }
    int widthOf(uint32_t t) { t = scalarOf(t); return op(t) == OpTypeBool ? 32 : (int)in(t)[2]; }
    bool isOpaque(uint32_t t) {
        uint32_t o = op(t);
        if (o == OpTypeArray || o == OpTypeRuntimeArray) return isOpaque(in(t)[2]);
        return o == OpTypeImage || o == OpTypeSampledImage || o == OpTypeSampler;
    }
    uint32_t arrayLen(uint32_t t) {
        uint64_t v;
        if (!constBits(in(t)[3], v)) fail("array length is not a constant");
        return (uint32_t)v;
    }
    bool isBuiltinStruct(uint32_t t) {
        if (!isStruct(t)) return false;
        for (uint32_t i = 0; i + 2 < len(t); i++)
            if (mdeco(t, i).builtin >= 0) return true;
        return false;
    }
    bool isBlockStruct(uint32_t t) { return isStruct(t) && (deco(t).block || deco(t).bufferBlock); }

    std::string scalarName(uint32_t t) {
        switch (op(t)) {
            case OpTypeBool: return "bool";
            case OpTypeInt:
                if (in(t)[2] != 32) fail(num(in(t)[2]) + "-bit integers are not supported");
                return in(t)[3] ? "int" : "uint";
            case OpTypeFloat:
                if (in(t)[2] == 64) { usesFp64_ = true; return "double"; }
                if (in(t)[2] != 32) fail(num(in(t)[2]) + "-bit floats are not supported");
                return "float";
            default: fail("not a scalar type");
        }
    }
    std::string vecPrefix(uint32_t s) {
        std::string n = scalarName(s);
        return n == "bool" ? "b" : n == "int" ? "i" : n == "uint" ? "u" : n == "double" ? "d" : "";
    }
    std::string vecName(uint32_t s, int n) { return n == 1 ? scalarName(s) : vecPrefix(s) + "vec" + num(n); }
    std::string typeBase(uint32_t t) {
        switch (op(t)) {
            case OpTypeVoid: return "void";
            case OpTypeBool:
            case OpTypeInt:
            case OpTypeFloat: return scalarName(t);
            case OpTypeVector: return vecName(in(t)[2], (int)in(t)[3]);
            case OpTypeMatrix: {
                uint32_t col = in(t)[2];
                int c = (int)in(t)[3], r = comps(col);
                std::string p = widthOf(col) == 64 ? "dmat" : "mat";
                if (widthOf(col) == 64) usesFp64_ = true;
                return c == r ? p + num(c) : p + num(c) + "x" + num(r);
            }
            case OpTypeArray:
            case OpTypeRuntimeArray: return typeBase(in(t)[2]);
            case OpTypeStruct: return "_S" + num(t);
            case OpTypeImage: return imageTypeName(t, false);
            case OpTypeSampledImage: return imageTypeName(in(t)[2], true);
            case OpTypePointer: return typeBase(in(t)[3]);
            default: fail("unsupported type (opcode " + num(op(t)) + ")");
        }
    }
    std::string typeDims(uint32_t t) {
        if (op(t) == OpTypeArray) return "[" + num(arrayLen(t)) + "]" + typeDims(in(t)[2]);
        if (op(t) == OpTypeRuntimeArray) return "[]" + typeDims(in(t)[2]);
        return "";
    }
    std::string typeName(uint32_t t) { return typeBase(t) + typeDims(t); }
    std::string decl(uint32_t t, const std::string& name) { return typeBase(t) + " " + name + typeDims(t); }

    std::string imageTypeName(uint32_t img, bool combined) {
        const uint32_t* w = in(img);
        uint32_t sampledT = w[2], dim = w[3], depth = w[4], arrayed = w[5], ms = w[6], sampled = w[7];
        std::string p = op(sampledT) == OpTypeInt ? (in(sampledT)[3] ? "i" : "u") : "";
        bool image = !combined && sampled == 2;
        std::string s = p + (image ? "image" : "sampler");
        switch (dim) {
            case Dim1D: s += "1D"; break;
            case Dim2D: s += "2D"; break;
            case Dim3D: s += "3D"; break;
            case DimCube: s += "Cube"; break;
            case DimRect: s += "2DRect"; break;
            case DimBuffer: s += "Buffer"; break;
            default: fail("unsupported image dimensionality " + num(dim));
        }
        if (ms) s += "MS";
        if (arrayed) s += "Array";
        if (!image && depth == 1) s += "Shadow";
        return s;
    }
    static const char* imageFormat(uint32_t f) {
        switch (f) {
            case ImageFormatRgba32f: return "rgba32f";
            case ImageFormatRgba16f: return "rgba16f";
            case ImageFormatR32f: return "r32f";
            case ImageFormatRgba8: return "rgba8";
            case ImageFormatRgba8Snorm: return "rgba8_snorm";
            case ImageFormatRg32f: return "rg32f";
            case ImageFormatRg16f: return "rg16f";
            case ImageFormatR11fG11fB10f: return "r11f_g11f_b10f";
            case ImageFormatR16f: return "r16f";
            case ImageFormatRgba16: return "rgba16";
            case ImageFormatRgb10A2: return "rgb10_a2";
            case ImageFormatRg16: return "rg16";
            case ImageFormatRg8: return "rg8";
            case ImageFormatR16: return "r16";
            case ImageFormatR8: return "r8";
            case ImageFormatRgba16Snorm: return "rgba16_snorm";
            case ImageFormatRg16Snorm: return "rg16_snorm";
            case ImageFormatRg8Snorm: return "rg8_snorm";
            case ImageFormatR16Snorm: return "r16_snorm";
            case ImageFormatR8Snorm: return "r8_snorm";
            case ImageFormatRgba32i: return "rgba32i";
            case ImageFormatRgba16i: return "rgba16i";
            case ImageFormatRgba8i: return "rgba8i";
            case ImageFormatR32i: return "r32i";
            case ImageFormatRg32i: return "rg32i";
            case ImageFormatRg16i: return "rg16i";
            case ImageFormatRg8i: return "rg8i";
            case ImageFormatR16i: return "r16i";
            case ImageFormatR8i: return "r8i";
            case ImageFormatRgba32ui: return "rgba32ui";
            case ImageFormatRgba16ui: return "rgba16ui";
            case ImageFormatRgba8ui: return "rgba8ui";
            case ImageFormatR32ui: return "r32ui";
            case ImageFormatRgb10a2ui: return "rgb10_a2ui";
            case ImageFormatRg32ui: return "rg32ui";
            case ImageFormatRg16ui: return "rg16ui";
            case ImageFormatRg8ui: return "rg8ui";
            case ImageFormatR16ui: return "r16ui";
            case ImageFormatR8ui: return "r8ui";
            default: return nullptr;
        }
    }
    // image type behind a sampler/image/sampled-image/pointer type
    uint32_t imageOf(uint32_t t) {
        for (;;) {
            switch (op(t)) {
                case OpTypePointer: t = in(t)[3]; continue;
                case OpTypeArray:
                case OpTypeRuntimeArray: t = in(t)[2]; continue;
                case OpTypeSampledImage: t = in(t)[2]; continue;
                case OpTypeImage: return t;
                default: fail("not an image type");
            }
        }
    }

    // ---- constants ----
    bool isConstOp(uint32_t o) {
        switch (o) {
            case OpConstantTrue: case OpConstantFalse: case OpConstant: case OpConstantComposite: case OpConstantNull:
            case OpSpecConstantTrue: case OpSpecConstantFalse: case OpSpecConstant: case OpSpecConstantComposite:
            case OpSpecConstantOp: case OpUndef: return true;
            default: return false;
        }
    }
    bool specOverride(uint32_t id, uint32_t& v) {
        int sid = deco(id).specId;
        if (sid < 0) return false;
        auto it = spec_.find((uint32_t)sid);
        if (it == spec_.end()) return false;
        v = it->second;
        return true;
    }
    // Scalar value of a constant (bit pattern, sign/zero extended to 64 bits for ints).
    bool constBits(uint32_t id, uint64_t& out) {
        switch (op(id)) {
            case OpConstantTrue: out = 1; return true;
            case OpConstantFalse: out = 0; return true;
            case OpSpecConstantTrue:
            case OpSpecConstantFalse: {
                uint32_t v;
                out = specOverride(id, v) ? (v != 0) : (op(id) == OpSpecConstantTrue);
                return true;
            }
            case OpConstant:
            case OpSpecConstant: {
                const uint32_t* w = in(id);
                uint32_t t = w[1];
                uint32_t v;
                if (op(id) == OpSpecConstant && specOverride(id, v)) {
                    out = v;
                } else {
                    out = w[3];
                    if (len(id) > 4) out |= (uint64_t)w[4] << 32;
                }
                if (op(t) == OpTypeInt && in(t)[3] && in(t)[2] == 32) out = (uint64_t)(int64_t)(int32_t)(uint32_t)out;
                return true;
            }
            case OpConstantNull: out = 0; return true;
            case OpUndef: out = 0; return true;
            case OpSpecConstantOp: return foldSpecOp(id, out);
            default: return false;
        }
    }
    bool foldSpecOp(uint32_t id, uint64_t& out) {
        const uint32_t* w = in(id);
        uint32_t t = w[1], sop = w[3];
        if (op(t) != OpTypeInt && op(t) != OpTypeBool) return false;
        uint32_t n = len(id) - 4;
        const uint32_t* a = w + 4;
        uint64_t x = 0, y = 0, z = 0;
        auto get = [&](int i, uint64_t& v) { return (uint32_t)i < n && constBits(a[i], v); };
        if (sop == OpCompositeExtract) {
            uint32_t c = a[0];
            for (uint32_t i = 1; i < n; i++) {
                if (op(c) != OpConstantComposite && op(c) != OpSpecConstantComposite) return false;
                c = in(c)[3 + a[i]];
            }
            return constBits(c, out);
        }
        if (sop == OpSelect) {
            if (!get(0, x) || !get(1, y) || !get(2, z)) return false;
            out = x ? y : z;
            return true;
        }
        if (!get(0, x)) return false;
        bool two = get(1, y);
        int64_t sx = (int64_t)x, sy = (int64_t)y;
        uint32_t ux = (uint32_t)x, uy = (uint32_t)y;
        uint64_t r;
        switch (sop) {
            case OpSConvert: case OpUConvert: r = x; break;
            case OpSNegate: r = (uint64_t)-sx; break;
            case OpNot: r = ~x; break;
            case OpLogicalNot: r = !x; break;
            case OpIAdd: if (!two) return false; r = x + y; break;
            case OpISub: if (!two) return false; r = x - y; break;
            case OpIMul: if (!two) return false; r = x * y; break;
            case OpUDiv: if (!two || !uy) return false; r = ux / uy; break;
            case OpSDiv: if (!two || !sy) return false; r = (uint64_t)(sx / sy); break;
            case OpUMod: if (!two || !uy) return false; r = ux % uy; break;
            case OpSRem: if (!two || !sy) return false; r = (uint64_t)(sx % sy); break;
            case OpSMod: {
                if (!two || !sy) return false;
                int64_t m = sx % sy;
                if (m && ((m < 0) != (sy < 0))) m += sy;
                r = (uint64_t)m;
                break;
            }
            case OpShiftRightLogical: if (!two) return false; r = ux >> (uy & 31); break;
            case OpShiftRightArithmetic: if (!two) return false; r = (uint64_t)((int32_t)ux >> (uy & 31)); break;
            case OpShiftLeftLogical: if (!two) return false; r = (uint32_t)(ux << (uy & 31)); break;
            case OpBitwiseOr: if (!two) return false; r = x | y; break;
            case OpBitwiseXor: if (!two) return false; r = x ^ y; break;
            case OpBitwiseAnd: if (!two) return false; r = x & y; break;
            case OpLogicalOr: if (!two) return false; r = x || y; break;
            case OpLogicalAnd: if (!two) return false; r = x && y; break;
            case OpLogicalEqual: case OpIEqual: if (!two) return false; r = ux == uy; break;
            case OpLogicalNotEqual: case OpINotEqual: if (!two) return false; r = ux != uy; break;
            case OpULessThan: if (!two) return false; r = ux < uy; break;
            case OpSLessThan: if (!two) return false; r = sx < sy; break;
            case OpUGreaterThan: if (!two) return false; r = ux > uy; break;
            case OpSGreaterThan: if (!two) return false; r = sx > sy; break;
            case OpULessThanEqual: if (!two) return false; r = ux <= uy; break;
            case OpSLessThanEqual: if (!two) return false; r = sx <= sy; break;
            case OpUGreaterThanEqual: if (!two) return false; r = ux >= uy; break;
            case OpSGreaterThanEqual: if (!two) return false; r = sx >= sy; break;
            default: return false;
        }
        if (op(t) == OpTypeInt && in(t)[3]) r = (uint64_t)(int64_t)(int32_t)(uint32_t)r;
        else if (op(t) == OpTypeInt) r = (uint32_t)r;
        out = r;
        return true;
    }
    std::string literal(uint32_t t, uint64_t bits) {
        switch (op(t)) {
            case OpTypeBool: return bits ? "true" : "false";
            case OpTypeInt: {
                if (in(t)[2] != 32) fail("64-bit integers are not supported");
                if (!in(t)[3]) return num((uint32_t)bits) + "u";
                int32_t v = (int32_t)(uint32_t)bits;
                if (v == INT32_MIN) return "(-2147483647 - 1)";
                return v < 0 ? "(" + num(v) + ")" : num(v);
            }
            case OpTypeFloat: {
                char buf[64];
                if (in(t)[2] == 64) {
                    usesFp64_ = true;
                    double d;
                    memcpy(&d, &bits, 8);
                    if (!std::isfinite(d)) {
                        snprintf(buf, sizeof buf, "packDouble2x32(uvec2(0x%08xu, 0x%08xu))", (uint32_t)bits, (uint32_t)(bits >> 32));
                        return buf;
                    }
                    snprintf(buf, sizeof buf, "%.17g", d);
                } else {
                    float f;
                    uint32_t b32 = (uint32_t)bits;
                    memcpy(&f, &b32, 4);
                    if (!std::isfinite(f)) {
                        snprintf(buf, sizeof buf, "uintBitsToFloat(0x%08xu)", b32);
                        return buf;
                    }
                    snprintf(buf, sizeof buf, "%.9g", f);
                }
                std::string s = buf;
                if (s.find_first_of(".en") == std::string::npos) s += ".0";
                if (in(t)[2] == 64) s += "lf";
                if (s[0] == '-') s = "(" + s + ")";
                return s;
            }
            default: fail("bad literal type");
        }
    }
    std::string zeroOf(uint32_t t) {
        switch (op(t)) {
            case OpTypeBool:
            case OpTypeInt:
            case OpTypeFloat: return literal(t, 0);
            case OpTypeVector:
            case OpTypeMatrix: return typeName(t) + "(" + literal(scalarOf(t), 0) + ")";
            case OpTypeArray: {
                std::string s = typeName(t) + "(";
                uint32_t n = arrayLen(t);
                for (uint32_t i = 0; i < n; i++) s += (i ? ", " : "") + zeroOf(in(t)[2]);
                return s + ")";
            }
            case OpTypeStruct: {
                std::string s = typeName(t) + "(";
                for (uint32_t i = 2; i < len(t); i++) s += (i > 2 ? ", " : "") + zeroOf(in(t)[i]);
                return s + ")";
            }
            default: fail("cannot build a null value of this type");
        }
    }
    bool globalConst(uint32_t id) {  // composite constants of array/struct type become named globals
        uint32_t o = op(id);
        if (o == OpSpecConstantOp) {
            uint64_t v;
            return !constBits(id, v);
        }
        return (o == OpConstantComposite || o == OpSpecConstantComposite) && (isArrayT(typeOf(id)) || isStruct(typeOf(id)));
    }
    std::string constExpr(uint32_t id) {
        uint32_t t = typeOf(id);
        if (globalConst(id)) {
            neededConsts_.insert(id);
            return "_" + num(id);
        }
        return constValue(id);
    }
    std::string constValue(uint32_t id) {
        uint32_t t = typeOf(id);
        switch (op(id)) {
            case OpConstantComposite:
            case OpSpecConstantComposite: {
                std::string s = typeName(t) + "(";
                for (uint32_t i = 3; i < len(id); i++) s += (i > 3 ? ", " : "") + ex(in(id)[i]);
                return s + ")";
            }
            case OpConstantNull:
            case OpUndef: return zeroOf(t);
            case OpSpecConstantOp: {
                uint64_t v;
                if (constBits(id, v)) return literal(t, v);
                const uint32_t* w = in(id);
                return valueExpr(w[3], t, w + 4, len(id) - 4);
            }
            default: {
                uint64_t v;
                if (!constBits(id, v)) fail("bad constant");
                return literal(t, v);
            }
        }
    }

    // ---- expressions ----
    std::string ex(uint32_t id) {
        auto it = inl_.find(id);
        if (it != inl_.end()) return it->second;
        if (isConstOp(op(id))) return constExpr(id);
        if (op(id) == OpVariable || ptr_.count(id)) fail("pointer %" + num(id) + " used as a value");
        return "_" + num(id);
    }
    static bool simple(const std::string& e) {
        for (char c : e)
            if (!isalnum((unsigned char)c) && c != '_') return false;
        return true;
    }
    static std::string paren(const std::string& e) { return simple(e) ? e : "(" + e + ")"; }
    // Converts value `e` of type `from` to the GLSL type `to` (bit-preserving between int and uint).
    std::string conv(const std::string& e, uint32_t from, const std::string& to) {
        if (typeName(from) == to) return e;
        return to + "(" + e + ")";
    }
    std::string conv(const std::string& e, uint32_t from, uint32_t to) { return conv(e, from, typeName(to)); }
    std::string val(uint32_t id, uint32_t to) { return conv(ex(id), typeOf(id), to); }
    std::string intName(uint32_t t, bool sgn) {
        int n = comps(t);
        return n == 1 ? std::string(sgn ? "int" : "uint") : std::string(sgn ? "ivec" : "uvec") + num(n);
    }
    std::string asInt(uint32_t id, bool sgn) { return conv(ex(id), typeOf(id), intName(typeOf(id), sgn)); }

    // ---- builtins ----
    std::string builtinName(int b, uint32_t storage) {
        bool input = storage == StorageClassInput;
        switch (b) {
            case BuiltInPosition: return "gl_Position";
            case BuiltInPointSize: return "gl_PointSize";
            case BuiltInClipDistance: return "gl_ClipDistance";
            case BuiltInCullDistance: return "gl_CullDistance";
            case BuiltInVertexId:
            case BuiltInVertexIndex: return "gl_VertexID";
            case BuiltInInstanceId: return "gl_InstanceID";
            case BuiltInInstanceIndex: return "(gl_InstanceID + gl_BaseInstance)";
            case BuiltInPrimitiveId: return stage_ == Stage::Geometry && input ? "gl_PrimitiveIDIn" : "gl_PrimitiveID";
            case BuiltInInvocationId: return "gl_InvocationID";
            case BuiltInLayer: return "gl_Layer";
            case BuiltInViewportIndex: return "gl_ViewportIndex";
            case BuiltInTessLevelOuter: return "gl_TessLevelOuter";
            case BuiltInTessLevelInner: return "gl_TessLevelInner";
            case BuiltInTessCoord: return "gl_TessCoord";
            case BuiltInPatchVertices: return "gl_PatchVerticesIn";
            case BuiltInFragCoord: return "gl_FragCoord";
            case BuiltInPointCoord: return "gl_PointCoord";
            case BuiltInFrontFacing: return "gl_FrontFacing";
            case BuiltInSampleId: return "gl_SampleID";
            case BuiltInSamplePosition: return "gl_SamplePosition";
            case BuiltInSampleMask: return input ? "gl_SampleMaskIn" : "gl_SampleMask";
            case BuiltInFragDepth: return "gl_FragDepth";
            case BuiltInHelperInvocation: return "gl_HelperInvocation";
            case BuiltInNumWorkgroups: return "gl_NumWorkGroups";
            case BuiltInWorkgroupSize: return "gl_WorkGroupSize";
            case BuiltInWorkgroupId: return "gl_WorkGroupID";
            case BuiltInLocalInvocationId: return "gl_LocalInvocationID";
            case BuiltInGlobalInvocationId: return "gl_GlobalInvocationID";
            case BuiltInLocalInvocationIndex: return "gl_LocalInvocationIndex";
            case BuiltInBaseVertex: return "gl_BaseVertex";
            case BuiltInBaseInstance: return "gl_BaseInstance";
            case BuiltInDrawIndex: return "gl_DrawID";
            default: fail("unsupported BuiltIn " + num(b));
        }
    }
    void useBuiltin(int b, uint32_t type, uint32_t storage) {
        if (b == BuiltInClipDistance || b == BuiltInCullDistance) usedBuiltins_[b * 2 + (storage == StorageClassOutput)] = type;
        else usedBuiltins_[b * 2 + (storage == StorageClassOutput)] = type;
    }

    // ---- pointers ----
    std::string globalName(uint32_t id);
    Ptr& ptrOf(uint32_t id);
    void stepMember(Ptr& p, uint32_t k) {
        uint32_t st = p.type;
        if (!isStruct(st)) fail("member access on a non-struct");
        uint32_t mt = in(st)[2 + k];
        if (p.flat) {
            p.expr = flatMemberName(p.flat, k) + p.pend;
            p.pend.clear();
            p.flat = 0;
        } else if (int b = mdeco(st, k).builtin; b >= 0) {
            std::string n = builtinName(b, p.storage);
            useBuiltin(b, mt, p.storage);
            p.expr = p.expr.empty() ? n : p.expr + "." + n;
        } else {
            p.expr += ".m" + num(k);
        }
        p.type = mt;
    }
    void stepIndex(Ptr& p, const std::string& idx) {
        uint32_t t = p.type;
        if (p.flat) {
            p.pend += "[" + idx + "]";
        } else {
            p.expr += "[" + idx + "]";
        }
        if (op(t) == OpTypeArray || op(t) == OpTypeRuntimeArray || op(t) == OpTypeVector || op(t) == OpTypeMatrix) p.type = in(t)[2];
        else fail("index into a non-composite");
    }
    std::string indexExpr(uint32_t id) {
        uint64_t v;
        if (isConstOp(op(id)) && constBits(id, v)) return num((int64_t)(int32_t)(uint32_t)v);
        return ex(id);
    }
    std::string flatMemberName(uint32_t var, uint32_t k) {
        uint32_t sc = in(typeOf(var))[2];
        return std::string(sc == StorageClassInput ? "_i" : "_o") + num(var) + "_" + num(k);
    }
    // Reads/writes whole values through pointers whose GLSL form is not a single variable
    // (flattened I/O blocks, builtin blocks, uniform/storage block instances).
    bool needsSplit(const Ptr& p) {
        if (p.flat) return true;
        if (!isStruct(p.type)) return false;
        return isBuiltinStruct(p.type) || isBlockStruct(p.type);
    }
    std::string load(Ptr p) {
        if (!needsSplit(p)) return p.expr;
        if (!isStruct(p.type)) fail("whole-array access to an interface block array is not supported");
        std::string s = typeName(p.type) + "(";
        for (uint32_t i = 0; i + 2 < len(p.type); i++) {
            Ptr q = p;
            stepMember(q, i);
            s += (i ? ", " : "") + load(q);
        }
        return s + ")";
    }
    void store(Ptr p, const std::string& v) {
        if (!needsSplit(p)) {
            line(p.expr + " = " + v + ";");
            return;
        }
        if (!isStruct(p.type)) fail("whole-array store to an interface block array is not supported");
        std::string tmp = paren(v);
        for (uint32_t i = 0; i + 2 < len(p.type); i++) {
            Ptr q = p;
            stepMember(q, i);
            store(q, tmp + ".m" + num(i));
        }
    }

    // ---- output ----
    void line(const std::string& s) {
        out_.append((size_t)indent_ * 4, ' ');
        out_ += s;
        out_ += '\n';
    }
    void set(uint32_t id, const std::string& e) { line("_" + num(id) + " = " + e + ";"); }

    // ---- instruction translation ----
    std::string valueExpr(uint32_t opc, uint32_t rt, const uint32_t* a, uint32_t n);
    std::string extInst(uint32_t rt, uint32_t inst, const uint32_t* a, uint32_t n, uint32_t id);
    void emitInsn(size_t o);
    void emitImage(size_t o);
    void emitAtomic(size_t o);

    // ---- control flow ----
    void collectBlocks(uint32_t fn);
    const Block& block(uint32_t label) {
        auto it = blk_.find(label);
        if (it == blk_.end()) fail("branch to unknown block %" + num(label));
        return it->second;
    }
    void phiCopies(uint32_t from, uint32_t to);
    int classify(uint32_t to, int* level);
    uint32_t jump(uint32_t to);
    void arm(uint32_t from, uint32_t to);
    void emitChain(uint32_t label);
    uint32_t emitBlock(uint32_t label);
    void emitContinue(int level);

    std::string emitFunction(uint32_t fn, bool isEntry);
    std::string declareGlobals();
};

// ---------------------------------------------------------------------------
// Pointers to globals
// ---------------------------------------------------------------------------
std::string Decompiler::globalName(uint32_t id) {
    uint32_t pt = typeOf(id), sc = in(pt)[2], t = in(pt)[3];
    const Deco& d = deco(id);
    switch (sc) {
        case StorageClassUniformConstant:
            if (d.location >= 0) return "_uL" + num(d.location);
            return "_u" + num(id);
        case StorageClassAtomicCounter: return "_ac" + num(std::max(0, d.binding)) + "_" + num(std::max(0, d.offset)) + "_" + num(id);
        case StorageClassUniform:
        case StorageClassStorageBuffer: {
            uint32_t st = t;
            while (isArrayT(st)) st = in(st)[2];
            bool ssbo = sc == StorageClassStorageBuffer || deco(st).bufferBlock;
            std::string base = ssbo ? "_sb" : "_ub";
            return d.binding >= 0 ? base + num(d.binding) : base + "n" + num(id);
        }
        case StorageClassInput: return "_i" + num(id);
        case StorageClassOutput: return "_o" + num(id);
        default: return "_g" + num(id);
    }
}

Ptr& Decompiler::ptrOf(uint32_t id) {
    auto it = ptr_.find(id);
    if (it != ptr_.end()) return it->second;
    if (op(id) != OpVariable) fail("%" + num(id) + " is not a pointer");
    // a global variable (function-scope ones are registered when declared)
    usedGlobals_.insert(id);
    Ptr p;
    uint32_t pt = typeOf(id);
    p.storage = in(pt)[2];
    p.type = in(pt)[3];
    const Deco& d = deco(id);
    if (d.builtin >= 0) {
        p.expr = builtinName(d.builtin, p.storage);
        useBuiltin(d.builtin, p.type, p.storage);
    } else if ((p.storage == StorageClassInput || p.storage == StorageClassOutput)) {
        uint32_t st = p.type;
        bool arrayed = false;
        if (op(st) == OpTypeArray && isStruct(in(st)[2])) {
            st = in(st)[2];
            arrayed = true;
        }
        if (isBuiltinStruct(st)) {
            p.expr = arrayed ? (p.storage == StorageClassInput ? "gl_in" : "gl_out") : "";
        } else if (isBlockStruct(st)) {
            p.flat = id;
            p.flatArrayed = arrayed;
        } else {
            p.expr = globalName(id);
        }
    } else {
        p.expr = globalName(id);
    }
    return ptr_[id] = p;
}

// ---------------------------------------------------------------------------
// Value instructions shared by function bodies and OpSpecConstantOp.
// `a` points at the operands after the result id; returns the GLSL expression.
// ---------------------------------------------------------------------------
std::string Decompiler::valueExpr(uint32_t opc, uint32_t rt, const uint32_t* a, uint32_t n) {
    std::string R = typeName(rt);
    auto A = [&](int i) { return ex(a[i]); };
    auto bin = [&](const char* o) { return "(" + val(a[0], rt) + " " + o + " " + val(a[1], rt) + ")"; };
    auto vecfn = [&](const char* scalarOp, const char* fn, const std::string& x, const std::string& y) {
        return comps(typeOf(a[0])) == 1 ? "(" + x + " " + scalarOp + " " + y + ")" : std::string(fn) + "(" + x + ", " + y + ")";
    };
    auto icmp = [&](const char* o, const char* fn, int sgn /*1 signed 0 unsigned -1 as-is*/) {
        std::string x = sgn < 0 ? ex(a[0]) : asInt(a[0], sgn), y = sgn < 0 ? val(a[1], typeOf(a[0])) : asInt(a[1], sgn);
        return vecfn(o, fn, x, y);
    };
    switch (opc) {
        case OpCopyObject: return A(0);
        case OpSNegate: return R + "(-" + asInt(a[0], true) + ")";
        case OpFNegate: return "(-" + A(0) + ")";
        case OpNot: return "(~" + val(a[0], rt) + ")";
        case OpIAdd: return bin("+");
        case OpFAdd: return bin("+");
        case OpISub: return bin("-");
        case OpFSub: return bin("-");
        case OpIMul: return bin("*");
        case OpFMul: return bin("*");
        case OpFDiv: return bin("/");
        case OpUDiv: return R + "(" + asInt(a[0], false) + " / " + asInt(a[1], false) + ")";
        case OpSDiv: return R + "(" + asInt(a[0], true) + " / " + asInt(a[1], true) + ")";
        case OpUMod: return R + "(" + asInt(a[0], false) + " % " + asInt(a[1], false) + ")";
        case OpSRem:
        case OpSMod: {
            if (opc == OpSRem) return R + "(" + asInt(a[0], true) + " % " + asInt(a[1], true) + ")";
            // sign of the divisor
            std::string x = asInt(a[0], true), y = asInt(a[1], true), it = intName(rt, true);
            std::string r = "(" + x + " % " + y + ")";
            if (comps(rt) == 1) return R + "((" + r + " != 0 && ((" + r + " < 0) != (" + y + " < 0))) ? " + r + " + " + y + " : " + r + ")";
            return R + "(" + r + ")";
        }
        case OpFMod: return "mod(" + A(0) + ", " + val(a[1], typeOf(a[0])) + ")";
        case OpFRem: return "(" + A(0) + " - " + A(1) + " * trunc(" + A(0) + " / " + A(1) + "))";
        case OpVectorTimesScalar:
        case OpMatrixTimesScalar:
        case OpVectorTimesMatrix:
        case OpMatrixTimesVector:
        case OpMatrixTimesMatrix: return "(" + A(0) + " * " + A(1) + ")";
        case OpOuterProduct: return "outerProduct(" + A(0) + ", " + A(1) + ")";
        case OpDot: return "dot(" + A(0) + ", " + A(1) + ")";
        case OpTranspose: return "transpose(" + A(0) + ")";
        case OpShiftLeftLogical: return "(" + val(a[0], rt) + " << " + A(1) + ")";
        case OpShiftRightLogical: return R + "(" + asInt(a[0], false) + " >> " + A(1) + ")";
        case OpShiftRightArithmetic: return R + "(" + asInt(a[0], true) + " >> " + A(1) + ")";
        case OpBitwiseOr: return bin("|");
        case OpBitwiseXor: return bin("^");
        case OpBitwiseAnd: return bin("&");
        case OpLogicalOr:
        case OpLogicalAnd: {
            const char* o = opc == OpLogicalOr ? "||" : "&&";
            if (comps(rt) == 1) return "(" + A(0) + " " + o + " " + A(1) + ")";
            std::string s = R + "(";
            static const char* sw = "xyzw";
            for (int i = 0; i < comps(rt); i++)
                s += std::string(i ? ", " : "") + paren(A(0)) + "." + sw[i] + " " + o + " " + paren(A(1)) + "." + sw[i];
            return s + ")";
        }
        case OpLogicalNot: return comps(rt) == 1 ? "(!" + A(0) + ")" : "not(" + A(0) + ")";
        case OpLogicalEqual: return vecfn("==", "equal", A(0), A(1));
        case OpLogicalNotEqual: return vecfn("!=", "notEqual", A(0), A(1));
        case OpIEqual: return icmp("==", "equal", -1);
        case OpINotEqual: return icmp("!=", "notEqual", -1);
        case OpUGreaterThan: return icmp(">", "greaterThan", 0);
        case OpSGreaterThan: return icmp(">", "greaterThan", 1);
        case OpUGreaterThanEqual: return icmp(">=", "greaterThanEqual", 0);
        case OpSGreaterThanEqual: return icmp(">=", "greaterThanEqual", 1);
        case OpULessThan: return icmp("<", "lessThan", 0);
        case OpSLessThan: return icmp("<", "lessThan", 1);
        case OpULessThanEqual: return icmp("<=", "lessThanEqual", 0);
        case OpSLessThanEqual: return icmp("<=", "lessThanEqual", 1);
        case OpFOrdEqual: return vecfn("==", "equal", A(0), A(1));
        case OpFUnordNotEqual: return vecfn("!=", "notEqual", A(0), A(1));
        case OpFOrdLessThan: return vecfn("<", "lessThan", A(0), A(1));
        case OpFOrdGreaterThan: return vecfn(">", "greaterThan", A(0), A(1));
        case OpFOrdLessThanEqual: return vecfn("<=", "lessThanEqual", A(0), A(1));
        case OpFOrdGreaterThanEqual: return vecfn(">=", "greaterThanEqual", A(0), A(1));
        case OpFOrdNotEqual: {
            if (comps(rt) == 1) return "(" + A(0) + " < " + A(1) + " || " + A(0) + " > " + A(1) + ")";
            return "notEqual(" + A(0) + ", " + A(1) + ")";
        }
        case OpFUnordEqual: return comps(rt) == 1 ? "(!(" + A(0) + " < " + A(1) + " || " + A(0) + " > " + A(1) + "))" : "equal(" + A(0) + ", " + A(1) + ")";
        case OpFUnordLessThan: return comps(rt) == 1 ? "(!(" + A(0) + " >= " + A(1) + "))" : "not(greaterThanEqual(" + A(0) + ", " + A(1) + "))";
        case OpFUnordGreaterThan: return comps(rt) == 1 ? "(!(" + A(0) + " <= " + A(1) + "))" : "not(lessThanEqual(" + A(0) + ", " + A(1) + "))";
        case OpFUnordLessThanEqual: return comps(rt) == 1 ? "(!(" + A(0) + " > " + A(1) + "))" : "not(greaterThan(" + A(0) + ", " + A(1) + "))";
        case OpFUnordGreaterThanEqual: return comps(rt) == 1 ? "(!(" + A(0) + " < " + A(1) + "))" : "not(lessThan(" + A(0) + ", " + A(1) + "))";
        case OpIsNan: return "isnan(" + A(0) + ")";
        case OpIsInf: return "isinf(" + A(0) + ")";
        case OpAny: return "any(" + A(0) + ")";
        case OpAll: return "all(" + A(0) + ")";
        case OpSelect: {
            uint32_t ct = typeOf(a[0]);
            if (comps(ct) == 1 || op(rt) != OpTypeVector) return "(" + A(0) + " ? " + val(a[1], rt) + " : " + val(a[2], rt) + ")";
            return "mix(" + val(a[2], rt) + ", " + val(a[1], rt) + ", " + A(0) + ")";
        }
        case OpConvertFToU:
        case OpConvertFToS: return R + "(" + A(0) + ")";
        case OpConvertSToF: return R + "(" + asInt(a[0], true) + ")";
        case OpConvertUToF: return R + "(" + asInt(a[0], false) + ")";
        case OpUConvert: return R + "(" + asInt(a[0], false) + ")";
        case OpSConvert: return R + "(" + asInt(a[0], true) + ")";
        case OpFConvert: return R + "(" + A(0) + ")";
        case OpQuantizeToF16: {
            if (comps(rt) == 1) return "unpackHalf2x16(packHalf2x16(vec2(" + A(0) + ", 0.0))).x";
            std::string s = R + "(";
            static const char* sw = "xyzw";
            for (int i = 0; i < comps(rt); i++) s += std::string(i ? ", " : "") + "unpackHalf2x16(packHalf2x16(vec2(" + paren(A(0)) + "." + sw[i] + ", 0.0))).x";
            return s + ")";
        }
        case OpBitcast: {
            uint32_t ft = typeOf(a[0]);
            if (isFloatT(rt) && widthOf(rt) == 32 && isIntT(ft)) return (isSignedT(ft) ? "intBitsToFloat(" : "uintBitsToFloat(") + A(0) + ")";
            if (isIntT(rt) && isFloatT(ft) && widthOf(ft) == 32) return conv((isSignedT(rt) ? "floatBitsToInt(" : "floatBitsToUint(") + A(0) + ")", rt, R);
            if (isFloatT(rt) && widthOf(rt) == 64 && isIntT(ft) && comps(ft) == 2 && comps(rt) == 1) return "packDouble2x32(" + asInt(a[0], false) + ")";
            if (isIntT(rt) && comps(rt) == 2 && isFloatT(ft) && widthOf(ft) == 64 && comps(ft) == 1) return R + "(unpackDouble2x32(" + A(0) + "))";
            if (isIntT(rt) && isIntT(ft)) return R + "(" + A(0) + ")";
            fail("unsupported OpBitcast");
        }
        case OpCompositeConstruct: {
            std::string s = R + "(";
            for (uint32_t i = 0; i < n; i++) s += (i ? ", " : "") + ex(a[i]);
            return s + ")";
        }
        case OpCompositeExtract: {
            std::string s = paren(A(0));
            uint32_t t = typeOf(a[0]);
            static const char* sw = "xyzw";
            for (uint32_t i = 1; i < n; i++) {
                if (isStruct(t)) {
                    s += ".m" + num(a[i]);
                    t = in(t)[2 + a[i]];
                } else if (op(t) == OpTypeVector) {
                    s += std::string(".") + sw[a[i] & 3];
                    t = in(t)[2];
                } else {
                    s += "[" + num(a[i]) + "]";
                    t = in(t)[2];
                }
            }
            return s;
        }
        case OpVectorShuffle: {
            uint32_t t1 = typeOf(a[0]);
            int n1 = comps(t1);
            static const char* sw = "xyzw";
            bool all1 = true, all2 = true;
            for (uint32_t i = 2; i < n; i++) {
                uint32_t c = a[i] == 0xFFFFFFFFu ? 0 : a[i];
                if ((int)c >= n1) all1 = false;
                else all2 = false;
            }
            if (all1 || all2) {
                std::string s = paren(all1 ? A(0) : A(1)) + ".";
                for (uint32_t i = 2; i < n; i++) {
                    uint32_t c = a[i] == 0xFFFFFFFFu ? 0 : a[i];
                    s += sw[all1 ? c : c - n1];
                }
                return s;
            }
            std::string s = R + "(";
            for (uint32_t i = 2; i < n; i++) {
                uint32_t c = a[i] == 0xFFFFFFFFu ? 0 : a[i];
                s += (i > 2 ? ", " : "") + ((int)c < n1 ? paren(A(0)) + "." + sw[c] : paren(A(1)) + "." + sw[c - n1]);
            }
            return s + ")";
        }
        case OpVectorExtractDynamic: return paren(A(0)) + "[" + A(1) + "]";
        case OpBitFieldInsert: return R + "(bitfieldInsert(" + val(a[0], rt) + ", " + val(a[1], rt) + ", int(" + A(2) + "), int(" + A(3) + ")))";
        case OpBitFieldSExtract: return R + "(bitfieldExtract(" + asInt(a[0], true) + ", int(" + A(1) + "), int(" + A(2) + ")))";
        case OpBitFieldUExtract: return R + "(bitfieldExtract(" + asInt(a[0], false) + ", int(" + A(1) + "), int(" + A(2) + ")))";
        case OpBitReverse: return "bitfieldReverse(" + val(a[0], rt) + ")";
        case OpBitCount: return R + "(bitCount(" + A(0) + "))";
        case OpDPdx: return "dFdx(" + A(0) + ")";
        case OpDPdy: return "dFdy(" + A(0) + ")";
        case OpFwidth: return "fwidth(" + A(0) + ")";
        case OpDPdxFine: return "dFdxFine(" + A(0) + ")";
        case OpDPdyFine: return "dFdyFine(" + A(0) + ")";
        case OpFwidthFine: return "fwidthFine(" + A(0) + ")";
        case OpDPdxCoarse: return "dFdxCoarse(" + A(0) + ")";
        case OpDPdyCoarse: return "dFdyCoarse(" + A(0) + ")";
        case OpFwidthCoarse: return "fwidthCoarse(" + A(0) + ")";
        default: fail("unsupported instruction (opcode " + num(opc) + ")");
    }
}

// GLSL.std.450
std::string Decompiler::extInst(uint32_t rt, uint32_t e, const uint32_t* a, uint32_t n, uint32_t id) {
    auto A = [&](int i) { return ex(a[i]); };
    auto F = [&](const char* f) {
        std::string s = std::string(f) + "(";
        for (uint32_t i = 0; i < n; i++) s += (i ? ", " : "") + ex(a[i]);
        return s + ")";
    };
    auto I = [&](const char* f, bool sgn) {  // integer function evaluated with the given signedness
        std::string s = std::string(f) + "(";
        for (uint32_t i = 0; i < n; i++) s += (i ? ", " : "") + asInt(a[i], sgn);
        return typeName(rt) + "(" + s + "))";
    };
    switch (e) {
        case GLSLRound: return F("round");
        case GLSLRoundEven: return F("roundEven");
        case GLSLTrunc: return F("trunc");
        case GLSLFAbs: return F("abs");
        case GLSLSAbs: return I("abs", true);
        case GLSLFSign: return F("sign");
        case GLSLSSign: return I("sign", true);
        case GLSLFloor: return F("floor");
        case GLSLCeil: return F("ceil");
        case GLSLFract: return F("fract");
        case GLSLRadians: return F("radians");
        case GLSLDegrees: return F("degrees");
        case GLSLSin: return F("sin");
        case GLSLCos: return F("cos");
        case GLSLTan: return F("tan");
        case GLSLAsin: return F("asin");
        case GLSLAcos: return F("acos");
        case GLSLAtan: return F("atan");
        case GLSLSinh: return F("sinh");
        case GLSLCosh: return F("cosh");
        case GLSLTanh: return F("tanh");
        case GLSLAsinh: return F("asinh");
        case GLSLAcosh: return F("acosh");
        case GLSLAtanh: return F("atanh");
        case GLSLAtan2: return F("atan");
        case GLSLPow: return F("pow");
        case GLSLExp: return F("exp");
        case GLSLLog: return F("log");
        case GLSLExp2: return F("exp2");
        case GLSLLog2: return F("log2");
        case GLSLSqrt: return F("sqrt");
        case GLSLInverseSqrt: return F("inversesqrt");
        case GLSLDeterminant: return F("determinant");
        case GLSLMatrixInverse: return F("inverse");
        case GLSLModf: return "modf(" + A(0) + ", " + ptrOf(a[1]).expr + ")";
        case GLSLFrexp: return "frexp(" + A(0) + ", " + ptrOf(a[1]).expr + ")";
        case GLSLModfStruct:
        case GLSLFrexpStruct: {
            std::string r = "_" + num(id);
            line(r + ".m0 = " + (e == GLSLModfStruct ? "modf(" : "frexp(") + A(0) + ", " + r + ".m1);");
            return "";
        }
        case GLSLFMin:
        case GLSLNMin: return F("min");
        case GLSLUMin: return I("min", false);
        case GLSLSMin: return I("min", true);
        case GLSLFMax:
        case GLSLNMax: return F("max");
        case GLSLUMax: return I("max", false);
        case GLSLSMax: return I("max", true);
        case GLSLFClamp:
        case GLSLNClamp: return F("clamp");
        case GLSLUClamp: return I("clamp", false);
        case GLSLSClamp: return I("clamp", true);
        case GLSLFMix: return F("mix");
        case GLSLStep: return F("step");
        case GLSLSmoothStep: return F("smoothstep");
        case GLSLFma: return F("fma");
        case GLSLLdexp: {
            uint32_t et = typeOf(a[1]);
            return "ldexp(" + A(0) + ", " + conv(A(1), et, intName(et, true)) + ")";
        }
        case GLSLPackSnorm4x8: return F("packSnorm4x8");
        case GLSLPackUnorm4x8: return F("packUnorm4x8");
        case GLSLPackSnorm2x16: return F("packSnorm2x16");
        case GLSLPackUnorm2x16: return F("packUnorm2x16");
        case GLSLPackHalf2x16: return F("packHalf2x16");
        case GLSLPackDouble2x32: return "packDouble2x32(" + asInt(a[0], false) + ")";
        case GLSLUnpackSnorm2x16: return F("unpackSnorm2x16");
        case GLSLUnpackUnorm2x16: return F("unpackUnorm2x16");
        case GLSLUnpackHalf2x16: return F("unpackHalf2x16");
        case GLSLUnpackSnorm4x8: return F("unpackSnorm4x8");
        case GLSLUnpackUnorm4x8: return F("unpackUnorm4x8");
        case GLSLUnpackDouble2x32: return typeName(rt) + "(unpackDouble2x32(" + A(0) + "))";
        case GLSLLength: return F("length");
        case GLSLDistance: return F("distance");
        case GLSLCross: return F("cross");
        case GLSLNormalize: return F("normalize");
        case GLSLFaceForward: return F("faceforward");
        case GLSLReflect: return F("reflect");
        case GLSLRefract: return F("refract");
        case GLSLFindILsb: return typeName(rt) + "(findLSB(" + A(0) + "))";
        case GLSLFindSMsb: return typeName(rt) + "(findMSB(" + asInt(a[0], true) + "))";
        case GLSLFindUMsb: return typeName(rt) + "(findMSB(" + asInt(a[0], false) + "))";
        case GLSLInterpolateAtCentroid: return "interpolateAtCentroid(" + ptrOf(a[0]).expr + ")";
        case GLSLInterpolateAtSample: return "interpolateAtSample(" + ptrOf(a[0]).expr + ", int(" + A(1) + "))";
        case GLSLInterpolateAtOffset: return "interpolateAtOffset(" + ptrOf(a[0]).expr + ", " + A(1) + ")";
        default: fail("unsupported GLSL.std.450 instruction " + num(e));
    }
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------
void Decompiler::emitImage(size_t o) {
    const uint32_t* w = &W[o];
    uint32_t opc = opAt(o), n = lenAt(o);
    if (opc == OpImageWrite) {
        uint32_t img = imageOf(typeOf(w[1]));
        uint32_t st = in(img)[2];
        std::string vt = vecName(st, 4);
        std::string args = ex(w[1]) + ", " + ex(w[2]);
        if (n > 4 && (w[4] & ImageOperandsSampleMask)) args += ", int(" + ex(w[5]) + ")";
        line("imageStore(" + args + ", " + conv(ex(w[3]), typeOf(w[3]), vt) + ");");
        return;
    }
    uint32_t rt = w[1], id = w[2];
    std::string R = typeName(rt);
    std::string s = ex(w[3]);
    uint32_t img = imageOf(typeOf(w[3]));
    const uint32_t* iw = in(img);
    uint32_t dim = iw[3], arrayed = iw[5], ms = iw[6];
    auto res = [&](const std::string& e, const std::string& from) { set(id, from == R ? e : R + "(" + e + ")"); };
    // image operands
    struct Ops {
        std::string bias, lod, dx, dy, offset, offsets, sample;
        bool constOffset = false;
    } ops;
    auto parseOps = [&](uint32_t at) {
        if (at >= n) return;
        uint32_t mask = w[at], k = at + 1;
        if (mask & ImageOperandsBiasMask) ops.bias = ex(w[k++]);
        if (mask & ImageOperandsLodMask) ops.lod = ex(w[k++]);
        if (mask & ImageOperandsGradMask) { ops.dx = ex(w[k++]); ops.dy = ex(w[k++]); }
        if (mask & ImageOperandsConstOffsetMask) { ops.offset = ex(w[k++]); ops.constOffset = true; }
        if (mask & ImageOperandsOffsetMask) ops.offset = ex(w[k++]);
        if (mask & ImageOperandsConstOffsetsMask) ops.offsets = ex(w[k++]);
        if (mask & ImageOperandsSampleMask) ops.sample = "int(" + ex(w[k++]) + ")";
        if (mask & ImageOperandsMinLodMask) fail("MinLod image operand is not supported");
    };
    std::string vt = vecName(in(img)[2], 4);
    switch (opc) {
        case OpImageSampleImplicitLod:
        case OpImageSampleExplicitLod:
        case OpImageSampleProjImplicitLod:
        case OpImageSampleProjExplicitLod:
        case OpImageSampleDrefImplicitLod:
        case OpImageSampleDrefExplicitLod:
        case OpImageSampleProjDrefImplicitLod:
        case OpImageSampleProjDrefExplicitLod: {
            bool proj = opc == OpImageSampleProjImplicitLod || opc == OpImageSampleProjExplicitLod || opc == OpImageSampleProjDrefImplicitLod ||
                        opc == OpImageSampleProjDrefExplicitLod;
            bool dref = opc == OpImageSampleDrefImplicitLod || opc == OpImageSampleDrefExplicitLod || opc == OpImageSampleProjDrefImplicitLod ||
                        opc == OpImageSampleProjDrefExplicitLod;
            parseOps(dref ? 6 : 5);
            std::string coord = ex(w[4]);
            uint32_t ct = typeOf(w[4]);
            int cn = comps(ct);
            std::string extra;
            if (dref) {
                std::string d = ex(w[5]);
                if (dim == DimCube && arrayed) extra = d;
                else if (proj) {
                    // GLSL: (s, [t,] [0,] ref, q)
                    std::string c = paren(coord);
                    if (dim == Dim1D) coord = "vec4(" + c + ".x, 0.0, " + d + ", " + c + ".y)";
                    else coord = "vec4(" + c + ".x, " + c + ".y, " + d + ", " + c + ".z)";
                } else if (dim == Dim1D && !arrayed) {
                    coord = "vec3(" + coord + ", 0.0, " + d + ")";
                } else {
                    coord = "vec" + num(cn + 1) + "(" + coord + ", " + d + ")";
                }
            }
            std::string fn = proj ? "textureProj" : "texture";
            std::string args = s + ", " + coord;
            if (!extra.empty()) args += ", " + extra;
            if (!ops.lod.empty()) { fn += "Lod"; args += ", " + ops.lod; }
            if (!ops.dx.empty()) { fn += "Grad"; args += ", " + ops.dx + ", " + ops.dy; }
            if (!ops.offset.empty()) { fn += "Offset"; args += ", " + ops.offset; }
            if (!ops.bias.empty()) args += ", " + ops.bias;
            res(fn + "(" + args + ")", dref ? "float" : vt);
            return;
        }
        case OpImageFetch: {
            parseOps(5);
            std::string args = s + ", " + ex(w[4]);
            if (ms) args += ", " + (ops.sample.empty() ? std::string("0") : ops.sample);
            else if (dim != DimBuffer && dim != DimRect) args += ", " + (ops.lod.empty() ? std::string("0") : "int(" + ops.lod + ")");
            std::string fn = "texelFetch";
            if (!ops.offset.empty()) { fn += "Offset"; args += ", " + ops.offset; }
            res(fn + "(" + args + ")", vt);
            return;
        }
        case OpImageGather:
        case OpImageDrefGather: {
            parseOps(6);
            std::string args = s + ", " + ex(w[4]);
            std::string fn = "textureGather";
            std::string last;
            if (opc == OpImageDrefGather) last = ex(w[5]);
            else {
                uint64_t c = 0;
                if (!constBits(w[5], c)) fail("textureGather component must be a constant");
                if (c) last = num((int)c);
            }
            if (opc == OpImageDrefGather) args += ", " + last, last.clear();
            if (!ops.offset.empty()) { fn += "Offset"; args += ", " + ops.offset; }
            if (!ops.offsets.empty()) { fn += "Offsets"; args += ", " + ops.offsets; }
            if (!last.empty()) args += ", " + last;
            res(fn + "(" + args + ")", opc == OpImageDrefGather ? "vec4" : vt);
            return;
        }
        case OpImageRead: {
            parseOps(5);
            std::string args = s + ", " + ex(w[4]);
            if (!ops.sample.empty()) args += ", " + ops.sample;
            std::string e = "imageLoad(" + args + ")";
            if (comps(rt) < 4) {
                static const char* sw[] = {"", ".x", ".xy", ".xyz"};
                e += sw[comps(rt)];
                res(e, vecName(in(img)[2], comps(rt)));
            } else {
                res(e, vt);
            }
            return;
        }
        case OpImageQuerySizeLod: {
            std::string e = "textureSize(" + s + ", int(" + ex(w[4]) + "))";
            res(e, comps(rt) == 1 ? "int" : "ivec" + num(comps(rt)));
            return;
        }
        case OpImageQuerySize: {
            bool storage = iw[7] == 2 && op(typeOf(w[3])) == OpTypeImage;
            std::string e = (storage ? "imageSize(" : "textureSize(") + s + ")";
            res(e, comps(rt) == 1 ? "int" : "ivec" + num(comps(rt)));
            return;
        }
        case OpImageQueryLod: res("textureQueryLod(" + s + ", " + ex(w[4]) + ")", "vec2"); return;
        case OpImageQueryLevels: res("textureQueryLevels(" + s + ")", "int"); return;
        case OpImageQuerySamples: {
            bool storage = iw[7] == 2 && op(typeOf(w[3])) == OpTypeImage;
            res(std::string(storage ? "imageSamples(" : "textureSamples(") + s + ")", "int");
            return;
        }
        default: fail("unsupported image instruction (opcode " + num(opc) + ")");
    }
}

void Decompiler::emitAtomic(size_t o) {
    const uint32_t* w = &W[o];
    uint32_t opc = opAt(o);
    bool hasResult = opc != OpAtomicStore;
    uint32_t rt = hasResult ? w[1] : 0, id = hasResult ? w[2] : 0;
    uint32_t p = hasResult ? w[3] : w[1];
    const uint32_t* v = hasResult ? w + 6 : w + 4;  // value operands (after scope/semantics)
    if (opc == OpAtomicCompareExchange) v = w + 7;  // value, comparator
    auto result = [&](const std::string& e, uint32_t from) {
        if (hasResult) set(id, conv(e, from, rt));
        else line(e + ";");
    };
    if (texel_.count(p)) {
        const TexelPtr& t = texel_[p];
        std::string T = scalarName(t.type);
        std::string base = t.image + ", " + t.coord + (t.sample.empty() ? "" : ", " + t.sample);
        auto V = [&](int i) { return conv(ex(v[i]), typeOf(v[i]), T); };
        std::string one = T == "int" ? "1" : "1u", mone = T == "int" ? "(-1)" : "0xFFFFFFFFu";
        switch (opc) {
            case OpAtomicLoad: result("imageLoad(" + base + ").x", t.type); return;
            case OpAtomicStore: line("imageAtomicExchange(" + base + ", " + conv(ex(w[4]), typeOf(w[4]), T) + ");"); return;
            case OpAtomicExchange: result("imageAtomicExchange(" + base + ", " + V(0) + ")", t.type); return;
            case OpAtomicCompareExchange: result("imageAtomicCompSwap(" + base + ", " + V(1) + ", " + V(0) + ")", t.type); return;
            case OpAtomicIIncrement: result("imageAtomicAdd(" + base + ", " + one + ")", t.type); return;
            case OpAtomicIDecrement: result("imageAtomicAdd(" + base + ", " + mone + ")", t.type); return;
            case OpAtomicIAdd: result("imageAtomicAdd(" + base + ", " + V(0) + ")", t.type); return;
            case OpAtomicISub: result("imageAtomicAdd(" + base + ", " + T + "(0) - " + V(0) + ")", t.type); return;
            case OpAtomicSMin:
            case OpAtomicUMin: result("imageAtomicMin(" + base + ", " + V(0) + ")", t.type); return;
            case OpAtomicSMax:
            case OpAtomicUMax: result("imageAtomicMax(" + base + ", " + V(0) + ")", t.type); return;
            case OpAtomicAnd: result("imageAtomicAnd(" + base + ", " + V(0) + ")", t.type); return;
            case OpAtomicOr: result("imageAtomicOr(" + base + ", " + V(0) + ")", t.type); return;
            case OpAtomicXor: result("imageAtomicXor(" + base + ", " + V(0) + ")", t.type); return;
            default: fail("unsupported image atomic");
        }
    }
    Ptr& P = ptrOf(p);
    uint32_t pt = P.type;
    std::string e = P.expr;
    if (P.storage == StorageClassAtomicCounter) {
        auto V = [&](int i) { return conv(ex(v[i]), typeOf(v[i]), "uint"); };
        switch (opc) {
            case OpAtomicLoad: result("atomicCounter(" + e + ")", pt); return;
            case OpAtomicIIncrement: result("atomicCounterIncrement(" + e + ")", pt); return;
            case OpAtomicIDecrement: result("(atomicCounterDecrement(" + e + ") + 1u)", pt); return;
            case OpAtomicIAdd: result("atomicCounterAdd(" + e + ", " + V(0) + ")", pt); return;
            case OpAtomicISub: result("atomicCounterSubtract(" + e + ", " + V(0) + ")", pt); return;
            case OpAtomicSMin:
            case OpAtomicUMin: result("atomicCounterMin(" + e + ", " + V(0) + ")", pt); return;
            case OpAtomicSMax:
            case OpAtomicUMax: result("atomicCounterMax(" + e + ", " + V(0) + ")", pt); return;
            case OpAtomicAnd: result("atomicCounterAnd(" + e + ", " + V(0) + ")", pt); return;
            case OpAtomicOr: result("atomicCounterOr(" + e + ", " + V(0) + ")", pt); return;
            case OpAtomicXor: result("atomicCounterXor(" + e + ", " + V(0) + ")", pt); return;
            case OpAtomicExchange: result("atomicCounterExchange(" + e + ", " + V(0) + ")", pt); return;
            case OpAtomicCompareExchange: result("atomicCounterCompSwap(" + e + ", " + V(1) + ", " + V(0) + ")", pt); return;
            default: fail("unsupported atomic counter operation");
        }
    }
    std::string T = typeName(pt);
    auto V = [&](int i) { return conv(ex(v[i]), typeOf(v[i]), T); };
    std::string one = T == "int" ? "1" : "1u", mone = T == "int" ? "(-1)" : "0xFFFFFFFFu";
    switch (opc) {
        case OpAtomicLoad: result(e, pt); return;
        case OpAtomicStore: line("atomicExchange(" + e + ", " + conv(ex(w[4]), typeOf(w[4]), T) + ");"); return;
        case OpAtomicExchange: result("atomicExchange(" + e + ", " + V(0) + ")", pt); return;
        case OpAtomicCompareExchange: result("atomicCompSwap(" + e + ", " + V(1) + ", " + V(0) + ")", pt); return;
        case OpAtomicIIncrement: result("atomicAdd(" + e + ", " + one + ")", pt); return;
        case OpAtomicIDecrement: result("atomicAdd(" + e + ", " + mone + ")", pt); return;
        case OpAtomicIAdd: result("atomicAdd(" + e + ", " + V(0) + ")", pt); return;
        case OpAtomicISub: result("atomicAdd(" + e + ", " + T + "(0) - " + V(0) + ")", pt); return;
        case OpAtomicSMin:
        case OpAtomicUMin: result("atomicMin(" + e + ", " + V(0) + ")", pt); return;
        case OpAtomicSMax:
        case OpAtomicUMax: result("atomicMax(" + e + ", " + V(0) + ")", pt); return;
        case OpAtomicAnd: result("atomicAnd(" + e + ", " + V(0) + ")", pt); return;
        case OpAtomicOr: result("atomicOr(" + e + ", " + V(0) + ")", pt); return;
        case OpAtomicXor: result("atomicXor(" + e + ", " + V(0) + ")", pt); return;
        default: fail("unsupported atomic operation");
    }
}

// ---------------------------------------------------------------------------
// Instructions in function bodies
// ---------------------------------------------------------------------------
void Decompiler::emitInsn(size_t o) {
    const uint32_t* w = &W[o];
    uint32_t opc = opAt(o), n = lenAt(o);
    switch (opc) {
        case OpNop:
        case OpLine:
        case OpNoLine:
        case OpVariable:  // declared at the top of the function
        case OpPhi:
        case OpSelectionMerge:
        case OpLoopMerge:
        case OpUndef:
        case OpModuleProcessed:
        case OpName:
        case OpMemberName:
        case OpString:
        case OpSource:
        case OpSourceContinued:
        case OpSourceExtension:
            return;
        case OpLoad: {
            Ptr& p = ptrOf(w[3]);
            if (isOpaque(p.type)) {
                inl_[w[2]] = p.expr;
                return;
            }
            // Input builtins are read-only: use them directly (TCS outputs must be indexed with the
            // literal gl_InvocationID in GLSL).
            if (p.storage == StorageClassInput && !p.flat && p.expr.compare(0, 3, "gl_") == 0 && simple(p.expr) &&
                typeName(p.type) == typeName(w[1])) {
                inl_[w[2]] = p.expr;
                return;
            }
            set(w[2], conv(load(p), p.type, w[1]));
            return;
        }
        case OpStore: {
            Ptr& p = ptrOf(w[1]);
            store(p, conv(ex(w[2]), typeOf(w[2]), p.type));
            return;
        }
        case OpCopyMemory: {
            Ptr& d = ptrOf(w[1]);
            Ptr& s = ptrOf(w[2]);
            store(d, load(s));
            return;
        }
        case OpAccessChain:
        case OpInBoundsAccessChain: {
            Ptr p = ptrOf(w[3]);
            for (uint32_t i = 4; i < n; i++) {
                if (isStruct(p.type)) {
                    uint64_t k;
                    if (!constBits(w[i], k)) fail("struct member index is not a constant");
                    stepMember(p, (uint32_t)k);
                } else {
                    stepIndex(p, indexExpr(w[i]));
                }
            }
            ptr_[w[2]] = p;
            return;
        }
        case OpPtrAccessChain:
        case OpInBoundsPtrAccessChain: fail("OpPtrAccessChain is not supported");
        case OpArrayLength: {
            Ptr p = ptrOf(w[3]);
            stepMember(p, w[4]);
            set(w[2], "uint(" + p.expr + ".length())");
            return;
        }
        case OpImageTexelPointer: {
            TexelPtr t;
            t.image = ptrOf(w[3]).expr;
            t.coord = ex(w[4]);
            uint32_t img = imageOf(typeOf(w[3]));
            if (in(img)[6]) t.sample = "int(" + ex(w[5]) + ")";
            t.type = in(w[1])[3];
            texel_[w[2]] = t;
            return;
        }
        case OpSampledImage:
        case OpImage: inl_[w[2]] = ex(w[3]); return;
        case OpCopyObject:
            if (op(w[1]) == OpTypePointer) {
                if (texel_.count(w[3])) texel_[w[2]] = texel_[w[3]];
                else ptr_[w[2]] = ptrOf(w[3]);
                return;
            }
            if (isOpaque(w[1])) { inl_[w[2]] = ex(w[3]); return; }
            set(w[2], ex(w[3]));
            return;
        case OpCompositeInsert: {
            std::string r = "_" + num(w[2]);
            line(r + " = " + ex(w[4]) + ";");
            std::string path = r;
            uint32_t t = w[1];
            static const char* sw = "xyzw";
            for (uint32_t i = 5; i < n; i++) {
                if (isStruct(t)) { path += ".m" + num(w[i]); t = in(t)[2 + w[i]]; }
                else if (op(t) == OpTypeVector) { path += std::string(".") + sw[w[i] & 3]; t = in(t)[2]; }
                else { path += "[" + num(w[i]) + "]"; t = in(t)[2]; }
            }
            line(path + " = " + ex(w[3]) + ";");
            return;
        }
        case OpVectorInsertDynamic: {
            std::string r = "_" + num(w[2]);
            line(r + " = " + ex(w[3]) + ";");
            line(r + "[" + ex(w[5]) + "] = " + ex(w[4]) + ";");
            return;
        }
        case OpIAddCarry:
        case OpISubBorrow: {
            std::string r = "_" + num(w[2]);
            uint32_t mt = in(w[1])[2];
            std::string T = typeName(mt);
            line(r + ".m0 = " + T + "(" + (opc == OpIAddCarry ? "uaddCarry(" : "usubBorrow(") + asInt(w[3], false) + ", " + asInt(w[4], false) +
                 ", " + r + ".m1));");
            return;
        }
        case OpUMulExtended:
        case OpSMulExtended: {
            std::string r = "_" + num(w[2]);
            bool s = opc == OpSMulExtended;
            line(std::string(s ? "imulExtended(" : "umulExtended(") + asInt(w[3], s) + ", " + asInt(w[4], s) + ", " + r + ".m1, " + r + ".m0);");
            return;
        }
        case OpExtInst: {
            if (ignoredExt_.count(w[3])) return;
            if (!glslExt_.count(w[3])) fail("unsupported extended instruction set");
            std::string e = extInst(w[1], w[4], w + 5, n - 5, w[2]);
            if (!e.empty()) {
                if (op(w[1]) == OpTypeVoid) line(e + ";");
                else set(w[2], e);
            }
            return;
        }
        case OpFunctionCall: {
            const uint32_t* fw = &W[fnBegin_.at(w[3])];
            (void)fw;
            std::string call = "_f" + num(w[3]) + "(";
            for (uint32_t i = 4; i < n; i++) {
                uint32_t arg = w[i];
                std::string a;
                if (op(typeOf(arg)) == OpTypePointer) a = ptrOf(arg).expr;
                else a = ex(arg);
                call += (i > 4 ? ", " : "") + a;
            }
            call += ")";
            if (op(w[1]) == OpTypeVoid) line(call + ";");
            else set(w[2], call);
            return;
        }
        case OpEmitVertex: line("EmitVertex();"); return;
        case OpEndPrimitive: line("EndPrimitive();"); return;
        case OpEmitStreamVertex:
        case OpEndStreamPrimitive: {
            uint64_t s = 0;
            if (!constBits(w[1], s)) fail("stream must be a constant");
            line(std::string(opc == OpEmitStreamVertex ? "EmitStreamVertex(" : "EndStreamPrimitive(") + num((int)s) + ");");
            return;
        }
        case OpControlBarrier: {
            uint64_t exec = 0;
            constBits(w[1], exec);
            if (exec == ScopeWorkgroup) line("barrier();");
            else {
                // a memory barrier at most
                uint64_t sem = 0;
                constBits(w[3], sem);
                if (sem) line("memoryBarrier();");
            }
            return;
        }
        case OpMemoryBarrier: {
            uint64_t scope = 0, sem = 0;
            constBits(w[1], scope);
            constBits(w[2], sem);
            uint32_t kinds = (uint32_t)sem & (MemorySemanticsUniformMemoryMask | MemorySemanticsWorkgroupMemoryMask |
                                              MemorySemanticsAtomicCounterMemoryMask | MemorySemanticsImageMemoryMask);
            if (scope == ScopeWorkgroup) line("groupMemoryBarrier();");
            else if (kinds == MemorySemanticsWorkgroupMemoryMask) line("memoryBarrierShared();");
            else if (kinds == MemorySemanticsImageMemoryMask) line("memoryBarrierImage();");
            else if (kinds == MemorySemanticsUniformMemoryMask) line("memoryBarrierBuffer();");
            else if (kinds == MemorySemanticsAtomicCounterMemoryMask) line("memoryBarrierAtomicCounter();");
            else line("memoryBarrier();");
            return;
        }
        case OpImageSampleImplicitLod:
        case OpImageSampleExplicitLod:
        case OpImageSampleDrefImplicitLod:
        case OpImageSampleDrefExplicitLod:
        case OpImageSampleProjImplicitLod:
        case OpImageSampleProjExplicitLod:
        case OpImageSampleProjDrefImplicitLod:
        case OpImageSampleProjDrefExplicitLod:
        case OpImageFetch:
        case OpImageGather:
        case OpImageDrefGather:
        case OpImageRead:
        case OpImageWrite:
        case OpImageQuerySizeLod:
        case OpImageQuerySize:
        case OpImageQueryLod:
        case OpImageQueryLevels:
        case OpImageQuerySamples: emitImage(o); return;
        case OpAtomicLoad:
        case OpAtomicStore:
        case OpAtomicExchange:
        case OpAtomicCompareExchange:
        case OpAtomicIIncrement:
        case OpAtomicIDecrement:
        case OpAtomicIAdd:
        case OpAtomicISub:
        case OpAtomicSMin:
        case OpAtomicUMin:
        case OpAtomicSMax:
        case OpAtomicUMax:
        case OpAtomicAnd:
        case OpAtomicOr:
        case OpAtomicXor: emitAtomic(o); return;
        default: break;
    }
    // plain value instruction: <type> <id> <operands...>
    if (n < 3) fail("unsupported instruction (opcode " + num(opc) + ")");
    uint32_t rt = w[1];
    if (isOpaque(rt)) fail("unsupported operation on an opaque value (opcode " + num(opc) + ")");
    std::string e = valueExpr(opc, rt, w + 3, n - 3);
    set(w[2], e);
}

// ---------------------------------------------------------------------------
// Control flow
// ---------------------------------------------------------------------------
void Decompiler::collectBlocks(uint32_t fn) {
    blk_.clear();
    size_t o = fnBegin_.at(fn);
    size_t end = W.size();
    Block* cur = nullptr;
    int order = 0;
    for (o += lenAt(o); o < end; o += lenAt(o)) {
        uint32_t opc = opAt(o);
        if (opc == OpFunctionEnd) break;
        if (opc == OpLabel) {
            uint32_t l = W[o + 1];
            Block& b = blk_[l];
            b.label = l;
            b.first = o + lenAt(o);
            b.order = order++;
            cur = &b;
            continue;
        }
        if (!cur) continue;
        if (opc == OpPhi) cur->phis.push_back(o);
        if (opc == OpSelectionMerge || opc == OpLoopMerge) cur->merge = o;
        if (opc == OpBranch || opc == OpBranchConditional || opc == OpSwitch || opc == OpReturn || opc == OpReturnValue || opc == OpKill ||
            opc == OpUnreachable || opc == OpTerminateInvocation) {
            cur->term = o;
            cur = nullptr;
        }
    }
}

void Decompiler::phiCopies(uint32_t from, uint32_t to) {
    const Block& b = block(to);
    std::vector<std::pair<uint32_t, uint32_t>> copies;  // phi id, value id
    for (size_t po : b.phis) {
        const uint32_t* w = &W[po];
        uint32_t n = lenAt(po);
        for (uint32_t i = 3; i + 1 < n; i += 2)
            if (w[i + 1] == from) copies.push_back({w[2], w[i]});
    }
    if (copies.empty()) return;
    if (copies.size() == 1) {
        set(copies[0].first, ex(copies[0].second));
        return;
    }
    // parallel copy
    line("{");
    indent_++;
    for (auto& c : copies) line(decl(typeOf(c.first), "_t" + num(c.first)) + " = " + ex(c.second) + ";");
    for (auto& c : copies) set(c.first, "_t" + num(c.first));
    indent_--;
    line("}");
}

// What a branch to `to` means at this point: 0 inline the block, 1 fall through (stop), 2 break,
// 3 continue (back edge), 4 run the continue construct of loop `*level` then continue.
int Decompiler::classify(uint32_t to, int* level) {
    for (int i = (int)cs_.size() - 1; i >= 0; i--) {
        const Ctx& c = cs_[i];
        if (level) *level = i;
        if (c.k == CK::Sel) {
            if (to == c.merge) {
                if (i == (int)cs_.size() - 1) return 1;
                fail("unstructured branch to an enclosing selection merge");
            }
            continue;
        }
        if (c.k == CK::Switch) {
            if (i == (int)cs_.size() - 1 && c.fallthrough && to == c.fallthrough) return 1;
            if (to == c.merge) return 2;
            continue;
        }
        // loop
        if (to == c.header) return 3;
        if (to == c.merge) {
            for (int j = i + 1; j < (int)cs_.size(); j++)
                if (cs_[j].k == CK::Switch) fail("breaking out of a loop from inside a switch is not supported");
            return 2;
        }
        if (to == c.cont) return 4;
        break;
    }
    return 0;
}

// Performs the structural action for a branch to `to`; returns `to` when the block must be inlined.
uint32_t Decompiler::jump(uint32_t to) {
    int level = 0;
    switch (classify(to, &level)) {
        case 0: return to;
        case 1: return 0;
        case 2: line("break;"); return 0;
        case 3: line("continue;"); return 0;
        case 4: emitContinue(level); return 0;
    }
    return 0;
}

void Decompiler::emitContinue(int level) {
    // Emit the continue construct inline; it ends with the back edge ("continue;") or a loop exit.
    Ctx c = cs_[level];
    emitChain(c.cont);
}

void Decompiler::arm(uint32_t from, uint32_t to) {
    phiCopies(from, to);
    if (uint32_t l = jump(to)) emitChain(l);
}

void Decompiler::emitChain(uint32_t label) {
    while (label) {
        const Block& b = block(label);
        if (b.merge && opAt(b.merge) == OpLoopMerge) {
            uint32_t M = W[b.merge + 1], C = W[b.merge + 2];
            line("for (;;) {");
            indent_++;
            cs_.push_back(Ctx{CK::Loop, M, C, label, 0});
            uint32_t next = emitBlock(label);
            if (next) emitChain(next);
            cs_.pop_back();
            indent_--;
            line("}");
            label = jump(M);
            continue;
        }
        label = emitBlock(label);
    }
}

// Emits the instructions and terminator of one block; returns the next block to inline (if any).
uint32_t Decompiler::emitBlock(uint32_t label) {
    const Block& b = block(label);
    if (!b.term) fail("block %" + num(label) + " has no terminator");
    for (size_t o = b.first; o < b.term; o += lenAt(o)) emitInsn(o);
    const uint32_t* w = &W[b.term];
    uint32_t opc = opAt(b.term);
    bool selection = b.merge && opAt(b.merge) == OpSelectionMerge;
    switch (opc) {
        case OpReturn: line("return;"); return 0;
        case OpReturnValue: line("return " + ex(w[1]) + ";"); return 0;
        case OpKill:
        case OpTerminateInvocation: line("discard;"); return 0;
        case OpUnreachable: return 0;
        case OpBranch:
            phiCopies(label, w[1]);
            return jump(w[1]);
        case OpBranchConditional: {
            uint32_t c = w[1], t = w[2], f = w[3];
            if (t == f && !selection) {
                phiCopies(label, t);
                return jump(t);
            }
            if (selection) {
                uint32_t M = W[b.merge + 1];
                cs_.push_back(Ctx{CK::Sel, M});
                line("if (" + ex(c) + ") {");
                indent_++;
                arm(label, t);
                indent_--;
                size_t elsePos = out_.size();
                line("} else {");
                size_t bodyPos = out_.size();
                indent_++;
                arm(label, f);
                indent_--;
                if (out_.size() == bodyPos) out_.resize(elsePos);  // empty else
                line("}");
                cs_.pop_back();
                return jump(M);
            }
            // no merge: a loop header's exit test or a conditional break/continue
            int lt = 0, lf = 0;
            int kt = classify(t, &lt), kf = classify(f, &lf);
            if (kf != 0 && kt == 0) {
                line("if (!" + paren(ex(c)) + ") {");
                indent_++;
                arm(label, f);
                indent_--;
                line("}");
                phiCopies(label, t);
                return jump(t);
            }
            if (kt != 0 && kf == 0) {
                line("if (" + ex(c) + ") {");
                indent_++;
                arm(label, t);
                indent_--;
                line("}");
                phiCopies(label, f);
                return jump(f);
            }
            line("if (" + ex(c) + ") {");
            indent_++;
            arm(label, t);
            indent_--;
            line("} else {");
            indent_++;
            arm(label, f);
            indent_--;
            line("}");
            return 0;
        }
        case OpSwitch: {
            if (!selection) fail("OpSwitch without OpSelectionMerge");
            uint32_t M = W[b.merge + 1];
            uint32_t n = lenAt(b.term);
            uint32_t sel = w[1], def = w[2];
            uint32_t st = typeOf(sel);
            bool sgn = isSignedT(st);
            std::map<uint32_t, std::vector<std::string>> lits;
            for (uint32_t i = 3; i + 1 < n; i += 2) {
                std::string l = sgn ? num((int32_t)w[i]) : num(w[i]) + "u";
                lits[w[i + 1]].push_back(l);
            }
            std::vector<uint32_t> targets;
            for (auto& [l, v] : lits) targets.push_back(l);
            if (!lits.count(def)) targets.push_back(def);
            std::sort(targets.begin(), targets.end(), [&](uint32_t x, uint32_t y) {
                if (x == M) return false;
                if (y == M) return true;
                return block(x).order < block(y).order;
            });
            cs_.push_back(Ctx{CK::Switch, M});
            line("switch (" + ex(sel) + ") {");
            for (size_t i = 0; i < targets.size(); i++) {
                uint32_t L = targets[i];
                for (auto& l : lits[L]) line("case " + l + ":");
                if (L == def) line("default:");
                line("{");
                indent_++;
                cs_.back().fallthrough = (i + 1 < targets.size() && targets[i + 1] != M) ? targets[i + 1] : 0;
                arm(label, L);
                indent_--;
                line("}");
            }
            line("}");
            cs_.pop_back();
            return jump(M);
        }
        default: fail("unsupported terminator");
    }
}

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------
std::string Decompiler::emitFunction(uint32_t fn, bool isEntry) {
    size_t o = fnBegin_.at(fn);
    uint32_t rt = W[o + 1];
    collectBlocks(fn);
    out_.clear();
    indent_ = 1;
    cs_.clear();
    // parameters
    std::vector<std::string> params;
    size_t p = o + lenAt(o);
    for (; opAt(p) == OpFunctionParameter; p += lenAt(p)) {
        uint32_t t = W[p + 1], id = W[p + 2];
        std::string name = "_" + num(id);
        if (op(t) == OpTypePointer) {
            uint32_t sc = in(t)[2], pt = in(t)[3];
            Ptr ptr;
            ptr.expr = name;
            ptr.type = pt;
            ptr.storage = sc;
            ptr_[id] = ptr;
            if (sc == StorageClassUniformConstant || isOpaque(pt)) params.push_back(decl(pt, name));
            else if (sc == StorageClassFunction || sc == StorageClassPrivate) params.push_back("inout " + decl(pt, name));
            else fail("pointer parameters to storage class " + num(sc) + " are not supported");
        } else {
            params.push_back(decl(t, name));
        }
    }
    // hoisted declarations: function variables and SSA values
    std::string decls;
    std::vector<std::pair<uint32_t, uint32_t>> inits;  // local variable initializers
    for (size_t q = p; q < W.size() && opAt(q) != OpFunctionEnd; q += lenAt(q)) {
        uint32_t opc = opAt(q);
        const uint32_t* w = &W[q];
        if (opc == OpVariable) {
            uint32_t pt = in(w[1])[3];
            Ptr ptr;
            ptr.expr = "_" + num(w[2]);
            ptr.type = pt;
            ptr.storage = StorageClassFunction;
            ptr_[w[2]] = ptr;
            decls += "    " + decl(pt, ptr.expr) + ";\n";
            if (lenAt(q) > 4) inits.push_back({w[2], w[4]});
            continue;
        }
        // result-bearing value instructions
        bool typed = false;
        switch (opc) {
            case OpLabel: case OpSelectionMerge: case OpLoopMerge: case OpBranch: case OpBranchConditional: case OpSwitch:
            case OpReturn: case OpReturnValue: case OpKill: case OpUnreachable: case OpStore: case OpCopyMemory: case OpLine:
            case OpNoLine: case OpNop: case OpEmitVertex: case OpEndPrimitive: case OpEmitStreamVertex: case OpEndStreamPrimitive:
            case OpControlBarrier: case OpMemoryBarrier: case OpImageWrite: case OpAtomicStore: case OpTerminateInvocation:
            case OpAccessChain: case OpInBoundsAccessChain: case OpImageTexelPointer: case OpSampledImage: case OpImage:
            case OpUndef:
                break;
            default: typed = lenAt(q) >= 3; break;
        }
        if (!typed) continue;
        uint32_t t = w[1];
        if (!has(t)) continue;
        uint32_t to = op(t);
        if (to == OpTypeVoid || to == OpTypePointer || isOpaque(t)) continue;
        if (to < OpTypeVoid || to > OpTypeForwardPointer) continue;
        if (opc == OpExtInst && ignoredExt_.count(w[3])) continue;
        decls += "    " + decl(t, "_" + num(w[2])) + ";\n";
    }
    for (auto& [v, init] : inits) out_ += "    _" + num(v) + " = " + ex(init) + ";\n";
    // body
    uint32_t entry = 0;
    for (auto& [l, b] : blk_)
        if (b.order == 0) entry = l;
    emitChain(entry);
    std::string sig = isEntry ? "void main(" : decl(rt, "_f" + num(fn)) + "(";
    if (!isEntry && op(rt) == OpTypeArray) sig = typeName(rt) + " _f" + num(fn) + "(";
    for (size_t i = 0; i < params.size(); i++) sig += (i ? ", " : "") + params[i];
    sig += ")";
    std::string body = decls + out_;
    // functions that end without a return on some path (unreachable) still need one for GLSL
    if (!isEntry && op(rt) != OpTypeVoid) body += "    return " + zeroOf(rt) + ";\n";
    return sig + " {\n" + body + "}\n";
}

// ---------------------------------------------------------------------------
// Global declarations
// ---------------------------------------------------------------------------
std::string Decompiler::declareGlobals() {
    std::string s;
    auto interp = [&](const Deco& d) {
        std::string q;
        if (d.flat) q += "flat ";
        if (d.noPersp) q += "noperspective ";
        if (d.centroid) q += "centroid ";
        if (d.sample) q += "sample ";
        if (d.patch) q += "patch ";
        return q;
    };
    auto layoutStr = [](const std::vector<std::string>& items) {
        if (items.empty()) return std::string();
        std::string q = "layout(";
        for (size_t i = 0; i < items.size(); i++) q += (i ? ", " : "") + items[i];
        return q + ") ";
    };
    auto memq = [&](const Deco& d) {
        std::string q;
        if (d.coherent) q += "coherent ";
        if (d.volatile_) q += "volatile ";
        if (d.restrict_) q += "restrict ";
        if (d.nonWritable) q += "readonly ";
        if (d.nonReadable) q += "writeonly ";
        return q;
    };
    // number of locations a type occupies
    std::function<int(uint32_t)> locs = [&](uint32_t t) -> int {
        switch (op(t)) {
            case OpTypeArray: return (int)arrayLen(t) * locs(in(t)[2]);
            case OpTypeMatrix: return (int)in(t)[3] * locs(in(t)[2]);
            case OpTypeStruct: {
                int n = 0;
                for (uint32_t i = 2; i < len(t); i++) n += locs(in(t)[i]);
                return n;
            }
            case OpTypeVector: return widthOf(t) == 64 && comps(t) > 2 ? 2 : 1;
            default: return 1;
        }
    };
    // structs (all non-builtin structs without runtime arrays, in module order)
    for (size_t o : globalsOrder_) {
        if (opAt(o) != OpTypeStruct) continue;
        uint32_t t = W[o + 1];
        if (isBuiltinStruct(t)) continue;
        bool ok = true;
        for (uint32_t i = 2; i < lenAt(o); i++)
            if (op(W[o + i]) == OpTypeRuntimeArray || isOpaque(W[o + i])) ok = false;
        if (!ok) continue;
        s += "struct _S" + num(t) + " {\n";
        for (uint32_t i = 2; i < lenAt(o); i++) s += "    " + decl(W[o + i], "m" + num(i - 2)) + ";\n";
        s += "};\n";
    }
    // constants that need a name
    for (size_t o : globalsOrder_) {
        uint32_t opc = opAt(o);
        if (!isConstOp(opc) || opc == OpUndef) continue;
        uint32_t id = W[o + 2];
        if (!neededConsts_.count(id)) continue;
        s += "const " + decl(W[o + 1], "_" + num(id)) + " = " + constValue(id) + ";\n";
    }
    // variables
    for (size_t o : globalsOrder_) {
        if (opAt(o) != OpVariable) continue;
        uint32_t id = W[o + 2];
        if (!usedGlobals_.count(id)) continue;
        uint32_t pt = W[o + 1], sc = W[o + 3], t = in(pt)[3];
        const Deco& d = deco(id);
        if (d.builtin >= 0) continue;
        {
            uint32_t st = t;
            while (isArrayT(st)) st = in(st)[2];
            if (isBuiltinStruct(st)) continue;  // gl_PerVertex: implicit in GLSL
        }
        Ptr& P = ptr_[id];
        switch (sc) {
            case StorageClassInput:
            case StorageClassOutput: {
                const char* dir = sc == StorageClassInput ? "in " : "out ";
                if (P.flat) {
                    uint32_t st = P.flatArrayed ? in(t)[2] : t;
                    std::string dims = P.flatArrayed ? "[" + num(arrayLen(t)) + "]" : "";
                    int loc = d.location;
                    for (uint32_t i = 0; i + 2 < len(st); i++) {
                        uint32_t mt = in(st)[2 + i];
                        const Deco& md = mdeco(st, i);
                        if (md.location >= 0) loc = md.location;
                        std::vector<std::string> L;
                        if (loc >= 0) L.push_back("location = " + num(loc));
                        if (md.component >= 0) L.push_back("component = " + num(md.component));
                        int xb = md.xfbBuffer >= 0 ? md.xfbBuffer : d.xfbBuffer;
                        if (sc == StorageClassOutput && md.offset >= 0) {
                            if (xb >= 0) L.push_back("xfb_buffer = " + num(xb));
                            L.push_back("xfb_offset = " + num(md.offset));
                            if (d.xfbStride >= 0) L.push_back("xfb_stride = " + num(d.xfbStride));
                        }
                        Deco q = md;
                        q.flat |= d.flat; q.noPersp |= d.noPersp; q.centroid |= d.centroid; q.sample |= d.sample; q.patch |= d.patch;
                        s += layoutStr(L) + interp(q) + dir + decl(mt, flatMemberName(id, i)) + dims + ";\n";
                        if (loc >= 0) loc += locs(mt);
                    }
                    break;
                }
                std::vector<std::string> L;
                if (d.location >= 0) L.push_back("location = " + num(d.location));
                if (d.component >= 0) L.push_back("component = " + num(d.component));
                if (d.index >= 0) L.push_back("index = " + num(d.index));
                if (sc == StorageClassOutput && (d.offset >= 0 || d.xfbBuffer >= 0 || d.xfbStride >= 0)) {
                    if (d.xfbBuffer >= 0) L.push_back("xfb_buffer = " + num(d.xfbBuffer));
                    if (d.offset >= 0) L.push_back("xfb_offset = " + num(d.offset));
                    if (d.xfbStride >= 0) L.push_back("xfb_stride = " + num(d.xfbStride));
                }
                if (d.stream > 0) L.push_back("stream = " + num(d.stream));
                s += layoutStr(L) + interp(d) + dir + decl(t, P.expr) + ";\n";
                break;
            }
            case StorageClassUniformConstant: {
                std::vector<std::string> L;
                if (d.location >= 0) L.push_back("location = " + num(d.location));
                if (d.binding >= 0) L.push_back("binding = " + num(d.binding));
                std::string q;
                if (isOpaque(t) && op(imageOf(t)) == OpTypeImage) {
                    uint32_t img = imageOf(t);
                    bool sampledImage = false;
                    for (uint32_t tt = t; op(tt) != OpTypeImage; tt = in(tt)[2])
                        if (op(tt) == OpTypeSampledImage) sampledImage = true;
                    if (!sampledImage && in(img)[7] == 2) {
                        if (const char* f = imageFormat(in(img)[8])) L.push_back(f);
                        q = memq(d);
                    }
                }
                std::string init;
                if (W[o] >> 16 > 4) init = " = " + ex(W[o + 4]);
                s += layoutStr(L) + q + "uniform " + decl(t, P.expr) + init + ";\n";
                break;
            }
            case StorageClassAtomicCounter: {
                std::vector<std::string> L;
                L.push_back("binding = " + num(std::max(0, d.binding)));
                if (d.offset >= 0) L.push_back("offset = " + num(d.offset));
                s += layoutStr(L) + "uniform atomic_uint " + P.expr + typeDims(t) + ";\n";
                break;
            }
            case StorageClassUniform:
            case StorageClassStorageBuffer: {
                uint32_t st = t;
                while (isArrayT(st)) st = in(st)[2];
                bool ssbo = sc == StorageClassStorageBuffer || deco(st).bufferBlock;
                // choose the packing that reproduces the strides (std140 vs std430)
                bool std140 = !ssbo;
                if (ssbo) {
                    std::function<bool(uint32_t)> looks140 = [&](uint32_t tt) -> bool {
                        if (op(tt) == OpTypeStruct) {
                            for (uint32_t i = 2; i < len(tt); i++) {
                                const Deco& md = mdeco(tt, i - 2);
                                uint32_t m = in(tt)[i];
                                if (isArrayT(m) && deco(m).arrayStride >= 0 && deco(m).arrayStride % 16) return false;
                                if (op(m) == OpTypeMatrix && md.matrixStride >= 0 && md.matrixStride % 16) return false;
                                if (isArrayT(m) && deco(m).arrayStride >= 0) {
                                    uint32_t e = in(m)[2];
                                    while (isArrayT(e)) e = in(e)[2];
                                    if (op(e) != OpTypeStruct && op(e) != OpTypeMatrix && !(op(e) == OpTypeVector && comps(e) >= 3) &&
                                        widthOf(e) * comps(e) / 8 < 16)
                                        return true;
                                }
                                if (op(m) == OpTypeStruct && looks140(m)) return true;
                            }
                        }
                        return false;
                    };
                    std140 = looks140(st);
                }
                std::vector<std::string> L;
                L.push_back(std140 ? "std140" : "std430");
                if (d.binding >= 0) L.push_back("binding = " + num(d.binding));
                std::string blockName = "_B" + P.expr.substr(1);
                std::string b = layoutStr(L) + memq(d) + (ssbo ? "buffer " : "uniform ") + blockName + " {\n";
                for (uint32_t i = 2; i < len(st); i++) {
                    const Deco& md = mdeco(st, i - 2);
                    std::vector<std::string> ML;
                    if (md.offset >= 0) ML.push_back("offset = " + num(md.offset));
                    if (md.rowMajor) ML.push_back("row_major");
                    else if (md.colMajor) ML.push_back("column_major");
                    b += "    " + layoutStr(ML) + memq(md) + decl(in(st)[i], "m" + num(i - 2)) + ";\n";
                }
                b += "} " + P.expr + typeDims(t) + ";\n";
                s += b;
                break;
            }
            case StorageClassWorkgroup: s += "shared " + decl(t, P.expr) + ";\n"; break;
            case StorageClassPrivate: {
                std::string init;
                if (W[o] >> 16 > 4) init = " = " + ex(W[o + 4]);
                s += decl(t, P.expr) + init + ";\n";
                break;
            }
            default: fail("unsupported storage class " + num(sc));
        }
    }
    return s;
}

std::string Decompiler::run() {
    // ---- index the module ----
    uint32_t bound = W[3];
    def_.assign(bound + 1, 0);
    size_t o = 5;
    bool inFunction = false;
    std::vector<std::pair<uint32_t, std::vector<std::pair<uint32_t, std::vector<uint32_t>>>>> groups;
    std::map<uint32_t, std::vector<std::pair<uint32_t, std::vector<uint32_t>>>> groupDecos;  // group id -> decorations
    std::map<uint32_t, std::vector<std::tuple<uint32_t, uint32_t, std::vector<uint32_t>>>> groupMDecos;
    std::vector<uint32_t> caps;
    std::vector<std::string> exts;
    for (; o < W.size(); o += lenAt(o)) {
        uint32_t opc = opAt(o), n = lenAt(o);
        const uint32_t* w = &W[o];
        auto defineAt = [&](uint32_t idWord) {
            uint32_t id = w[idWord];
            if (id && id <= bound) def_[id] = o;
        };
        switch (opc) {
            case OpCapability: caps.push_back(w[1]); continue;
            case OpExtension: exts.push_back(literalString(w + 1, n - 1, nullptr)); continue;
            case OpExtInstImport: {
                std::string name = literalString(w + 2, n - 2, nullptr);
                if (name == "GLSL.std.450") glslExt_.insert(w[1]);
                else if (name.compare(0, 12, "NonSemantic.") == 0) ignoredExt_.insert(w[1]);
                else fail("unsupported extended instruction set \"" + name + "\"");
                defineAt(1);
                continue;
            }
            case OpMemoryModel:
                if (w[1] != AddressingModelLogical) fail("only the Logical addressing model is supported");
                continue;
            case OpExecutionMode:
            case OpExecutionModeId:
                if (w[1] == ep_.fn) execModes_.push_back({w[2], std::vector<uint32_t>(w + 3, w + n)});
                continue;
            case OpDecorate:
                deco_[w[1]].apply(w[2], w + 3, n - 3);
                groupDecos[w[1]].push_back({w[2], std::vector<uint32_t>(w + 3, w + n)});
                continue;
            case OpMemberDecorate:
                mdeco_[{w[1], w[2]}].apply(w[3], w + 4, n - 4);
                groupMDecos[w[1]].push_back({w[2], w[3], std::vector<uint32_t>(w + 4, w + n)});
                continue;
            case OpDecorationGroup: continue;
            case OpGroupDecorate:
                for (uint32_t i = 2; i < n; i++) {
                    for (auto& [dd, args] : groupDecos[w[1]]) deco_[w[i]].apply(dd, args.data(), args.size());
                    for (auto& [m, dd, args] : groupMDecos[w[1]]) mdeco_[{w[i], m}].apply(dd, args.data(), args.size());
                }
                continue;
            case OpGroupMemberDecorate:
                for (uint32_t i = 2; i + 1 < n; i += 2)
                    for (auto& [dd, args] : groupDecos[w[1]]) mdeco_[{w[i], w[i + 1]}].apply(dd, args.data(), args.size());
                continue;
            case OpFunction:
                inFunction = true;
                fnBegin_[w[2]] = o;
                defineAt(2);
                continue;
            case OpFunctionEnd: inFunction = false; continue;
            default: break;
        }
        if (opc >= OpTypeVoid && opc <= OpTypeForwardPointer && opc != OpTypeForwardPointer) {
            defineAt(1);
            globalsOrder_.push_back(o);
            continue;
        }
        if (n >= 3 && opc != OpName && opc != OpMemberName && opc != OpEntryPoint && opc != OpSource && opc != OpString &&
            opc != OpSourceExtension && opc != OpLine && opc != OpExecutionMode && opc != OpSelectionMerge && opc != OpLoopMerge &&
            opc != OpStore && opc != OpBranchConditional && opc != OpSwitch && opc != OpCopyMemory && opc != OpImageWrite &&
            opc != OpAtomicStore && opc != OpMemoryBarrier && opc != OpControlBarrier && opc != OpModuleProcessed &&
            opc != OpEmitStreamVertex && opc != OpEndStreamPrimitive) {
            defineAt(2);
            if (!inFunction) globalsOrder_.push_back(o);
        } else if (opc == OpLabel) {
            defineAt(1);
        }
    }
    // ---- capabilities / extensions ----
    static const std::set<uint32_t> okCaps = {
        CapabilityMatrix, CapabilityShader, CapabilityGeometry, CapabilityTessellation, CapabilityFloat64, CapabilityAtomicStorage,
        CapabilityTessellationPointSize, CapabilityGeometryPointSize, CapabilityImageGatherExtended, CapabilityStorageImageMultisample,
        CapabilityUniformBufferArrayDynamicIndexing, CapabilitySampledImageArrayDynamicIndexing,
        CapabilityStorageBufferArrayDynamicIndexing, CapabilityStorageImageArrayDynamicIndexing, CapabilityClipDistance,
        CapabilityCullDistance, CapabilityImageCubeArray, CapabilitySampleRateShading, CapabilityImageRect, CapabilitySampledRect,
        CapabilitySampled1D, CapabilityImage1D, CapabilitySampledCubeArray, CapabilitySampledBuffer, CapabilityImageBuffer,
        CapabilityImageMSArray, CapabilityStorageImageExtendedFormats, CapabilityImageQuery, CapabilityDerivativeControl,
        CapabilityInterpolationFunction, CapabilityTransformFeedback, CapabilityGeometryStreams, CapabilityStorageImageReadWithoutFormat,
        CapabilityStorageImageWriteWithoutFormat, CapabilityMultiViewport, CapabilityDrawParameters};
    for (uint32_t c : caps)
        if (!okCaps.count(c)) fail("unsupported capability " + num(c));
    for (auto& e : exts) {
        auto& ok = supportedSpirvExtensions();
        if (std::find(ok.begin(), ok.end(), e) == ok.end()) fail("unsupported extension " + e);
    }
    // ---- functions reachable from the entry point (callees first) ----
    std::vector<uint32_t> order;
    std::set<uint32_t> visiting, done;
    std::function<void(uint32_t)> visit = [&](uint32_t f) {
        if (done.count(f)) return;
        if (visiting.count(f)) fail("recursion is not allowed");
        if (!fnBegin_.count(f)) fail("call to an undefined function");
        visiting.insert(f);
        size_t q = fnBegin_[f];
        for (q += lenAt(q); q < W.size() && opAt(q) != OpFunctionEnd; q += lenAt(q))
            if (opAt(q) == OpFunctionCall) visit(W[q + 3]);
        visiting.erase(f);
        done.insert(f);
        order.push_back(f);
    };
    visit(ep_.fn);
    // interface variables are declared even when unused (they still occupy their locations)
    for (uint32_t v : ep_.interface)
        if (op(v) == OpVariable) ptrOf(v);
    std::string functions;
    for (uint32_t f : order) {
        if (f != ep_.fn && f == ep_.fn) continue;
        functions += emitFunction(f, f == ep_.fn) + "\n";
    }

    // ---- stage layout ----
    std::string layouts;
    int localSize[3] = {1, 1, 1};
    bool hasLocal = false;
    std::vector<std::string> inL, outL;
    int outVerts = -1;
    std::string gsOutPrim;
    uint32_t tessPrim = 0, spacing = 0, order_ = 0;
    bool pointMode = false;
    std::string depthLayout;
    bool originUpper = false, pixelInteger = false;
    for (auto& [mode, args] : execModes_) {
        auto arg = [&](size_t i) -> uint32_t { return i < args.size() ? args[i] : 0; };
        switch (mode) {
            case ExecutionModeLocalSize: hasLocal = true; for (int i = 0; i < 3; i++) localSize[i] = (int)arg(i); break;
            case ExecutionModeLocalSizeId:
                hasLocal = true;
                for (int i = 0; i < 3; i++) {
                    uint64_t v = 1;
                    if (!constBits(arg(i), v)) fail("LocalSizeId is not a constant");
                    localSize[i] = (int)v;
                }
                break;
            case ExecutionModeInvocations: inL.push_back("invocations = " + num(arg(0))); break;
            case ExecutionModeInputPoints: inL.push_back("points"); break;
            case ExecutionModeInputLines: inL.push_back("lines"); break;
            case ExecutionModeInputLinesAdjacency: inL.push_back("lines_adjacency"); break;
            case ExecutionModeTriangles:
                if (stage_ == Stage::Geometry) inL.push_back("triangles");
                else tessPrim = 1;
                break;
            case ExecutionModeInputTrianglesAdjacency: inL.push_back("triangles_adjacency"); break;
            case ExecutionModeQuads: tessPrim = 2; break;
            case ExecutionModeIsolines: tessPrim = 3; break;
            case ExecutionModeOutputVertices: outVerts = (int)arg(0); break;
            case ExecutionModeOutputPoints: gsOutPrim = "points"; break;
            case ExecutionModeOutputLineStrip: gsOutPrim = "line_strip"; break;
            case ExecutionModeOutputTriangleStrip: gsOutPrim = "triangle_strip"; break;
            case ExecutionModeSpacingEqual: spacing = 1; break;
            case ExecutionModeSpacingFractionalEven: spacing = 2; break;
            case ExecutionModeSpacingFractionalOdd: spacing = 3; break;
            case ExecutionModeVertexOrderCw: order_ = 1; break;
            case ExecutionModeVertexOrderCcw: order_ = 2; break;
            case ExecutionModePointMode: pointMode = true; break;
            case ExecutionModeOriginUpperLeft: originUpper = true; break;
            case ExecutionModeOriginLowerLeft: break;
            case ExecutionModePixelCenterInteger: pixelInteger = true; break;
            case ExecutionModeEarlyFragmentTests: layouts += "layout(early_fragment_tests) in;\n"; break;
            case ExecutionModeDepthGreater: depthLayout = "depth_greater"; break;
            case ExecutionModeDepthLess: depthLayout = "depth_less"; break;
            case ExecutionModeDepthUnchanged: depthLayout = "depth_unchanged"; break;
            default: break;
        }
    }
    // WorkgroupSize builtin constant overrides the local size
    for (size_t q : globalsOrder_) {
        uint32_t opc = opAt(q);
        if ((opc == OpConstantComposite || opc == OpSpecConstantComposite) && deco(W[q + 2]).builtin == BuiltInWorkgroupSize) {
            hasLocal = true;
            for (int i = 0; i < 3 && 3 + i < (int)lenAt(q); i++) {
                uint64_t v = 1;
                if (!constBits(W[q + 3 + i], v)) fail("WorkgroupSize is not a constant");
                localSize[i] = (int)v;
            }
        }
    }
    if (stage_ == Stage::Compute) {
        if (!hasLocal) fail("compute entry point without a local size");
        layouts += "layout(local_size_x = " + num(localSize[0]) + ", local_size_y = " + num(localSize[1]) + ", local_size_z = " + num(localSize[2]) +
                   ") in;\n";
    }
    if (stage_ == Stage::Geometry) {
        if (!inL.empty()) layouts += "layout(" + [&] { std::string x; for (size_t i = 0; i < inL.size(); i++) x += (i ? ", " : "") + inL[i]; return x; }() + ") in;\n";
        if (!gsOutPrim.empty() || outVerts >= 0) {
            std::string x = gsOutPrim;
            if (outVerts >= 0) x += (x.empty() ? "" : ", ") + std::string("max_vertices = ") + num(outVerts);
            layouts += "layout(" + x + ") out;\n";
        }
    }
    if (stage_ == Stage::TessControl && outVerts >= 0) layouts += "layout(vertices = " + num(outVerts) + ") out;\n";
    if (stage_ == Stage::TessEval) {
        std::vector<std::string> L;
        if (tessPrim) L.push_back(tessPrim == 1 ? "triangles" : tessPrim == 2 ? "quads" : "isolines");
        if (spacing) L.push_back(spacing == 1 ? "equal_spacing" : spacing == 2 ? "fractional_even_spacing" : "fractional_odd_spacing");
        if (order_) L.push_back(order_ == 1 ? "cw" : "ccw");
        if (pointMode) L.push_back("point_mode");
        if (!L.empty()) {
            std::string x;
            for (size_t i = 0; i < L.size(); i++) x += (i ? ", " : "") + L[i];
            layouts += "layout(" + x + ") in;\n";
        }
    }
    if (stage_ == Stage::Fragment) {
        if (originUpper || pixelInteger) {
            std::string x = originUpper ? "origin_upper_left" : "";
            if (pixelInteger) x += std::string(x.empty() ? "" : ", ") + "pixel_center_integer";
            layouts += "layout(" + x + ") in vec4 gl_FragCoord;\n";
        }
        if (!depthLayout.empty()) layouts += "layout(" + depthLayout + ") out float gl_FragDepth;\n";
    }
    // builtin redeclarations: clip/cull distance array sizes, invariance
    for (auto& [key, t] : usedBuiltins_) {
        int b = key / 2;
        bool output = key & 1;
        if ((b == BuiltInClipDistance || b == BuiltInCullDistance) && op(t) == OpTypeArray) {
            bool perVertexArray = !output ? (stage_ != Stage::Vertex && stage_ != Stage::Fragment) : stage_ == Stage::TessControl;
            if (perVertexArray) continue;
            layouts += std::string(output ? "out" : "in") + " float " + (b == BuiltInClipDistance ? "gl_ClipDistance" : "gl_CullDistance") + "[" +
                       num(arrayLen(t)) + "];\n";
        }
    }
    // transform feedback of builtin outputs: gl_PerVertex redeclaration carrying the xfb qualifiers
    for (uint32_t id : usedGlobals_) {
        uint32_t pt = typeOf(id), st = in(pt)[3];
        if (in(pt)[2] != StorageClassOutput || !isBuiltinStruct(st)) continue;
        bool xfb = false;
        for (uint32_t i = 0; i + 2 < len(st); i++)
            if (mdeco(st, i).offset >= 0) xfb = true;
        if (!xfb) continue;
        const Deco& d = deco(id);
        std::vector<std::string> L;
        if (d.xfbBuffer >= 0) L.push_back("xfb_buffer = " + num(d.xfbBuffer));
        if (d.xfbStride >= 0) L.push_back("xfb_stride = " + num(d.xfbStride));
        std::string x = L.empty() ? "" : "layout(" + L[0] + (L.size() > 1 ? ", " + L[1] : "") + ") ";
        x += "out gl_PerVertex {\n";
        for (uint32_t i = 0; i + 2 < len(st); i++) {
            const Deco& md = mdeco(st, i);
            std::string ml = md.offset >= 0 ? "layout(xfb_offset = " + num(md.offset) + ") " : "";
            x += "    " + ml + decl(in(st)[2 + i], builtinName(md.builtin, StorageClassOutput)) + ";\n";
        }
        layouts += x + "};\n";
    }
    for (auto& [id, d] : deco_)
        if (d.builtin == BuiltInPosition && d.invariant && usedGlobals_.count(id)) layouts += "invariant gl_Position;\n";
    for (auto& [k, d] : mdeco_)
        if (d.builtin == BuiltInPosition && d.invariant && stage_ != Stage::Fragment) layouts += "invariant gl_Position;\n";

    std::string globals = declareGlobals();
    std::string src = "#version 460 core\n";
    src += layouts + globals + "\n" + functions;
    return src;
}

}  // namespace

std::shared_ptr<const SpirvModule> parseSpirv(const void* data, size_t bytes, std::string* error) {
    auto bad = [&](const std::string& m) -> std::shared_ptr<const SpirvModule> {
        if (error) *error = m;
        return nullptr;
    };
    if (!data || bytes < 20 || bytes % 4) return bad("not a SPIR-V module");
    auto m = std::make_shared<SpirvModule>();
    m->words.resize(bytes / 4);
    memcpy(m->words.data(), data, bytes);
    auto& w = m->words;
    if (w[0] == 0x03022307u)
        for (auto& x : w) x = __builtin_bswap32(x);
    if (w[0] != 0x07230203u) return bad("bad SPIR-V magic number");
    if ((w[1] >> 16) != 1) return bad("unsupported SPIR-V version");
    if (w[3] == 0 || w[3] > (1u << 22)) return bad("bad SPIR-V id bound");
    size_t o = 5;
    while (o < w.size()) {
        uint32_t n = w[o] >> 16, opc = w[o] & 0xffff;
        if (n == 0 || o + n > w.size()) return bad("truncated SPIR-V instruction");
        if (opc == OpEntryPoint && n >= 4) {
            SpirvModule::EntryPoint ep;
            ep.model = w[o + 1];
            ep.fn = w[o + 2];
            size_t used = 0;
            ep.name = literalString(&w[o + 3], n - 3, &used);
            for (size_t i = 3 + used; i < n; i++) ep.interface.push_back(w[o + i]);
            m->entryPoints.push_back(ep);
        } else if (opc == OpDecorate && n >= 4 && w[o + 2] == DecorationSpecId) {
            m->specIds.insert(w[o + 3]);
        }
        o += n;
    }
    if (m->entryPoints.empty()) return bad("SPIR-V module has no entry points");
    return m;
}

SpecializeResult specializeSpirv(const SpirvModule& module, Stage stage, const std::string& entryPoint,
                                 const std::vector<std::pair<uint32_t, uint32_t>>& constants, const CompileOptions& opts) {
    SpecializeResult r;
    static const uint32_t models[] = {ExecutionModelVertex, ExecutionModelTessellationControl, ExecutionModelTessellationEvaluation,
                                      ExecutionModelGeometry, ExecutionModelFragment, ExecutionModelGLCompute};
    const SpirvModule::EntryPoint* ep = nullptr;
    for (auto& e : module.entryPoints)
        if (e.name == entryPoint && e.model == models[(int)stage]) ep = &e;
    if (!ep) {
        r.status = SpecializeStatus::BadEntryPoint;
        return r;
    }
    std::map<uint32_t, uint32_t> spec;
    for (auto& [id, v] : constants) {
        if (!module.specIds.count(id)) {
            r.status = SpecializeStatus::BadConstantId;
            return r;
        }
        spec[id] = v;
    }
    try {
        Decompiler d(module, stage, *ep, spec);
        r.glsl = d.run();
    } catch (const SpvError& e) {
        r.status = SpecializeStatus::Failed;
        r.compile.infoLog = "error: SPIR-V: " + e.msg + "\n";
        return r;
    } catch (const std::exception& e) {
        r.status = SpecializeStatus::Failed;
        r.compile.infoLog = std::string("error: SPIR-V: malformed module (") + e.what() + ")\n";
        return r;
    }
    if (getenv("METALGLDRV_DUMP_SPIRV")) fprintf(stderr, "---- SPIR-V entry point '%s' as GLSL ----\n%s\n", entryPoint.c_str(), r.glsl.c_str());
    CompileOptions o = opts;
    o.compatibilityProfile = false;
    r.compile = compile(stage, {r.glsl}, o);
    if (r.compile.shader) {
        const_cast<Shader*>(r.compile.shader.get())->spirv = true;
        r.status = SpecializeStatus::Ok;
    } else {
        r.status = SpecializeStatus::Failed;
        r.compile.infoLog = "error: SPIR-V module translation failed to compile:\n" + r.compile.infoLog;
    }
    return r;
}

}  // namespace glsl
