#include <cstdlib>
// GLSL AST -> Metal Shading Language.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <unordered_map>

#include "glsl/builtins.h"
#include "glsl/fold.h"
#include "glsl/gl_enums.h"
#include "glsl/program.h"
#include "glsl/types.h"

namespace glsl {

std::string buildMSLPrelude(const std::string& body);

namespace {

struct EmitError {
    std::string msg;
};

struct Acc {
    std::string ptr, off;
    Type type;
    TL layout;
    bool device = false;
    uint32_t compStride = 0;  // vector whose components are strided (row-major matrix column)
    bool blockArray = false;  // Var of an array of blocks, needs indexing
    const BlockBinding* bb = nullptr;
    std::string sizeExpr;     // SSBO bound size expression
};

bool isSimpleIdent(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s)
        if (!(std::isalnum((unsigned char)c) || c == '_' || c == '.' || c == '>' || c == '-')) return false;
    return !std::isdigit((unsigned char)s[0]);
}

const char* kSwz = "xyzw";
static const char* const kStageNs[] = {"vs", "tcs", "tes", "gs", "fs", "cs"};

class Emitter {
public:
    Emitter(const LinkedProgram& p, const VariantKey& k) : P(&p), P0(&p), K(k) {
        for (auto& x : SP) x = &p;
        stageMask_ = p.stageMask;
        usesFp64_ = p.refl.usesFp64;
    }
    // Program pipeline: each stage from its own (separable) program, each with its own
    // argument buffer (MGLArgs at buffer 0 then holds one pointer per stage).
    Emitter(const LinkedProgram* const stages[(int)Stage::Count], const VariantKey& k) : K(k) {
        multi_ = true;
        for (int i = 0; i < (int)Stage::Count; i++) {
            SP[i] = stages[i];
            if (stages[i] && (stages[i]->stageMask & (1u << i))) {
                stageMask_ |= 1u << i;
                if (!P0) P0 = stages[i];
                usesFp64_ |= stages[i]->refl.usesFp64;
            }
        }
        for (auto& x : SP) if (!x) x = P0;
        P = P0;
    }
    MSLOutput run();
    std::string recordSizes(MultiPassInfo& mp);

private:
    const LinkedProgram* P = nullptr;   // program of the current stage
    const LinkedProgram* P0 = nullptr;
    const LinkedProgram* SP[(int)Stage::Count] = {};
    bool multi_ = false;
    bool argsEmitted_[(int)Stage::Count] = {};
    uint32_t stageMask_ = 0;
    bool usesFp64_ = false;
    const StageCode& SC(Stage s) const { return SP[(int)s]->stages[(int)s]; }
    const Reflection& RF(Stage s) const { return SP[(int)s]->refl; }
    Stage lastPreRaster() const {
        for (Stage st : {Stage::Geometry, Stage::TessEval, Stage::Vertex})
            if (stageMask_ & (1u << (int)st)) return st;
        return Stage::Vertex;
    }
    // Expression giving a stage's argument buffer pointer inside entry points.
    std::string argsOf(Stage s) const { return multi_ ? std::string("A.") + kStageNs[(int)s] : "&A"; }
    int discardCount_ = 0;  // METALGLDRV_DEBUG_NODISCARD bisecting
    const VariantKey& K;
    MSLOutput out_;

    // per stage
    Stage stage_ = Stage::Vertex;
    const StageCode* sc_ = nullptr;
    std::string ns_;
    std::string helpers_;
    std::map<std::string, std::string> helperByKey_;
    int helperN_ = 0;
    std::vector<std::string> temps_;
    std::string retWriteback_;  // copy-out of out/inout parameters, emitted before every return
    const Function* curFn_ = nullptr;
    int tempN_ = 0;
    std::map<const Function*, std::string> fnNames_;
    std::set<std::string> fragInputsAvailable_;  // from the vertex stage
    bool haveVsOutputs_ = false;
    bool hasShared_ = false;
    std::map<std::string, std::string> interpLeaf_;  // FS input leaf accessor (without "g.") -> In member, for interpolateAt*
    bool needEq_ = false;

    [[noreturn]] void fail(const std::string& m) { throw EmitError{m}; }

    // ---- types -------------------------------------------------------------------
    static std::string scalarName(BT b) {
        switch (b) {
            case BT::Float: return "float";
            case BT::Double: return "ulong";
            case BT::Int: return "int";
            case BT::UInt: return "uint";
            case BT::Bool: return "bool";
            default: return "void";
        }
    }
    static std::string vecName(BT b, int n) {
        if (n == 1) return scalarName(b);
        return scalarName(b) + std::to_string(n);
    }
    // in and out blocks of one stage may share a block name (separate interfaces)
    std::string structName(const StructDef* sd) {
        if (!sd->isBlock) return "S_" + sd->name;
        return std::string(sd->blockQ.storage == Storage::In ? "Bi_" : sd->blockQ.storage == Storage::Out ? "Bo_" : "B_") + sd->name;
    }
    std::string texType(const Type& t, uint8_t memory) {
        std::string T = scalarName(t.sampled == BT::Float ? BT::Float : t.sampled);
        bool image = t.base == BT::Image;
        std::string acc;
        if (image) acc = (memory & MemReadOnly) ? ", access::read" : (memory & MemWriteOnly) ? ", access::write" : ", access::read_write";
        if (t.base == BT::Sampler && t.shadow) {
            if (t.dim == Dim::Cube) return t.arrayed ? "depthcube_array<float>" : "depthcube<float>";
            return t.arrayed ? "depth2d_array<float>" : "depth2d<float>";
        }
        if (t.ms) return (t.arrayed ? "texture2d_ms_array<" : "texture2d_ms<") + T + (image ? ", access::read" : "") + ">";
        switch (t.dim) {
            case Dim::D1:
            case Dim::D2:
            case Dim::Rect: return (t.arrayed ? "texture2d_array<" : "texture2d<") + T + acc + ">";
            case Dim::D3: return "texture3d<" + T + acc + ">";
            case Dim::Cube:
                if (image) return "texture2d_array<" + T + acc + ">";
                return (t.arrayed ? "texturecube_array<" : "texturecube<") + T + ">";
            case Dim::Buffer: return "texture_buffer<" + T + (image ? acc : ", access::read") + ">";
            default: fail("bad texture type");
        }
    }
    std::string mt(const Type& t) {
        if (!t.dims.empty()) {
            if (t.dims[0] <= 0) fail("unsized array type '" + t.str() + "' cannot be used here");
            return "mgl_array<" + mt(t.elem()) + ", " + std::to_string(t.dims[0]) + ">";
        }
        switch (t.base) {
            case BT::Struct: return structName(t.sdef);
            case BT::Sampler:
            case BT::Image: return texType(t, 0);
            case BT::AtomicUint: return "device atomic_uint*";
            case BT::Void: return "void";
            default: break;
        }
        if (t.cols) {
            if (t.base == BT::Double) return "mgl_dmat<" + std::to_string(t.cols) + ", " + std::to_string(t.vec) + ">";
            return "float" + std::to_string(t.cols) + "x" + std::to_string(t.vec);
        }
        return vecName(t.base, t.vec);
    }

    // ---- literals ----------------------------------------------------------------
    std::string scalarLit(BT b, const Scalar& s) {
        char buf[64];
        switch (b) {
            case BT::Double: {
                uint64_t bits;
                std::memcpy(&bits, &s.f, 8);
                std::snprintf(buf, sizeof buf, "0x%016llXul", (unsigned long long)bits);
                return buf;
            }
            case BT::Float: {
                double v = s.f;
                if (std::isnan(v)) return "NAN";
                if (std::isinf(v)) return v > 0 ? "INFINITY" : "(-INFINITY)";
                std::snprintf(buf, sizeof buf, "%.9g", (double)(float)v);
                std::string r = buf;
                if (r.find_first_of(".e") == std::string::npos) r += ".0";
                r += "f";
                if (v < 0 || (v == 0 && std::signbit(v))) r = "(" + r + ")";
                return r;
            }
            case BT::Int: {
                int32_t v = (int32_t)s.i;
                if (v == INT32_MIN) return "(-2147483647 - 1)";
                return v < 0 ? "(" + std::to_string(v) + ")" : std::to_string(v);
            }
            case BT::UInt: return std::to_string((uint32_t)s.u) + "u";
            case BT::Bool: return s.b ? "true" : "false";
            default: return "0";
        }
    }
    std::string lit(const Type& t, const ConstVal& cv, size_t& i) {
        if (!t.dims.empty()) {
            std::string r = mt(t) + "{{";
            for (int k = 0; k < t.dims[0]; k++) r += (k ? ", " : "") + lit(t.elem(), cv, i);
            return r + "}}";
        }
        if (t.base == BT::Struct) {
            std::string r = mt(t) + "{";
            for (size_t k = 0; k < t.sdef->members.size(); k++) r += (k ? ", " : "") + lit(t.sdef->members[k].type, cv, i);
            return r + "}";
        }
        if (t.cols) {
            bool d = t.base == BT::Double;
            std::string r = mt(t) + (d ? "{{" : "(");
            for (int c = 0; c < t.cols; c++) {
                r += (c ? ", " : "") + vecName(t.base, t.vec) + "(";
                for (int k = 0; k < t.vec; k++) r += (k ? ", " : "") + scalarLit(t.base, cv.c[i++]);
                r += ")";
            }
            return r + (d ? "}}" : ")");
        }
        if (t.vec > 1) {
            std::string r = mt(t) + "(";
            for (int k = 0; k < t.vec; k++) r += (k ? ", " : "") + scalarLit(t.base, cv.c[i++]);
            return r + ")";
        }
        return scalarLit(t.base, cv.c[i++]);
    }

    // Base-type conversion of a same-shaped value (scalar, vector or matrix). Doubles go through the fp64 library.
    std::string conv(const std::string& s, const Type& from, const Type& to) {
        if (from.base == to.base) return s;
        if (to.base == BT::Double) {
            if (to.cols) return "mgl_dmat_from_float(" + (from.base == BT::Float ? s : mt(from.withBase(BT::Float)) + "(" + s + ")") + ")";
            switch (from.base) {
                case BT::Float: return "mgl_f64_from_float(" + s + ")";
                case BT::Int: return "mgl_f64_from_int(" + s + ")";
                case BT::UInt: return "mgl_f64_from_uint(" + s + ")";
                case BT::Bool: return "mgl_f64_from_bool(" + s + ")";
                default: fail("bad conversion to double");
            }
        }
        if (from.base == BT::Double) {
            if (from.cols) return conv("mgl_dmat_to_float(" + s + ")", from.withBase(BT::Float), to);
            switch (to.base) {
                case BT::Float: return "mgl_f64_to_float(" + s + ")";
                case BT::Int: return "mgl_f64_to_int(" + s + ")";
                case BT::UInt: return "mgl_f64_to_uint(" + s + ")";
                case BT::Bool: return "mgl_f64_to_bool(" + s + ")";
                default: fail("bad conversion from double");
            }
        }
        return mt(to) + "(" + s + ")";
    }
    std::string f64one(const Type& t) {
        std::string one = "0x3FF0000000000000ul";
        return t.isVector() ? vecName(BT::Double, t.vec) + "(" + one + ")" : one;
    }
    std::string binaryF64(Op op, std::string a, const Type& ta, std::string b, const Type& tb);
    std::string builtinF64(const Expr* e);

    // ---- temporaries -------------------------------------------------------------
    std::string newTemp(const Type& t) {
        std::string n = "_t" + std::to_string(tempN_++);
        temps_.push_back(mt(t) + " " + n + " = {};");  // zeroed: see SK::Decl
        return n;
    }
    std::string newTempS(const std::string& type) {
        std::string n = "_t" + std::to_string(tempN_++);
        temps_.push_back(type + " " + n + " = {};");
        return n;
    }
    std::string bind(const std::string& expr, const std::string& type, std::string& pre) {
        if (isSimpleIdent(expr)) return expr;
        std::string t = newTempS(type);
        pre += t + " = " + expr + ", ";
        return t;
    }

    // ---- variables -----------------------------------------------------------------
    // Variables living in threadgroup memory (shared, and the patch-wide interface
    // variables of tessellation stages) are reached through G::S.
    bool inThreadgroup(const Variable* v) const {
        if (!v->global) return false;
        if (v->q.storage == Storage::Shared) return true;
        bool perThread = v->builtin == BuiltinVar::InvocationID || v->builtin == BuiltinVar::PrimitiveID ||
                         v->builtin == BuiltinVar::PatchVerticesIn || v->builtin == BuiltinVar::TessCoord ||
                         v->builtin == BuiltinVar::CompatUniform || v->builtin == BuiltinVar::DepthRange ||
                         v->builtin == BuiltinVar::NumSamples;
        if (perThread) return false;
        if (stage_ == Stage::TessControl) return v->q.storage == Storage::In || v->q.storage == Storage::Out;
        if (stage_ == Stage::TessEval) return v->q.storage == Storage::In;
        return false;
    }
    bool rootInThreadgroup(const Expr* e) const {
        while (e->kind == EK::Index || e->kind == EK::Field || e->kind == EK::Swizzle) e = e->args[0];
        return e->kind == EK::Var && inThreadgroup(e->var);
    }
    std::string varName(const Variable* v) {
        if (v->builtin != BuiltinVar::None) return (inThreadgroup(v) ? "S->bi_" : "bi_") + v->name;
        if (v->global) {
            if (inThreadgroup(v)) return "S->g_" + v->name;
            return "g_" + v->name;
        }
        if (v->name.empty()) return "_p" + std::to_string(v->id);
        return v->name + "_" + std::to_string(v->id);
    }
    bool isDefaultUniform(const Variable* v) {
        return v->global && v->q.storage == Storage::Uniform && !v->block &&
               (!v->type.containsOpaque() || v->type.stripArrays().base == BT::Struct) && v->type.base != BT::Subroutine &&
               v->builtin != BuiltinVar::DepthRange && v->builtin != BuiltinVar::NumSamples;
    }
    const BlockBinding* blockOf(const Variable* v) {
        if (!v->block || !(v->q.storage == Storage::Uniform || v->q.storage == Storage::Buffer)) return nullptr;
        auto it = P->blocks.find(std::string(v->q.storage == Storage::Buffer ? "B:" : "U:") + v->block->name);
        if (it == P->blocks.end()) fail("internal: block '" + v->block->name + "' has no binding");
        return &it->second;
    }

    // ---- buffer access paths -----------------------------------------------------------
    bool bufAccess(const Expr* e, Acc& a);
    std::string loadAcc(const Acc& a);
    std::string storeAcc(const Acc& a, const std::string& v);
    std::string loaderFor(const Type& t, const TL& L);
    std::string storerFor(const Type& t, const TL& L);
    std::string layoutSig(const TL& L);

    // ---- expressions ---------------------------------------------------------------------
    std::string ex(const Expr* e);
    std::string binary(Op op, const std::string& a, const Type& ta, const std::string& b, const Type& tb, const Type& rt);
    std::string assign(const Expr* e);
    std::string incdec(const Expr* e);
    std::string storeTo(const Expr* l, const std::string& v);
    std::string ctor(const Expr* e);
    std::string callUser(const Expr* e);
    std::string callBuiltin(const Expr* e);
    std::string textureFn(const Expr* e, const std::string& name);
    std::string imageFn(const Expr* e, const std::string& name);
    std::string atomicPtr(const Expr* e, std::string& pre);
    bool addressable(const Expr* e);
    bool isBufferRooted(const Expr* e);
    std::pair<std::string, std::string> texRef(const Expr* e);
    std::string borderRef(const Expr* e);
    std::string withOuts(const std::string& callee, const Expr* e, const std::vector<uint8_t>& outs, const std::vector<Type>& ptypes,
                         const Type& ret, const std::function<std::string(size_t, const Expr*)>& argFn, const std::string& firstArg = "");
    std::string subroutineDispatchers();

    // ---- statements ------------------------------------------------------------------
    void stmt(const Stmt* s, std::string& o, int ind);
    void block(const Stmt* s, std::string& o, int ind);
    std::string fnSignature(const Function* f);
    std::string emitFunction(const Function* f);

    // ---- stage -------------------------------------------------------------------------
    struct StageInfo {
        std::set<const Variable*> builtins;
        std::map<BuiltinVar, const Variable*> bi;
        std::map<const Variable*, std::set<int>> usedMembers;  // members of gl_in/gl_out referenced by the stage
        std::string mainName;
        bool hasShared = false;
        bool prepared = false;
    };
    StageInfo si_[(int)Stage::Count];
    std::string entries_;  // entry point functions, appended after all namespaces
    struct Field {
        std::string name;            // varying leaf name (also the [[user()]] attribute)
        Type type;
        const Variable* var;         // null: suffix is the complete accessor
        std::string suffix;          // accessor suffix, "$J" = per-vertex array element
        std::string key;
    };
    void useStage(Stage s) {
        stage_ = s;
        P = SP[(int)s] ? SP[(int)s] : P0;
        sc_ = &P->stages[(int)s];
    }
    void prepareStage(Stage s);
    void emitBody(Stage s, std::string& o, const std::string& preG = "", const std::string& extraMembers = "",
                  const std::string& extraMethods = "");
    static std::string builtinKey(BuiltinVar b);
    void fieldsOfVar(const Variable* v, Stage st, bool arrayed, std::vector<Field>& out);
    std::vector<Field> producerFields(Stage st);
    std::vector<Field> arrayedFields(Stage st, Storage storage);
    std::vector<Field> patchFields(Stage st, Storage storage);
    std::string acc(const Field& f, const std::string& prefix, const std::string& j = "");
    std::string vsInputs(bool stageIn, std::string& inStruct, bool& anyIn, const std::string& g);
    std::string runVS(const std::string& gv, const std::string& vid, const std::string& inst);
    struct RasterOut {
        std::string vertexStruct, primStruct, vertexCopy, primCopy;
        std::set<std::string> names;
        bool hasPrim = false;
    };
    RasterOut rasterOutput(Stage producer, const std::string& g, bool mesh, bool points);
    void emitClassicVS(std::string& o);
    void emitFS(std::string& o, bool meshInput);
    // Mesh render pipelines need a fragment function; programs without a fragment shader
    // (GL leaves color undefined, depth/stencil still update) get an empty one.
    void emitNullFS() {
        entries_ += "fragment void mgl_fs() {}\n\n";
        out_.entryPoints.push_back({"mgl_fs", Stage::Fragment, EntryPoint::Fragment});
    }
    void emitCS(std::string& o);
    void emitGeometry(std::string& o, bool fromTes);
    std::string gsUnpack(const std::vector<Field>& src, const std::string& rec, const std::string& gp);
    struct TessState {
        bool hasTCS = false;
        int OUTV = 0;
        size_t VB = 16, PB = 32;
        std::map<std::string, size_t> vslot, pslot;
        std::string cps, setup;
    } ts_;
    std::vector<Field> tesRec_;  // TES output record layout (tessellation feeding GS / xfb)
    int tesRB_ = 16, tesIpp_ = 3;
    void emitTessControlPass(std::string& o);
    std::string tesLoadControlPoints(const std::string& gp, const std::string& cps, int NT);
    std::string tesLoadPatch();
    std::string tesRun(const std::string& coord, const std::string& patch, const std::string& cps);
    void emitTessEvalMesh(std::string& o);
    void emitTessEvalRecords(std::string& o);
    void emitTesXfb(std::string& o);
    void emitXfb(std::string& o);
    void emitGeometryXfb(std::string& o, bool fromTes);
    std::string xfbVertexStores();
    void captureWords(Stage st, const std::string& g, std::vector<std::tuple<int, uint32_t, std::string>>& out);

    void collectStructs(std::vector<const StructDef*>& order);
    void emitArgs(std::string& o);
    void flatten(const Type& t, const std::string& acc, std::vector<std::pair<Type, std::string>>& leaves);
    // double vectors travel as one flat uint2 varying per component
    static void splitDoubleLeaves(std::vector<std::pair<Type, std::string>>& leaves) {
        std::vector<std::pair<Type, std::string>> out;
        for (auto& [t, acc] : leaves) {
            if (t.base == BT::Double && t.isVector())
                for (int c = 0; c < t.vec; c++) out.push_back({Type::scalar(BT::Double), acc + "." + kSwz[c]});
            else
                out.push_back({t, acc});
        }
        leaves.swap(out);
    }
    std::string interpAttr(const Qualifiers& q, const Type& leaf);
};

// ===========================================================================
// Buffer access
// ===========================================================================
bool Emitter::bufAccess(const Expr* e, Acc& a) {
    switch (e->kind) {
        case EK::Var: {
            const Variable* v = e->var;
            if (isDefaultUniform(v)) {
                auto it = P->defaultOffsets.find(v->name);
                if (it == P->defaultOffsets.end()) fail("internal: uniform '" + v->name + "' has no offset");
                a.ptr = "A->s" + std::to_string(P->defaultSlot);
                a.off = std::to_string(it->second) + "u";
                a.type = v->type;
                a.layout = computeLayout(v->type, Packing::Std430, v->q.layout.matrix == 2);
                a.device = false;
                return true;
            }
            const BlockBinding* bb = blockOf(v);
            if (!bb) return false;
            a.bb = bb;
            a.device = bb->storage;
            if (v->type.isArray()) {
                a.blockArray = true;
                a.type = v->type;
                return true;
            }
            a.ptr = "A->s" + std::to_string(bb->slot);
            a.off = "0u";
            a.type = v->type;
            a.layout = bb->layout;
            if (bb->storage) a.sizeExpr = "A->s" + std::to_string(bb->sizeSlot);
            return true;
        }
        case EK::Field: {
            if (!bufAccess(e->args[0], a)) return false;
            if (a.blockArray) fail("internal: field of unindexed block array");
            int f = e->field;
            a.off = "(" + a.off + " + " + std::to_string(a.layout->offsets[f]) + "u)";
            a.layout = a.layout->members[f];
            a.type = e->type;
            return true;
        }
        case EK::Index: {
            if (!bufAccess(e->args[0], a)) return false;
            std::string idx;
            const Expr* ie = e->args[1];
            if (ie->cv) idx = std::to_string(ie->type.base == BT::UInt ? (uint32_t)ie->cv->c[0].u : (uint32_t)ie->cv->c[0].i) + "u";
            else idx = "uint(" + ex(ie) + ")";
            if (a.blockArray) {
                a.blockArray = false;
                std::string sub = "[" + idx + "]";
                a.ptr = "A->s" + std::to_string(a.bb->slot) + sub;
                a.off = "0u";
                a.type = e->type;
                a.layout = a.bb->layout;
                if (a.bb->storage) a.sizeExpr = "A->s" + std::to_string(a.bb->sizeSlot) + sub;
                return true;
            }
            const Type& bt = a.type;
            if (bt.isArray()) {
                a.off = "(" + a.off + " + " + idx + " * " + std::to_string(a.layout->arrayStride) + "u)";
                a.layout = a.layout->elem;
            } else if (bt.isMatrix()) {
                uint32_t N = scalarSize(bt.base);
                if (!a.layout->rowMajor) {
                    a.off = "(" + a.off + " + " + idx + " * " + std::to_string(a.layout->matrixStride) + "u)";
                    auto vl = std::make_shared<TypeLayout>();
                    vl->size = N * bt.vec;
                    a.layout = vl;
                } else {
                    a.off = "(" + a.off + " + " + idx + " * " + std::to_string(N) + "u)";
                    a.compStride = a.layout->matrixStride;
                    auto vl = std::make_shared<TypeLayout>();
                    a.layout = vl;
                }
            } else if (bt.isVector()) {
                uint32_t step = a.compStride ? a.compStride : scalarSize(bt.base);
                a.off = "(" + a.off + " + " + idx + " * " + std::to_string(step) + "u)";
                a.compStride = 0;
                a.layout = std::make_shared<TypeLayout>();
            } else {
                return false;
            }
            a.type = e->type;
            return true;
        }
        default:
            return false;
    }
}

std::string Emitter::layoutSig(const TL& L) {
    std::string s = std::to_string(L->size) + "/" + std::to_string(L->arrayStride) + "/" + std::to_string(L->matrixStride) +
                    (L->rowMajor ? "r" : "c");
    if (L->elem) s += "[" + layoutSig(L->elem) + "]";
    if (!L->members.empty()) {
        s += "{";
        for (size_t i = 0; i < L->members.size(); i++) s += std::to_string(L->offsets[i]) + ":" + layoutSig(L->members[i]) + ",";
        s += "}";
    }
    return s;
}

std::string Emitter::loaderFor(const Type& t, const TL& L) {
    if (t.dims.empty() && t.isNumeric() && !t.cols) return "mgl_ld_" + mt(t);
    std::string key = "ld:" + t.str() + ":" + (t.sdef ? std::to_string(t.sdef->id) : "") + ":" + layoutSig(L);
    auto it = helperByKey_.find(key);
    if (it != helperByKey_.end()) return it->second;
    std::string name = "mgl_ld" + std::to_string(helperN_++);
    std::string T = mt(t);
    std::string body;
    if (!t.dims.empty()) {
        std::string el = loaderFor(t.elem(), L->elem);
        body = "    " + T + " r;\n    for (int i = 0; i < " + std::to_string(t.dims[0]) + "; i++) r[i] = " + el + "(p, o + uint(i) * " +
               std::to_string(L->arrayStride) + "u);\n    return r;\n";
    } else if (t.base == BT::Struct) {
        body = "    " + T + " r;\n";
        for (size_t i = 0; i < t.sdef->members.size(); i++) {
            const Member& m = t.sdef->members[i];
            if (m.type.isArray() && m.type.dims[0] == 0) continue;
            if (m.type.containsOpaque()) fail("structures containing opaque types cannot be used as values");
            body += "    r.m_" + m.name + " = " + loaderFor(m.type, L->members[i]) + "(p, o + " + std::to_string(L->offsets[i]) + "u);\n";
        }
        body += "    return r;\n";
    } else if (t.cols) {
        uint32_t N = scalarSize(t.base);
        std::string col = vecName(t.base, t.vec);
        body = "    " + T + " r;\n";
        for (int c = 0; c < t.cols; c++) {
            body += "    r[" + std::to_string(c) + "] = ";
            if (!L->rowMajor) body += "mgl_ld_" + col + "(p, o + " + std::to_string(c * L->matrixStride) + "u)";
            else {
                body += col + "(";
                for (int r = 0; r < t.vec; r++)
                    body += (r ? ", " : "") + std::string("mgl_ld_") + scalarName(t.base) + "(p, o + " + std::to_string(r * L->matrixStride + c * N) + "u)";
                body += ")";
            }
            body += ";\n";
        }
        body += "    return r;\n";
    } else {
        fail("cannot load type '" + t.str() + "' from a buffer");
    }
    helpers_ += "template <typename P> inline " + T + " " + name + "(P p, uint o) {\n" + body + "}\n";
    helperByKey_[key] = name;
    return name;
}

std::string Emitter::storerFor(const Type& t, const TL& L) {
    if (t.dims.empty() && t.isNumeric() && !t.cols) return "mgl_st_" + mt(t);
    std::string key = "st:" + t.str() + ":" + (t.sdef ? std::to_string(t.sdef->id) : "") + ":" + layoutSig(L);
    auto it = helperByKey_.find(key);
    if (it != helperByKey_.end()) return it->second;
    std::string name = "mgl_st" + std::to_string(helperN_++);
    std::string T = mt(t);
    std::string body;
    if (!t.dims.empty()) {
        std::string el = storerFor(t.elem(), L->elem);
        body = "    for (int i = 0; i < " + std::to_string(t.dims[0]) + "; i++) " + el + "(p, o + uint(i) * " +
               std::to_string(L->arrayStride) + "u, v[i]);\n";
    } else if (t.base == BT::Struct) {
        for (size_t i = 0; i < t.sdef->members.size(); i++) {
            const Member& m = t.sdef->members[i];
            if (m.type.isArray() && m.type.dims[0] == 0) continue;
            body += "    " + storerFor(m.type, L->members[i]) + "(p, o + " + std::to_string(L->offsets[i]) + "u, v.m_" + m.name + ");\n";
        }
    } else if (t.cols) {
        uint32_t N = scalarSize(t.base);
        std::string col = vecName(t.base, t.vec);
        for (int c = 0; c < t.cols; c++) {
            if (!L->rowMajor) body += "    mgl_st_" + col + "(p, o + " + std::to_string(c * L->matrixStride) + "u, v[" + std::to_string(c) + "]);\n";
            else
                for (int r = 0; r < t.vec; r++)
                    body += "    mgl_st_" + scalarName(t.base) + "(p, o + " + std::to_string(r * L->matrixStride + c * N) + "u, v[" +
                            std::to_string(c) + "][" + std::to_string(r) + "]);\n";
        }
    } else {
        fail("cannot store type '" + t.str() + "' to a buffer");
    }
    helpers_ += "inline void " + name + "(device uchar* p, uint o, " + T + " v) {\n" + body + "}\n";
    helperByKey_[key] = name;
    return name;
}

std::string Emitter::loadAcc(const Acc& a) {
    if (a.blockArray) fail("cannot use an array of blocks as a value");
    if (a.compStride) {
        return "mgl_ldcs_" + mt(a.type) + "(" + a.ptr + ", " + a.off + ", " + std::to_string(a.compStride) + "u)";
    }
    return loaderFor(a.type, a.layout) + "(" + a.ptr + ", " + a.off + ")";
}

std::string Emitter::storeAcc(const Acc& a, const std::string& v) {
    if (!a.device) fail("cannot write to a uniform");
    if (a.compStride) {
        return "mgl_stcs_" + mt(a.type) + "(" + a.ptr + ", " + a.off + ", " + std::to_string(a.compStride) + "u, " + v + ")";
    }
    return storerFor(a.type, a.layout) + "(" + a.ptr + ", " + a.off + ", " + v + ")";
}

bool Emitter::isBufferRooted(const Expr* e) {
    while (e->kind == EK::Index || e->kind == EK::Field || e->kind == EK::Swizzle) e = e->args[0];
    if (e->kind != EK::Var) return false;
    return isDefaultUniform(e->var) || blockOf(e->var) != nullptr;
}

bool Emitter::addressable(const Expr* e) {
    switch (e->kind) {
        case EK::Var: {
            const Variable* v = e->var;
            if (inThreadgroup(v) || v->q.storage == Storage::Uniform || v->q.storage == Storage::Buffer) return false;
            return true;
        }
        case EK::Index:
            if (e->args[0]->type.isVector()) return false;
            return addressable(e->args[0]);
        case EK::Field:
            return addressable(e->args[0]);
        default:
            return false;
    }
}

// ===========================================================================
// Expressions
// ===========================================================================
std::string Emitter::binaryF64(Op op, std::string a, const Type& ta, std::string b, const Type& tb) {
    bool ma = ta.isMatrix(), mb = tb.isMatrix();
    if (ma || mb) {
        switch (op) {
            case Op::Add: return (ma && mb ? "mgl_dmat_add(" : "mgl_dmat_adds(") + a + ", " + b + ")";
            case Op::Sub: return (ma && mb ? "mgl_dmat_sub(" : "mgl_dmat_subs(") + a + ", " + b + ")";
            case Op::Div: return (ma && mb ? "mgl_dmat_div(" : "mgl_dmat_divs(") + a + ", " + b + ")";
            case Op::Mul:
                if (ma && mb) return "mgl_dmat_mm(" + a + ", " + b + ")";
                if (ma && tb.isVector()) return "mgl_dmat_mv(" + a + ", " + b + ")";
                if (mb && ta.isVector()) return "mgl_dmat_vm(" + a + ", " + b + ")";
                return "mgl_dmat_muls(" + a + ", " + b + ")";
            case Op::Eq: needEq_ = true; return "mgl_eq(" + a + ", " + b + ")";
            case Op::Ne: needEq_ = true; return "(!mgl_eq(" + a + ", " + b + "))";
            default: fail("invalid double matrix operation");
        }
    }
    if (ta.isScalar() && tb.isVector()) a = vecName(BT::Double, tb.vec) + "(" + a + ")";
    if (tb.isScalar() && ta.isVector()) b = vecName(BT::Double, ta.vec) + "(" + b + ")";
    bool vec = ta.isVector() || tb.isVector();
    switch (op) {
        case Op::Add: return "mgl_f64_add(" + a + ", " + b + ")";
        case Op::Sub: return "mgl_f64_sub(" + a + ", " + b + ")";
        case Op::Mul: return "mgl_f64_mul(" + a + ", " + b + ")";
        case Op::Div: return "mgl_f64_div(" + a + ", " + b + ")";
        case Op::Lt: return "mgl_f64_lt(" + a + ", " + b + ")";
        case Op::Gt: return "mgl_f64_gt(" + a + ", " + b + ")";
        case Op::Le: return "mgl_f64_le(" + a + ", " + b + ")";
        case Op::Ge: return "mgl_f64_ge(" + a + ", " + b + ")";
        case Op::Eq: return vec ? "all(mgl_f64_eq(" + a + ", " + b + "))" : "mgl_f64_eq(" + a + ", " + b + ")";
        case Op::Ne: return vec ? "(!all(mgl_f64_eq(" + a + ", " + b + ")))" : "mgl_f64_ne(" + a + ", " + b + ")";
        default: fail("invalid operation on doubles");
    }
}

std::string Emitter::binary(Op op, const std::string& a, const Type& ta, const std::string& b, const Type& tb, const Type& rt) {
    bool ma = ta.isMatrix(), mb = tb.isMatrix();
    if ((ta.base == BT::Double || tb.base == BT::Double) && op != Op::LogAnd && op != Op::LogOr && op != Op::LogXor)
        return binaryF64(op, a, ta, b, tb);
    switch (op) {
        case Op::Add:
            if (ma != mb) return "mgl_madd(" + a + ", " + b + ")";
            return "(" + a + " + " + b + ")";
        case Op::Sub:
            if (ma != mb) return "mgl_msub(" + a + ", " + b + ")";
            return "(" + a + " - " + b + ")";
        case Op::Mul: return "(" + a + " * " + b + ")";
        case Op::Div:
            if (ma || mb) return "mgl_mdiv(" + a + ", " + b + ")";
            return "(" + a + " / " + b + ")";
        case Op::Mod: return "(" + a + " % " + b + ")";
        case Op::Shl: return "(" + a + " << " + b + ")";
        case Op::Shr: return "(" + a + " >> " + b + ")";
        case Op::BitAnd: return "(" + a + " & " + b + ")";
        case Op::BitOr: return "(" + a + " | " + b + ")";
        case Op::BitXor: return "(" + a + " ^ " + b + ")";
        case Op::LogAnd: return "(" + a + " && " + b + ")";
        case Op::LogOr: return "(" + a + " || " + b + ")";
        case Op::LogXor: return "(" + a + " != " + b + ")";
        case Op::Lt: return "(" + a + " < " + b + ")";
        case Op::Gt: return "(" + a + " > " + b + ")";
        case Op::Le: return "(" + a + " <= " + b + ")";
        case Op::Ge: return "(" + a + " >= " + b + ")";
        case Op::Eq:
            if (ta.isScalar()) return "(" + a + " == " + b + ")";
            needEq_ = true;
            return "mgl_eq(" + a + ", " + b + ")";
        case Op::Ne:
            if (ta.isScalar()) return "(" + a + " != " + b + ")";
            needEq_ = true;
            return "(!mgl_eq(" + a + ", " + b + "))";
        default: fail("internal: bad binary op");
    }
}

static Op compoundBase(Op op) {
    switch (op) {
        case Op::AddAssign: return Op::Add;
        case Op::SubAssign: return Op::Sub;
        case Op::MulAssign: return Op::Mul;
        case Op::DivAssign: return Op::Div;
        case Op::ModAssign: return Op::Mod;
        case Op::ShlAssign: return Op::Shl;
        case Op::ShrAssign: return Op::Shr;
        case Op::AndAssign: return Op::BitAnd;
        case Op::OrAssign: return Op::BitOr;
        case Op::XorAssign: return Op::BitXor;
        default: return Op::None;
    }
}

std::string Emitter::storeTo(const Expr* l, const std::string& v) {
    Acc a;
    if (bufAccess(l, a)) return storeAcc(a, v);
    if (l->kind == EK::Swizzle && isBufferRooted(l->args[0])) {
        const Expr* base = l->args[0];
        std::string t = newTemp(base->type);
        std::string sw;
        for (int i = 0; i < l->nswz; i++) sw += kSwz[l->swz[i]];
        return "(" + t + " = " + ex(base) + ", " + t + "." + sw + " = " + v + ", " + storeTo(base, t) + ")";
    }
    if (l->kind == EK::Index && l->args[0]->type.isVector() && isBufferRooted(l->args[0])) {
        // vector element of a buffer vector reached through a swizzle chain is handled by bufAccess; this is a fallback
        fail("unsupported buffer element store");
    }
    return "(" + ex(l) + " = " + v + ")";
}

std::string Emitter::assign(const Expr* e) {
    const Expr* l = e->args[0];
    const Expr* r = e->args[1];
    bool buf = isBufferRooted(l);
    if (e->op == Op::Assign) {
        if (!buf) return "(" + ex(l) + " = " + ex(r) + ")";
        std::string t = newTemp(l->type);
        return "(" + t + " = " + ex(r) + ", " + storeTo(l, t) + ", " + t + ")";
    }
    Op bop = compoundBase(e->op);
    if (!buf) {
        bool expand = l->type.isMatrix() || r->type.isMatrix() || l->type.base == BT::Double ||
                      (bop == Op::Mul && l->type.isVector() && r->type.isMatrix());
        if (!expand) {
            static const std::map<Op, const char*> s = {{Op::Add, "+="}, {Op::Sub, "-="}, {Op::Mul, "*="}, {Op::Div, "/="},
                                                        {Op::Mod, "%="}, {Op::Shl, "<<="}, {Op::Shr, ">>="}, {Op::BitAnd, "&="},
                                                        {Op::BitOr, "|="}, {Op::BitXor, "^="}};
            return "(" + ex(l) + " " + s.at(bop) + " " + ex(r) + ")";
        }
        std::string L = ex(l);
        return "(" + L + " = " + binary(bop, L, l->type, ex(r), r->type, l->type) + ")";
    }
    std::string t = newTemp(l->type);
    return "(" + t + " = " + binary(bop, ex(l), l->type, ex(r), r->type, l->type) + ", " + storeTo(l, t) + ", " + t + ")";
}

std::string Emitter::incdec(const Expr* e) {
    const Expr* a = e->args[0];
    bool inc = e->op == Op::PreInc || e->op == Op::PostInc;
    bool pre = e->op == Op::PreInc || e->op == Op::PreDec;
    if (a->type.base == BT::Double) {
        std::string fn = inc ? "mgl_f64_add(" : "mgl_f64_sub(";
        std::string t = newTemp(a->type);
        std::string cur = ex(a);
        if (pre) return "(" + t + " = " + fn + cur + ", " + f64one(a->type) + "), " + storeTo(a, t) + ", " + t + ")";
        return "(" + t + " = " + cur + ", " + storeTo(a, fn + t + ", " + f64one(a->type) + ")") + ", " + t + ")";
    }
    if (!isBufferRooted(a)) return pre ? "(" + std::string(inc ? "++" : "--") + ex(a) + ")" : "(" + ex(a) + (inc ? "++" : "--") + ")";
    std::string t = newTemp(a->type);
    std::string one = mt(a->type) + "(1)";
    std::string opS = inc ? " + " : " - ";
    if (pre) return "(" + t + " = " + ex(a) + opS + one + ", " + storeTo(a, t) + ", " + t + ")";
    return "(" + t + " = " + ex(a) + ", " + storeTo(a, t + opS + one) + ", " + t + ")";
}

std::string Emitter::ex(const Expr* e) {
    if (e->cv && (e->type.isNumeric()) && e->kind != EK::Assign && e->kind != EK::Call) {
        size_t i = 0;
        return lit(e->type, *e->cv, i);
    }
    switch (e->kind) {
        case EK::Const: {
            size_t i = 0;
            return lit(e->type, *e->cv, i);
        }
        case EK::Var:
        case EK::Index:
        case EK::Field: {
            Acc a;
            if (bufAccess(e, a)) return loadAcc(a);
            if (e->kind == EK::Var) {
                const Variable* v = e->var;
                if (v->builtin == BuiltinVar::DepthRange)
                    return "S_gl_DepthRangeParameters{sys->depthRangeNear, sys->depthRangeFar, sys->depthRangeFar - sys->depthRangeNear}";
                if (v->builtin == BuiltinVar::NumSamples) return "int(get_num_samples())";
                if (v->builtin == BuiltinVar::WorkGroupSize) {
                    const auto& r = P->refl;
                    return "uint3(" + std::to_string(r.localSize[0]) + "u, " + std::to_string(r.localSize[1]) + "u, " +
                           std::to_string(r.localSize[2]) + "u)";
                }
                if (v->type.isOpaque()) fail("opaque variable '" + v->name + "' used as a value");
                if (v->constVal && v->q.storage == Storage::Const && !v->param) {
                    size_t i = 0;
                    return lit(v->type, *v->constVal, i);
                }
                return varName(v);
            }
            if (e->kind == EK::Field) return ex(e->args[0]) + ".m_" + e->args[0]->type.sdef->members[e->field].name;
            const Expr* b = e->args[0];
            std::string idx = ex(e->args[1]);
            if (b->type.isOpaque() || b->type.stripArrays().isOpaque()) fail("opaque array used as a value");
            if (b->type.isVector() && !b->type.isArray() && rootInThreadgroup(b)) {
                // Address the component as a scalar so concurrent invocations writing different
                // components of one shared vector don't race through a whole-vector store.
                return "((threadgroup " + scalarName(b->type.base) + "*)&(" + ex(b) + "))[" + idx + "]";
            }
            return ex(b) + "[" + idx + "]";
        }
        case EK::Swizzle: {
            const Expr* b = e->args[0];
            std::string s = ex(b);
            if (b->type.isScalar()) {
                if (e->nswz == 1) return s;
                return mt(e->type) + "(" + s + ")";
            }
            std::string sw;
            for (int i = 0; i < e->nswz; i++) sw += kSwz[e->swz[i]];
            return s + "." + sw;
        }
        case EK::Binary:
            return binary(e->op, ex(e->args[0]), e->args[0]->type, ex(e->args[1]), e->args[1]->type, e->type);
        case EK::Unary: {
            const Expr* a = e->args[0];
            switch (e->op) {
                case Op::Neg:
                    if (a->type.base == BT::Double) return (a->type.isMatrix() ? "mgl_dmat_neg(" : "mgl_f64_neg(") + ex(a) + ")";
                    if (a->type.isMatrix()) return "mgl_mneg(" + ex(a) + ")";
                    return "(-" + ex(a) + ")";
                case Op::Plus: return ex(a);
                case Op::Not: return "(!" + ex(a) + ")";
                case Op::BitNot: return "(~" + ex(a) + ")";
                default: return incdec(e);
            }
        }
        case EK::Assign: return assign(e);
        case EK::Ternary:
            return "(" + ex(e->args[0]) + " ? " + ex(e->args[1]) + " : " + ex(e->args[2]) + ")";
        case EK::Comma: return "(" + ex(e->args[0]) + ", " + ex(e->args[1]) + ")";
        case EK::Ctor: return ctor(e);
        case EK::Call: return callUser(e);
        case EK::Builtin: return callBuiltin(e);
        case EK::SubroutineCall: {
            const auto& ss = P->refl.subroutines[(int)stage_];
            auto it = P->subroutineLocation.find({(int)stage_, e->var->name});
            if (ss.slot < 0 || it == P->subroutineLocation.end()) fail("internal: subroutine uniform without location");
            std::string sel = "A->s" + std::to_string(ss.slot) + "[" + std::to_string(it->second) + "u" +
                              (e->aux ? " + uint(" + ex(e->aux) + ")" : "") + "]";
            std::vector<uint8_t> outs;
            std::vector<Type> ptypes;
            for (auto* p : e->fn->params) {
                outs.push_back(p->q.storage == Storage::Out ? 1 : p->q.storage == Storage::InOut ? 2 : 0);
                ptypes.push_back(p->type);
            }
            return withOuts("mgl_sub_" + e->var->name, e, outs, ptypes, e->type, [&](size_t, const Expr* a) { return ex(a); }, sel);
        }
        case EK::Length: {
            const Expr* a = e->args[0];
            Acc acc;
            if (bufAccess(a, acc) && acc.type.isArray() && acc.type.dims[0] == 0) {
                if (acc.sizeExpr.empty()) fail("length() of runtime array outside a storage block");
                return "int((uint(" + acc.sizeExpr + ") - " + acc.off + ") / " + std::to_string(acc.layout->arrayStride) + "u)";
            }
            const Type& t = a->type;
            int n = t.isArray() ? t.dims[0] : t.isMatrix() ? t.cols : t.vec;
            return std::to_string(n);
        }
    }
    fail("internal: unhandled expression");
}

std::string Emitter::ctor(const Expr* e) {
    const Type& T = e->type;
    const auto& args = e->args;
    if (T.isArray()) {
        std::string r = mt(T) + "{{";
        for (size_t i = 0; i < args.size(); i++) r += (i ? ", " : "") + ex(args[i]);
        return r + "}}";
    }
    if (T.base == BT::Struct) {
        std::string r = mt(T) + "{";
        for (size_t i = 0; i < args.size(); i++) r += (i ? ", " : "") + ex(args[i]);
        return r + "}";
    }
    std::string sname = scalarName(T.base);
    if (T.isScalar()) {
        const Expr* a = args[0];
        std::string s = ex(a);
        if (a->type.isVector()) s += ".x";
        else if (a->type.isMatrix()) s += "[0][0]";
        if (a->type.base == T.base && a->type.isScalar()) return s;
        return conv(s, a->type.scalarType(), T);
    }
    int n = T.components();
    if (args.size() == 1) {
        const Expr* a = args[0];
        const Type& at = a->type;
        std::string CR = std::to_string(T.cols) + ", " + std::to_string(T.vec);
        if (at.isScalar()) {
            std::string sc = conv(ex(a), at, T.scalarType());
            if (T.isMatrix()) {
                if (T.base == BT::Double) return "mgl_dmat_diag<" + CR + ">(" + sc + ")";
                return "mgl_mdiag<" + sname + ", " + CR + ">(" + sc + ")";
            }
            return mt(T) + "(" + sc + ")";
        }
        if (T.isVector() && at.isVector()) {
            if (at.vec == T.vec) return at.base == T.base ? ex(a) : conv(ex(a), at, T);
            if (at.vec > T.vec) return conv(ex(a) + "." + std::string(kSwz, T.vec), Type::vector(at.base, T.vec), T);
        }
        if (T.isMatrix() && at.isMatrix()) {
            std::string src = conv(ex(a), at, at.withBase(T.base));
            if (at.cols == T.cols && at.vec == T.vec) return src;
            if (T.base == BT::Double) return "mgl_dmat_resize<" + CR + ">(" + src + ")";
            return "mgl_mresize<" + sname + ", " + CR + ">(" + src + ")";
        }
    }
    bool anyMat = false;
    int total = 0;
    for (auto* a : args) {
        anyMat |= a->type.isMatrix();
        total += a->type.components();
    }
    if (!anyMat && T.isVector()) {
        std::string r = mt(T) + "(";
        int used = 0;
        for (size_t i = 0; i < args.size() && used < n; i++) {
            const Type& at = args[i]->type;
            int take = std::min(at.components(), n - used);
            std::string s = ex(args[i]);
            if (at.isVector() && take < at.vec) s += "." + std::string(kSwz, take);
            Type part = take == 1 ? at.scalarType() : Type::vector(at.base, take);
            Type tpart = take == 1 ? T.scalarType() : Type::vector(T.base, take);
            r += (i ? ", " : "") + (at.base == T.base ? s : conv(s, part, tpart));
            used += take;
        }
        return r + ")";
    }
    if (T.isMatrix() && !anyMat && (int)args.size() == T.cols) {
        bool cols = true;
        for (auto* a : args) if (!(a->type.isVector() && a->type.vec == T.vec)) cols = false;
        if (cols) {
            bool d = T.base == BT::Double;
            std::string r = mt(T) + (d ? "{{" : "(");
            for (size_t i = 0; i < args.size(); i++) r += (i ? ", " : "") + conv(ex(args[i]), args[i]->type, T.column());
            return r + (d ? "}}" : ")");
        }
    }
    // general: flatten components
    std::string pre;
    std::vector<std::string> comps;
    for (auto* a : args) {
        std::string s = bind(ex(a), mt(a->type), pre);
        const Type& at = a->type;
        if (at.isScalar()) comps.push_back(s);
        else if (at.isVector()) for (int k = 0; k < at.vec; k++) comps.push_back(s + "." + kSwz[k]);
        else for (int c = 0; c < at.cols; c++) for (int r = 0; r < at.vec; r++) comps.push_back(s + "[" + std::to_string(c) + "][" + std::to_string(r) + "]");
    }
    if ((int)comps.size() < n) fail("not enough constructor components");
    std::string r;
    std::vector<BT> compBase;
    for (auto* a : args)
        for (int k = 0; k < a->type.components(); k++) compBase.push_back(a->type.base);
    auto cc = [&](int k) { return conv(comps[k], Type::scalar(compBase[k]), T.scalarType()); };
    if (T.isVector()) {
        r = mt(T) + "(";
        for (int k = 0; k < n; k++) r += (k ? ", " : "") + cc(k);
        r += ")";
    } else {
        bool d = T.base == BT::Double;
        r = mt(T) + (d ? "{{" : "(");
        for (int c = 0; c < T.cols; c++) {
            r += (c ? ", " : "") + vecName(T.base, T.vec) + "(";
            for (int k = 0; k < T.vec; k++) r += (k ? ", " : "") + cc(c * T.vec + k);
            r += ")";
        }
        r += d ? "}}" : ")";
    }
    (void)total;
    return pre.empty() ? r : "(" + pre + r + ")";
}

std::string Emitter::withOuts(const std::string& callee, const Expr* e, const std::vector<uint8_t>& outs,
                              const std::vector<Type>& ptypes, const Type& ret,
                              const std::function<std::string(size_t, const Expr*)>& argFn, const std::string& firstArg) {
    std::string pre, post, args = firstArg;
    for (size_t i = 0; i < e->args.size(); i++) {
        const Expr* a = e->args[i];
        std::string s;
        if (i < outs.size() && outs[i]) {
            if (addressable(a)) s = ex(a);
            else {
                std::string t = newTemp(ptypes[i]);
                if (outs[i] == 2) pre += t + " = " + ex(a) + ", ";
                post += storeTo(a, t) + ", ";
                s = t;
            }
        } else {
            s = argFn(i, a);
        }
        if (!s.empty()) args += (args.empty() ? "" : ", ") + s;  // firstArg (if any) already in args
    }
    std::string call = callee + "(" + args + ")";
    if (post.empty()) return pre.empty() ? call : "(" + pre + call + ")";
    post.resize(post.size() - 2);
    if (ret.isVoid()) return "(" + pre + call + ", " + post + ")";
    std::string r = newTemp(ret);
    return "(" + pre + r + " = " + call + ", " + post + ", " + r + ")";
}

// Argument slot holding the emulated border color of an integer sampler uniform, or ""
// (sampler function parameters, dynamically indexed arrays, struct members).
std::string Emitter::borderRef(const Expr* e) {
    const Expr* v = e;
    int idx = 0;
    if (e->kind == EK::Index) {
        if (!e->args[1]->cv) return "";
        idx = (int)e->args[1]->cv->c[0].i;
        v = e->args[0];
    }
    if (v->kind != EK::Var || v->var->param) return "";
    auto it = P->opaque.find(v->var->name);
    if (it == P->opaque.end() || it->second.borderSlot < 0 || idx < 0 || idx >= it->second.count) return "";
    return "A->s" + std::to_string(it->second.borderSlot + idx);
}

std::pair<std::string, std::string> Emitter::texRef(const Expr* e) {
    // opaque members of uniform structs: resolve the constant access path to its binding
    if (e->kind == EK::Field || (e->kind == EK::Index && e->args[0]->kind == EK::Field)) {
        const Expr* leaf = e->kind == EK::Field ? e : e->args[0];
        std::function<bool(const Expr*, std::string&)> path = [&](const Expr* y, std::string& out) -> bool {
            if (y->kind == EK::Var) { out = y->var->name; return !y->var->param; }
            if (y->kind == EK::Field) {
                if (!path(y->args[0], out)) return false;
                out += "." + y->args[0]->type.sdef->members[y->field].name;
                return true;
            }
            if (y->kind == EK::Index) {
                if (!y->args[1]->cv) fail("dynamic indexing of uniform structures containing opaque types is not supported");
                if (!path(y->args[0], out)) return false;
                out += "[" + std::to_string(y->args[1]->cv->c[0].i) + "]";
                return true;
            }
            return false;
        };
        std::string p;
        if (!path(leaf, p)) fail("unsupported opaque member expression");
        auto it = P->opaque.find(p);
        if (it == P->opaque.end()) fail("internal: opaque uniform '" + p + "' has no binding");
        const OpaqueBinding& ob = it->second;
        std::string t = "A->s" + std::to_string(ob.slot), sm = ob.samplerSlot >= 0 ? "A->s" + std::to_string(ob.samplerSlot) : "";
        if (e->kind == EK::Index) {
            std::string idx = ex(e->args[1]);
            return {t + "[" + idx + "]", sm.empty() ? "" : sm + "[" + idx + "]"};
        }
        return {t, sm};
    }
    if (e->kind == EK::Var) {
        const Variable* v = e->var;
        if (v->param) {
            std::string n = varName(v);
            return {n + "_t", n + "_s"};
        }
        auto it = P->opaque.find(v->name);
        if (it == P->opaque.end()) fail("internal: opaque uniform '" + v->name + "' has no binding");
        const OpaqueBinding& ob = it->second;
        return {"A->s" + std::to_string(ob.slot), ob.samplerSlot >= 0 ? "A->s" + std::to_string(ob.samplerSlot) : ""};
    }
    if (e->kind == EK::Index && e->args[0]->kind == EK::Var) {
        auto r = texRef(e->args[0]);
        std::string idx = ex(e->args[1]);
        return {r.first + "[" + idx + "]", r.second.empty() ? "" : r.second + "[" + idx + "]"};
    }
    fail("unsupported sampler/image expression");
}

std::string Emitter::callUser(const Expr* e) {
    auto it = sc_->defs.find(e->fn->mangled);
    if (it == sc_->defs.end()) fail("internal: undefined function " + e->fn->name);
    const Function* f = it->second;
    std::vector<uint8_t> outs;
    std::vector<Type> ptypes;
    for (auto* p : f->params) {
        outs.push_back(p->q.storage == Storage::Out ? 1 : p->q.storage == Storage::InOut ? 2 : 0);
        ptypes.push_back(p->type);
    }
    return withOuts(fnNames_.at(f), e, outs, ptypes, f->ret, [&](size_t i, const Expr* a) -> std::string {
        if (a->type.base == BT::Sampler) {
            auto r = texRef(a);
            return r.first + ", " + r.second;
        }
        if (a->type.base == BT::Image) return texRef(a).first;
        if (a->type.base == BT::AtomicUint) {
            std::string pre;
            return atomicPtr(a, pre);
        }
        return ex(a);
    });
}

std::string Emitter::atomicPtr(const Expr* a, std::string& pre) {
    if (a->type.base == BT::AtomicUint) {
        const Expr* base = a;
        std::string idx;
        if (a->kind == EK::Index) { base = a->args[0]; idx = "uint(" + ex(a->args[1]) + ")"; }
        if (base->kind != EK::Var) fail("unsupported atomic counter expression");
        const Variable* v = base->var;
        if (v->param) return varName(v);
        const OpaqueBinding& ob = P->opaque.at(v->name);
        std::string off = std::to_string(ob.acOffset) + "u";
        if (!idx.empty()) off = "(" + off + " + " + idx + " * 4u)";
        return "((device atomic_uint*)(A->s" + std::to_string(ob.slot) + " + " + off + "))";
    }
    std::string T = a->type.base == BT::Int ? "atomic_int" : "atomic_uint";
    if (a->kind == EK::Swizzle && a->nswz == 1) {
        // one component of a shared-memory vector (MSL cannot take the address of a swizzle)
        const Expr* r = a->args[0];
        while (r->kind == EK::Index || r->kind == EK::Field) r = r->args[0];
        if (r->kind == EK::Var && inThreadgroup(r->var))
            return "((threadgroup " + T + "*)&(" + ex(a->args[0]) + ") + " + std::to_string(a->swz[0]) + ")";
    }
    Acc acc;
    if (a->kind == EK::Swizzle && a->nswz == 1 && bufAccess(a->args[0], acc) && acc.type.isVector()) {
        // single component of a buffer vector, e.g. atomicAdd(buf.v.y, 1)
        uint32_t step = acc.compStride ? acc.compStride : scalarSize(acc.type.base);
        if (!acc.device) fail("atomic operation on a uniform");
        return "((device " + T + "*)(" + acc.ptr + " + (" + acc.off + " + " + std::to_string(a->swz[0] * step) + "u)))";
    }
    acc = Acc();
    if (bufAccess(a, acc)) {
        if (!acc.device) fail("atomic operation on a uniform");
        return "((device " + T + "*)(" + acc.ptr + " + " + acc.off + "))";
    }
    const Expr* r = a;
    while (r->kind == EK::Index || r->kind == EK::Field || r->kind == EK::Swizzle) r = r->args[0];
    if (r->kind == EK::Var && inThreadgroup(r->var)) return "((threadgroup " + T + "*)&(" + ex(a) + "))";
    fail("atomic operation requires a buffer or shared variable");
}

// ---------------------------------------------------------------------------
// Builtin functions
// ---------------------------------------------------------------------------
std::string Emitter::callBuiltin(const Expr* e) {
    const std::string& n = e->fn->name;
    const Builtin* bi = e->fn->builtin;
    const auto& A = e->args;
    const Type& rt = e->type;
    auto a = [&](size_t i) { return ex(A[i]); };
    auto bc = [&](size_t i) {  // broadcast scalar arg to the result vector shape
        std::string s = ex(A[i]);
        if (rt.isVector() && A[i]->type.isScalar()) return vecName(A[i]->type.base, rt.vec) + "(" + s + ")";
        return s;
    };
    auto simple = [&](const char* f) {
        std::string r = std::string(f) + "(";
        for (size_t i = 0; i < A.size(); i++) r += (i ? ", " : "") + a(i);
        return r + ")";
    };
    if (n.compare(0, 7, "texture") == 0 || n.compare(0, 6, "shadow") == 0 || n.compare(0, 10, "texelFetch") == 0)
        return textureFn(e, n);
    bool anyDouble = rt.stripArrays().base == BT::Double;
    for (auto* x : A) anyDouble |= x->type.stripArrays().base == BT::Double;
    if (anyDouble) return builtinF64(e);
    if (n.compare(0, 5, "image") == 0) return imageFn(e, n);
    static const std::unordered_map<std::string, const char*> direct = {
        {"sin", "sin"}, {"cos", "cos"}, {"tan", "tan"}, {"asin", "asin"}, {"acos", "acos"}, {"sinh", "sinh"}, {"cosh", "cosh"},
        {"tanh", "tanh"}, {"asinh", "asinh"}, {"acosh", "acosh"}, {"atanh", "atanh"}, {"pow", "pow"}, {"exp", "exp"},
        {"log", "log"}, {"exp2", "exp2"}, {"log2", "log2"}, {"sqrt", "sqrt"}, {"inversesqrt", "rsqrt"}, {"abs", "abs"},
        {"floor", "floor"}, {"trunc", "trunc"}, {"round", "round"}, {"roundEven", "rint"}, {"ceil", "ceil"},
        {"fract", "fract"}, {"isnan", "isnan"}, {"isinf", "isinf"}, {"fma", "fma"}, {"cross", "cross"}, {"transpose", "transpose"},
        {"determinant", "determinant"}, {"any", "any"}, {"all", "all"}, {"bitfieldReverse", "reverse_bits"},
        {"dFdx", "dfdx"}, {"dFdy", "dfdy"}, {"dFdxFine", "dfdx"}, {"dFdyFine", "dfdy"}, {"dFdxCoarse", "dfdx"},
        {"dFdyCoarse", "dfdy"}, {"fwidth", "fwidth"}, {"fwidthFine", "fwidth"}, {"fwidthCoarse", "fwidth"},
        {"packUnorm2x16", "pack_float_to_unorm2x16"}, {"packSnorm2x16", "pack_float_to_snorm2x16"},
        {"packUnorm4x8", "pack_float_to_unorm4x8"}, {"packSnorm4x8", "pack_float_to_snorm4x8"},
        {"unpackUnorm2x16", "unpack_unorm2x16_to_float"}, {"unpackSnorm2x16", "unpack_snorm2x16_to_float"},
        {"unpackUnorm4x8", "unpack_unorm4x8_to_float"}, {"unpackSnorm4x8", "unpack_snorm4x8_to_float"},
        {"matrixCompMult", "mgl_mcm"}, {"outerProduct", "mgl_outer"}, {"inverse", "mgl_inverse"},
        {"findLSB", "mgl_findLSB"}, {"findMSB", "mgl_findMSB"}, {"packDouble2x32", "mgl_pack_f64"},
        {"unpackDouble2x32", "mgl_unpack_f64"}, {"anyInvocation", "simd_any"}, {"allInvocations", "simd_all"},
        {"anyInvocationARB", "simd_any"}, {"allInvocationsARB", "simd_all"},
    };
    if (auto it = direct.find(n); it != direct.end()) return simple(it->second);
    if (n == "sign") return (A[0]->type.base == BT::Int ? "mgl_isign(" : "sign(") + a(0) + ")";
    if (n == "radians") return "(" + a(0) + " * 0.0174532925199432958f)";
    if (n == "degrees") return "(" + a(0) + " * 57.2957795130823209f)";
    if (n == "atan") return A.size() == 2 ? "atan2(" + a(0) + ", " + a(1) + ")" : "atan(" + a(0) + ")";
    if (n == "mod") return "mgl_mod(" + bc(0) + ", " + bc(1) + ")";
    if (n == "min" || n == "max" || n == "clamp" || n == "step" || n == "smoothstep") {
        std::string r = n + "(";
        for (size_t i = 0; i < A.size(); i++) r += (i ? ", " : "") + bc(i);
        return r + ")";
    }
    if (n == "mix") {
        if (A[2]->type.base == BT::Bool) return "select(" + a(0) + ", " + a(1) + ", " + a(2) + ")";
        return "mix(" + bc(0) + ", " + bc(1) + ", " + bc(2) + ")";
    }
    if (n == "floatBitsToInt") return "as_type<" + mt(rt) + ">(" + a(0) + ")";
    if (n == "floatBitsToUint") return "as_type<" + mt(rt) + ">(" + a(0) + ")";
    if (n == "intBitsToFloat" || n == "uintBitsToFloat") return "as_type<" + mt(rt) + ">(" + a(0) + ")";
    if (n == "packHalf2x16") return "as_type<uint>(half2(" + a(0) + "))";
    if (n == "unpackHalf2x16") return "float2(as_type<half2>(" + a(0) + "))";
    if (n == "length") return A[0]->type.isScalar() ? "abs(" + a(0) + ")" : simple("length");
    if (n == "distance") return A[0]->type.isScalar() ? "abs(" + a(0) + " - " + a(1) + ")" : simple("distance");
    if (n == "dot") return A[0]->type.isScalar() ? "(" + a(0) + " * " + a(1) + ")" : simple("dot");
    if (n == "normalize") return A[0]->type.isScalar() ? "sign(" + a(0) + ")" : simple("normalize");
    if (n == "faceforward") {
        if (A[0]->type.isScalar()) return "((" + a(2) + " * " + a(1) + ") < 0.0f ? " + a(0) + " : -" + a(0) + ")";
        return simple("faceforward");
    }
    if (n == "reflect") {
        if (A[0]->type.isScalar()) return "(" + a(0) + " - 2.0f * " + a(1) + " * " + a(1) + " * " + a(0) + ")";
        return simple("reflect");
    }
    if (n == "refract") {
        if (A[0]->type.isScalar()) fail("scalar refract() is not supported");
        return simple("refract");
    }
    if (n == "lessThan") return "(" + a(0) + " < " + a(1) + ")";
    if (n == "lessThanEqual") return "(" + a(0) + " <= " + a(1) + ")";
    if (n == "greaterThan") return "(" + a(0) + " > " + a(1) + ")";
    if (n == "greaterThanEqual") return "(" + a(0) + " >= " + a(1) + ")";
    if (n == "equal") return "(" + a(0) + " == " + a(1) + ")";
    if (n == "notEqual") return "(" + a(0) + " != " + a(1) + ")";
    if (n == "not") return "(!" + a(0) + ")";
    if (n == "bitfieldExtract") return "extract_bits(" + a(0) + ", uint(" + a(1) + "), uint(" + a(2) + "))";
    if (n == "bitfieldInsert") return "insert_bits(" + a(0) + ", " + a(1) + ", uint(" + a(2) + "), uint(" + a(3) + "))";
    if (n == "bitCount") return mt(rt) + "(popcount(" + a(0) + "))";
    if (n == "interpolateAtCentroid" || n == "interpolateAtSample" || n == "interpolateAtOffset") {
        const Expr* x = A[0];
        std::string post;
        if (x->kind == EK::Swizzle) {
            post = ".";
            for (int i = 0; i < x->nswz; i++) post += kSwz[x->swz[i]];
            if (x->args[0]->type.isScalar()) post.clear();
            x = x->args[0];
        } else if (x->kind == EK::Index && x->args[0]->type.isVector() && x->args[1]->cv) {
            post = "[" + std::to_string(x->args[1]->cv->c[0].i) + "]";
            x = x->args[0];
        }
        std::function<bool(const Expr*, std::string&)> path = [&](const Expr* y, std::string& out) -> bool {
            if (y->kind == EK::Var) { out = varName(y->var); return true; }
            if (y->kind == EK::Field) {
                if (!path(y->args[0], out)) return false;
                out += ".m_" + y->args[0]->type.sdef->members[y->field].name;
                return true;
            }
            if (y->kind == EK::Index && y->args[1]->cv && !y->args[0]->type.isVector()) {
                if (!path(y->args[0], out)) return false;
                out += "[" + std::to_string(y->args[1]->cv->c[0].i) + "]";
                return true;
            }
            return false;
        };
        std::string pth;
        auto it = path(x, pth) ? interpLeaf_.find(pth) : interpLeaf_.end();
        if (it == interpLeaf_.end()) return a(0);  // not resolvable to one interpolant: value at the default location
        std::string call = "mgl_in->" + it->second;
        if (n == "interpolateAtCentroid") call += ".interpolate_at_centroid()";
        else if (n == "interpolateAtSample") call += ".interpolate_at_sample(uint(" + a(1) + "))";
        else call += ".interpolate_at_offset(" + a(1) + " + 0.5f)";  // Metal offsets are relative to the pixel's top-left corner
        return call + post;
    }
    if (n == "noise1") return "0.0f";
    if (n == "noise2" || n == "noise3" || n == "noise4") return mt(rt) + "(0.0f)";
    if (n == "allInvocationsEqual" || n == "allInvocationsEqualARB") {
        std::string pre;
        std::string v = bind(a(0), "bool", pre);
        // simd_broadcast_first has no bool overload
        return "(" + pre + "simd_all(int(" + v + ") == simd_broadcast_first(int(" + v + "))))";
    }
    if (n == "modf" || n == "frexp" || n == "uaddCarry" || n == "usubBorrow" || n == "umulExtended" || n == "imulExtended") {
        std::string callee = n == "modf" ? "modf" : n == "frexp" ? "frexp" : n == "uaddCarry" ? "mgl_uaddCarry"
                           : n == "usubBorrow" ? "mgl_usubBorrow" : "mgl_mulExtended";
        return withOuts(callee, e, bi->paramOut, bi->params, rt, [&](size_t i, const Expr* x) { return ex(x); });
    }
    if (n == "ldexp") return "ldexp(" + a(0) + ", " + a(1) + ")";
    if (n == "barrier") {
        if (stage_ == Stage::TessControl) return "threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)";
        if (stage_ == Stage::Compute) return "threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device | mem_flags::mem_texture)";
        return "((void)0)";
    }
    if (n == "memoryBarrier" || n == "memoryBarrierBuffer" || n == "memoryBarrierImage" || n == "memoryBarrierAtomicCounter" ||
        n == "memoryBarrierShared" || n == "groupMemoryBarrier") {
        std::string flags = n == "memoryBarrierShared" ? "mem_flags::mem_threadgroup"
                          : n == "memoryBarrierImage" ? "mem_flags::mem_texture"
                          : n == "memoryBarrier" || n == "groupMemoryBarrier" ? "mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture"
                          : "mem_flags::mem_device";
        if (stage_ != Stage::Compute && n == "memoryBarrierShared") return "((void)0)";
        if (stage_ != Stage::Compute) flags = "mem_flags::mem_device | mem_flags::mem_texture";
        return "atomic_thread_fence(" + flags + ", memory_order_seq_cst)";
    }
    if (n.compare(0, 13, "atomicCounter") == 0) {
        std::string pre;
        std::string p = atomicPtr(A[0], pre);
        if (n == "atomicCounterIncrement") return "atomic_fetch_add_explicit(" + p + ", 1u, memory_order_relaxed)";
        if (n == "atomicCounterDecrement") return "(atomic_fetch_sub_explicit(" + p + ", 1u, memory_order_relaxed) - 1u)";
        if (n == "atomicCounter") return "atomic_load_explicit(" + p + ", memory_order_relaxed)";
        static const std::map<std::string, std::string> ops = {
            {"atomicCounterAdd", "atomic_fetch_add_explicit"}, {"atomicCounterSubtract", "atomic_fetch_sub_explicit"},
            {"atomicCounterMin", "atomic_fetch_min_explicit"}, {"atomicCounterMax", "atomic_fetch_max_explicit"},
            {"atomicCounterAnd", "atomic_fetch_and_explicit"}, {"atomicCounterOr", "atomic_fetch_or_explicit"},
            {"atomicCounterXor", "atomic_fetch_xor_explicit"}, {"atomicCounterExchange", "atomic_exchange_explicit"}};
        if (n == "atomicCounterCompSwap") return "mgl_atomic_cas(" + p + ", " + a(1) + ", " + a(2) + ")";
        return ops.at(n) + "(" + p + ", " + a(1) + ", memory_order_relaxed)";
    }
    if (n.compare(0, 6, "atomic") == 0) {
        std::string pre;
        std::string p = atomicPtr(A[0], pre);
        static const std::map<std::string, std::string> ops = {
            {"atomicAdd", "atomic_fetch_add_explicit"}, {"atomicMin", "atomic_fetch_min_explicit"},
            {"atomicMax", "atomic_fetch_max_explicit"}, {"atomicAnd", "atomic_fetch_and_explicit"},
            {"atomicOr", "atomic_fetch_or_explicit"},   {"atomicXor", "atomic_fetch_xor_explicit"},
            {"atomicExchange", "atomic_exchange_explicit"}};
        if (n == "atomicCompSwap") return "mgl_atomic_cas(" + p + ", " + a(1) + ", " + a(2) + ")";
        return ops.at(n) + "(" + p + ", " + a(1) + ", memory_order_relaxed)";
    }
    if (n == "ftransform") {
        auto it = P->defaultOffsets.find("gl_ModelViewProjectionMatrix");
        if (it == P->defaultOffsets.end()) fail("internal: ftransform without gl_ModelViewProjectionMatrix");
        Type m4 = Type::matrix(BT::Float, 4, 4);
        return "(" + loaderFor(m4, computeLayout(m4, Packing::Std430, false)) + "(A->s" + std::to_string(P->defaultSlot) + ", " +
               std::to_string(it->second) + "u) * bi_gl_Vertex)";
    }
    if (n == "EmitVertex") return "mgl_EmitVertex(0)";
    if (n == "EndPrimitive") return "mgl_EndPrimitive(0)";
    if (n == "EmitStreamVertex") return "mgl_EmitVertex(int(" + a(0) + "))";
    if (n == "EndStreamPrimitive") return "mgl_EndPrimitive(int(" + a(0) + "))";
    fail("builtin function '" + n + "' is not supported yet");
}

std::string Emitter::builtinF64(const Expr* e) {
    const std::string& n = e->fn->name;
    const auto& A = e->args;
    const Type& rt = e->type;
    auto a = [&](size_t i) { return ex(A[i]); };
    auto bc = [&](size_t i) {
        std::string s = ex(A[i]);
        if (rt.isVector() && A[i]->type.isScalar()) return vecName(A[i]->type.base, rt.vec) + "(" + s + ")";
        return s;
    };
    auto call = [&](const std::string& f, bool broadcast) {
        std::string r = f + "(";
        for (size_t i = 0; i < A.size(); i++) r += (i ? ", " : "") + (broadcast ? bc(i) : a(i));
        return r + ")";
    };
    static const std::set<std::string> direct = {"abs", "sign", "floor", "ceil", "trunc", "round", "roundEven", "fract", "sqrt",
                                                 "inversesqrt", "isnan", "isinf", "dot", "length", "distance", "normalize",
                                                 "cross", "faceforward", "reflect", "refract", "fma"};
    if (direct.count(n)) return call("mgl_f64_" + n, false);
    if (n == "mod" || n == "min" || n == "max" || n == "clamp" || n == "step" || n == "smoothstep") return call("mgl_f64_" + n, true);
    if (n == "mix") {
        if (A[2]->type.base == BT::Bool) return "select(" + a(0) + ", " + a(1) + ", " + a(2) + ")";
        return call("mgl_f64_mix", true);
    }
    if (n == "frexp" || n == "modf" || n == "ldexp") {
        const Builtin* bi = e->fn->builtin;
        return withOuts("mgl_f64_" + n, e, bi->paramOut, bi->params, rt, [&](size_t, const Expr* x) { return ex(x); });
    }
    static const std::map<std::string, std::string> rel = {{"lessThan", "lt"}, {"lessThanEqual", "le"}, {"greaterThan", "gt"},
                                                           {"greaterThanEqual", "ge"}, {"equal", "eq"}, {"notEqual", "ne"}};
    if (auto it = rel.find(n); it != rel.end()) return "mgl_f64_" + it->second + "(" + a(0) + ", " + a(1) + ")";
    if (n == "packDouble2x32") return "mgl_f64_pack2x32(" + a(0) + ")";
    if (n == "unpackDouble2x32") return "mgl_f64_unpack2x32(" + a(0) + ")";
    if (n == "transpose") return "mgl_dmat_transpose(" + a(0) + ")";
    if (n == "determinant") return "mgl_dmat_det(" + a(0) + ")";
    if (n == "inverse") return "mgl_dmat_inverse(" + a(0) + ")";
    if (n == "outerProduct") return "mgl_dmat_outer(" + a(0) + ", " + a(1) + ")";
    if (n == "matrixCompMult") return "mgl_dmat_mcm(" + a(0) + ", " + a(1) + ")";
    if (n == "uaddCarry" || n == "any" || n == "all") return call(n, false);
    fail("builtin '" + n + "' is not supported for double arguments");
}

// Legacy texture function names -> (modern name, legacy vec4 shadow result)
static std::pair<std::string, bool> canonicalTexName(const std::string& n) {
    static const std::unordered_map<std::string, std::string> m = {
        {"texture1D", "texture"}, {"texture2D", "texture"}, {"texture3D", "texture"}, {"textureCube", "texture"},
        {"texture2DRect", "texture"}, {"texture1DArray", "texture"}, {"texture2DArray", "texture"},
        {"shadow1D", "texture"}, {"shadow2D", "texture"}, {"shadow2DRect", "texture"}, {"shadow1DArray", "texture"},
        {"shadow2DArray", "texture"},
        {"texture1DProj", "textureProj"}, {"texture2DProj", "textureProj"}, {"texture3DProj", "textureProj"},
        {"texture2DRectProj", "textureProj"}, {"shadow1DProj", "textureProj"}, {"shadow2DProj", "textureProj"},
        {"shadow2DRectProj", "textureProj"},
        {"texture1DLod", "textureLod"}, {"texture2DLod", "textureLod"}, {"texture3DLod", "textureLod"},
        {"textureCubeLod", "textureLod"}, {"shadow1DLod", "textureLod"}, {"shadow2DLod", "textureLod"},
        {"texture2DArrayLod", "textureLod"}, {"texture1DArrayLod", "textureLod"}, {"texture2DLodEXT", "textureLod"},
        {"textureCubeLodEXT", "textureLod"}, {"texture2DLodARB", "textureLod"},
        {"texture1DProjLod", "textureProjLod"}, {"texture2DProjLod", "textureProjLod"}, {"texture3DProjLod", "textureProjLod"},
        {"shadow1DProjLod", "textureProjLod"}, {"shadow2DProjLod", "textureProjLod"}, {"texture2DProjLodEXT", "textureProjLod"},
        {"texture1DGradARB", "textureGrad"}, {"texture2DGradARB", "textureGrad"}, {"texture3DGradARB", "textureGrad"},
        {"textureCubeGradARB", "textureGrad"}, {"shadow1DGradARB", "textureGrad"}, {"shadow2DGradARB", "textureGrad"},
        {"texture2DRectGradARB", "textureGrad"}, {"texture2DGradEXT", "textureGrad"}, {"textureCubeGradEXT", "textureGrad"},
        {"texture1DProjGradARB", "textureProjGrad"}, {"texture2DProjGradARB", "textureProjGrad"},
        {"texture3DProjGradARB", "textureProjGrad"}, {"shadow2DProjGradARB", "textureProjGrad"},
        {"texture2DProjGradEXT", "textureProjGrad"},
        {"texelFetch1D", "texelFetch"}, {"texelFetch2D", "texelFetch"}, {"texelFetch3D", "texelFetch"},
        {"texelFetch2DRect", "texelFetch"}, {"texelFetch1DArray", "texelFetch"}, {"texelFetch2DArray", "texelFetch"},
        {"texelFetchBuffer", "texelFetch"},
        {"textureSize1D", "textureSize"}, {"textureSize2D", "textureSize"}, {"textureSize3D", "textureSize"},
        {"textureSizeCube", "textureSize"}, {"textureSize2DRect", "textureSize"}, {"textureSize2DArray", "textureSize"},
        {"textureSizeBuffer", "textureSize"},
    };
    auto it = m.find(n);
    if (it == m.end()) return {n, false};
    return {it->second, n.compare(0, 6, "shadow") == 0};
}

std::string Emitter::textureFn(const Expr* e, const std::string& rawName) {
    auto [name, legacyShadow] = canonicalTexName(rawName);
    const auto& A = e->args;
    const Type T = A[0]->type.stripArrays();
    auto [tex, smp] = texRef(A[0]);
    std::string pre;
    const bool sh = T.shadow, arr = T.arrayed, rect = T.dim == Dim::Rect;
    const Dim dim = T.dim;
    auto finish = [&](std::string r) {
        if (legacyShadow) {
            std::string v = bind(r, "float", pre);
            r = "float4(" + v + ", " + v + ", " + v + ", 1.0f)";
        }
        return pre.empty() ? r : "(" + pre + r + ")";
    };
    auto lodArg = [&](size_t i) { return "uint(" + ex(A[i]) + ")"; };

    if (name == "textureSize") {
        std::string lod = (A.size() > 1) ? bind(lodArg(1), "uint", pre) : "";
        std::string w = tex + ".get_width(" + lod + ")", h = tex + ".get_height(" + lod + ")";
        std::string r;
        if (T.ms) r = arr ? "int3(" + tex + ".get_width(), " + tex + ".get_height(), " + tex + ".get_array_size())" : "int2(" + tex + ".get_width(), " + tex + ".get_height())";
        else if (dim == Dim::Buffer) r = "int(" + tex + ".get_width())";
        else if (dim == Dim::D1) r = arr ? "int2(" + w + ", " + tex + ".get_array_size())" : "int(" + w + ")";
        else if (dim == Dim::D3) r = "int3(" + w + ", " + h + ", " + tex + ".get_depth(" + lod + "))";
        else r = arr ? "int3(" + w + ", " + h + ", " + tex + ".get_array_size())" : "int2(" + w + ", " + h + ")";
        return finish(r);
    }
    if (name == "textureQueryLevels") return finish("int(" + tex + ".get_num_mip_levels())");
    if (name == "textureSamples") return finish("int(" + tex + ".get_num_samples())");
    if (name == "texelFetch" || name == "texelFetchOffset") {
        std::string P = ex(A[1]);
        bool off = name == "texelFetchOffset";
        if (off) {
            // Arrayed textures: the offset applies to the coordinates, not the layer.
            const Type& Pt = A[1]->type;
            int pn = Pt.isScalar() ? 1 : Pt.vec;
            const Type& Ot = A.back()->type;
            int on = Ot.isScalar() ? 1 : Ot.vec;
            std::string o = ex(A.back());
            if (on < pn) {
                o = vecName(BT::Int, pn) + "(" + o;
                for (int i = on; i < pn; i++) o += ", 0";
                o += ")";
            }
            P = "(" + P + " + " + o + ")";
        }
        std::string r;
        if (T.ms) {
            std::string s = "uint(" + ex(A[2]) + ")";
            if (arr) { P = bind(P, "int3", pre); r = tex + ".read(uint2(" + P + ".xy), uint(" + P + ".z), " + s + ")"; }
            else r = tex + ".read(uint2(" + P + "), " + s + ")";
            return finish(r);
        }
        // Out-of-range texel coordinates or levels read as zero (as robust GL implementations
        // do; Metal's read() is undefined there). Casting to uint folds the < 0 test in.
        if (dim == Dim::Buffer) {
            P = bind(P, "int", pre);
            r = tex + ".read(uint(" + P + "))";
            return finish("(uint(" + P + ") < " + tex + ".get_width() ? " + r + " : decltype(" + r + ")(0))");
        }
        bool hasLod = !(dim == Dim::Rect || A.size() < 3 || (off && A.size() < 4));
        std::string L = hasLod ? bind("uint(" + ex(A[2]) + ")", "uint", pre) : "0u";
        std::string lod = hasLod ? ", " + L : "";
        std::string inb;
        if (dim == Dim::D1) {
            if (arr) {
                P = bind(P, "int2", pre);
                r = tex + ".read(uint2(uint(" + P + ".x), 0u), uint(" + P + ".y)" + lod + ")";
                inb = "uint(" + P + ".x) < " + tex + ".get_width(" + L + ") && uint(" + P + ".y) < " + tex + ".get_array_size()";
            } else {
                P = bind(P, "int", pre);
                r = tex + ".read(uint2(uint(" + P + "), 0u)" + lod + ")";
                inb = "uint(" + P + ") < " + tex + ".get_width(" + L + ")";
            }
        } else if (dim == Dim::D3) {
            P = bind(P, "int3", pre);
            r = tex + ".read(uint3(" + P + ")" + lod + ")";
            inb = "all(uint3(" + P + ") < uint3(" + tex + ".get_width(" + L + "), " + tex + ".get_height(" + L + "), " + tex + ".get_depth(" + L + ")))";
        } else if (arr) {
            P = bind(P, "int3", pre);
            r = tex + ".read(uint2(" + P + ".xy), uint(" + P + ".z)" + lod + ")";
            inb = "all(uint3(" + P + ") < uint3(" + tex + ".get_width(" + L + "), " + tex + ".get_height(" + L + "), " + tex + ".get_array_size()))";
        } else {
            P = bind(P, "int2", pre);
            r = tex + ".read(uint2(" + P + ")" + lod + ")";
            inb = "all(uint2(" + P + ") < uint2(" + tex + ".get_width(" + L + "), " + tex + ".get_height(" + L + ")))";
        }
        if (hasLod) inb = L + " < " + tex + ".get_num_mip_levels() && " + inb;
        return finish("(" + inb + " ? " + r + " : decltype(" + r + ")(0))");
    }
    // --- sampling functions ---
    bool proj = name.find("Proj") != std::string::npos;
    bool isLod = name.find("Lod") != std::string::npos;
    bool isGrad = name.find("Grad") != std::string::npos;
    bool isOffset = name.find("Offset") != std::string::npos;
    bool isGather = name.find("Gather") != std::string::npos;
    bool isQueryLod = name == "textureQueryLod";
    const Type& Pt = A[1]->type;
    int pn = Pt.isScalar() ? 1 : Pt.vec;
    std::string P = bind(ex(A[1]), mt(Pt), pre);
    auto comp = [&](int i) { return pn == 1 ? P : P + "." + kSwz[i]; };
    std::string q = proj ? comp(pn - 1) : "";
    auto div = [&](const std::string& s) { return proj ? "(" + s + " / " + q + ")" : s; };
    std::string coord, layer, ref;
    switch (dim) {
        case Dim::D1:
            coord = "float2(" + div(comp(0)) + ", 0.5f)";
            if (arr) layer = comp(1);
            if (sh) ref = arr ? comp(2) : div(comp(2));
            break;
        case Dim::D2:
        case Dim::Rect:
            coord = div(pn == 2 && !proj ? P : P + ".xy");
            if (arr) layer = comp(2);
            if (sh) ref = arr ? comp(3) : div(comp(2));
            break;
        case Dim::D3:
            coord = div(P + ".xyz");
            break;
        case Dim::Cube:
            coord = P + ".xyz";
            if (arr) layer = comp(3);
            if (sh && !arr) ref = comp(3);
            break;
        default: fail("unsupported sampler for " + rawName);
    }
    if (isQueryLod) {
        coord = dim == Dim::D1 ? "float2(" + comp(0) + ", 0.5f)" : (dim == Dim::D2 || dim == Dim::Rect) ? (pn == 2 ? P : P + ".xy") : P + ".xyz";
        std::string c = bind(coord, (dim == Dim::D3 || dim == Dim::Cube) ? "float3" : "float2", pre);
        return finish("float2(" + tex + ".calculate_clamped_lod(" + smp + ", " + c + "), " + tex + ".calculate_unclamped_lod(" + smp + ", " + c + "))");
    }
    if (rect) coord = "(" + coord + " / float2(" + tex + ".get_width(), " + tex + ".get_height()))";
    if (!layer.empty()) layer = "uint(max(rint(" + layer + "), 0.0f))";
    size_t ai = 2;
    if (sh && dim == Dim::Cube && arr && !isGather) ref = ex(A[ai++]);
    if (isGather) {
        // textureGather(s, P [, refZ | comp]) / textureGatherOffset(s, P, [refZ,] offset [, comp])
        std::string refZ;
        if (sh) refZ = ex(A[ai++]);
        std::string offset;
        if (isOffset) {
            if (name == "textureGatherOffsets") {
                // each result texel k is texel i0j0 (.w) of the footprint selected by offsets[k]
                const Expr* offs = A[ai++];
                if (!offs->cv) fail("textureGatherOffsets offsets must be a constant expression");
                int compIdx2 = 0;
                if (!sh && ai < A.size()) {
                    if (!A[ai]->cv) fail("textureGather component must be a constant expression");
                    compIdx2 = (int)A[ai]->cv->c[0].i;
                }
                std::string r = "float4(";
                if (T.sampled == BT::Int) r = "int4(";
                if (T.sampled == BT::UInt) r = "uint4(";
                for (int k = 0; k < 4; k++) {
                    std::string o2 = "int2(" + std::to_string((int)offs->cv->c[2 * k].i) + ", " + std::to_string((int)offs->cv->c[2 * k + 1].i) + ")";
                    std::string args = smp + ", " + coord;
                    if (!layer.empty()) args += ", " + layer;
                    if (sh) args += ", " + ex(A[2]);
                    args += ", " + o2;
                    if (!sh) args += std::string(", component::") + kSwz[compIdx2];
                    r += (k ? ", " : "") + tex + (sh ? ".gather_compare(" : ".gather(") + args + ").w";
                }
                return finish(r + ")");
            }
            offset = ex(A[ai++]);
        }
        int compIdx = 0;
        if (!sh && ai < A.size()) {
            if (!A[ai]->cv) fail("textureGather component must be a constant expression");
            compIdx = (int)A[ai]->cv->c[0].i;
        }
        std::string args = smp + ", " + coord;
        if (!layer.empty()) args += ", " + layer;
        if (sh) args += ", " + refZ;
        if (dim != Dim::Cube) args += ", " + (offset.empty() ? std::string("int2(0)") : offset);
        if (!sh) args += std::string(", component::") + kSwz[compIdx];
        return finish(tex + (sh ? ".gather_compare(" : ".gather(") + args + ")");
    }
    std::string options, offset;
    if (isLod) options = "level(" + ex(A[ai++]) + ")";
    else if (isGrad) {
        std::string dx = ex(A[ai++]), dy = ex(A[ai++]);
        if (dim == Dim::D1) options = "gradient2d(float2(" + dx + ", 0.0f), float2(" + dy + ", 0.0f))";
        else if (dim == Dim::D3) options = "gradient3d(" + dx + ", " + dy + ")";
        else if (dim == Dim::Cube) options = "gradientcube(" + dx + ", " + dy + ")";
        else options = "gradient2d(" + dx + ", " + dy + ")";
    }
    if (isOffset) {
        std::string o = ex(A[ai++]);
        offset = dim == Dim::D1 ? "int2(" + o + ", 0)" : o;
    }
    if (ai < A.size() && !isLod && !isGrad) options = "bias(" + ex(A[ai++]) + ")";
    std::string args = smp + ", " + coord;
    if (!layer.empty()) args += ", " + layer;
    if (sh) {
        args += ", " + ref;
        if (options.compare(0, 4, "bias") == 0) options.clear();
    }
    if (!options.empty()) args += ", " + options;
    if (!offset.empty()) args += ", " + offset;
    if (!sh && (T.sampled == BT::Int || T.sampled == BT::UInt) && dim != Dim::Cube) {
        std::string br = borderRef(A[0]);
        if (!br.empty()) {
            // Integer textures with GL_CLAMP_TO_BORDER: Metal can only return black/white.
            std::string c = bind(coord, dim == Dim::D3 ? "float3" : "float2", pre);
            args = smp + ", " + c + args.substr(smp.size() + 2 + coord.size());
            return finish("mgl_int_border(" + tex + ".sample(" + args + "), " + c + ", " + br + ")");
        }
    }
    return finish(tex + (sh ? ".sample_compare(" : ".sample(") + args + ")");
}

std::string Emitter::imageFn(const Expr* e, const std::string& n) {
    const auto& A = e->args;
    const Type T = A[0]->type.stripArrays();
    std::string tex = texRef(A[0]).first;
    std::string pre;
    auto finish = [&](const std::string& r) { return pre.empty() ? r : "(" + pre + r + ")"; };
    if (n == "imageSize") {
        std::string w = tex + ".get_width()", h = tex + ".get_height()";
        std::string r;
        if (T.dim == Dim::Buffer) r = "int(" + w + ")";
        else if (T.dim == Dim::D1) r = T.arrayed ? "int2(" + w + ", " + tex + ".get_array_size())" : "int(" + w + ")";
        else if (T.dim == Dim::D3) r = "int3(" + w + ", " + h + ", " + tex + ".get_depth())";
        else if (T.dim == Dim::Cube) r = T.arrayed ? "int3(" + w + ", " + h + ", int(" + tex + ".get_array_size()) / 6)" : "int2(" + w + ", " + h + ")";
        else r = T.arrayed ? "int3(" + w + ", " + h + ", " + tex + ".get_array_size())" : "int2(" + w + ", " + h + ")";
        return r;
    }
    if (n == "imageSamples") return "int(" + tex + ".get_num_samples())";
    std::string P = bind(ex(A[1]), mt(A[1]->type), pre);
    std::string coord, layer;
    bool layered = T.arrayed || T.dim == Dim::Cube;
    switch (T.dim) {
        case Dim::D1:
            coord = T.arrayed ? "uint2(uint(" + P + ".x), 0u)" : "uint2(uint(" + P + "), 0u)";
            if (T.arrayed) layer = "uint(" + P + ".y)";
            break;
        case Dim::D2:
        case Dim::Rect:
        case Dim::Cube:
            coord = layered ? "uint2(" + P + ".xy)" : "uint2(" + P + ")";
            if (layered) layer = "uint(" + P + ".z)";
            break;
        case Dim::D3: coord = "uint3(" + P + ")"; break;
        case Dim::Buffer: coord = "uint(" + P + ")"; break;
        default: fail("bad image");
    }
    size_t ai = 2;
    std::string sample;
    if (T.ms) sample = "uint(" + ex(A[ai++]) + ")";
    std::string loc = coord + (layer.empty() ? "" : ", " + layer);
    if (n == "imageLoad") return finish(tex + ".read(" + loc + (sample.empty() ? "" : ", " + sample) + ")");
    // read_write textures need fence() so the invocation sees its own writes (GLSL guarantees this)
    const Expr* imgRoot = A[0];
    while (imgRoot->kind == EK::Index) imgRoot = imgRoot->args[0];
    uint8_t imgMem = imgRoot->kind == EK::Var ? imgRoot->var->q.memory : 0;
    bool rw = !(imgMem & (MemReadOnly | MemWriteOnly));
    if (n == "imageStore") {
        // Metal cannot write multisample textures: the store is dropped (the program still links).
        if (T.ms) return finish("((void)(" + ex(A[ai]) + "))");
        std::string w = tex + ".write(" + ex(A[ai]) + ", " + loc + ")";
        if (rw) w = "(" + w + ", " + tex + ".fence())";
        return finish(w);
    }
    std::string S = scalarName(T.sampled);
    std::string V = vecName(T.sampled, 4);
    if (T.sampled == BT::Float) fail("float image atomics are not supported by Metal");
    auto fenced = [&](const std::string& v) {
        if (!rw) return finish(v);
        std::string t = newTempS(S);
        return finish("(" + t + " = " + v + ", " + tex + ".fence(), " + t + ")");
    };
    if (n == "imageAtomicCompSwap") {
        if (layer.empty()) return fenced("mgl_img_cas(" + tex + ", " + coord + ", " + S + "(" + ex(A[ai]) + "), " + S + "(" + ex(A[ai + 1]) + "))");
        return fenced("mgl_img_cas_a(" + tex + ", " + coord + ", " + layer + ", " + S + "(" + ex(A[ai]) + "), " + S + "(" + ex(A[ai + 1]) + "))");
    }
    static const std::map<std::string, std::string> ops = {
        {"imageAtomicAdd", "atomic_fetch_add"}, {"imageAtomicMin", "atomic_fetch_min"}, {"imageAtomicMax", "atomic_fetch_max"},
        {"imageAtomicAnd", "atomic_fetch_and"}, {"imageAtomicOr", "atomic_fetch_or"},   {"imageAtomicXor", "atomic_fetch_xor"},
        {"imageAtomicExchange", "atomic_exchange"}};
    auto it = ops.find(n);
    if (it == ops.end()) fail("image function '" + n + "' is not supported");
    return fenced(tex + "." + it->second + "(" + loc + ", " + V + "(" + ex(A[ai]) + ")).x");
}

// ===========================================================================
// Statements & functions
// ===========================================================================
static std::string ind(int n) { return std::string(n * 4, ' '); }

void Emitter::block(const Stmt* s, std::string& o, int in) {
    if (s->kind == SK::Block) {
        o += "{\n";
        for (auto* b : s->body) stmt(b, o, in + 1);
        o += ind(in) + "}";
    } else {
        o += "{\n";
        stmt(s, o, in + 1);
        o += ind(in) + "}";
    }
}

void Emitter::stmt(const Stmt* s, std::string& o, int in) {
    switch (s->kind) {
        case SK::Empty: return;
        case SK::Block:
            o += ind(in);
            block(s, o, in);
            o += "\n";
            return;
        case SK::Expr: o += ind(in) + ex(s->expr) + ";\n"; return;
        case SK::Decl:
            for (auto* v : s->vars) {
                if (v->type.isOpaque()) fail("local variables of opaque type are not supported");
                o += ind(in) + mt(v->type) + " " + varName(v);
                // GLSL leaves uninitialized locals undefined; in MSL reading one is UB that the
                // optimizer may exploit (e.g. treating a path as unreachable), so zero them.
                o += v->init ? " = " + ex(v->init) : " = {}";
                o += ";\n";
            }
            return;
        case SK::If:
            o += ind(in) + "if (" + ex(s->expr) + ") ";
            block(s->then, o, in);
            if (s->els) {
                o += " else ";
                block(s->els, o, in);
            }
            o += "\n";
            return;
        case SK::For: {
            o += ind(in) + "{\n";
            if (s->init) stmt(s->init, o, in + 1);
            o += ind(in + 1) + "for (; " + (s->cond ? ex(s->cond) : "") + "; " + (s->incr ? ex(s->incr) : "") + ") ";
            block(s->then, o, in + 1);
            o += "\n" + ind(in) + "}\n";
            return;
        }
        case SK::While:
            o += ind(in) + "while (" + ex(s->expr) + ") ";
            block(s->then, o, in);
            o += "\n";
            return;
        case SK::DoWhile:
            o += ind(in) + "do ";
            block(s->then, o, in);
            o += " while (" + ex(s->expr) + ");\n";
            return;
        case SK::Switch: {
            // Declarations inside a switch are hoisted so case labels never jump past initializations.
            o += ind(in) + "{\n";
            for (auto* b : s->body)
                if (b->kind == SK::Decl)
                    for (auto* v : b->vars) o += ind(in + 1) + mt(v->type) + " " + varName(v) + " = {};\n";
            o += ind(in + 1) + "switch (" + ex(s->expr) + ") {\n";
            for (auto* b : s->body) {
                if (b->kind == SK::Decl) {
                    for (auto* v : b->vars)
                        if (v->init) o += ind(in + 2) + varName(v) + " = " + ex(v->init) + ";\n";
                    continue;
                }
                stmt(b, o, in + 2);
            }
            o += ind(in + 1) + "}\n" + ind(in) + "}\n";
            return;
        }
        case SK::Case: {
            size_t i = 0;
            o += ind(in - 1) + "case " + lit(s->expr->type, *s->expr->cv, i) + ":\n";
            return;
        }
        case SK::Default: o += ind(in - 1) + "default:\n"; return;
        case SK::Break: o += ind(in) + "break;\n"; return;
        case SK::Continue: o += ind(in) + "continue;\n"; return;
        case SK::Return:
            if (!retWriteback_.empty()) {
                // out/inout parameters are copied back on return (GLSL value-return semantics)
                if (s->expr) {
                    o += ind(in) + "{ " + mt(curFn_->ret) + " mgl_ret = " + ex(s->expr) + "; " + retWriteback_ + "return mgl_ret; }\n";
                } else {
                    o += ind(in) + "{ " + retWriteback_ + "return; }\n";
                }
                return;
            }
            o += ind(in) + "return" + (s->expr ? " " + ex(s->expr) : "") + ";\n";
            return;
        case SK::Discard: {
            // Field-debugging bisect switch (METALGLDRV_DEBUG_NODISCARD=1): drop shader discards.
            // 1 = all; N >= 2 = only the (N-1)th discard statement of each shader.
            static const int noDiscard = getenv("METALGLDRV_DEBUG_NODISCARD") ? atoi(getenv("METALGLDRV_DEBUG_NODISCARD")) : 0;
            int k = ++discardCount_;
            if (!(noDiscard == 1 || (noDiscard >= 2 && k == noDiscard - 1))) o += ind(in) + "discard_fragment();\n";
            return;
        }
    }
}

std::string Emitter::subroutineDispatchers() {
    std::string out;
    std::set<std::string> done;
    for (auto* v : sc_->used) {
        if (v->type.base != BT::Subroutine || !done.insert(v->name).second) continue;
        const Function* proto = v->type.subr;
        std::string params = "uint mgl_sel", args;
        for (size_t i = 0; i < proto->params.size(); i++) {
            const Variable* p = proto->params[i];
            std::string n = "a" + std::to_string(i);
            if (p->type.isOpaque()) fail("opaque subroutine parameters are not supported");
            bool ref = p->q.storage == Storage::Out || p->q.storage == Storage::InOut;
            params += ", " + (ref ? "thread " + mt(p->type) + "& " : mt(p->type) + " ") + n;
            args += (i ? ", " : "") + n;
        }
        out += "    " + mt(proto->ret) + " mgl_sub_" + v->name + "(" + params + ") {\n        switch (mgl_sel) {\n";
        for (auto& [mangled, f] : sc_->defs) {
            if (std::find(f->subroutineTypes.begin(), f->subroutineTypes.end(), proto->name) == f->subroutineTypes.end()) continue;
            auto it = P->subroutineIndex.find({(int)stage_, f->name});
            if (it == P->subroutineIndex.end() || !fnNames_.count(f)) continue;
            std::string call = fnNames_.at(f) + "(" + args + ")";
            out += "            case " + std::to_string(it->second) + "u: " + (proto->ret.isVoid() ? call + "; return;" : "return " + call + ";") + "\n";
        }
        out += "            default: break;\n        }\n";
        if (!proto->ret.isVoid()) out += "        return " + mt(proto->ret) + "{};\n";
        out += "    }\n";
    }
    return out;
}

std::string Emitter::fnSignature(const Function* f) {
    std::string r = mt(f->ret) + " " + fnNames_.at(f) + "(";
    bool first = true;
    for (auto* p : f->params) {
        if (!first) r += ", ";
        first = false;
        const Type& t = p->type;
        std::string n = varName(p);
        if (t.base == BT::Sampler) {
            if (t.dims.size() > 1) fail("arrays of arrays of samplers as function parameters are not supported");
            if (t.isArray()) {
                std::string N = std::to_string(t.dims[0]);
                r += "constant array<" + texType(t.stripArrays(), 0) + ", " + N + ">& " + n + "_t, constant array<sampler, " + N + ">& " + n + "_s";
            } else {
                r += texType(t, 0) + " " + n + "_t, sampler " + n + "_s";
            }
            continue;
        }
        if (t.base == BT::Image) {
            if (t.dims.size() > 1) fail("arrays of arrays of images as function parameters are not supported");
            if (t.isArray())
                r += "constant array<" + texType(t.stripArrays(), p->q.memory) + ", " + std::to_string(t.dims[0]) + ">& " + n + "_t";
            else
                r += texType(t, p->q.memory) + " " + n + "_t";
            continue;
        }
        if (t.base == BT::AtomicUint) {
            r += "device atomic_uint* " + n;
            continue;
        }
        if (t.isArray() && t.dims[0] == 0) fail("unsized array parameter");
        if (p->q.storage == Storage::Out || p->q.storage == Storage::InOut) r += "thread " + mt(t) + "& " + n + "_ref";
        else r += mt(t) + " " + n;
    }
    return r + ")";
}

std::string Emitter::emitFunction(const Function* f) {
    temps_.clear();
    // out/inout parameters are local copies (copy-in for inout) written back on return, so that a
    // global passed as an argument is not aliased by the parameter inside the function.
    std::string prologue;
    retWriteback_.clear();
    for (auto* p : f->params) {
        if (p->type.isOpaque() || !(p->q.storage == Storage::Out || p->q.storage == Storage::InOut)) continue;
        std::string n = varName(p);
        prologue += "        " + mt(p->type) + " " + n + (p->q.storage == Storage::InOut ? " = " + n + "_ref;\n" : "{};\n");
        retWriteback_ += n + "_ref = " + n + "; ";
    }
    curFn_ = f;
    std::string bodyText;
    for (auto* s : f->body->body) stmt(s, bodyText, 2);
    std::string r = "    " + fnSignature(f) + " {\n";
    r += prologue;
    for (auto& t : temps_) r += "        " + t + "\n";
    r += bodyText;
    if (!retWriteback_.empty() && f->ret.isVoid()) r += "        " + retWriteback_ + "\n";
    r += "    }\n";
    retWriteback_.clear();
    curFn_ = nullptr;
    return r;
}

// ===========================================================================
// Stage emission
// ===========================================================================
void Emitter::flatten(const Type& t, const std::string& acc, std::vector<std::pair<Type, std::string>>& leaves) {
    if (!t.dims.empty()) {
        for (int i = 0; i < t.dims[0]; i++) flatten(t.elem(), acc + "[" + std::to_string(i) + "]", leaves);
        return;
    }
    if (t.base == BT::Struct) {
        for (auto& m : t.sdef->members) flatten(m.type, acc + ".m_" + m.name, leaves);
        return;
    }
    if (t.cols) {
        for (int c = 0; c < t.cols; c++) leaves.push_back({t.column(), acc + "[" + std::to_string(c) + "]"});
        return;
    }
    leaves.push_back({t, acc});
}

std::string Emitter::interpAttr(const Qualifiers& q, const Type& leaf) {
    if (leaf.base == BT::Int || leaf.base == BT::UInt || leaf.base == BT::Bool || q.interp == Interp::Flat) return ", flat";
    if (q.interp == Interp::NoPerspective)
        return q.centroid ? ", centroid_no_perspective" : q.sample ? ", sample_no_perspective" : ", center_no_perspective";
    if (q.centroid) return ", centroid_perspective";
    if (q.sample) return ", sample_perspective";
    return "";
}

void Emitter::collectStructs(std::vector<const StructDef*>& order) {
    std::set<std::string> seen;
    std::function<void(const StructDef*)> visit = [&](const StructDef* sd) {
        std::string n = structName(sd);
        if (seen.count(n)) return;
        for (auto& m : sd->members)
            if (m.type.containsOpaque()) return;
        seen.insert(n);
        for (auto& m : sd->members) {
            Type b = m.type.stripArrays();
            if (b.base == BT::Struct) visit(b.sdef);
        }
        order.push_back(sd);
    };
    auto visitType = [&](const Type& t) {
        Type b = t.stripArrays();
        if (b.base != BT::Struct || !b.sdef) return;
        if (b.sdef->isBlock) {
            // only in/out blocks become MSL structs; buffer blocks are accessed by offset
            for (auto& m : b.sdef->members) {
                Type mb = m.type.stripArrays();
                if (mb.base == BT::Struct && !mb.sdef->isBlock) visit(mb.sdef);
            }
            return;
        }
        visit(b.sdef);
    };
    for (auto* v : sc_->globals) {
        if (v->block && (v->q.storage == Storage::In || v->q.storage == Storage::Out)) visit(v->block);
        else visitType(v->type);
    }
    for (auto* v : sc_->used) {
        if (v->block && (v->q.storage == Storage::In || v->q.storage == Storage::Out)) visit(v->block);
        else visitType(v->type);
    }
    std::function<void(const Expr*)> ve = [&](const Expr* e) {
        if (!e) return;
        visitType(e->type);
        for (auto* a : e->args) ve(a);
        ve(e->aux);
    };
    std::function<void(const Stmt*)> vs = [&](const Stmt* s) {
        if (!s) return;
        for (auto* b : s->body) vs(b);
        ve(s->expr);
        vs(s->init);
        ve(s->cond);
        ve(s->incr);
        vs(s->then);
        vs(s->els);
        for (auto* v : s->vars) { visitType(v->type); ve(v->init); }
    };
    for (auto* f : sc_->reachable) {
        visitType(f->ret);
        for (auto* p : f->params) visitType(p->type);
        vs(f->body);
    }
    for (auto* v : sc_->globals) ve(v->init);
}

void Emitter::emitArgs(std::string& o) {
    const auto& slots = P->refl.slots;
    std::vector<std::string> decl(slots.size());
    std::vector<bool> skip(slots.size(), false);
    if (P->defaultSlot >= 0) decl[P->defaultSlot] = "constant uchar* s" + std::to_string(P->defaultSlot) + ";";
    for (auto& [key, bb] : P->blocks) {
        std::string arr = bb.isArray ? "[" + std::to_string(bb.count) + "]" : "";
        decl[bb.slot] = std::string(bb.storage ? "device uchar* s" : "constant uchar* s") + std::to_string(bb.slot) + arr + ";";
        for (int i = 1; i < bb.count; i++) skip[bb.slot + i] = true;
        if (bb.storage) {
            decl[bb.sizeSlot] = "ulong s" + std::to_string(bb.sizeSlot) + arr + ";";
            for (int i = 1; i < bb.count; i++) skip[bb.sizeSlot + i] = true;
        }
    }
    for (auto& [name, ob] : P->opaque) {
        auto group = [&](int slot, const std::string& type) {
            if (ob.count > 1 || ob.isArray) decl[slot] = "array<" + type + ", " + std::to_string(ob.count) + "> s" + std::to_string(slot) + ";";
            else decl[slot] = type + " s" + std::to_string(slot) + ";";
            for (int i = 1; i < ob.count; i++) skip[slot + i] = true;
        };
        if (ob.kind == SlotKind::Texture) {
            group(ob.slot, texType(ob.type, 0));
            group(ob.samplerSlot, "sampler");
        } else if (ob.kind == SlotKind::Image) {
            group(ob.slot, texType(ob.type, ob.memory));
        }
    }
    for (auto& acb : P->refl.atomicCounterBuffers) decl[acb.slot] = "device uchar* s" + std::to_string(acb.slot) + ";";
    for (auto& ss : P->refl.subroutines)
        if (ss.slot >= 0) decl[ss.slot] = "device const uint* s" + std::to_string(ss.slot) + ";";
    o += "struct MGLArgs {\n";
    bool any = false;
    for (size_t i = 0; i < slots.size(); i++) {
        if (skip[i]) continue;
        if (decl[i].empty()) decl[i] = "ulong s" + std::to_string(i) + ";";
        o += "    " + decl[i] + "\n";
        any = true;
    }
    if (!any) o += "    ulong mgl_unused;\n";
    o += "};\n\n";
}

static const char* kNsNames[] = {"vs", "tcs", "tes", "gs", "fs", "cs"};

static std::string replaceJ(const std::string& s, const std::string& j) {
    std::string r = s;
    size_t p;
    while ((p = r.find("$J")) != std::string::npos) r.replace(p, 2, j);
    return r;
}

static std::string indentBy(const std::string& code, int n) {
    std::string pad(n, ' '), r;
    size_t a = 0;
    while (a < code.size()) {
        size_t b = code.find('\n', a);
        if (b == std::string::npos) b = code.size();
        if (b > a) r += pad + code.substr(a, b - a);
        r += "\n";
        a = b + 1;
    }
    return r;
}

void Emitter::prepareStage(Stage s) {
    useStage(s);
    StageInfo& si = si_[(int)s];
    if (si.prepared) return;
    si.prepared = true;
    std::vector<const Variable*> cand;
    for (auto* v : sc_->used) {
        if (v->builtin == BuiltinVar::None || v->builtin == BuiltinVar::CompatUniform || v->builtin == BuiltinVar::DepthRange ||
            v->builtin == BuiltinVar::NumSamples || v->builtin == BuiltinVar::WorkGroupSize || v->q.storage == Storage::Const || v->q.storage == Storage::Uniform)
            continue;
        cand.push_back(v);
    }
    std::sort(cand.begin(), cand.end(), [](const Variable* a, const Variable* b) { return a->name < b->name; });
    // member usage of builtin interface blocks
    std::function<void(const Expr*)> ve = [&](const Expr* e) {
        if (!e) return;
        if (e->kind == EK::Field) {
            const Expr* b = e->args[0];
            if (b->kind == EK::Index) b = b->args[0];
            if (b->kind == EK::Var && b->var->block && (b->var->builtin == BuiltinVar::In || b->var->builtin == BuiltinVar::Out))
                si.usedMembers[b->var].insert(e->field);
        }
        for (auto* a : e->args) ve(a);
        ve(e->aux);
    };
    std::function<void(const Stmt*)> vs = [&](const Stmt* st) {
        if (!st) return;
        for (auto* b : st->body) vs(b);
        ve(st->expr);
        vs(st->init);
        ve(st->cond);
        vs(st->condDecl);
        ve(st->incr);
        vs(st->then);
        vs(st->els);
        for (auto* v : st->vars) ve(v->init);
    };
    for (auto* f : sc_->reachable) vs(f->body);
    std::set<std::string> names;
    for (auto* v : cand) {
        if (!names.insert(v->name).second) continue;
        si.builtins.insert(v);
        if (!si.bi.count(v->builtin)) si.bi[v->builtin] = v;
    }
}

void Emitter::emitBody(Stage s, std::string& o, const std::string& preG, const std::string& extraMembers,
                       const std::string& extraMethods) {
    prepareStage(s);
    useStage(s);
    StageInfo& si = si_[(int)s];
    helpers_.clear();
    helperByKey_.clear();
    needEq_ = false;
    fnNames_.clear();
    tempN_ = 0;
    ns_ = kNsNames[(int)s];

    std::map<std::string, int> counts;
    for (auto& [m, f] : sc_->defs) {
        if (!sc_->reachable.count(f)) continue;
        fnNames_[f] = "f_" + f->name + "_" + std::to_string(counts[f->name]++);
    }

    std::string structs, members, funcs, init, zero;
    std::vector<const StructDef*> order;
    collectStructs(order);
    for (auto* sd : order) {
        structs += "struct " + structName(sd) + " {\n";
        for (auto& m : sd->members) {
            if (m.type.isArray() && m.type.dims[0] == 0) continue;
            structs += "    " + mt(m.type) + " m_" + m.name + ";\n";
        }
        structs += "};\n";
    }

    members += "    constant MGLArgs* A;\n    constant MGLSys* sys;\n";
    std::string sharedDecl;
    std::set<std::string> declared;
    for (auto* v : sc_->globals) {
        if (declared.count(v->name)) continue;
        declared.insert(v->name);
        if (v->q.storage == Storage::Uniform || v->q.storage == Storage::Buffer) continue;
        if (v->type.isArray() && v->type.dims[0] == 0) fail("unsized global array '" + v->name + "'");
        if (inThreadgroup(v)) {
            sharedDecl += "    " + mt(v->type) + " g_" + v->name + ";\n";
            continue;
        }
        members += "    " + mt(v->type) + " " + varName(v) + ";\n";
        // inputs are copied in by the entry point and initialized globals are set by mgl_init
        if (!v->init && v->q.storage != Storage::In) zero += "        " + varName(v) + " = {};\n";
    }
    for (auto* v : si.builtins) {
        if (inThreadgroup(v)) sharedDecl += "    " + mt(v->type) + " bi_" + v->name + ";\n";
        else {
            members += "    " + mt(v->type) + " bi_" + v->name + ";\n";
            if (v->q.storage != Storage::In) zero += "        bi_" + v->name + " = {};\n";
        }
    }
    if ((s == Stage::Vertex || s == Stage::Geometry || s == Stage::TessEval) && !si.bi.count(BuiltinVar::Position)) {
        members += "    float4 bi_gl_Position;\n";
        zero += "        bi_gl_Position = {};\n";
    }
    si.hasShared = hasShared_ = !sharedDecl.empty();
    if (hasShared_) members += "    threadgroup MGLShared* S;\n";

    for (auto& [m, f] : sc_->defs)
        if (sc_->reachable.count(f)) funcs += emitFunction(f);
    funcs += subroutineDispatchers();
    temps_.clear();
    std::string initBody;
    declared.clear();
    for (auto* v : sc_->globals) {
        if (!v->init || declared.count(v->name)) continue;
        declared.insert(v->name);
        if (v->q.storage == Storage::Uniform) continue;
        initBody += "        " + varName(v) + " = " + ex(v->init) + ";\n";
    }
    init = "    void mgl_init() {\n";
    for (auto& t : temps_) init += "        " + t + "\n";
    init += initBody + "    }\n";
    // Zeroes the members nothing else initializes (outputs, globals without initializer). Entry
    // points call this instead of value-initializing G: on Apple GPUs, zeroing all of a large G
    // (whether `G g = {}` or member by member) broke fragment shaders that can discard, which
    // then lost every fragment of the draw (Complementary's gbuffers_terrain).
    init += "    void mgl_zero() {\n" + zero + "    }\n";

    o += "namespace " + ns_ + " {\n";
    if (multi_) {
        emitArgs(o);
        argsEmitted_[(int)s] = true;
    }
    if (needEq_) o += "using ::mgl_eq;\n";
    if (needEq_ && usesFp64_) o += "// mgl_eq_f64_overloads\n";
    o += structs;
    for (auto* sd : needEq_ ? order : std::vector<const StructDef*>{}) {
        std::string n = structName(sd);
        o += "inline bool mgl_eq(" + n + " a, " + n + " b) { return true";
        for (auto& m : sd->members)
            if (!(m.type.isArray() && m.type.dims[0] == 0)) o += " && mgl_eq(a.m_" + m.name + ", b.m_" + m.name + ")";
        o += "; }\n";
    }
    if (hasShared_) o += "struct MGLShared {\n" + sharedDecl + "};\n";
    o += helpers_;
    o += preG;
    o += "struct G {\n" + members + extraMembers + init + funcs + extraMethods + "};\n";
    si.mainName = fnNames_.at(sc_->main);
}

// ---------------------------------------------------------------------------
// Varying fields
// ---------------------------------------------------------------------------
std::string Emitter::builtinKey(BuiltinVar b) {
    switch (b) {
        case BuiltinVar::Position: return "pos";
        case BuiltinVar::PointSize: return "psize";
        case BuiltinVar::ClipDistance: return "clipd";
        case BuiltinVar::CullDistance: return "culld";
        case BuiltinVar::ClipVertex: return "clipv";
        case BuiltinVar::FrontColor:
        case BuiltinVar::FragColorIn: return "c0";
        case BuiltinVar::FrontSecondaryColor:
        case BuiltinVar::SecondaryColorIn: return "c1";
        case BuiltinVar::BackColor: return "bc0";
        case BuiltinVar::BackSecondaryColor: return "bc1";
        case BuiltinVar::TexCoord: return "tc";
        case BuiltinVar::FogFragCoord: return "fog";
        default: return "";
    }
}

void Emitter::fieldsOfVar(const Variable* v, Stage st, bool arrayed, std::vector<Field>& out) {
    Type t = arrayed && v->type.isArray() ? v->type.elem() : v->type;
    std::string base = arrayed ? "[$J]" : "";
    auto add = [&](const std::string& key, const Type& ty, const std::string& suffix) {
        std::vector<std::pair<Type, std::string>> L;
        flatten(ty, suffix, L);
        splitDoubleLeaves(L);
        for (size_t i = 0; i < L.size(); i++) out.push_back({key + "_" + std::to_string(i), L[i].first, v, L[i].second, key});
    };
    if (v->block && (v->builtin == BuiltinVar::In || v->builtin == BuiltinVar::Out)) {
        auto& used = si_[(int)st].usedMembers[v];
        for (size_t mi = 0; mi < v->block->members.size(); mi++) {
            const Member& m = v->block->members[mi];
            if (!used.count((int)mi)) continue;
            std::string key = builtinKey(builtinVarId(m.name, st));
            if (!key.empty()) add(key, m.type, base + ".m_" + m.name);
        }
        return;
    }
    std::string key = v->builtin != BuiltinVar::None ? builtinKey(v->builtin) : varyingKey(v, st);
    if (key.empty()) return;
    add(key, t, base);
}

std::vector<Emitter::Field> Emitter::producerFields(Stage st) {
    prepareStage(st);
    StageInfo& si = si_[(int)st];
    std::vector<Field> f;
    if (si.bi.count(BuiltinVar::Position)) fieldsOfVar(si.bi[BuiltinVar::Position], st, false, f);
    else f.push_back({"pos_0", Type::vector(BT::Float, 4), nullptr, "bi_gl_Position", "pos"});
    for (BuiltinVar b : {BuiltinVar::PointSize, BuiltinVar::ClipDistance, BuiltinVar::ClipVertex, BuiltinVar::CullDistance,
                         BuiltinVar::FrontColor, BuiltinVar::FrontSecondaryColor, BuiltinVar::BackColor, BuiltinVar::BackSecondaryColor,
                         BuiltinVar::TexCoord, BuiltinVar::FogFragCoord})
        if (si.bi.count(b) && si.bi[b]->q.storage == Storage::Out) fieldsOfVar(si.bi[b], st, false, f);
    for (auto* v : sc_->globals)
        if (v->q.storage == Storage::Out && v->builtin == BuiltinVar::None && !v->q.patch) fieldsOfVar(v, st, false, f);
    // de-duplicate (shader objects of one stage may redeclare outputs)
    std::vector<Field> r;
    std::set<std::string> seen;
    for (auto& x : f)
        if (seen.insert(x.name).second) r.push_back(x);
    return r;
}

std::vector<Emitter::Field> Emitter::arrayedFields(Stage st, Storage storage) {
    prepareStage(st);
    StageInfo& si = si_[(int)st];
    std::vector<Field> f;
    for (auto& [b, v] : si.bi)
        if ((b == BuiltinVar::In || b == BuiltinVar::Out) && v->q.storage == storage) fieldsOfVar(v, st, true, f);
    for (auto* v : sc_->globals)
        if (v->q.storage == storage && v->builtin == BuiltinVar::None && !v->q.patch && v->type.isArray()) fieldsOfVar(v, st, true, f);
    std::vector<Field> r;
    std::set<std::string> seen;
    for (auto& x : f)
        if (seen.insert(x.name).second) r.push_back(x);
    return r;
}

std::vector<Emitter::Field> Emitter::patchFields(Stage st, Storage storage) {
    prepareStage(st);
    std::vector<Field> f;
    for (auto* v : sc_->globals)
        if (v->q.storage == storage && v->builtin == BuiltinVar::None && v->q.patch) fieldsOfVar(v, st, false, f);
    return f;
}

std::string Emitter::acc(const Field& f, const std::string& prefix, const std::string& j) {
    return prefix + (f.var ? varName(f.var) : "") + replaceJ(f.suffix, j);
}


// ---------------------------------------------------------------------------
// Vertex inputs (stage_in or manual fetch from MGLDrawInfo)
// ---------------------------------------------------------------------------
std::string Emitter::vsInputs(bool stageIn, std::string& inS, bool& anyIn, const std::string& g) {
    useStage(Stage::Vertex);
    std::string code;
    anyIn = false;
    std::map<int, Type> declaredLoc;  // compatibility attributes may alias generic ones
    std::vector<const Variable*> attrs(sc_->used.begin(), sc_->used.end());
    std::sort(attrs.begin(), attrs.end(), [](const Variable* a, const Variable* b) { return a->id < b->id; });
    for (auto* v : attrs) {
        if (v->q.storage != Storage::In || !(v->builtin == BuiltinVar::None || v->builtin == BuiltinVar::Vertex ||
                                              v->builtin == BuiltinVar::Normal || v->builtin == BuiltinVar::Color ||
                                              v->builtin == BuiltinVar::SecondaryColor || v->builtin == BuiltinVar::MultiTexCoord ||
                                              v->builtin == BuiltinVar::FogCoord))
            continue;
        auto it = P->attribLocations.find(v->name);
        if (it == P->attribLocations.end()) continue;
        int loc = it->second;
        std::vector<std::pair<Type, std::string>> leaves;
        flatten(v->type, g + varName(v), leaves);
        for (auto& [lt, a] : leaves) {
            std::string L0 = std::to_string(loc);
            if (lt.base == BT::Bool) fail("boolean vertex attributes are not allowed");
            int n = lt.vec;
            std::string sw = n == 1 ? ".x" : n == 4 ? "" : "." + std::string(kSwz, n);
            if (stageIn) {
                if (lt.base == BT::Double) {
                    std::string L1 = std::to_string(16 + loc);  // second half: Metal attribute 16 + L
                    if (n <= 2) {
                        inS += "    uint" + std::to_string(2 * n) + " a" + L0 + " [[attribute(" + L0 + ")]];\n";
                        code += "    " + a + " = as_type<" + vecName(BT::Double, n) + ">(in.a" + L0 + ");\n";
                        loc++;
                    } else {
                        inS += "    uint4 a" + L0 + " [[attribute(" + L0 + ")]];\n";
                        inS += "    uint" + std::to_string(2 * (n - 2)) + " a" + L1 + " [[attribute(" + L1 + ")]];\n";
                        code += "    " + a + " = " + vecName(BT::Double, n) + "(as_type<ulong2>(in.a" + L0 + "), as_type<" +
                                vecName(BT::Double, n - 2) + ">(in.a" + L1 + "));\n";
                        loc += 1;
                    }
                    anyIn = true;
                    continue;
                }
                auto dl = declaredLoc.find(loc);
                if (dl != declaredLoc.end()) {
                    std::string src = "in.a" + L0;
                    int dn = dl->second.vec;
                    if (dn > n) src += "." + std::string(kSwz, n);
                    code += "    " + a + " = " + mt(lt) + "(" + src + ");\n";
                    loc++;
                    continue;
                }
                declaredLoc[loc] = lt;
                BT declBase = lt.base;
                if (loc < 32 && (K.attribSignFlip >> loc & 1) && (lt.base == BT::Int || lt.base == BT::UInt))
                    declBase = lt.base == BT::Int ? BT::UInt : BT::Int;
                inS += "    " + vecName(declBase, n) + " a" + L0 + " [[attribute(" + L0 + ")]];\n";
                code += "    " + a + " = " + mt(lt) + "(in.a" + L0 + ");\n";
                loc++;
                anyIn = true;
                continue;
            }
            // manual fetch
            const VertexFetch& f = K.attribs[loc < 32 ? loc : 31];
            std::string ptr = "mgl_vptr(D, " + std::to_string(f.binding) + "u, " + std::to_string(f.relativeOffset) + "u, mgl_vid, mgl_vinst)";
            std::string cur = "sys.currentAttrib[" + L0 + "]";
            std::string val;
            if (lt.base == BT::Double) {
                std::string d4 = f.enabled ? (f.isDouble ? "mgl_fetch_d(" + ptr + ", " + std::to_string(f.size) + "u)"
                                                         : "mgl_f64_from_float(mgl_fetch_f(" + ptr + ", " + std::to_string(f.glType) + "u, " +
                                                               std::to_string(f.size) + "u, " + (f.normalized ? "true" : "false") + "))")
                                           : "ulong4(as_type<ulong2>(sys.currentAttribD[2 * " + L0 + "]), as_type<ulong2>(sys.currentAttribD[2 * " + L0 +
                                                 " + 1]))";
                val = d4 + sw;
                loc += 1;
            } else if (lt.base == BT::Float) {
                val = (f.enabled ? "mgl_fetch_f(" + ptr + ", " + std::to_string(f.glType) + "u, " + std::to_string(f.size) + "u, " +
                                       (f.normalized ? "true" : "false") + ")"
                                 : cur) + sw;
                loc++;
            } else {
                std::string i4 = f.enabled ? "mgl_fetch_i(" + ptr + ", " + std::to_string(f.glType) + "u, " + std::to_string(f.size) + "u)"
                                           : "as_type<int4>(" + cur + ")";
                val = (lt.base == BT::UInt ? "as_type<uint4>(" + i4 + ")" : i4) + sw;
                loc++;
            }
            code += "    " + a + " = " + val + ";\n";
            anyIn = true;
        }
    }
    return code;
}

std::string Emitter::runVS(const std::string& gv, const std::string& vid, const std::string& inst) {
    useStage(Stage::Vertex);
    StageInfo& si = si_[(int)Stage::Vertex];
    std::string c = "vs::G " + gv + " = {};\n" + gv + ".A = " + argsOf(Stage::Vertex) + ";\n" + gv + ".sys = &sys;\n";
    c += "{\n    uint mgl_vid = " + vid + ", mgl_vinst = " + inst + ";\n";
    std::string dummy;
    bool any;
    c += indentBy(vsInputs(false, dummy, any, gv + "."), 0);
    for (auto* v : si.builtins) {
        std::string m = gv + ".bi_" + v->name;
        switch (v->builtin) {
            case BuiltinVar::VertexID: c += "    " + m + " = int(mgl_vid);\n"; break;
            case BuiltinVar::InstanceID: c += "    " + m + " = int(mgl_vinst);\n"; break;
            case BuiltinVar::BaseVertex: c += "    " + m + " = D.baseVertex;\n"; break;
            case BuiltinVar::BaseInstance: c += "    " + m + " = int(D.baseInstance);\n"; break;
            case BuiltinVar::DrawID: c += "    " + m + " = sys.drawID;\n"; break;
            case BuiltinVar::PointSize: c += "    " + m + " = sys.pointSize;\n"; break;
            default: break;
        }
    }
    c += "}\n" + gv + ".mgl_init();\n" + gv + "." + si.mainName + "();\n";
    return c;
}

// ---------------------------------------------------------------------------
// Rasterizer-facing outputs (classic vertex out struct or mesh vertex/primitive)
// ---------------------------------------------------------------------------
Emitter::RasterOut Emitter::rasterOutput(Stage st, const std::string& g, bool mesh, bool points) {
    RasterOut r;
    auto fields = producerFields(st);
    useStage(st);
    StageInfo& si = si_[(int)st];
    const Variable* posV = si.bi.count(BuiltinVar::Position) ? si.bi[BuiltinVar::Position] : nullptr;
    std::string sys = g + "sys->";
    r.vertexStruct += std::string("    float4 mgl_pos [[position") + (posV && posV->q.invariant ? ", invariant" : "") + "]];\n";
    r.vertexCopy += "v.mgl_pos = " + g + "bi_gl_Position;\n";
    r.vertexCopy += "if ((" + sys + "flags & 4u) == 0u) v.mgl_pos.y = -v.mgl_pos.y;\n";
    r.vertexCopy += "if (" + sys + "depthMode == 0u) v.mgl_pos.z = (v.mgl_pos.z + v.mgl_pos.w) * 0.5f;\n";
    if (!mesh || points) {
        r.vertexStruct += "    float mgl_psize [[point_size]];\n";
        if (si.bi.count(BuiltinVar::PointSize) && si.bi[BuiltinVar::PointSize]->q.storage == Storage::Out)
            r.vertexCopy += "v.mgl_psize = (" + sys + "flags & 1u) != 0u ? " + g + "bi_gl_PointSize : " + sys + "pointSize;\n";
        else
            r.vertexCopy += "v.mgl_psize = " + sys + "pointSize;\n";
    }
    bool hasClipD = si.bi.count(BuiltinVar::ClipDistance) && si.bi[BuiltinVar::ClipDistance]->q.storage == Storage::Out;
    bool hasClipV = si.bi.count(BuiltinVar::ClipVertex);
    bool hasCullD = si.bi.count(BuiltinVar::CullDistance) && si.bi[BuiltinVar::CullDistance]->q.storage == Storage::Out;
    int nclip = (hasClipD || hasClipV) ? (hasClipD ? si.bi[BuiltinVar::ClipDistance]->type.dims[0] : 8) : 0;
    int ncull = hasCullD ? si.bi[BuiltinVar::CullDistance]->type.dims[0] : 0;
    if (nclip + ncull > 0)
        r.vertexStruct += "    float mgl_clip [[clip_distance]] [" + std::to_string(nclip + ncull) + "];\n";
    // Cull distances (Metal has none) ride on extra clip distances: a vertex with a negative
    // cull distance outputs -1, others a huge positive value. A primitive whose vertices are all
    // negative is clipped away entirely (= culled); otherwise the clipped sliver next to the
    // negative vertices is ~1e-20 of the primitive, i.e. it is drawn whole as GL requires.
    for (int i = 0; i < ncull; i++)
        r.vertexCopy += "v.mgl_clip[" + std::to_string(nclip + i) + "] = " + g + varName(si.bi[BuiltinVar::CullDistance]) + "[" +
                        std::to_string(i) + "] < 0.0f ? -1.0f : 1e20f;\n";
    if (nclip > 0) {
        for (int i = 0; i < nclip; i++) {
            std::string val;
            if (hasClipD) val = g + varName(si.bi[BuiltinVar::ClipDistance]) + "[" + std::to_string(i) + "]";
            else {
                auto it = P->defaultOffsets.find("gl_ClipPlane");
                if (it == P->defaultOffsets.end()) fail("internal: gl_ClipVertex without gl_ClipPlane");
                val = "dot(" + g + "bi_gl_ClipVertex, mgl_ld_float4(" + g + "A->s" + std::to_string(P->defaultSlot) + ", " +
                      std::to_string(it->second + 16 * i) + "u))";
            }
            r.vertexCopy += "v.mgl_clip[" + std::to_string(i) + "] = ((" + sys + "clipDistanceMask >> " + std::to_string(i) +
                            "u) & 1u) != 0u ? " + val + " : 1.0f;\n";
        }
    }
    auto perPrim = [&](const char* type, const char* name, const char* attr, const std::string& value) {
        if (mesh) {
            r.primStruct += std::string("    ") + type + " " + name + " [[" + attr + "]];\n";
            r.primCopy += std::string("p.") + name + " = " + value + ";\n";
            r.hasPrim = true;
        } else {
            r.vertexStruct += std::string("    ") + type + " " + name + " [[" + attr + "]];\n";
            r.vertexCopy += std::string("v.") + name + " = " + value + ";\n";
        }
    };
    if (si.bi.count(BuiltinVar::Layer) && si.bi[BuiltinVar::Layer]->q.storage == Storage::Out)
        perPrim("uint", "mgl_layer", "render_target_array_index", "uint(" + g + "bi_gl_Layer)");
    if (si.bi.count(BuiltinVar::ViewportIndex) && si.bi[BuiltinVar::ViewportIndex]->q.storage == Storage::Out)
        perPrim("uint", "mgl_vp", "viewport_array_index", "uint(" + g + "bi_gl_ViewportIndex)");
    const StageCode& fsc = SC(Stage::Fragment);
    bool fsClip = fsc.main && fsc.usedNames.count("gl_ClipDistance");
    bool fsCull = fsc.main && fsc.usedNames.count("gl_CullDistance");
    // Mesh vertex outputs are limited (124 scalars): only pass varyings the fragment shader reads.
    std::set<std::string> fsNames;
    bool filter = mesh && fsc.main;
    if (filter) {
        Stage keep = stage_;
        useStage(Stage::Fragment);
        prepareStage(Stage::Fragment);
        for (auto* v : sc_->globals) {
            if (v->q.storage != Storage::In || v->builtin != BuiltinVar::None) continue;
            std::vector<std::pair<Type, std::string>> leaves;
            flatten(v->type, "", leaves);
            splitDoubleLeaves(leaves);
            std::string key = varyingKey(v, Stage::Fragment);
            for (size_t i = 0; i < leaves.size(); i++) fsNames.insert(key + "_" + std::to_string(i));
        }
        for (auto& [k, v] : si_[(int)Stage::Fragment].bi)
            if (k == BuiltinVar::FragColorIn || k == BuiltinVar::SecondaryColorIn || k == BuiltinVar::TexCoord || k == BuiltinVar::FogFragCoord) {
                std::vector<std::pair<Type, std::string>> leaves;
                flatten(v->type, "", leaves);
                std::string key = builtinKey(k);
                for (size_t i = 0; i < leaves.size(); i++) fsNames.insert(key + "_" + std::to_string(i));
            }
        useStage(keep);
    }
    for (auto& f : fields) {
        if (f.key == "pos" || f.key == "psize" || f.key == "clipv") continue;
        if (filter && f.key != "clipd" && f.key != "culld" && !fsNames.count(f.name)) continue;
        if (f.key == "clipd" && !fsClip) continue;
        if (f.key == "culld" && !fsCull) continue;
        if (f.type.base == BT::Bool) fail("boolean varyings are not allowed");
        std::string a = acc(f, g);
        if (f.type.base == BT::Double) {
            r.vertexStruct += "    uint2 " + f.name + " [[user(" + f.name + ")]];\n";
            r.vertexCopy += "v." + f.name + " = as_type<uint2>(" + a + ");\n";
        } else {
            r.vertexStruct += "    " + mt(f.type) + " " + f.name + " [[user(" + f.name + ")]];\n";
            r.vertexCopy += "v." + f.name + " = " + a + ";\n";
        }
        r.names.insert(f.name);
    }
    return r;
}

// ---------------------------------------------------------------------------
// Classic pipelines
// ---------------------------------------------------------------------------
void Emitter::emitClassicVS(std::string& o) {
    emitBody(Stage::Vertex, o);
    StageInfo& si = si_[(int)Stage::Vertex];
    std::string inS;
    bool anyIn = false;
    std::string copyIn;
    if (K.manualFetch) {
        bool any = false;
        std::string dummy;
        copyIn = "    uint mgl_vinst = mgl_iid - mgl_bi;\n" + vsInputs(false, dummy, any, "g.");
    } else {
        copyIn = vsInputs(true, inS, anyIn, "g.");
    }
    RasterOut ro = rasterOutput(Stage::Vertex, "g.", false, true);
    o += "struct In {\n" + inS + "};\nstruct Out {\n" + ro.vertexStruct + "};\n}  // namespace vs\n\n";
    if (K.xfbInVertex) {
        // Transform feedback with vertex shader side effects: the vertex shader runs once per
        // vertex of the decomposed primitive list (non-indexed draw) and captures its outputs
        // itself, so its side effects happen exactly once.
        std::string e = "vertex vs::Out mgl_vs(constant MGLArgs& A [[buffer(0)]], constant MGLSys& sys [[buffer(1)]], "
                        "constant MGLDrawInfo& D [[buffer(22)]], uint mgl_lv [[vertex_id]], uint mgl_li [[instance_id]]) {\n";
        e += "    uint mgl_t = mgl_li * D.count + mgl_lv;\n";
        e += indentBy(runVS("g", "D.indexBuffer[mgl_lv] + uint(D.baseVertex)", "mgl_li"), 4);
        e += xfbVertexStores();
        e += "    vs::Out v;\n" + indentBy(ro.vertexCopy, 4) + "    return v;\n}\n\n";
        entries_ += e;
        fragInputsAvailable_ = ro.names;
        haveVsOutputs_ = true;
        out_.entryPoints.push_back({"mgl_vs", Stage::Vertex, EntryPoint::Vertex});
        return;
    }
    // Rasterizer discard: Metal needs a void vertex function when rasterization is disabled.
    std::string e = K.rasterDiscard ? "vertex void mgl_vs(" : "vertex vs::Out mgl_vs(";
    if (anyIn) e += "vs::In in [[stage_in]], ";
    e += "constant MGLArgs& A [[buffer(0)]], constant MGLSys& sys [[buffer(1)]], uint mgl_vid [[vertex_id]], uint mgl_iid [[instance_id]]";
    e += ", uint mgl_bi [[base_instance]]";
    if (K.manualFetch) e += ", constant MGLDrawInfo& D [[buffer(22)]]";
    if (si.bi.count(BuiltinVar::BaseVertex)) e += ", uint mgl_bv [[base_vertex]]";
    e += ") {\n    vs::G g;\n    g.mgl_zero();\n    g.A = " + argsOf(Stage::Vertex) + ";\n    g.sys = &sys;\n";
    e += copyIn;
    for (auto* v : si.builtins) {
        if (v->builtin == BuiltinVar::VertexID) e += "    g.bi_" + v->name + " = int(mgl_vid);\n";
        if (v->builtin == BuiltinVar::InstanceID) e += "    g.bi_" + v->name + " = int(mgl_iid - mgl_bi);\n";
        if (v->builtin == BuiltinVar::BaseVertex) e += "    g.bi_" + v->name + " = int(mgl_bv);\n";
        if (v->builtin == BuiltinVar::BaseInstance) e += "    g.bi_" + v->name + " = int(mgl_bi);\n";
        if (v->builtin == BuiltinVar::DrawID) e += "    g.bi_" + v->name + " = sys.drawID;\n";
        if (v->builtin == BuiltinVar::PointSize) e += "    g.bi_" + v->name + " = sys.pointSize;\n";
    }
    if (K.rasterDiscard) e += "    g.mgl_init();\n    g." + si.mainName + "();\n}\n\n";
    else e += "    g.mgl_init();\n    g." + si.mainName + "();\n    vs::Out v;\n" + indentBy(ro.vertexCopy, 4) + "    return v;\n}\n\n";
    entries_ += e;
    fragInputsAvailable_ = ro.names;
    haveVsOutputs_ = true;
    out_.entryPoints.push_back({"mgl_vs", Stage::Vertex, EntryPoint::Vertex});
}

void Emitter::emitFS(std::string& o, bool meshInput) {
    prepareStage(Stage::Fragment);
    useStage(Stage::Fragment);
    StageInfo& si = si_[(int)Stage::Fragment];
    // inputs used with interpolateAt* stay interpolants, reachable from G through mgl_in
    std::set<const Variable*> interpVars;
    {
        std::function<void(const Expr*)> ve = [&](const Expr* e) {
            if (!e) return;
            if (e->kind == EK::Builtin && e->fn->name.compare(0, 13, "interpolateAt") == 0) {
                const Expr* r = e->args[0];
                while (r->kind == EK::Swizzle || r->kind == EK::Index || r->kind == EK::Field) r = r->args[0];
                if (r->kind == EK::Var && r->var->q.storage == Storage::In && r->var->builtin == BuiltinVar::None) interpVars.insert(r->var);
            }
            for (auto* a : e->args) ve(a);
            ve(e->aux);
        };
        std::function<void(const Stmt*)> vs = [&](const Stmt* st) {
            if (!st) return;
            for (auto* b : st->body) vs(b);
            ve(st->expr);
            vs(st->init);
            ve(st->cond);
            vs(st->condDecl);
            ve(st->incr);
            vs(st->then);
            vs(st->els);
            for (auto* v : st->vars) ve(v->init);
        };
        for (auto* f : sc_->reachable) vs(f->body);
    }
    interpLeaf_.clear();
    auto& biByKind = si.bi;
    std::string inS = "struct In {\n", outS = "struct Out {\n", copyIn, copyOut, params;
    bool anyIn = false, anyOut = false;
    // Per-sample shading: forced by the state (GL_SAMPLE_SHADING), by gl_SampleID /
    // gl_SamplePosition, or by sample-qualified inputs. Metal runs a fragment function at
    // sample rate when it takes [[sample_id]].
    bool perSample = K.sampleShading || biByKind.count(BuiltinVar::SampleID) || biByKind.count(BuiltinVar::SamplePosition);
    bool haveSid = false;
    auto needSid = [&]() {
        if (!haveSid) params += ", uint mgl_sid [[sample_id]]";
        haveSid = true;
    };
    std::vector<const Variable*> ins;
    for (auto* v : sc_->globals)
        if (v->q.storage == Storage::In && v->builtin == BuiltinVar::None) ins.push_back(v);
    for (auto& [k, v] : biByKind)
        if (k == BuiltinVar::FragColorIn || k == BuiltinVar::SecondaryColorIn || k == BuiltinVar::TexCoord || k == BuiltinVar::FogFragCoord)
            ins.push_back(v);
    std::set<std::string> names;
    for (auto* v : ins) {
        std::string key = varyingKey(v, Stage::Fragment);
        std::vector<std::pair<Type, std::string>> leaves;
        flatten(v->type, "g." + varName(v), leaves);
        splitDoubleLeaves(leaves);
        const Qualifiers& q = v->q;
        for (size_t i = 0; i < leaves.size(); i++) {
            std::string nm = key + "_" + std::to_string(i);
            if (names.count(nm)) continue;
            names.insert(nm);
            if (haveVsOutputs_ && !fragInputsAvailable_.count(nm)) {
                if (v->builtin == BuiltinVar::FragColorIn || v->builtin == BuiltinVar::SecondaryColorIn)
                    copyIn += "    " + leaves[i].second + " = float4(1.0f);\n";
                continue;
            }
            Type lt = leaves[i].first;
            Qualifiers lq = q;
            if (v->block) {
                size_t dot = leaves[i].second.find(".m_", 2 + varName(v).size());
                if (dot != std::string::npos) {
                    std::string mn = leaves[i].second.substr(dot + 3);
                    mn = mn.substr(0, mn.find_first_of(".["));
                    int mi = v->block->findMember(mn);
                    if (mi >= 0) lq = v->block->members[mi].q;
                }
            }
            if (lt.base == BT::Double) {
                inS += "    uint2 " + nm + " [[user(" + nm + "), flat]];\n";
                copyIn += "    " + leaves[i].second + " = as_type<ulong>(in." + nm + ");\n";
                anyIn = true;
                continue;
            }
            // Sample shading evaluates every interpolated input at the invocation's sample.
            if (K.sampleShading && lq.interp != Interp::Flat) lq.sample = true, lq.centroid = false;
            if (lq.sample && lq.interp != Interp::Flat) perSample = true;
            if (interpVars.count(v) && lt.base == BT::Float && lq.interp != Interp::Flat) {
                std::string mode = lq.interp == Interp::NoPerspective ? "no_perspective" : "perspective";
                inS += "    interpolant<" + mt(lt) + ", interpolation::" + mode + "> " + nm + " [[user(" + nm + ")]];\n";
                std::string at = lq.centroid ? "interpolate_at_centroid()" : "interpolate_at_center()";
                if (lq.sample) {
                    needSid();
                    at = "interpolate_at_sample(mgl_sid)";
                }
                copyIn += "    " + leaves[i].second + " = in." + nm + "." + at + ";\n";
                interpLeaf_[leaves[i].second.substr(2)] = nm;  // strip "g."
                anyIn = true;
                continue;
            }
            inS += "    " + mt(lt) + " " + nm + " [[user(" + nm + ")" + interpAttr(lq, lt) + "]];\n";
            copyIn += "    " + leaves[i].second + " = in." + nm + ";\n";
            anyIn = true;
        }
    }
    for (auto* v : si.builtins) {
        switch (v->builtin) {
            case BuiltinVar::FragCoord: {
                params += ", float4 mgl_fc [[position]]";
                std::string x = "mgl_fc.x", y = "mgl_fc.y";
                if (v->q.layout.originUpperLeft) y = "(sys.renderTargetHeight - mgl_fc.y)";
                if (v->q.layout.pixelCenterInteger) { x = "(" + x + " - 0.5f)"; y = "(" + y + " - 0.5f)"; }
                copyIn += "    g.bi_" + v->name + " = float4(" + x + ", " + y + ", mgl_fc.z, mgl_fc.w);\n";
                break;
            }
            case BuiltinVar::FrontFacing:
                params += ", bool mgl_ff [[front_facing]]";
                copyIn += "    g.bi_" + v->name + " = mgl_ff;\n";
                break;
            case BuiltinVar::PointCoord:
                params += ", float2 mgl_pc [[point_coord]]";
                copyIn += "    g.bi_" + v->name + " = float2(mgl_pc.x, (sys.flags & 2u) != 0u ? mgl_pc.y : 1.0f - mgl_pc.y);\n";
                break;
            case BuiltinVar::SampleID:
                needSid();
                copyIn += "    g.bi_" + v->name + " = int(mgl_sid);\n";
                break;
            case BuiltinVar::SamplePosition:
                needSid();
                copyIn += "    g.bi_" + v->name + " = get_sample_position(mgl_sid);\n";
                break;
            case BuiltinVar::SampleMaskIn:
                params += ", uint mgl_smi [[sample_mask]]";
                // With per-sample shading each invocation covers only its own sample.
                if (perSample) {
                    needSid();
                    copyIn += "    g.bi_" + v->name + "[0] = int(mgl_smi & (1u << mgl_sid));\n";
                } else {
                    copyIn += "    g.bi_" + v->name + "[0] = int(mgl_smi);\n";
                }
                break;
            case BuiltinVar::PrimitiveID:
                if (meshInput) {
                    inS += "    int mgl_primid [[user(mgl_primid), flat]];\n";
                    copyIn += "    g.bi_" + v->name + " = in.mgl_primid;\n";
                    anyIn = true;
                } else {
                    params += ", uint mgl_pid [[primitive_id]]";
                    copyIn += "    g.bi_" + v->name + " = int(mgl_pid);\n";
                }
                break;
            case BuiltinVar::Layer:
                params += ", uint mgl_layer [[render_target_array_index]]";
                copyIn += "    g.bi_" + v->name + " = int(mgl_layer);\n";
                break;
            case BuiltinVar::ViewportIndex:
                params += ", uint mgl_vpi [[viewport_array_index]]";
                copyIn += "    g.bi_" + v->name + " = int(mgl_vpi);\n";
                break;
            case BuiltinVar::HelperInvocation:
                copyIn += "    g.bi_" + v->name + " = simd_is_helper_thread();\n";
                break;
            case BuiltinVar::ClipDistance:
            case BuiltinVar::CullDistance: {
                std::string key = v->builtin == BuiltinVar::ClipDistance ? "clipd" : "culld";
                int n = v->type.isArray() ? v->type.dims[0] : 1;
                for (int i = 0; i < n; i++) {
                    std::string nm = key + "_" + std::to_string(i);
                    if (haveVsOutputs_ && !fragInputsAvailable_.count(nm)) {
                        copyIn += "    g.bi_" + v->name + "[" + std::to_string(i) + "] = 0.0f;\n";
                        continue;
                    }
                    inS += "    float " + nm + " [[user(" + nm + ")]];\n";
                    copyIn += "    g.bi_" + v->name + "[" + std::to_string(i) + "] = in." + nm + ";\n";
                    anyIn = true;
                }
                break;
            }
            default: break;
        }
    }
    if (K.sampleShading) needSid();
    bool fragDepth = biByKind.count(BuiltinVar::FragDepth) && sc_->writesFragDepth;
    if (fragDepth && !biByKind.count(BuiltinVar::FragCoord)) params += ", float4 mgl_fc [[position]]";
    if (fragDepth) copyIn += "    g.bi_gl_FragDepth = mgl_fc.z;\n";
    if (biByKind.count(BuiltinVar::SampleMask)) copyIn += "    g.bi_gl_SampleMask[0] = -1;\n";
    bool color0Float = false;
    std::string color0;
    for (auto* v : sc_->globals) {
        if (v->q.storage != Storage::Out || v->builtin != BuiltinVar::None) continue;
        auto it = P->fragOutputs.find(v->name);
        if (it == P->fragOutputs.end()) continue;
        auto [loc, idx] = it->second;
        int n = v->type.isArray() ? v->type.dims[0] : 1;
        Type et = v->type.stripArrays();
        for (int i = 0; i < n; i++) {
            std::string a = "g." + varName(v) + (v->type.isArray() ? "[" + std::to_string(i) + "]" : "");
            std::string fn = "mgl_c" + std::to_string(loc + i) + (idx ? "_1" : "");
            std::string vt = vecName(et.base == BT::Double ? BT::Float : et.base, 4);
            outS += "    " + vt + " " + fn + " [[color(" + std::to_string(loc + i) + ")" + (idx ? ", index(1)" : "") + "]];\n";
            std::string val = a;
            if (et.vec == 1) val = vt + "(" + a + ", 0, 0, 1)";
            else if (et.vec == 2) val = vt + "(" + a + ", 0, 1)";
            else if (et.vec == 3) val = vt + "(" + a + ", 1)";
            copyOut += "    o." + fn + " = " + val + ";\n";
            if (loc + i == 0 && idx == 0 && et.base == BT::Float) { color0Float = true; color0 = "o." + fn; }
            anyOut = true;
        }
    }
    if (biByKind.count(BuiltinVar::FragColor)) {
        outS += "    float4 mgl_c0 [[color(0)]];\n";
        copyOut += "    o.mgl_c0 = g.bi_gl_FragColor;\n";
        color0Float = true;
        color0 = "o.mgl_c0";
        anyOut = true;
    }
    if (biByKind.count(BuiltinVar::FragData)) {
        int n = biByKind[BuiltinVar::FragData]->type.dims[0];
        for (int i = 0; i < n; i++) {
            if (i == 0 && biByKind.count(BuiltinVar::FragColor)) continue;
            outS += "    float4 mgl_c" + std::to_string(i) + " [[color(" + std::to_string(i) + ")]];\n";
            copyOut += "    o.mgl_c" + std::to_string(i) + " = g.bi_gl_FragData[" + std::to_string(i) + "];\n";
        }
        color0Float = true;
        color0 = "o.mgl_c0";
        anyOut = true;
    }
    if (fragDepth) {
        const Variable* fd = biByKind[BuiltinVar::FragDepth];
        const char* mode = fd->q.layout.depthLayout == 2 ? "greater" : fd->q.layout.depthLayout == 3 ? "less" : "any";
        outS += std::string("    float mgl_depth [[depth(") + mode + ")]];\n";
        copyOut += "    o.mgl_depth = g.bi_gl_FragDepth;\n";
        anyOut = true;
    }
    if (biByKind.count(BuiltinVar::SampleMask) || K.sampleMask) {
        // gl_SampleMask and GL_SAMPLE_MASK only apply to multisample framebuffers.
        std::string m = biByKind.count(BuiltinVar::SampleMask) ? "uint(g.bi_gl_SampleMask[0])" : "0xffffffffu";
        if (K.sampleMask) m = "(" + m + " & sys.sampleMask & mgl_coverage_mask(sys.sampleCoverage, (sys.flags & 8u) != 0u))";
        outS += "    uint mgl_mask [[sample_mask]];\n";
        copyOut += "    o.mgl_mask = get_num_samples() > 1u ? " + m + " : 0xffffffffu;\n";
        anyOut = true;
    }
    outS += "};\n";
    inS += "};\n";
    std::string alpha;
    if (color0Float && sc_->primary->compat)
        alpha = "    if (sys.alphaFunc != 0x0207u && !mgl_alpha_pass(" + color0 + ".a, sys.alphaFunc, sys.alphaRef)) discard_fragment();\n";
    bool useInterp = !interpLeaf_.empty();
    emitBody(Stage::Fragment, o, inS, useInterp ? "    thread In* mgl_in;\n" : "");
    if (useInterp) copyIn += "    g.mgl_in = &in;\n";
    o += outS + "}  // namespace fs\n\n";
    std::string e;
    if (sc_->inLayout.earlyFragmentTests) e += "[[early_fragment_tests]]\n";
    e += std::string("fragment ") + (anyOut ? "fs::Out" : "void") + " mgl_fs(";
    if (anyIn) e += "fs::In in [[stage_in]], ";
    e += "constant MGLArgs& A [[buffer(0)]], constant MGLSys& sys [[buffer(1)]]" + params + ") {\n";
    e += "    fs::G g;\n    g.mgl_zero();\n    g.A = " + argsOf(Stage::Fragment) + ";\n    g.sys = &sys;\n" + copyIn;
    e += "    g.mgl_init();\n    g." + si.mainName + "();\n";
    if (anyOut) e += "    fs::Out o;\n" + copyOut + alpha + "    return o;\n";
    e += "}\n\n";
    entries_ += e;
    out_.entryPoints.push_back({"mgl_fs", Stage::Fragment, EntryPoint::Fragment});
}

void Emitter::emitCS(std::string& o) {
    emitBody(Stage::Compute, o);
    StageInfo& si = si_[(int)Stage::Compute];
    o += "}  // namespace cs\n\n";
    const auto& r = P->refl;
    std::string e = "kernel void mgl_cs(constant MGLArgs& A [[buffer(0)]], constant MGLSys& sys [[buffer(1)]], uint3 mgl_gid [[thread_position_in_grid]], "
                    "uint3 mgl_lid [[thread_position_in_threadgroup]], uint3 mgl_wid [[threadgroup_position_in_grid]], uint mgl_lidx [[thread_index_in_threadgroup]], "
                    "uint3 mgl_nwg [[threadgroups_per_grid]]) {\n";
    if (si.hasShared) e += "    threadgroup cs::MGLShared mgl_shared;\n";
    e += "    cs::G g;\n    g.A = &A;\n    g.sys = &sys;\n";
    if (si.hasShared) e += "    g.S = &mgl_shared;\n";
    for (auto* v : si.builtins) {
        switch (v->builtin) {
            case BuiltinVar::NumWorkGroups: e += "    g.bi_" + v->name + " = mgl_nwg;\n"; break;
            case BuiltinVar::WorkGroupID: e += "    g.bi_" + v->name + " = mgl_wid;\n"; break;
            case BuiltinVar::LocalInvocationID: e += "    g.bi_" + v->name + " = mgl_lid;\n"; break;
            case BuiltinVar::GlobalInvocationID: e += "    g.bi_" + v->name + " = mgl_gid;\n"; break;
            case BuiltinVar::LocalInvocationIndex:
                // GL's linearization of the local id (z * X * Y + y * X + x).
                e += "    g.bi_" + v->name + " = (mgl_lid.z * " + std::to_string(r.localSize[1]) + "u + mgl_lid.y) * " +
                     std::to_string(r.localSize[0]) + "u + mgl_lid.x;\n";
                break;
            default: break;
        }
    }
    e += "    g.mgl_init();\n    g." + si.mainName + "();\n}\n\n";
    entries_ += e;
    EntryPoint ep{"mgl_cs", Stage::Compute, EntryPoint::Kernel};
    ep.threadsPerThreadgroup[0] = r.localSize[0];
    ep.threadsPerThreadgroup[1] = r.localSize[1];
    ep.threadsPerThreadgroup[2] = r.localSize[2];
    out_.entryPoints.push_back(ep);
}

// ---------------------------------------------------------------------------
// Geometry shaders on mesh shaders
// ---------------------------------------------------------------------------
static int primVerts(uint32_t prim) {
    switch (prim) {
        case gl::POINTS: return 1;
        case gl::LINES: return 2;
        case gl::LINES_ADJACENCY: return 4;
        case gl::TRIANGLES: return 3;
        case gl::TRIANGLES_ADJACENCY: return 6;
        default: return 0;
    }
}

// Unpacks the per-vertex inputs of the geometry shader (accessor prefix `gp`) for vertex j from
// varying records laid out like `src` (one 16-byte slot per field); `rec` is the expression of
// slot $I of vertex j.
std::string Emitter::gsUnpack(const std::vector<Field>& src, const std::string& rec, const std::string& gp) {
    std::map<std::string, size_t> slot;
    for (size_t i = 0; i < src.size(); i++) slot[src[i].name] = i;
    auto gin = arrayedFields(Stage::Geometry, Storage::In);
    useStage(Stage::Geometry);
    std::string c;
    for (auto& f : gin) {
        auto it = slot.find(f.name);
        if (it == slot.end()) continue;
        std::string r = rec;
        for (size_t p; (p = r.find("$I")) != std::string::npos;) r.replace(p, 2, std::to_string(it->second));
        c += acc(f, gp, "j") + " = mgl_upk_" + mt(f.type) + "(" + r + ");\n";
    }
    return c;
}

// Geometry shader rasterized through a mesh shader. Inputs come from the vertex shader (run per
// input vertex) or, after tessellation, from the TES output records.
void Emitter::emitGeometry(std::string& o, bool fromTes) {
    const Reflection& R = RF(Stage::Geometry);
    int NV = primVerts(R.gsInputPrimitive);
    if (!NV) fail("unsupported geometry shader input primitive");
    uint32_t outPrim = R.gsOutputPrimitive;
    bool points = outPrim == gl::POINTS, lines = outPrim == gl::LINE_STRIP;
    int maxV = std::clamp(R.gsMaxVertices, 1, 256);
    int maxP = points ? maxV : lines ? std::max(1, maxV - 1) : std::max(1, maxV - 2);
    maxP = std::min(maxP, 512);
    std::string topo = points ? "point" : lines ? "line" : "triangle";
    std::vector<Field> vf;
    if (!fromTes) {
        emitBody(Stage::Vertex, o);
        vf = producerFields(Stage::Vertex);
        o += "struct Rec { uint4 s[" + std::to_string(std::max<size_t>(1, vf.size())) + "]; };\n}  // namespace vs\n\n";
    } else {
        vf = tesRec_;
    }
    // geometry stage
    prepareStage(Stage::Geometry);
    StageInfo& gsi = si_[(int)Stage::Geometry];
    bool writesPrimId = gsi.bi.count(BuiltinVar::PrimitiveID) && gsi.bi[BuiltinVar::PrimitiveID]->q.storage == Storage::Out;
    RasterOut ro = rasterOutput(Stage::Geometry, "", true, points);
    ro.primStruct += "    int mgl_primid [[user(mgl_primid)]];\n";
    ro.primCopy += std::string("p.mgl_primid = ") + (writesPrimId ? "bi_gl_PrimitiveID" : "mgl_primIn") + ";\n";
    std::string preG = "struct MV {\n" + ro.vertexStruct + "};\nstruct MP {\n" + ro.primStruct + "};\n";
    preG += "using MeshT = mesh<MV, MP, " + std::to_string(maxV) + ", " + std::to_string(maxP) + ", topology::" + topo + ">;\n";
    std::string members = "    thread MeshT* mgl_mo;\n    uint mgl_nv, mgl_np, mgl_strip;\n    int mgl_primIn;\n";
    std::string MV = std::to_string(maxV), MP = std::to_string(maxP);
    // only vertex stream 0 is rasterized
    std::string m = "    void mgl_EmitVertex(int s) {\n        if (s != 0 || mgl_nv >= " + MV + "u) return;\n        MV v;\n" +
                    indentBy(ro.vertexCopy, 8) + "        mgl_mo->set_vertex(mgl_nv, v);\n        uint c = mgl_nv++;\n        mgl_strip++;\n";
    std::string prim = "            MP p;\n" + indentBy(ro.primCopy, 12) + "            mgl_mo->set_primitive(mgl_np, p);\n            mgl_np++;\n";
    if (points)
        m += "        if (mgl_np < " + MP + "u) {\n            mgl_mo->set_index(mgl_np, c);\n" + prim + "        }\n";
    else if (lines)
        m += "        if (mgl_strip >= 2u && mgl_np < " + MP + "u) {\n            mgl_mo->set_index(2u * mgl_np, c);\n"
             "            mgl_mo->set_index(2u * mgl_np + 1u, c - 1u);\n" + prim + "        }\n";
    else
        m += "        if (mgl_strip >= 3u && mgl_np < " + MP + "u) {\n            uint a = c - 2u, b = c - 1u;\n"
             "            if (((mgl_strip - 3u) & 1u) != 0u) { uint t = a; a = b; b = t; }\n"
             "            mgl_mo->set_index(3u * mgl_np, c);\n            mgl_mo->set_index(3u * mgl_np + 1u, a);\n"
             "            mgl_mo->set_index(3u * mgl_np + 2u, b);\n" + prim + "        }\n";
    m += "    }\n    void mgl_EndPrimitive(int s) { if (s == 0) mgl_strip = 0u; }\n";
    emitBody(Stage::Geometry, o, preG, members, m);
    o += "}  // namespace gs\n\n";
    std::string e = "[[mesh]] void mgl_gs(gs::MeshT mgl_mo, constant MGLArgs& A [[buffer(0)]], constant MGLSys& sys [[buffer(1)]], "
                    "constant MGLDrawInfo& D [[buffer(22)]], uint3 mgl_tg [[threadgroup_position_in_grid]], uint mgl_tid [[thread_index_in_threadgroup]]) {\n";
    std::string N = std::to_string(NV);
    std::string loop = "        for (uint j = 0; j < " + N + "u; j++) {\n";
    if (!fromTes) {
        e += "    threadgroup vs::Rec mgl_vin[" + N + "];\n    uint mgl_prim = mgl_tg.x, mgl_inst = mgl_tg.y;\n";
        e += "    if (mgl_tid < " + N + "u) {\n";
        e += indentBy(runVS("vg", "D.indexBuffer[mgl_prim * " + N + "u + mgl_tid] + uint(D.baseVertex)", "mgl_inst"), 8);
        useStage(Stage::Vertex);
        for (size_t i = 0; i < vf.size(); i++) e += "        mgl_vin[mgl_tid].s[" + std::to_string(i) + "] = mgl_pk(" + acc(vf[i], "vg.") + ");\n";
        e += "    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        loop += indentBy(gsUnpack(vf, "mgl_vin[j].s[$I]", "g."), 12);
    } else {
        e += "    uint mgl_prim = mgl_tg.x;\n";
        loop = "        for (uint j = 0; j < " + std::to_string(std::min(NV, tesIpp_)) + "u; j++) {\n";
        loop += "            device const uint4* r = (device const uint4*)(D.tesVertexOut + D.tesPrimOut[mgl_prim * " + std::to_string(tesIpp_) +
                "u + j] * " + std::to_string(tesRB_) + "u);\n";
        loop += indentBy(gsUnpack(vf, "r[$I]", "g."), 12);
    }
    loop += "        }\n";
    e += "    if (mgl_tid == 0u) {\n";
    e += "        gs::G g = {};\n        g.A = " + argsOf(Stage::Geometry) + ";\n        g.sys = &sys;\n        g.mgl_mo = &mgl_mo;\n        g.mgl_nv = 0u;\n"
         "        g.mgl_np = 0u;\n        g.mgl_strip = 0u;\n        g.mgl_primIn = int(mgl_prim);\n";
    e += loop;
    useStage(Stage::Geometry);
    for (auto* v : gsi.builtins) {
        if (v->builtin == BuiltinVar::PrimitiveIDIn) e += "        g.bi_" + v->name + " = int(mgl_prim);\n";
        if (v->builtin == BuiltinVar::InvocationID) e += "        g.bi_" + v->name + " = int(mgl_tg.z);\n";
        if (v->builtin == BuiltinVar::PrimitiveID && v->q.storage == Storage::Out) e += "        g.bi_" + v->name + " = int(mgl_prim);\n";
    }
    e += "        g.mgl_init();\n        g." + gsi.mainName + "();\n        mgl_mo.set_primitive_count(g.mgl_np);\n    }\n}\n\n";
    entries_ += e;
    EntryPoint ep{"mgl_gs", Stage::Geometry, EntryPoint::Mesh};
    ep.threadsPerThreadgroup[0] = NV;
    ep.maxMeshVertices = maxV;
    ep.maxMeshPrimitives = maxP;
    out_.entryPoints.push_back(ep);
    fragInputsAvailable_ = ro.names;
    haveVsOutputs_ = true;
}

// ---------------------------------------------------------------------------
// Tessellation: pass 1 = VS + TCS kernel, pass 2 = object (patch) + mesh (TES), or
// TES records (mgl_tes_count / mgl_tes_gen) feeding a geometry shader or transform feedback.
// ---------------------------------------------------------------------------
void Emitter::emitTessControlPass(std::string& o) {
    const Reflection& R = RF(Stage::TessEval);
    ts_.hasTCS = SC(Stage::TessControl).main != nullptr;
    int OUTV = ts_.hasTCS ? RF(Stage::TessControl).tcsOutputVertices : 0;
    ts_.OUTV = OUTV;
    if (ts_.hasTCS && (OUTV < 1 || OUTV > 32)) fail("invalid tessellation control output vertex count");
    // VS
    emitBody(Stage::Vertex, o);
    auto vf = producerFields(Stage::Vertex);
    o += "}  // namespace vs\n\n";
    // TCS
    std::vector<Field> recV, recP;
    if (ts_.hasTCS) {
        emitBody(Stage::TessControl, o);
        o += "}  // namespace tcs\n\n";
        recV = arrayedFields(Stage::TessControl, Storage::Out);
        recP = patchFields(Stage::TessControl, Storage::Out);
    } else {
        recV = vf;
    }
    ts_.VB = 16 * std::max<size_t>(1, recV.size());
    ts_.PB = 32 + 16 * recP.size();
    std::string sVB = std::to_string(ts_.VB) + "u", sPB = std::to_string(ts_.PB) + "u";
    ts_.vslot.clear();
    ts_.pslot.clear();
    for (size_t i = 0; i < recV.size(); i++) ts_.vslot[recV[i].name] = i;
    for (size_t i = 0; i < recP.size(); i++) ts_.pslot[recP[i].name] = i;
    ts_.cps = ts_.hasTCS ? std::to_string(OUTV) + "u" : "D.patchVertices";

    std::string e = "kernel void mgl_tcs(constant MGLArgs& A [[buffer(0)]], constant MGLSys& sys [[buffer(1)]], constant MGLDrawInfo& D [[buffer(22)]], "
                    "uint3 mgl_tg [[threadgroup_position_in_grid]], uint mgl_tid [[thread_index_in_threadgroup]]) {\n";
    e += "    uint mgl_patch = mgl_tg.x, mgl_inst = mgl_tg.y, mgl_gp = mgl_inst * D.primitiveCount + mgl_patch, mgl_pv = D.patchVertices;\n";
    e += "    device uchar* mgl_pout = D.tcsPatchOut + mgl_gp * " + sPB + ";\n";
    std::string vsRun = runVS("vg", "D.indexBuffer[mgl_patch * mgl_pv + mgl_cp] + uint(D.baseVertex)", "mgl_inst");
    int threads1;
    if (ts_.hasTCS) {
        StageInfo& tsi = si_[(int)Stage::TessControl];
        threads1 = OUTV;
        if (tsi.hasShared) e += "    threadgroup tcs::MGLShared mgl_sh;\n    threadgroup tcs::MGLShared* S = &mgl_sh;\n";
        auto tin = arrayedFields(Stage::TessControl, Storage::In);
        std::map<std::string, size_t> vsIdx;
        for (size_t i = 0; i < vf.size(); i++) vsIdx[vf[i].name] = i;
        e += "    for (uint mgl_cp = mgl_tid; mgl_cp < mgl_pv; mgl_cp += " + std::to_string(OUTV) + "u) {\n";
        e += indentBy(vsRun, 8);
        for (auto& f : tin) {
            auto it = vsIdx.find(f.name);
            if (it == vsIdx.end()) continue;
            useStage(Stage::Vertex);
            std::string src = acc(vf[it->second], "vg.");
            useStage(Stage::TessControl);
            e += "        " + acc(f, "", "mgl_cp") + " = " + src + ";\n";
        }
        e += "    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        e += "    tcs::G g = {};\n    g.A = " + argsOf(Stage::TessControl) + ";\n    g.sys = &sys;\n";
        if (tsi.hasShared) e += "    g.S = S;\n";
        for (auto* v : tsi.builtins) {
            if (v->builtin == BuiltinVar::InvocationID) e += "    g.bi_" + v->name + " = int(mgl_tid);\n";
            if (v->builtin == BuiltinVar::PrimitiveID) e += "    g.bi_" + v->name + " = int(mgl_patch);\n";
            if (v->builtin == BuiltinVar::PatchVerticesIn) e += "    g.bi_" + v->name + " = int(mgl_pv);\n";
        }
        e += "    g.mgl_init();\n    g." + tsi.mainName + "();\n";
        e += "    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);\n";
        useStage(Stage::TessControl);
        e += "    {\n        device uint4* r = (device uint4*)(D.tcsVertexOut + (mgl_gp * " + std::to_string(OUTV) + "u + mgl_tid) * " + sVB + ");\n";
        for (size_t i = 0; i < recV.size(); i++) e += "        r[" + std::to_string(i) + "] = mgl_pk(" + acc(recV[i], "", "mgl_tid") + ");\n";
        e += "    }\n    if (mgl_tid == 0u) {\n        device float* lv = (device float*)mgl_pout;\n";
        const Variable* lo = tsi.bi.count(BuiltinVar::TessLevelOuter) ? tsi.bi[BuiltinVar::TessLevelOuter] : nullptr;
        const Variable* li = tsi.bi.count(BuiltinVar::TessLevelInner) ? tsi.bi[BuiltinVar::TessLevelInner] : nullptr;
        for (int i = 0; i < 4; i++) e += "        lv[" + std::to_string(i) + "] = " + (lo ? varName(lo) + "[" + std::to_string(i) + "]" : "1.0f") + ";\n";
        for (int i = 0; i < 2; i++) e += "        lv[" + std::to_string(4 + i) + "] = " + (li ? varName(li) + "[" + std::to_string(i) + "]" : "1.0f") + ";\n";
        e += "        device uint4* pr = (device uint4*)(mgl_pout + 32u);\n";
        for (size_t i = 0; i < recP.size(); i++) e += "        pr[" + std::to_string(i) + "] = mgl_pk(" + acc(recP[i], "") + ");\n";
        e += "    }\n}\n\n";
    } else {
        threads1 = 32;
        e += "    uint mgl_cp = mgl_tid;\n    if (mgl_cp < mgl_pv) {\n" + indentBy(vsRun, 8);
        useStage(Stage::Vertex);
        e += "        device uint4* r = (device uint4*)(D.tcsVertexOut + (mgl_gp * mgl_pv + mgl_cp) * " + sVB + ");\n";
        for (size_t i = 0; i < recV.size(); i++) e += "        r[" + std::to_string(i) + "] = mgl_pk(" + acc(recV[i], "vg.") + ");\n";
        e += "    }\n    if (mgl_tid == 0u) {\n        device float* lv = (device float*)mgl_pout;\n";
        e += "        for (int i = 0; i < 4; i++) lv[i] = sys.tessDefaultOuter[i];\n        lv[4] = sys.tessDefaultInner[0];\n"
             "        lv[5] = sys.tessDefaultInner[1];\n    }\n}\n\n";
    }
    entries_ += e;
    EntryPoint k1{"mgl_tcs", Stage::TessControl, EntryPoint::Kernel};
    k1.threadsPerThreadgroup[0] = threads1;
    out_.entryPoints.push_back(k1);
    // tessellator configuration shared by the TES passes
    uint32_t pm = R.tesPrimitiveMode;
    uint32_t domain = pm == gl::TRIANGLES ? 0 : pm == gl::QUADS ? 1 : 2;
    uint32_t spacing = R.tesSpacing == gl::FRACTIONAL_EVEN ? 1 : R.tesSpacing == gl::FRACTIONAL_ODD ? 2 : 0;
    ts_.setup = "mgl_tess_setup(" + std::to_string(domain) + "u, " + std::to_string(spacing) + "u, " + (R.tesPointMode ? "1u" : "0u") + ", " +
                (R.tesOrder == gl::CW ? "0u" : "1u") + ", float4(lv[0], lv[1], lv[2], lv[3]), float2(lv[4], lv[5]))";
    tesIpp_ = R.tesPointMode ? 1 : domain == 2 ? 2 : 3;
}

// TES input loading. Control points: loop over j (threads of the threadgroup); patch data and
// levels: statements for one thread (expects `lv` = the patch record).
std::string Emitter::tesLoadControlPoints(const std::string& gp, const std::string& cps, int NT) {
    useStage(Stage::TessEval);
    auto ein = arrayedFields(Stage::TessEval, Storage::In);
    std::string e = "for (uint j = mgl_tid; j < " + cps + "; j += " + std::to_string(NT) + "u) {\n";
    e += "    device const uint4* r = (device const uint4*)(D.tcsVertexOut + (" + gp + " * " + cps + " + j) * " + std::to_string(ts_.VB) + "u);\n";
    for (auto& f : ein) {
        auto it = ts_.vslot.find(f.name);
        if (it == ts_.vslot.end()) continue;
        e += "    " + acc(f, "", "j") + " = mgl_upk_" + mt(f.type) + "(r[" + std::to_string(it->second) + "]);\n";
    }
    return e + "}\n";
}

std::string Emitter::tesLoadPatch() {
    useStage(Stage::TessEval);
    StageInfo& esi = si_[(int)Stage::TessEval];
    std::string e;
    if (esi.bi.count(BuiltinVar::TessLevelOuter))
        for (int i = 0; i < 4; i++) e += varName(esi.bi[BuiltinVar::TessLevelOuter]) + "[" + std::to_string(i) + "] = lv[" + std::to_string(i) + "];\n";
    if (esi.bi.count(BuiltinVar::TessLevelInner))
        for (int i = 0; i < 2; i++) e += varName(esi.bi[BuiltinVar::TessLevelInner]) + "[" + std::to_string(i) + "] = lv[" + std::to_string(4 + i) + "];\n";
    auto epin = patchFields(Stage::TessEval, Storage::In);
    e += "device const uint4* pr = (device const uint4*)((device const uchar*)lv + 32u);\n";
    for (auto& f : epin) {
        auto it = ts_.pslot.find(f.name);
        if (it == ts_.pslot.end()) continue;
        e += acc(f, "") + " = mgl_upk_" + mt(f.type) + "(pr[" + std::to_string(it->second) + "]);\n";
    }
    return e;
}

// Runs the TES for one domain point into `tes::G g`.
std::string Emitter::tesRun(const std::string& coord, const std::string& patch, const std::string& cps) {
    useStage(Stage::TessEval);
    StageInfo& esi = si_[(int)Stage::TessEval];
    std::string e = "tes::G g = {};\ng.A = " + argsOf(Stage::TessEval) + ";\ng.sys = &sys;\n";
    if (esi.hasShared) e += "g.S = S;\n";
    for (auto* v : esi.builtins) {
        if (v->builtin == BuiltinVar::TessCoord) e += "g.bi_" + v->name + " = " + coord + ";\n";
        if (v->builtin == BuiltinVar::PrimitiveID) e += "g.bi_" + v->name + " = int(" + patch + ");\n";
        if (v->builtin == BuiltinVar::PatchVerticesIn) e += "g.bi_" + v->name + " = int(" + cps + ");\n";
    }
    return e + "g.mgl_init();\ng." + esi.mainName + "();\n";
}

void Emitter::emitTessEvalMesh(std::string& o) {
    const Reflection& R = RF(Stage::TessEval);
    bool points = R.tesPointMode;
    bool isolines = R.tesPrimitiveMode != gl::TRIANGLES && R.tesPrimitiveMode != gl::QUADS;
    std::string topo = points ? "point" : isolines ? "line" : "triangle";
    int ipp = tesIpp_;
    const int MAXV = 130, MAXP = 128, NT = 64;
    std::string sPB = std::to_string(ts_.PB) + "u";
    prepareStage(Stage::TessEval);
    StageInfo& esi = si_[(int)Stage::TessEval];
    RasterOut ro = rasterOutput(Stage::TessEval, "g.", true, points);
    ro.primStruct += "    int mgl_primid [[user(mgl_primid)]];\n";
    ro.primCopy += "p.mgl_primid = int(pl.patch);\n";
    std::string preG = "struct MV {\n" + ro.vertexStruct + "};\nstruct MP {\n" + ro.primStruct + "};\n";
    preG += "using MeshT = mesh<MV, MP, " + std::to_string(MAXV) + ", " + std::to_string(MAXP) + ", topology::" + topo + ">;\n";
    emitBody(Stage::TessEval, o, preG);
    o += "}  // namespace tes\n\n";
    o += "struct mgl_TPayload {\n    uint gp, patch, inst, cps;\n    mgl_tess T;\n};\n\n";
    std::string e = "[[object]] void mgl_tes_obj(object_data mgl_TPayload& pl [[payload]], mesh_grid_properties mgp, constant MGLSys& sys [[buffer(1)]], "
                    "constant MGLDrawInfo& D [[buffer(22)]], uint3 mgl_tg [[threadgroup_position_in_grid]], uint mgl_tid [[thread_index_in_threadgroup]]) {\n";
    e += "    if (mgl_tid != 0u) return;\n";
    e += "    uint gp = mgl_tg.y * D.primitiveCount + mgl_tg.x;\n";
    e += "    device const float* lv = (device const float*)(D.tcsPatchOut + gp * " + sPB + ");\n";
    e += "    pl.gp = gp;\n    pl.patch = mgl_tg.x;\n    pl.inst = mgl_tg.y;\n    pl.cps = " + ts_.cps + ";\n";
    e += "    pl.T = " + ts_.setup + ";\n";
    e += "    mgp.set_threadgroups_per_grid(uint3(pl.T.units, 1u, 1u));\n}\n\n";
    e += "[[mesh]] void mgl_tes(tes::MeshT mgl_mo, const object_data mgl_TPayload& pl [[payload]], constant MGLArgs& A [[buffer(0)]], "
         "constant MGLSys& sys [[buffer(1)]], constant MGLDrawInfo& D [[buffer(22)]], uint3 mgl_tg [[threadgroup_position_in_grid]], "
         "uint mgl_tid [[thread_index_in_threadgroup]]) {\n";
    if (esi.hasShared) e += "    threadgroup tes::MGLShared mgl_sh;\n    threadgroup tes::MGLShared* S = &mgl_sh;\n";
    e += "    threadgroup float3 mgl_co[" + std::to_string(MAXV) + "];\n    threadgroup float mgl_tp[" + std::to_string(MAXV) + "];\n";
    e += "    threadgroup ushort mgl_idx[" + std::to_string(MAXP * 3) + "];\n    threadgroup uint mgl_cnt[2];\n";
    e += "    device const float* lv = (device const float*)(D.tcsPatchOut + pl.gp * " + sPB + ");\n";
    e += indentBy(tesLoadControlPoints("pl.gp", "pl.cps", NT), 4);
    e += "    if (mgl_tid == 0u) {\n" + indentBy(tesLoadPatch(), 8);
    e += "        mgl_tess T = pl.T;\n        uint nv, np;\n        mgl_tess_unit(T, mgl_tg.x, mgl_co, mgl_tp, mgl_idx, nv, np);\n";
    e += "        mgl_cnt[0] = nv;\n        mgl_cnt[1] = np;\n    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    e += "    uint nv = mgl_cnt[0], np = mgl_cnt[1];\n";
    e += "    for (uint k = mgl_tid; k < nv; k += " + std::to_string(NT) + "u) {\n";
    e += indentBy(tesRun("mgl_co[k]", "pl.patch", "pl.cps"), 8);
    e += "        tes::MV v;\n" + indentBy(ro.vertexCopy, 8);
    e += "        mgl_mo.set_vertex(k, v);\n    }\n";
    e += "    for (uint k = mgl_tid; k < np * " + std::to_string(ipp) + "u; k += " + std::to_string(NT) + "u) mgl_mo.set_index(k, mgl_idx[k]);\n";
    e += "    for (uint k = mgl_tid; k < np; k += " + std::to_string(NT) + "u) {\n        tes::MP p;\n" + indentBy(ro.primCopy, 8) +
         "        mgl_mo.set_primitive(k, p);\n    }\n";
    e += "    if (mgl_tid == 0u) mgl_mo.set_primitive_count(np);\n}\n\n";
    entries_ += e;
    EntryPoint obj{"mgl_tes_obj", Stage::TessEval, EntryPoint::Object};
    obj.payloadBytes = 128;
    out_.entryPoints.push_back(obj);
    EntryPoint ms{"mgl_tes", Stage::TessEval, EntryPoint::Mesh};
    ms.threadsPerThreadgroup[0] = NT;
    ms.maxMeshVertices = MAXV;
    ms.maxMeshPrimitives = MAXP;
    out_.entryPoints.push_back(ms);
    fragInputsAvailable_ = ro.names;
    haveVsOutputs_ = true;
}

// TES output written to records (one per generated domain point of each unit) plus a primitive
// list in GL order: mgl_tes_count sizes each patch (itemCounts = primitives, vertices, units),
// the runtime computes prefix sums (itemOffsets = first primitive, first vertex) and the largest
// unit count, then mgl_tes_gen runs one threadgroup per (patch, unit).
void Emitter::emitTessEvalRecords(std::string& o) {
    const int MAXV = 130, MAXP = 128, NT = 64;
    std::string sPB = std::to_string(ts_.PB) + "u";
    prepareStage(Stage::TessEval);
    StageInfo& esi = si_[(int)Stage::TessEval];
    emitBody(Stage::TessEval, o);
    o += "}  // namespace tes\n\n";
    tesRec_ = producerFields(Stage::TessEval);
    tesRB_ = 16 * std::max<int>(1, (int)tesRec_.size());
    std::string ipp = std::to_string(tesIpp_) + "u";
    std::string e = "kernel void mgl_tes_count(constant MGLDrawInfo& D [[buffer(22)]], uint mgl_t [[thread_position_in_grid]]) {\n";
    e += "    if (mgl_t >= D.itemCount) return;\n";
    e += "    device const float* lv = (device const float*)(D.tcsPatchOut + mgl_t * " + sPB + ");\n";
    e += "    mgl_tess T = " + ts_.setup + ";\n";
    e += "    uint nv = 0u, np = 0u;\n    for (uint u = 0; u < T.units; u++) {\n        uint2 sz = mgl_tess_unit_size(T, u);\n"
         "        nv += sz.x;\n        np += sz.y;\n    }\n";
    e += "    D.itemCounts[mgl_t] = uint4(np, nv, T.units, 0u);\n}\n\n";
    e += "kernel void mgl_tes_gen(constant MGLArgs& A [[buffer(0)]], constant MGLSys& sys [[buffer(1)]], constant MGLDrawInfo& D [[buffer(22)]], "
         "uint3 mgl_tg [[threadgroup_position_in_grid]], uint mgl_tid [[thread_index_in_threadgroup]]) {\n";
    if (esi.hasShared) e += "    threadgroup tes::MGLShared mgl_sh;\n    threadgroup tes::MGLShared* S = &mgl_sh;\n";
    e += "    threadgroup float3 mgl_co[" + std::to_string(MAXV) + "];\n    threadgroup float mgl_tp[" + std::to_string(MAXV) + "];\n";
    e += "    threadgroup ushort mgl_idx[" + std::to_string(MAXP * 3) + "];\n    threadgroup uint mgl_cnt[4];\n";
    e += "    uint gp = mgl_tg.x, unit = mgl_tg.y;\n    if (gp >= D.itemCount) return;\n";
    e += "    device const float* lv = (device const float*)(D.tcsPatchOut + gp * " + sPB + ");\n";
    e += "    mgl_tess T = " + ts_.setup + ";\n    if (unit >= T.units) return;\n";
    e += "    uint cps = " + ts_.cps + ", patch = gp % D.primitiveCount;\n";
    e += indentBy(tesLoadControlPoints("gp", "cps", NT), 4);
    e += "    if (mgl_tid == 0u) {\n" + indentBy(tesLoadPatch(), 8);
    e += "        uint4 off = D.itemOffsets[gp];\n        uint vo = off.y, po = off.x;\n";
    e += "        for (uint u = 0; u < unit; u++) {\n            uint2 sz = mgl_tess_unit_size(T, u);\n            vo += sz.x;\n            po += sz.y;\n        }\n";
    e += "        uint nv, np;\n        mgl_tess_unit(T, unit, mgl_co, mgl_tp, mgl_idx, nv, np);\n";
    e += "        mgl_cnt[0] = nv;\n        mgl_cnt[1] = np;\n        mgl_cnt[2] = vo;\n        mgl_cnt[3] = po;\n    }\n";
    e += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    e += "    uint nv = mgl_cnt[0], np = mgl_cnt[1], vo = mgl_cnt[2], po = mgl_cnt[3];\n";
    e += "    for (uint k = mgl_tid; k < nv; k += " + std::to_string(NT) + "u) {\n";
    e += indentBy(tesRun("mgl_co[k]", "patch", "cps"), 8);
    e += "        device uint4* rec = (device uint4*)(D.tesVertexOut + (vo + k) * " + std::to_string(tesRB_) + "u);\n";
    useStage(Stage::TessEval);
    for (size_t i = 0; i < tesRec_.size(); i++) e += "        rec[" + std::to_string(i) + "] = mgl_pk(" + acc(tesRec_[i], "g.") + ");\n";
    e += "    }\n";
    e += "    for (uint k = mgl_tid; k < np * " + ipp + "; k += " + std::to_string(NT) + "u) D.tesPrimOut[po * " + ipp + " + k] = vo + uint(mgl_idx[k]);\n}\n\n";
    entries_ += e;
    EntryPoint c{"mgl_tes_count", Stage::TessEval, EntryPoint::Kernel};
    c.threadsPerThreadgroup[0] = 64;
    out_.entryPoints.push_back(c);
    EntryPoint gk{"mgl_tes_gen", Stage::TessEval, EntryPoint::Kernel};
    gk.threadsPerThreadgroup[0] = NT;
    out_.entryPoints.push_back(gk);
}

// Transform feedback of TES output (no geometry shader): one thread per TES primitive.
void Emitter::emitTesXfb(std::string& o) {
    (void)o;
    const MultiPassInfo& mp = RF(lastPreRaster()).multipass;
    std::string ipp = std::to_string(tesIpp_) + "u";
    std::string e = "kernel void mgl_tes_xfb(constant MGLArgs& A [[buffer(0)]], constant MGLSys& sys [[buffer(1)]], constant MGLDrawInfo& D [[buffer(22)]], "
                    "uint mgl_p [[thread_position_in_grid]]) {\n";
    e += "    if (mgl_p >= D.tesPrimCount || mgl_p >= D.xfbPrimLimit[0]) return;\n";
    e += "    for (uint k = 0; k < " + ipp + "; k++) {\n";
    e += "        device const uint4* r = (device const uint4*)(D.tesVertexOut + D.tesPrimOut[mgl_p * " + ipp + " + k] * " + std::to_string(tesRB_) + "u);\n";
    e += "        tes::G g = {};\n";
    useStage(Stage::TessEval);
    for (size_t i = 0; i < tesRec_.size(); i++) e += "        " + acc(tesRec_[i], "g.") + " = mgl_upk_" + mt(tesRec_[i].type) + "(r[" + std::to_string(i) + "]);\n";
    e += "        uint vtx = D.xfbStreamBase[0] + mgl_p * " + ipp + " + k;\n";
    std::vector<std::tuple<int, uint32_t, std::string>> words;
    captureWords(Stage::TessEval, "g.", words);
    for (auto& [b, off, w] : words)
        e += "        mgl_st_uint(D.xfbBuffer[" + std::to_string(b) + "], vtx * " + std::to_string(mp.xfbStride[b]) + "u + " + std::to_string(off) + "u, " + w + ");\n";
    e += "    }\n}\n\n";
    entries_ += e;
    EntryPoint k{"mgl_tes_xfb", Stage::TessEval, EntryPoint::Kernel};
    k.threadsPerThreadgroup[0] = 64;
    out_.entryPoints.push_back(k);
}

// ---------------------------------------------------------------------------
// Transform feedback capture: vertex shader as a kernel
// ---------------------------------------------------------------------------
void Emitter::captureWords(Stage st, const std::string& g, std::vector<std::tuple<int, uint32_t, std::string>>& out) {
    const Reflection& R = P->refl;
    prepareStage(st);
    useStage(st);
    StageInfo& si = si_[(int)st];
    for (auto& x : R.xfbVaryings) {
        const std::string& name = x.name;
        size_t p = 0;
        while (p < name.size() && name[p] != '.' && name[p] != '[') p++;
        std::string base = name.substr(0, p);
        std::string a;
        Type t;
        bool found = false;
        for (auto* v : si.builtins)
            if (v->name == base && v->q.storage == Storage::Out) { a = g + varName(v); t = v->type; found = true; }
        if (!found && base == "gl_Position") { a = g + "bi_gl_Position"; t = Type::vector(BT::Float, 4); found = true; }
        if (!found)
            for (auto* v : sc_->globals) {
                if (v->q.storage != Storage::Out) continue;
                if (v->name == base && !v->anonBlock) { a = g + varName(v); t = v->type; found = true; break; }
                if (v->block && v->block->name == base) { a = g + varName(v); t = v->type; found = true; break; }
                if (v->anonBlock && v->block->findMember(base) >= 0) {
                    int mi = v->block->findMember(base);
                    a = g + varName(v) + ".m_" + base;
                    t = v->block->members[mi].type;
                    found = true;
                    break;
                }
            }
        if (!found) fail("transform feedback varying '" + name + "' not found");
        while (p < name.size()) {
            if (name[p] == '[') {
                size_t q = name.find(']', p);
                a += name.substr(p, q - p + 1);
                t = t.elem();
                p = q + 1;
            } else if (name[p] == '.') {
                size_t q = p + 1;
                while (q < name.size() && name[q] != '.' && name[q] != '[') q++;
                std::string mem = name.substr(p + 1, q - p - 1);
                if (t.base != BT::Struct || t.sdef->findMember(mem) < 0) fail("bad transform feedback member '" + name + "'");
                t = t.sdef->members[t.sdef->findMember(mem)].type;
                a += ".m_" + mem;
                p = q;
            } else break;
        }
        std::vector<std::pair<Type, std::string>> L;
        flatten(t, a, L);
        uint32_t off = (uint32_t)x.offset;
        for (auto& [lt, la] : L) {
            int n = lt.vec;
            for (int c = 0; c < n; c++) {
                std::string comp = n == 1 ? la : la + "." + kSwz[c];
                if (lt.base == BT::Double) {
                    out.push_back({x.buffer, off, "as_type<uint2>(" + comp + ").x"});
                    out.push_back({x.buffer, off + 4, "as_type<uint2>(" + comp + ").y"});
                    off += 8;
                } else {
                    out.push_back({x.buffer, off, lt.base == BT::Bool ? "uint(" + comp + ")" : "as_type<uint>(" + comp + ")"});
                    off += 4;
                }
            }
        }
    }
}

// Transform feedback capture of the vertex shader outputs in `g` for list position mgl_t
// (instance-major) of the decomposed primitive list described by D. Primitives past
// D.xfbPrimLimit[0] (the CPU-computed buffer capacity) are not recorded.
std::string Emitter::xfbVertexStores() {
    const MultiPassInfo& mp = RF(lastPreRaster()).multipass;
    std::string e = "    uint vpp = max(D.verticesPerPrimitive, 1u);\n    if (mgl_t / vpp < D.xfbPrimLimit[0]) {\n";
    e += "        uint vtx = D.xfbVertexBase + mgl_t;\n";
    std::vector<std::tuple<int, uint32_t, std::string>> words;
    captureWords(Stage::Vertex, "g.", words);
    for (auto& [b, off, w] : words)
        e += "        mgl_st_uint(D.xfbBuffer[" + std::to_string(b) + "], vtx * " + std::to_string(mp.xfbStride[b]) + "u + " + std::to_string(off) + "u, " + w + ");\n";
    e += "    }\n";
    return e;
}

void Emitter::emitXfb(std::string& o) {
    const Reflection& R = RF(lastPreRaster());
    bool gs = SC(Stage::Geometry).main != nullptr;
    if (SC(Stage::TessEval).main) {
        emitTessControlPass(o);
        emitTessEvalRecords(o);
        if (gs) emitGeometryXfb(o, true);
        else emitTesXfb(o);
        return;
    }
    if (gs) {
        emitGeometryXfb(o, false);
        return;
    }
    emitBody(Stage::Vertex, o);
    o += "}  // namespace vs\n\n";
    useStage(Stage::Vertex);
    std::string e = "kernel void mgl_xfb(constant MGLArgs& A [[buffer(0)]], constant MGLSys& sys [[buffer(1)]], constant MGLDrawInfo& D [[buffer(22)]], "
                    "uint mgl_t [[thread_position_in_grid]]) {\n";
    e += "    uint mgl_total = D.count * D.instanceCount;\n    if (mgl_t >= mgl_total) return;\n";
    e += "    uint mgl_pos = mgl_t % D.count, mgl_ii = mgl_t / D.count;\n";
    e += indentBy(runVS("g", "D.indexBuffer[mgl_pos] + uint(D.baseVertex)", "mgl_ii"), 4);
    e += xfbVertexStores();
    e += "}\n\n";
    entries_ += e;
    EntryPoint k{"mgl_xfb", Stage::Vertex, EntryPoint::Kernel};
    k.threadsPerThreadgroup[0] = 64;
    out_.entryPoints.push_back(k);
}

// Geometry shader output captured in GL order, in two passes over the work items (input
// primitive x GS invocation): mgl_gs_count runs the GS counting the primitives it emits per
// vertex stream (itemCounts), the runtime computes per-stream prefix sums (itemOffsets) and how
// many primitives fit (xfbPrimLimit), and mgl_gs_write runs it again writing the captured
// vertices of each completed primitive (strips decomposed into lists) at its final position.
void Emitter::emitGeometryXfb(std::string& o, bool fromTes) {
    const Reflection& R = RF(Stage::Geometry);
    const MultiPassInfo& mp = RF(lastPreRaster()).multipass;
    int NV = primVerts(R.gsInputPrimitive);
    if (!NV) fail("unsupported geometry shader input primitive");
    uint32_t outPrim = R.gsOutputPrimitive;
    int vpp = outPrim == gl::POINTS ? 1 : outPrim == gl::LINE_STRIP ? 2 : 3;
    std::string VPP = std::to_string(vpp) + "u";
    std::vector<Field> vf;
    if (!fromTes) {
        emitBody(Stage::Vertex, o);
        vf = producerFields(Stage::Vertex);
        o += "struct Rec { uint4 s[" + std::to_string(std::max<size_t>(1, vf.size())) + "]; };\n}  // namespace vs\n\n";
    } else {
        vf = tesRec_;
    }
    prepareStage(Stage::Geometry);
    useStage(Stage::Geometry);
    std::vector<std::tuple<int, uint32_t, std::string>> words;
    captureWords(Stage::Geometry, "", words);
    // captured words per vertex stream
    std::vector<size_t> sw[4];
    for (size_t i = 0; i < words.size(); i++) sw[mp.xfbBufferStream[std::get<0>(words[i]) & 3] & 3].push_back(i);
    std::string members = "    constant MGLDrawInfo* mgl_D;\n    uint mgl_mode, mgl_t;\n    uint mgl_cnt[4], mgl_strip[4], mgl_np[4];\n    uint4 mgl_off;\n"
                          "    int mgl_primIn;\n";
    // GL order of the vertices of the primitive completed by the last emitted vertex (strips decomposed)
    auto orderCode = [](int vpp, const std::string& pad) -> std::string {
        if (vpp == 1) return pad + "uint order[1] = {(e - 1u) % 3u};\n";
        if (vpp == 2) return pad + "uint order[2] = {(e - 2u) % 3u, (e - 1u) % 3u};\n";
        return pad + "uint a = (e - 3u) % 3u, b = (e - 2u) % 3u;\n" + pad + "if (((st - 3u) & 1u) != 0u) { uint t = a; a = b; b = t; }\n" + pad +
               "uint order[3] = {a, b, (e - 1u) % 3u};\n";
    };
    // primitives one item can emit per stream (max_vertices bounds them)
    const int maxPrimsRec = std::clamp(R.gsMaxVertices, 1, 256);
    for (int s = 0; s < 4; s++)
        if (!sw[s].empty()) members += "    uint mgl_ring" + std::to_string(s) + "[3][" + std::to_string(sw[s].size()) + "];\n";
    std::string m = "    void mgl_EmitVertex(int s) {\n        if (s < 0 || s > 3) return;\n        uint su = uint(s);\n";
    for (int s = 0; s < 4; s++) {
        if (sw[s].empty()) continue;
        std::string S = std::to_string(s);
        m += "        if (su == " + S + "u && (mgl_mode != 0u || mgl_D->gsRecord != nullptr)) {\n            uint slot = mgl_cnt[" + S + "] % 3u;\n";
        for (size_t i = 0; i < sw[s].size(); i++)
            m += "            mgl_ring" + S + "[slot][" + std::to_string(i) + "] = " + std::get<2>(words[sw[s][i]]) + ";\n";
        m += "        }\n";
    }
    m += "        mgl_cnt[su]++;\n        mgl_strip[su]++;\n        if (mgl_strip[su] < " + VPP + ") return;\n";
    // record layout per item: stream s area at recBase[s], local primitive p, vertex k, word i
    uint32_t recBase[4], recWords = 0;
    for (int s = 0; s < 4; s++) {
        recBase[s] = recWords;
        recWords += (uint32_t)(maxPrimsRec * vpp * sw[s].size());
    }
    m += "        uint pi = mgl_np[su]++;\n";
    m += "        if (mgl_mode == 0u) {\n            if (mgl_D->gsRecord == nullptr) return;\n";
    for (int s = 0; s < 4; s++) {
        if (sw[s].empty()) continue;
        std::string S = std::to_string(s);
        m += "            if (su == " + S + "u && pi < " + std::to_string(maxPrimsRec) + "u) {\n";
        m += "                uint e = mgl_cnt[" + S + "], st = mgl_strip[" + S + "];\n";
        m += orderCode(vpp, "                ");
        m += "                device uint* r = mgl_D->gsRecord + mgl_t * " + std::to_string(recWords) + "u + " + std::to_string(recBase[s]) +
             "u + pi * " + std::to_string(vpp * sw[s].size()) + "u;\n";
        m += "                for (uint k = 0; k < " + VPP + "; k++) {\n";
        for (size_t i = 0; i < sw[s].size(); i++)
            m += "                    r[k * " + std::to_string(sw[s].size()) + "u + " + std::to_string(i) + "u] = mgl_ring" + S + "[order[k]][" + std::to_string(i) + "];\n";
        m += "                }\n            }\n";
    }
    m += "            return;\n        }\n        pi += mgl_off[su];\n"
         "        if (pi >= mgl_D->xfbPrimLimit[su]) return;\n";
    for (int s = 0; s < 4; s++) {
        if (sw[s].empty()) continue;
        std::string S = std::to_string(s);
        m += "        if (su == " + S + "u) {\n            uint e = mgl_cnt[" + S + "], st = mgl_strip[" + S + "];\n";
        m += orderCode(vpp, "            ");
        m += "            for (uint k = 0; k < " + VPP + "; k++) {\n                uint vtx = mgl_D->xfbStreamBase[" + S + "] + pi * " + VPP + " + k;\n";
        for (size_t i = 0; i < sw[s].size(); i++) {
            auto& [b, off, w] = words[sw[s][i]];
            (void)w;
            m += "                mgl_st_uint(mgl_D->xfbBuffer[" + std::to_string(b) + "], vtx * " + std::to_string(mp.xfbStride[b]) + "u + " +
                 std::to_string(off) + "u, mgl_ring" + S + "[order[k]][" + std::to_string(i) + "]);\n";
        }
        m += "            }\n        }\n";
    }
    m += "    }\n    void mgl_EndPrimitive(int s) { if (s >= 0 && s < 4) mgl_strip[uint(s)] = 0u; }\n";
    emitBody(Stage::Geometry, o, "", members, m);
    o += "}  // namespace gs\n\n";
    StageInfo& gsi = si_[(int)Stage::Geometry];
    std::string N = std::to_string(NV);
    int INV = std::max(1, R.gsInvocations);
    std::string e = "static inline void mgl_gs_xfb_run(uint mgl_mode, uint mgl_t, constant MGLArgs& A, constant MGLSys& sys, constant MGLDrawInfo& D) {\n";
    e += "    uint mgl_invoc = mgl_t % " + std::to_string(INV) + "u, mgl_pi = mgl_t / " + std::to_string(INV) + "u;\n";
    std::string loop;
    if (!fromTes) {
        e += "    uint mgl_prim = mgl_pi % D.primitiveCount, mgl_inst = mgl_pi / D.primitiveCount;\n";
        e += "    vs::Rec mgl_vin[" + N + "];\n    for (uint mgl_j = 0; mgl_j < " + N + "u; mgl_j++) {\n";
        e += indentBy(runVS("vg", "D.indexBuffer[mgl_prim * " + N + "u + mgl_j] + uint(D.baseVertex)", "mgl_inst"), 8);
        useStage(Stage::Vertex);
        for (size_t i = 0; i < vf.size(); i++) e += "        mgl_vin[mgl_j].s[" + std::to_string(i) + "] = mgl_pk(" + acc(vf[i], "vg.") + ");\n";
        e += "    }\n";
        loop = "    for (uint j = 0; j < " + N + "u; j++) {\n" + indentBy(gsUnpack(vf, "mgl_vin[j].s[$I]", "g."), 8) + "    }\n";
    } else {
        e += "    uint mgl_prim = mgl_pi;\n";
        loop = "    for (uint j = 0; j < " + std::to_string(std::min(NV, tesIpp_)) + "u; j++) {\n";
        loop += "        device const uint4* r = (device const uint4*)(D.tesVertexOut + D.tesPrimOut[mgl_prim * " + std::to_string(tesIpp_) +
                "u + j] * " + std::to_string(tesRB_) + "u);\n";
        loop += indentBy(gsUnpack(vf, "r[$I]", "g."), 8) + "    }\n";
    }
    e += "    gs::G g = {};\n    g.A = " + argsOf(Stage::Geometry) + ";\n    g.sys = &sys;\n    g.mgl_D = &D;\n    g.mgl_mode = mgl_mode;\n    g.mgl_t = mgl_t;\n"
         "    for (int k = 0; k < 4; k++) { g.mgl_cnt[k] = 0u; g.mgl_strip[k] = 0u; g.mgl_np[k] = 0u; }\n"
         "    g.mgl_off = mgl_mode != 0u ? D.itemOffsets[mgl_t] : uint4(0u);\n    g.mgl_primIn = int(mgl_prim);\n";
    e += loop;
    useStage(Stage::Geometry);
    for (auto* v : gsi.builtins) {
        if (v->builtin == BuiltinVar::PrimitiveIDIn) e += "    g.bi_" + v->name + " = int(mgl_prim);\n";
        if (v->builtin == BuiltinVar::InvocationID) e += "    g.bi_" + v->name + " = int(mgl_invoc);\n";
    }
    e += "    g.mgl_init();\n    g." + gsi.mainName + "();\n";
    e += "    if (mgl_mode == 0u) D.itemCounts[mgl_t] = uint4(g.mgl_np[0], g.mgl_np[1], g.mgl_np[2], g.mgl_np[3]);\n}\n\n";
    // mgl_gs_copy: places the primitives recorded by mgl_gs_count (one GS execution in total, so
    // side effects such as atomic counters happen exactly once per invocation)
    {
        e += "kernel void mgl_gs_copy(constant MGLDrawInfo& D [[buffer(22)]], uint mgl_t [[thread_position_in_grid]]) {\n"
             "    if (mgl_t >= D.itemCount) return;\n    uint4 n = D.itemCounts[mgl_t], off = D.itemOffsets[mgl_t];\n";
        for (int s = 0; s < 4; s++) {
            if (sw[s].empty()) continue;
            std::string S = std::to_string(s), WS = std::to_string(sw[s].size());
            e += "    for (uint p = 0; p < min(n[" + S + "], " + std::to_string(maxPrimsRec) + "u); p++) {\n";
            e += "        uint pi = off[" + S + "] + p;\n        if (pi >= D.xfbPrimLimit[" + S + "]) break;\n";
            e += "        device const uint* r = D.gsRecord + mgl_t * " + std::to_string(recWords) + "u + " + std::to_string(recBase[s]) +
                 "u + p * " + std::to_string(vpp * sw[s].size()) + "u;\n";
            e += "        for (uint k = 0; k < " + VPP + "; k++) {\n            uint vtx = D.xfbStreamBase[" + S + "] + pi * " + VPP + " + k;\n";
            for (size_t i = 0; i < sw[s].size(); i++) {
                auto& [b, off, w] = words[sw[s][i]];
                (void)w;
                e += "            mgl_st_uint(D.xfbBuffer[" + std::to_string(b) + "], vtx * " + std::to_string(mp.xfbStride[b]) + "u + " +
                     std::to_string(off) + "u, r[k * " + WS + "u + " + std::to_string(i) + "u]);\n";
            }
            e += "        }\n    }\n";
        }
        e += "}\n\n";
        EntryPoint k{"mgl_gs_copy", Stage::Geometry, EntryPoint::Kernel};
        k.threadsPerThreadgroup[0] = 64;
        k.scratchWordsPerItem = recWords;
        out_.entryPoints.push_back(k);
    }
    for (const char* nm : {"mgl_gs_count", "mgl_gs_write"}) {
        e += std::string("kernel void ") + nm + "(constant MGLArgs& A [[buffer(0)]], constant MGLSys& sys [[buffer(1)]], constant MGLDrawInfo& D [[buffer(22)]], "
             "uint mgl_t [[thread_position_in_grid]]) {\n    if (mgl_t < D.itemCount) mgl_gs_xfb_run(" + (std::string(nm) == "mgl_gs_count" ? "0u" : "1u") +
             ", mgl_t, A, sys, D);\n}\n\n";
        EntryPoint k{nm, Stage::Geometry, EntryPoint::Kernel};
        k.threadsPerThreadgroup[0] = 64;
        out_.entryPoints.push_back(k);
    }
    entries_ += e;
}

MSLOutput Emitter::run() {
    try {
        uint32_t m = stageMask_;
        bool gs = m & (1u << (int)Stage::Geometry), tess = m & (1u << (int)Stage::TessEval);
        std::string o;
        if (!multi_) emitArgs(o);
        switch (K.kind) {
            case PipelineKind::Compute:
                if (!SC(Stage::Compute).main) fail("program has no compute shader");
                emitCS(o);
                break;
            case PipelineKind::Classic:
                if (gs || tess) fail("geometry/tessellation programs need PipelineKind::MeshGeometry/Tessellation");
                if (SC(Stage::Vertex).main) emitClassicVS(o);
                if (SC(Stage::Fragment).main) emitFS(o, false);
                break;
            case PipelineKind::MeshGeometry:
                if (!gs) fail("program has no geometry shader");
                if (tess) fail("tessellation programs need PipelineKind::Tessellation");
                emitGeometry(o, false);
                if (SC(Stage::Fragment).main) emitFS(o, true);
                else emitNullFS();
                break;
            case PipelineKind::Tessellation:
                if (!tess) fail("program has no tessellation evaluation shader");
                emitTessControlPass(o);
                if (gs) {
                    emitTessEvalRecords(o);
                    emitGeometry(o, true);
                } else {
                    emitTessEvalMesh(o);
                }
                if (SC(Stage::Fragment).main) emitFS(o, true);
                else emitNullFS();
                break;
            case PipelineKind::XfbCapture:
                emitXfb(o);
                break;
        }
        if (multi_) {
            // one argument buffer per stage program; unused stages keep a placeholder
            o += "struct MGLArgs {\n";
            for (int i = 0; i < (int)Stage::Count; i++)
                o += argsEmitted_[i] ? std::string("    constant ") + kStageNs[i] + "::MGLArgs* " + kStageNs[i] + ";\n"
                                     : std::string("    ulong ") + kStageNs[i] + ";\n";
            o += "};\n\n";
        }
        o += entries_;
        out_.source = buildMSLPrelude(o) + "\n" + o;
    } catch (EmitError& e) {
        out_.error = e.msg;
        out_.source.clear();
        out_.entryPoints.clear();
    }
    return std::move(out_);
}

std::string Emitter::recordSizes(MultiPassInfo& mp) {
    try {
        if (!SC(Stage::TessEval).main) return "";
        bool hasTCS = SC(Stage::TessControl).main != nullptr;
        size_t nv, np;
        if (hasTCS) {
            nv = arrayedFields(Stage::TessControl, Storage::Out).size();
            np = patchFields(Stage::TessControl, Storage::Out).size();
            mp.tcsOutputVertices = (uint32_t)RF(Stage::TessControl).tcsOutputVertices;
        } else {
            nv = producerFields(Stage::Vertex).size();
            np = 0;
            mp.tcsOutputVertices = 0;
        }
        mp.tcsVertexOutBytes = (uint32_t)(16 * std::max<size_t>(1, nv));
        mp.tcsPatchOutBytes = (uint32_t)(32 + 16 * np);
        mp.tesVertexOutBytes = (uint32_t)(16 * std::max<size_t>(1, producerFields(Stage::TessEval).size()));
    } catch (EmitError& e) {
        return e.msg;
    }
    return "";
}

}  // namespace

std::string fillMultipassRecords(const LinkedProgram& prog, MultiPassInfo& mp) {
    static const VariantKey key;
    Emitter e(prog, key);
    return e.recordSizes(mp);
}

MSLOutput generateMSL(const LinkedProgram& program, const VariantKey& key) {
    Emitter e(program, key);
    return e.run();
}

}  // namespace glsl

namespace glsl {

MSLOutput generateMSL(const LinkedProgram* const stages[(int)Stage::Count], const VariantKey& key) {
    Emitter e(stages, key);
    return e.run();
}

MultiPassInfo multipassInfo(const LinkedProgram* const stages[(int)Stage::Count]) {
    static const VariantKey key;
    // transform feedback layout comes from the program of the last vertex-processing stage
    MultiPassInfo mp;
    for (Stage st : {Stage::Geometry, Stage::TessEval, Stage::Vertex}) {
        const LinkedProgram* p = stages[(int)st];
        if (p && (p->stageMask & (1u << (int)st))) {
            mp = p->refl.multipass;
            break;
        }
    }
    Emitter e(stages, key);
    e.recordSizes(mp);
    return mp;
}

}  // namespace glsl
