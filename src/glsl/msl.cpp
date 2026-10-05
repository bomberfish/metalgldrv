// GLSL AST -> Metal Shading Language.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <unordered_map>

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

class Emitter {
public:
    Emitter(const LinkedProgram& p, const VariantKey& k) : P(p), K(k) {}
    MSLOutput run();

private:
    const LinkedProgram& P;
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
    int tempN_ = 0;
    std::map<const Function*, std::string> fnNames_;
    std::set<std::string> fragInputsAvailable_;  // from the vertex stage
    bool haveVsOutputs_ = false;
    bool hasShared_ = false;
    bool needEq_ = false;

    [[noreturn]] void fail(const std::string& m) { throw EmitError{m}; }

    // ---- types -------------------------------------------------------------------
    static std::string scalarName(BT b) {
        switch (b) {
            case BT::Float: return "float";
            case BT::Double: return "mgl_f64";
            case BT::Int: return "int";
            case BT::UInt: return "uint";
            case BT::Bool: return "bool";
            default: return "void";
        }
    }
    static std::string vecName(BT b, int n) {
        if (n == 1) return scalarName(b);
        if (b == BT::Double) return "mgl_f64v" + std::to_string(n);
        return scalarName(b) + std::to_string(n);
    }
    std::string structName(const StructDef* sd) { return (sd->isBlock ? "B_" : "S_") + sd->name; }
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
            if (t.base == BT::Double) return "mgl_f64m" + std::to_string(t.cols) + "x" + std::to_string(t.vec);
            return "float" + std::to_string(t.cols) + "x" + std::to_string(t.vec);
        }
        return vecName(t.base, t.vec);
    }

    // ---- literals ----------------------------------------------------------------
    std::string scalarLit(BT b, const Scalar& s) {
        char buf[64];
        switch (b) {
            case BT::Float:
            case BT::Double: {
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
            std::string r = mt(t) + "(";
            for (int c = 0; c < t.cols; c++) {
                r += (c ? ", " : "") + vecName(t.base, t.vec) + "(";
                for (int k = 0; k < t.vec; k++) r += (k ? ", " : "") + scalarLit(t.base, cv.c[i++]);
                r += ")";
            }
            return r + ")";
        }
        if (t.vec > 1) {
            std::string r = mt(t) + "(";
            for (int k = 0; k < t.vec; k++) r += (k ? ", " : "") + scalarLit(t.base, cv.c[i++]);
            return r + ")";
        }
        return scalarLit(t.base, cv.c[i++]);
    }

    // ---- temporaries -------------------------------------------------------------
    std::string newTemp(const Type& t) {
        std::string n = "_t" + std::to_string(tempN_++);
        temps_.push_back(mt(t) + " " + n + ";");
        return n;
    }
    std::string newTempS(const std::string& type) {
        std::string n = "_t" + std::to_string(tempN_++);
        temps_.push_back(type + " " + n + ";");
        return n;
    }
    std::string bind(const std::string& expr, const std::string& type, std::string& pre) {
        if (isSimpleIdent(expr)) return expr;
        std::string t = newTempS(type);
        pre += t + " = " + expr + ", ";
        return t;
    }

    // ---- variables -----------------------------------------------------------------
    std::string varName(const Variable* v) {
        if (v->builtin != BuiltinVar::None) return "bi_" + v->name;
        if (v->global) {
            if (v->q.storage == Storage::Shared) return "S->g_" + v->name;
            return "g_" + v->name;
        }
        if (v->name.empty()) return "_p" + std::to_string(v->id);
        return v->name + "_" + std::to_string(v->id);
    }
    bool isDefaultUniform(const Variable* v) {
        return v->global && v->q.storage == Storage::Uniform && !v->block && !v->type.containsOpaque() &&
               v->builtin != BuiltinVar::DepthRange;
    }
    const BlockBinding* blockOf(const Variable* v) {
        if (!v->block || !(v->q.storage == Storage::Uniform || v->q.storage == Storage::Buffer)) return nullptr;
        auto it = P.blocks.find(std::string(v->q.storage == Storage::Buffer ? "B:" : "U:") + v->block->name);
        if (it == P.blocks.end()) fail("internal: block '" + v->block->name + "' has no binding");
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
    std::string withOuts(const std::string& callee, const Expr* e, const std::vector<uint8_t>& outs, const std::vector<Type>& ptypes,
                         const Type& ret, const std::function<std::string(size_t, const Expr*)>& argFn);

    // ---- statements ------------------------------------------------------------------
    void stmt(const Stmt* s, std::string& o, int ind);
    void block(const Stmt* s, std::string& o, int ind);
    std::string fnSignature(const Function* f);
    std::string emitFunction(const Function* f);

    // ---- stage -------------------------------------------------------------------------
    void emitStage(Stage s, std::string& o);
    void collectStructs(std::vector<const StructDef*>& order);
    void emitArgs(std::string& o);
    void flatten(const Type& t, const std::string& acc, std::vector<std::pair<Type, std::string>>& leaves);
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
                auto it = P.defaultOffsets.find(v->name);
                if (it == P.defaultOffsets.end()) fail("internal: uniform '" + v->name + "' has no offset");
                a.ptr = "A->s" + std::to_string(P.defaultSlot);
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
                a.ptr = "A->s" + std::to_string(a.bb->slot) + "[" + idx + "]";
                a.off = "0u";
                a.type = e->type;
                a.layout = a.bb->layout;
                if (a.bb->storage) a.sizeExpr = "A->s" + std::to_string(a.bb->sizeSlot) + "[" + idx + "]";
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
            body += "    r.m_" + m.name + " = " + loaderFor(m.type, L->members[i]) + "(p, o + " + std::to_string(L->offsets[i]) + "u);\n";
        }
        body += "    return r;\n";
    } else if (t.cols) {
        uint32_t N = scalarSize(t.base);
        std::string col = vecName(t.base, t.vec);
        body = "    return " + T + "(";
        for (int c = 0; c < t.cols; c++) {
            if (c) body += ", ";
            if (!L->rowMajor) body += "mgl_ld_" + col + "(p, o + " + std::to_string(c * L->matrixStride) + "u)";
            else {
                body += col + "(";
                for (int r = 0; r < t.vec; r++)
                    body += (r ? ", " : "") + std::string("mgl_ld_") + scalarName(t.base) + "(p, o + " + std::to_string(r * L->matrixStride + c * N) + "u)";
                body += ")";
            }
        }
        body += ");\n";
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
        if (a.type.base != BT::Float) fail("row-major double matrices are not supported yet");
        return "mgl_ldcs_" + mt(a.type) + "(" + a.ptr + ", " + a.off + ", " + std::to_string(a.compStride) + "u)";
    }
    return loaderFor(a.type, a.layout) + "(" + a.ptr + ", " + a.off + ")";
}

std::string Emitter::storeAcc(const Acc& a, const std::string& v) {
    if (!a.device) fail("cannot write to a uniform");
    if (a.compStride) {
        if (a.type.base != BT::Float) fail("row-major double matrices are not supported yet");
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
            if (v->q.storage == Storage::Shared || v->q.storage == Storage::Uniform || v->q.storage == Storage::Buffer) return false;
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
std::string Emitter::binary(Op op, const std::string& a, const Type& ta, const std::string& b, const Type& tb, const Type& rt) {
    bool ma = ta.isMatrix(), mb = tb.isMatrix();
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
        bool expand = l->type.isMatrix() || r->type.isMatrix() || (bop == Op::Mul && l->type.isVector() && r->type.isMatrix());
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
                if (v->builtin == BuiltinVar::WorkGroupSize) {
                    const auto& r = P.refl;
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
        return sname + "(" + s + ")";
    }
    int n = T.components();
    if (args.size() == 1) {
        const Expr* a = args[0];
        const Type& at = a->type;
        if (at.isScalar()) {
            if (T.isMatrix()) return "mgl_mdiag<" + sname + ", " + std::to_string(T.cols) + ", " + std::to_string(T.vec) + ">(" + sname + "(" + ex(a) + "))";
            return mt(T) + "(" + ex(a) + ")";
        }
        if (T.isVector() && at.isVector()) {
            if (at.vec == T.vec) return mt(T) + "(" + ex(a) + ")";
            if (at.vec > T.vec) return mt(T) + "(" + ex(a) + "." + std::string(kSwz, T.vec) + ")";
        }
        if (T.isMatrix() && at.isMatrix()) {
            if (at.cols == T.cols && at.vec == T.vec) return mt(T) + "(" + ex(a) + ")";
            std::string src = ex(a);
            if (at.base != T.base) src = mt(at.withBase(T.base)) + "(" + src + ")";
            return "mgl_mresize<" + sname + ", " + std::to_string(T.cols) + ", " + std::to_string(T.vec) + ">(" + src + ")";
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
            r += (i ? ", " : "") + vecName(T.base, take) + "(" + s + ")";
            used += take;
        }
        return r + ")";
    }
    if (T.isMatrix() && !anyMat && (int)args.size() == T.cols) {
        bool cols = true;
        for (auto* a : args) if (!(a->type.isVector() && a->type.vec == T.vec)) cols = false;
        if (cols) {
            std::string r = mt(T) + "(";
            for (size_t i = 0; i < args.size(); i++) r += (i ? ", " : "") + vecName(T.base, T.vec) + "(" + ex(args[i]) + ")";
            return r + ")";
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
    if (T.isVector()) {
        r = mt(T) + "(";
        for (int k = 0; k < n; k++) r += (k ? ", " : "") + sname + "(" + comps[k] + ")";
        r += ")";
    } else {
        r = mt(T) + "(";
        for (int c = 0; c < T.cols; c++) {
            r += (c ? ", " : "") + vecName(T.base, T.vec) + "(";
            for (int k = 0; k < T.vec; k++) r += (k ? ", " : "") + sname + "(" + comps[c * T.vec + k] + ")";
            r += ")";
        }
        r += ")";
    }
    (void)total;
    return pre.empty() ? r : "(" + pre + r + ")";
}

std::string Emitter::withOuts(const std::string& callee, const Expr* e, const std::vector<uint8_t>& outs,
                              const std::vector<Type>& ptypes, const Type& ret,
                              const std::function<std::string(size_t, const Expr*)>& argFn) {
    std::string pre, post, args;
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
        if (!s.empty()) args += (args.empty() ? "" : ", ") + s;
    }
    std::string call = callee + "(" + args + ")";
    if (post.empty()) return pre.empty() ? call : "(" + pre + call + ")";
    post.resize(post.size() - 2);
    if (ret.isVoid()) return "(" + pre + call + ", " + post + ")";
    std::string r = newTemp(ret);
    return "(" + pre + r + " = " + call + ", " + post + ", " + r + ")";
}

std::pair<std::string, std::string> Emitter::texRef(const Expr* e) {
    if (e->kind == EK::Var) {
        const Variable* v = e->var;
        if (v->param) {
            std::string n = varName(v);
            return {n + "_t", n + "_s"};
        }
        auto it = P.opaque.find(v->name);
        if (it == P.opaque.end()) fail("internal: opaque uniform '" + v->name + "' has no binding");
        const OpaqueBinding& ob = it->second;
        return {"A->s" + std::to_string(ob.slot), ob.samplerSlot >= 0 ? "A->s" + std::to_string(ob.samplerSlot) : ""};
    }
    if (e->kind == EK::Index && e->args[0]->kind == EK::Var && !e->args[0]->var->param) {
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
        const OpaqueBinding& ob = P.opaque.at(v->name);
        std::string off = std::to_string(ob.acOffset) + "u";
        if (!idx.empty()) off = "(" + off + " + " + idx + " * 4u)";
        return "((device atomic_uint*)(A->s" + std::to_string(ob.slot) + " + " + off + "))";
    }
    std::string T = a->type.base == BT::Int ? "atomic_int" : "atomic_uint";
    Acc acc;
    if (bufAccess(a, acc)) {
        if (!acc.device) fail("atomic operation on a uniform");
        return "((device " + T + "*)(" + acc.ptr + " + " + acc.off + "))";
    }
    const Expr* r = a;
    while (r->kind == EK::Index || r->kind == EK::Field) r = r->args[0];
    if (r->kind == EK::Var && r->var->q.storage == Storage::Shared) return "((threadgroup " + T + "*)&(" + ex(a) + "))";
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
    if (n == "interpolateAtCentroid" || n == "interpolateAtSample" || n == "interpolateAtOffset") return a(0);
    if (n == "noise1") return "0.0f";
    if (n == "noise2" || n == "noise3" || n == "noise4") return mt(rt) + "(0.0f)";
    if (n == "allInvocationsEqual" || n == "allInvocationsEqualARB") {
        std::string pre;
        std::string v = bind(a(0), "bool", pre);
        return "(" + pre + "simd_all(" + v + " == simd_broadcast_first(" + v + ")))";
    }
    if (n == "modf" || n == "frexp" || n == "uaddCarry" || n == "usubBorrow" || n == "umulExtended" || n == "imulExtended") {
        std::string callee = n == "modf" ? "modf" : n == "frexp" ? "frexp" : n == "uaddCarry" ? "mgl_uaddCarry"
                           : n == "usubBorrow" ? "mgl_usubBorrow" : "mgl_mulExtended";
        return withOuts(callee, e, bi->paramOut, bi->params, rt, [&](size_t i, const Expr* x) { return ex(x); });
    }
    if (n == "ldexp") return "ldexp(" + a(0) + ", " + a(1) + ")";
    if (n == "barrier") {
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
        auto it = P.defaultOffsets.find("gl_ModelViewProjectionMatrix");
        if (it == P.defaultOffsets.end()) fail("internal: ftransform without gl_ModelViewProjectionMatrix");
        Type m4 = Type::matrix(BT::Float, 4, 4);
        return "(" + loaderFor(m4, computeLayout(m4, Packing::Std430, false)) + "(A->s" + std::to_string(P.defaultSlot) + ", " +
               std::to_string(it->second) + "u) * bi_gl_Vertex)";
    }
    if (n == "EmitVertex" || n == "EndPrimitive" || n == "EmitStreamVertex" || n == "EndStreamPrimitive")
        fail("geometry shaders are not supported yet");
    fail("builtin function '" + n + "' is not supported yet");
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
        if (off) P = "(" + P + " + " + ex(A.back()) + ")";
        std::string r;
        if (T.ms) {
            std::string s = "uint(" + ex(A[2]) + ")";
            if (arr) { P = bind(P, "int3", pre); r = tex + ".read(uint2(" + P + ".xy), uint(" + P + ".z), " + s + ")"; }
            else r = tex + ".read(uint2(" + P + "), " + s + ")";
            return finish(r);
        }
        if (dim == Dim::Buffer) return finish(tex + ".read(uint(" + P + "))");
        std::string lod = (dim == Dim::Rect || A.size() < 3 || (off && A.size() < 4)) ? "" : ", " + lodArg(2);
        if (dim == Dim::D1) {
            if (arr) { P = bind(P, "int2", pre); r = tex + ".read(uint2(uint(" + P + ".x), 0u), uint(" + P + ".y)" + lod + ")"; }
            else r = tex + ".read(uint2(uint(" + P + "), 0u)" + lod + ")";
        } else if (dim == Dim::D3) {
            r = tex + ".read(uint3(" + P + ")" + lod + ")";
        } else if (arr) {
            P = bind(P, "int3", pre);
            r = tex + ".read(uint2(" + P + ".xy), uint(" + P + ".z)" + lod + ")";
        } else {
            r = tex + ".read(uint2(" + P + ")" + lod + ")";
        }
        return finish(r);
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
            if (name == "textureGatherOffsets") fail("textureGatherOffsets is not supported yet");
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
    if (n == "imageStore") {
        if (T.ms) fail("imageStore to multisample images is not supported");
        return finish(tex + ".write(" + ex(A[ai]) + ", " + loc + ")");
    }
    std::string S = scalarName(T.sampled);
    std::string V = vecName(T.sampled, 4);
    if (T.sampled == BT::Float) fail("float image atomics are not supported by Metal");
    if (n == "imageAtomicCompSwap") {
        if (layer.empty()) return finish("mgl_img_cas(" + tex + ", " + coord + ", " + S + "(" + ex(A[ai]) + "), " + S + "(" + ex(A[ai + 1]) + "))");
        return finish("mgl_img_cas_a(" + tex + ", " + coord + ", " + layer + ", " + S + "(" + ex(A[ai]) + "), " + S + "(" + ex(A[ai + 1]) + "))");
    }
    static const std::map<std::string, std::string> ops = {
        {"imageAtomicAdd", "atomic_fetch_add"}, {"imageAtomicMin", "atomic_fetch_min"}, {"imageAtomicMax", "atomic_fetch_max"},
        {"imageAtomicAnd", "atomic_fetch_and"}, {"imageAtomicOr", "atomic_fetch_or"},   {"imageAtomicXor", "atomic_fetch_xor"},
        {"imageAtomicExchange", "atomic_exchange"}};
    auto it = ops.find(n);
    if (it == ops.end()) fail("image function '" + n + "' is not supported");
    return finish(tex + "." + it->second + "(" + loc + ", " + V + "(" + ex(A[ai]) + ")).x");
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
                if (v->init) o += " = " + ex(v->init);
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
                    for (auto* v : b->vars) o += ind(in + 1) + mt(v->type) + " " + varName(v) + ";\n";
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
            o += ind(in) + "return" + (s->expr ? " " + ex(s->expr) : "") + ";\n";
            return;
        case SK::Discard: o += ind(in) + "discard_fragment();\n"; return;
    }
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
            if (t.isArray()) fail("arrays of samplers as function parameters are not supported");
            r += texType(t, 0) + " " + n + "_t, sampler " + n + "_s";
            continue;
        }
        if (t.base == BT::Image) {
            if (t.isArray()) fail("arrays of images as function parameters are not supported");
            r += texType(t, p->q.memory) + " " + n + "_t";
            continue;
        }
        if (t.base == BT::AtomicUint) {
            r += "device atomic_uint* " + n;
            continue;
        }
        if (t.isArray() && t.dims[0] == 0) fail("unsized array parameter");
        if (p->q.storage == Storage::Out || p->q.storage == Storage::InOut) r += "thread " + mt(t) + "& " + n;
        else r += mt(t) + " " + n;
    }
    return r + ")";
}

std::string Emitter::emitFunction(const Function* f) {
    temps_.clear();
    std::string bodyText;
    for (auto* s : f->body->body) stmt(s, bodyText, 2);
    std::string r = "    " + fnSignature(f) + " {\n";
    for (auto& t : temps_) r += "        " + t + "\n";
    r += bodyText;
    r += "    }\n";
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
    for (auto* v : sc_->used) visitType(v->type);
    std::function<void(const Expr*)> ve = [&](const Expr* e) {
        if (!e) return;
        visitType(e->type);
        for (auto* a : e->args) ve(a);
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
    const auto& slots = P.refl.slots;
    std::vector<std::string> decl(slots.size());
    std::vector<bool> skip(slots.size(), false);
    if (P.defaultSlot >= 0) decl[P.defaultSlot] = "constant uchar* s" + std::to_string(P.defaultSlot) + ";";
    for (auto& [key, bb] : P.blocks) {
        std::string arr = bb.count > 1 ? "[" + std::to_string(bb.count) + "]" : "";
        decl[bb.slot] = std::string(bb.storage ? "device uchar* s" : "constant uchar* s") + std::to_string(bb.slot) + arr + ";";
        for (int i = 1; i < bb.count; i++) skip[bb.slot + i] = true;
        if (bb.storage) {
            decl[bb.sizeSlot] = "ulong s" + std::to_string(bb.sizeSlot) + arr + ";";
            for (int i = 1; i < bb.count; i++) skip[bb.sizeSlot + i] = true;
        }
    }
    for (auto& [name, ob] : P.opaque) {
        auto group = [&](int slot, const std::string& type) {
            if (ob.count > 1) decl[slot] = "array<" + type + ", " + std::to_string(ob.count) + "> s" + std::to_string(slot) + ";";
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
    for (auto& acb : P.refl.atomicCounterBuffers) decl[acb.slot] = "device uchar* s" + std::to_string(acb.slot) + ";";
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

void Emitter::emitStage(Stage s, std::string& o) {
    stage_ = s;
    sc_ = &P.stages[(int)s];
    helpers_.clear();
    helperByKey_.clear();
    needEq_ = false;
    fnNames_.clear();
    tempN_ = 0;
    static const char* nsNames[] = {"vs", "tcs", "tes", "gs", "fs", "cs"};
    ns_ = nsNames[(int)s];

    // function names
    std::map<std::string, int> counts;
    for (auto& [m, f] : sc_->defs) {
        if (!sc_->reachable.count(f)) continue;
        fnNames_[f] = "f_" + f->name + "_" + std::to_string(counts[f->name]++);
    }

    std::string structs, members, funcs, init;
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

    // members of the global struct
    members += "    constant MGLArgs* A;\n    constant MGLSys* sys;\n";
    std::string sharedDecl;
    std::set<std::string> declared;
    for (auto* v : sc_->globals) {
        if (declared.count(v->name)) continue;
        declared.insert(v->name);
        switch (v->q.storage) {
            case Storage::Uniform:
            case Storage::Buffer: continue;
            case Storage::Shared:
                sharedDecl += "    " + mt(v->type) + " g_" + v->name + ";\n";
                continue;
            default: break;
        }
        if (v->type.isArray() && v->type.dims[0] == 0) fail("unsized global array '" + v->name + "'");
        members += "    " + mt(v->type) + " " + varName(v) + ";\n";
    }
    hasShared_ = !sharedDecl.empty();
    if (hasShared_) members += "    threadgroup MGLShared* S;\n";
    // builtin variables
    std::set<const Variable*> builtins;
    std::map<BuiltinVar, const Variable*> biByKind;
    for (auto* v : sc_->used) {
        if (v->builtin == BuiltinVar::None || v->builtin == BuiltinVar::CompatUniform || v->builtin == BuiltinVar::DepthRange ||
            v->builtin == BuiltinVar::WorkGroupSize || v->q.storage == Storage::Const || v->q.storage == Storage::Uniform)
            continue;
        if (v->builtin == BuiltinVar::In || v->builtin == BuiltinVar::Out) fail("gl_in/gl_out are not supported yet");
        builtins.insert(v);
    }
    // ftransform needs gl_Vertex
    std::set<std::string> biNames;
    for (auto* v : builtins) {
        if (biNames.count(v->name)) continue;
        biNames.insert(v->name);
        biByKind[v->builtin] = v;
        members += "    " + mt(v->type) + " bi_" + v->name + ";\n";
    }
    bool needPosition = s == Stage::Vertex && !biNames.count("gl_Position");
    if (needPosition) members += "    float4 bi_gl_Position;\n";

    // functions
    for (auto& [m, f] : sc_->defs)
        if (sc_->reachable.count(f)) funcs += emitFunction(f);
    // global initializers
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

    o += "namespace " + ns_ + " {\n";
    if (needEq_) o += "using ::mgl_eq;\n";
    o += structs;
    for (auto* sd : needEq_ ? order : std::vector<const StructDef*>{}) {
        // equality helper for structs (used by == on aggregates)
        std::string n = structName(sd);
        o += "inline bool mgl_eq(" + n + " a, " + n + " b) { return true";
        for (auto& m : sd->members)
            if (!(m.type.isArray() && m.type.dims[0] == 0)) o += " && mgl_eq(a.m_" + m.name + ", b.m_" + m.name + ")";
        o += "; }\n";
    }
    if (hasShared_) o += "struct MGLShared {\n" + sharedDecl + "};\n";
    o += helpers_;
    o += "struct G {\n" + members + init + funcs + "};\n";

    // ---------------- entry point ----------------
    auto mainName = fnNames_.at(sc_->main);
    std::string e;
    if (s == Stage::Vertex) {
        std::string inS = "struct In {\n", outS = "struct Out {\n", copyIn, copyOut;
        bool anyIn = false;
        // attributes
        for (auto* v : sc_->used) {
            if (v->q.storage != Storage::In || !(v->builtin == BuiltinVar::None || v->builtin == BuiltinVar::Vertex ||
                                                  v->builtin == BuiltinVar::Normal || v->builtin == BuiltinVar::Color ||
                                                  v->builtin == BuiltinVar::SecondaryColor || v->builtin == BuiltinVar::MultiTexCoord ||
                                                  v->builtin == BuiltinVar::FogCoord))
                continue;
            auto it = P.attribLocations.find(v->name);
            if (it == P.attribLocations.end()) continue;
            int loc = it->second;
            std::vector<std::pair<Type, std::string>> leaves;
            flatten(v->type, "g." + varName(v), leaves);
            for (auto& [lt, acc] : leaves) {
                Type at = lt;
                if (at.base == BT::Double) at.base = BT::Float;
                if (at.base == BT::Bool) fail("boolean vertex attributes are not allowed");
                inS += "    " + vecName(at.base, at.vec) + " a" + std::to_string(loc) + " [[attribute(" + std::to_string(loc) + ")]];\n";
                copyIn += "    " + acc + " = " + mt(lt) + "(in.a" + std::to_string(loc) + ");\n";
                loc++;
                anyIn = true;
            }
        }
        inS += "};\n";
        // outputs
        auto posV = biByKind.count(BuiltinVar::Position) ? biByKind[BuiltinVar::Position] : nullptr;
        outS += std::string("    float4 mgl_pos [[position") + (posV && posV->q.invariant ? ", invariant" : "") + "]];\n";
        outS += "    float mgl_psize [[point_size]];\n";
        copyOut += "    o.mgl_pos = g.bi_gl_Position;\n    o.mgl_pos.y = -o.mgl_pos.y;\n";
        copyOut += "    if (sys.depthMode == 0u) o.mgl_pos.z = (o.mgl_pos.z + o.mgl_pos.w) * 0.5f;\n";
        if (biByKind.count(BuiltinVar::PointSize))
            copyOut += "    o.mgl_psize = (sys.flags & 1u) != 0u ? g.bi_gl_PointSize : sys.pointSize;\n";
        else
            copyOut += "    o.mgl_psize = sys.pointSize;\n";
        if (biByKind.count(BuiltinVar::ClipDistance) || biByKind.count(BuiltinVar::ClipVertex)) {
            int nclip = biByKind.count(BuiltinVar::ClipDistance) ? biByKind[BuiltinVar::ClipDistance]->type.dims[0] : 8;
            outS += "    float mgl_clip [[clip_distance]] [" + std::to_string(nclip) + "];\n";
            for (int i = 0; i < nclip; i++) {
                std::string val;
                if (biByKind.count(BuiltinVar::ClipDistance)) val = "g.bi_gl_ClipDistance[" + std::to_string(i) + "]";
                else {
                    auto it = P.defaultOffsets.find("gl_ClipPlane");
                    if (it == P.defaultOffsets.end()) fail("internal: gl_ClipVertex without gl_ClipPlane");
                    val = "dot(g.bi_gl_ClipVertex, mgl_ld_float4(A.s" + std::to_string(P.defaultSlot) + ", " + std::to_string(it->second + 16 * i) + "u))";
                }
                copyOut += "    o.mgl_clip[" + std::to_string(i) + "] = ((sys.clipDistanceMask >> " + std::to_string(i) + "u) & 1u) != 0u ? " + val + " : 1.0f;\n";
            }
        }
        if (biByKind.count(BuiltinVar::Layer)) {
            outS += "    uint mgl_layer [[render_target_array_index]];\n";
            copyOut += "    o.mgl_layer = uint(g.bi_gl_Layer);\n";
        }
        if (biByKind.count(BuiltinVar::ViewportIndex)) {
            outS += "    uint mgl_vp [[viewport_array_index]];\n";
            copyOut += "    o.mgl_vp = uint(g.bi_gl_ViewportIndex);\n";
        }
        // varyings
        std::vector<const Variable*> outs;
        for (auto* v : sc_->globals)
            if (v->q.storage == Storage::Out && v->builtin == BuiltinVar::None) outs.push_back(v);
        for (auto& [k, v] : biByKind)
            if (k == BuiltinVar::FrontColor || k == BuiltinVar::FrontSecondaryColor || k == BuiltinVar::BackColor ||
                k == BuiltinVar::BackSecondaryColor || k == BuiltinVar::TexCoord || k == BuiltinVar::FogFragCoord)
                outs.push_back(v);
        std::set<std::string> names;
        for (auto* v : outs) {
            std::string key = varyingKey(v, s);
            std::vector<std::pair<Type, std::string>> leaves;
            flatten(v->type, "g." + varName(v), leaves);
            for (size_t i = 0; i < leaves.size(); i++) {
                std::string nm = key + "_" + std::to_string(i);
                if (names.count(nm)) continue;
                names.insert(nm);
                Type lt = leaves[i].first;
                if (lt.base == BT::Bool) fail("boolean varyings are not allowed");
                outS += "    " + mt(lt) + " " + nm + " [[user(" + nm + ")]];\n";
                copyOut += "    o." + nm + " = " + leaves[i].second + ";\n";
            }
        }
        fragInputsAvailable_ = names;
        haveVsOutputs_ = true;
        outS += "};\n";
        o += inS + outS + "}  // namespace " + ns_ + "\n\n";
        e += "vertex vs::Out mgl_vs(";
        if (anyIn) e += "vs::In in [[stage_in]], ";
        e += "constant MGLArgs& A [[buffer(0)]], constant MGLSys& sys [[buffer(1)]], uint mgl_vid [[vertex_id]], uint mgl_iid [[instance_id]]";
        e += ", uint mgl_bi [[base_instance]]";
        if (biByKind.count(BuiltinVar::BaseVertex)) e += ", uint mgl_bv [[base_vertex]]";
        e += ") {\n    vs::G g = {};\n    g.A = &A;\n    g.sys = &sys;\n";
        e += copyIn;
        for (auto* v : builtins) {
            if (v->builtin == BuiltinVar::VertexID) e += "    g.bi_" + v->name + " = int(mgl_vid);\n";
            if (v->builtin == BuiltinVar::InstanceID) e += "    g.bi_" + v->name + " = int(mgl_iid - mgl_bi);\n";
            if (v->builtin == BuiltinVar::BaseVertex) e += "    g.bi_" + v->name + " = int(mgl_bv);\n";
            if (v->builtin == BuiltinVar::BaseInstance) e += "    g.bi_" + v->name + " = int(mgl_bi);\n";
            if (v->builtin == BuiltinVar::DrawID) e += "    g.bi_" + v->name + " = sys.drawID;\n";
            if (v->builtin == BuiltinVar::PointSize) e += "    g.bi_" + v->name + " = sys.pointSize;\n";
        }
        e += "    g.mgl_init();\n    g." + mainName + "();\n    vs::Out o;\n" + copyOut + "    return o;\n}\n\n";
        out_.entryPoints.push_back({"mgl_vs", Stage::Vertex, EntryPoint::Vertex});
    } else if (s == Stage::Fragment) {
        std::string inS = "struct In {\n", outS = "struct Out {\n", copyIn, copyOut, params;
        bool anyIn = false, anyOut = false;
        // varying inputs
        std::vector<const Variable*> ins;
        for (auto* v : sc_->globals)
            if (v->q.storage == Storage::In && v->builtin == BuiltinVar::None) ins.push_back(v);
        for (auto& [k, v] : biByKind)
            if (k == BuiltinVar::FragColorIn || k == BuiltinVar::SecondaryColorIn || k == BuiltinVar::TexCoord ||
                k == BuiltinVar::FogFragCoord)
                ins.push_back(v);
        std::set<std::string> names;
        for (auto* v : ins) {
            std::string key = varyingKey(v, s);
            std::vector<std::pair<Type, std::string>> leaves;
            flatten(v->type, "g." + varName(v), leaves);
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
                    // per-member interpolation of interface blocks
                    size_t dot = leaves[i].second.find(".m_", 2 + varName(v).size());
                    if (dot != std::string::npos) {
                        std::string mn = leaves[i].second.substr(dot + 3);
                        mn = mn.substr(0, mn.find_first_of(".["));
                        int mi = v->block->findMember(mn);
                        if (mi >= 0) lq = v->block->members[mi].q;
                    }
                }
                inS += "    " + mt(lt) + " " + nm + " [[user(" + nm + ")" + interpAttr(lq, lt) + "]];\n";
                copyIn += "    " + leaves[i].second + " = in." + nm + ";\n";
                anyIn = true;
            }
        }
        inS += "};\n";
        // builtins
        for (auto* v : builtins) {
            switch (v->builtin) {
                case BuiltinVar::FragCoord: {
                    params += ", float4 mgl_fc [[position]]";
                    std::string fc = "mgl_fc";
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
                    params += ", uint mgl_sid [[sample_id]]";
                    copyIn += "    g.bi_" + v->name + " = int(mgl_sid);\n";
                    break;
                case BuiltinVar::SamplePosition:
                    if (!biByKind.count(BuiltinVar::SampleID)) params += ", uint mgl_sid [[sample_id]]";
                    params += ", float4 mgl_fc2 [[position]]";
                    copyIn += "    g.bi_" + v->name + " = fract(mgl_fc2.xy);\n";
                    break;
                case BuiltinVar::SampleMaskIn:
                    params += ", uint mgl_smi [[sample_mask]]";
                    copyIn += "    g.bi_" + v->name + "[0] = int(mgl_smi);\n";
                    break;
                case BuiltinVar::PrimitiveID:
                    params += ", uint mgl_pid [[primitive_id]]";
                    copyIn += "    g.bi_" + v->name + " = int(mgl_pid);\n";
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
                case BuiltinVar::CullDistance:
                    break;  // not available as fragment inputs in Metal; reads return 0
                default: break;
            }
        }
        bool fragDepth = biByKind.count(BuiltinVar::FragDepth) && sc_->writesFragDepth;
        if (fragDepth && !biByKind.count(BuiltinVar::FragCoord)) params += ", float4 mgl_fc [[position]]";
        if (fragDepth) copyIn += "    g.bi_gl_FragDepth = mgl_fc.z;\n";
        if (biByKind.count(BuiltinVar::SampleMask)) copyIn += "    g.bi_gl_SampleMask[0] = -1;\n";
        // outputs
        bool color0Float = false;
        std::string color0;
        for (auto* v : sc_->globals) {
            if (v->q.storage != Storage::Out || v->builtin != BuiltinVar::None) continue;
            auto it = P.fragOutputs.find(v->name);
            if (it == P.fragOutputs.end()) continue;
            auto [loc, idx] = it->second;
            int n = v->type.isArray() ? v->type.dims[0] : 1;
            Type et = v->type.stripArrays();
            for (int i = 0; i < n; i++) {
                std::string acc = "g." + varName(v) + (v->type.isArray() ? "[" + std::to_string(i) + "]" : "");
                std::string fn = "mgl_c" + std::to_string(loc + i) + (idx ? "_1" : "");
                std::string vt = vecName(et.base == BT::Double ? BT::Float : et.base, 4);
                outS += "    " + vt + " " + fn + " [[color(" + std::to_string(loc + i) + ")" + (idx ? ", index(1)" : "") + "]];\n";
                std::string val = acc;
                if (et.vec == 1) val = vt + "(" + acc + ", 0, 0, 1)";
                else if (et.vec == 2) val = vt + "(" + acc + ", 0, 1)";
                else if (et.vec == 3) val = vt + "(" + acc + ", 1)";
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
        if (biByKind.count(BuiltinVar::SampleMask)) {
            outS += "    uint mgl_mask [[sample_mask]];\n";
            copyOut += "    o.mgl_mask = uint(g.bi_gl_SampleMask[0]);\n";
            anyOut = true;
        }
        outS += "};\n";
        std::string alpha;
        if (color0Float && sc_->primary->compat)
            alpha = "    if (sys.alphaFunc != 0x0207u && !mgl_alpha_pass(" + color0 + ".a, sys.alphaFunc, sys.alphaRef)) discard_fragment();\n";
        o += inS + outS + "}  // namespace " + ns_ + "\n\n";
        if (sc_->inLayout.earlyFragmentTests) e += "[[early_fragment_tests]]\n";
        e += std::string("fragment ") + (anyOut ? "fs::Out" : "void") + " mgl_fs(";
        if (anyIn) e += "fs::In in [[stage_in]], ";
        e += "constant MGLArgs& A [[buffer(0)]], constant MGLSys& sys [[buffer(1)]]" + params + ") {\n";
        e += "    fs::G g = {};\n    g.A = &A;\n    g.sys = &sys;\n" + copyIn;
        e += "    g.mgl_init();\n    g." + mainName + "();\n";
        if (anyOut) e += "    fs::Out o;\n" + copyOut + alpha + "    return o;\n";
        e += "}\n\n";
        out_.entryPoints.push_back({"mgl_fs", Stage::Fragment, EntryPoint::Fragment});
    } else if (s == Stage::Compute) {
        o += "}  // namespace " + ns_ + "\n\n";
        const auto& r = P.refl;
        e += "kernel void mgl_cs(constant MGLArgs& A [[buffer(0)]], constant MGLSys& sys [[buffer(1)]], uint3 mgl_gid [[thread_position_in_grid]], "
             "uint3 mgl_lid [[thread_position_in_threadgroup]], uint3 mgl_wid [[threadgroup_position_in_grid]], uint mgl_lidx [[thread_index_in_threadgroup]], "
             "uint3 mgl_nwg [[threadgroups_per_grid]]) {\n";
        if (hasShared_) e += "    threadgroup cs::MGLShared mgl_shared;\n";
        e += "    cs::G g;\n    g.A = &A;\n    g.sys = &sys;\n";
        if (hasShared_) e += "    g.S = &mgl_shared;\n";
        for (auto* v : builtins) {
            switch (v->builtin) {
                case BuiltinVar::NumWorkGroups: e += "    g.bi_" + v->name + " = mgl_nwg;\n"; break;
                case BuiltinVar::WorkGroupID: e += "    g.bi_" + v->name + " = mgl_wid;\n"; break;
                case BuiltinVar::LocalInvocationID: e += "    g.bi_" + v->name + " = mgl_lid;\n"; break;
                case BuiltinVar::GlobalInvocationID: e += "    g.bi_" + v->name + " = mgl_gid;\n"; break;
                case BuiltinVar::LocalInvocationIndex: e += "    g.bi_" + v->name + " = mgl_lidx;\n"; break;
                default: break;
            }
        }
        e += "    g.mgl_init();\n    g." + mainName + "();\n}\n\n";
        EntryPoint ep{"mgl_cs", Stage::Compute, EntryPoint::Kernel};
        ep.threadsPerThreadgroup[0] = r.localSize[0];
        ep.threadsPerThreadgroup[1] = r.localSize[1];
        ep.threadsPerThreadgroup[2] = r.localSize[2];
        out_.entryPoints.push_back(ep);
    } else {
        fail("this shader stage is not supported yet");
    }
    o += e;
}

MSLOutput Emitter::run() {
    try {
        if (K.kind != PipelineKind::Classic && K.kind != PipelineKind::Compute)
            fail("pipeline kind not supported yet (geometry/tessellation/transform feedback)");
        uint32_t m = P.stageMask;
        if (m & ((1u << (int)Stage::Geometry) | (1u << (int)Stage::TessControl) | (1u << (int)Stage::TessEval)))
            fail("geometry and tessellation shaders are not supported yet");
        std::string o;
        emitArgs(o);
        for (Stage s : {Stage::Vertex, Stage::Fragment, Stage::Compute})
            if (P.stages[(int)s].main) emitStage(s, o);
        out_.source = buildMSLPrelude(o) + "\n" + o;
    } catch (EmitError& e) {
        out_.error = e.msg;
        out_.source.clear();
        out_.entryPoints.clear();
    }
    return std::move(out_);
}

}  // namespace

MSLOutput generateMSL(const LinkedProgram& program, const VariantKey& key) {
    Emitter e(program, key);
    return e.run();
}

}  // namespace glsl
