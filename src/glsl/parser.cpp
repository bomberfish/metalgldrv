#include <functional>
#include <map>
// Recursive-descent GLSL parser with integrated semantic analysis.
#include "glsl/parser.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

#include "glsl/builtins.h"
#include "glsl/fold.h"
#include "glsl/gl_enums.h"
#include "glsl/types.h"

namespace glsl {

namespace {
// Implementation limits (gl_Max* constants), for compile-time checks.
enum : int {
#define X(name, glenum, value) kLimit_##name = value,
#include "glsl/limits.def"
#undef X
};
constexpr int kMaxAtomicCounterBufferSize = kLimit_gl_MaxAtomicCounterBufferSize;
}  // namespace

bool exprIsLvalue(const Expr* e) {
    switch (e->kind) {
        case EK::Var: {
            const Variable* v = e->var;
            switch (v->q.storage) {
                case Storage::Const:
                case Storage::Uniform:
                    return false;
                case Storage::In:
                case Storage::Attribute:
                    return v->param;  // 'in' parameters are writable local copies
                case Storage::Buffer:
                    return !(v->q.memory & MemReadOnly);
                default:
                    return true;
            }
        }
        case EK::Field: {
            const StructDef* sd = e->args[0]->type.sdef;
            if (sd && sd->isBlock && e->field >= 0 && (sd->members[e->field].q.memory & MemReadOnly)) return false;
            return exprIsLvalue(e->args[0]);
        }
        case EK::Index:
            return exprIsLvalue(e->args[0]);
        case EK::Swizzle: {
            for (int i = 0; i < e->nswz; i++)
                for (int j = i + 1; j < e->nswz; j++)
                    if (e->swz[i] == e->swz[j]) return false;
            return exprIsLvalue(e->args[0]);
        }
        default:
            return false;
    }
}

bool Expr::isLvalue() const { return exprIsLvalue(this); }

namespace {

int primitiveVertexCount(uint32_t prim);

struct ParseError {};

struct Sym {
    Variable* var = nullptr;
    int member = -1;  // anonymous block member index
    StructDef* type = nullptr;
    std::vector<Function*> funcs;
    Function* subrType = nullptr;  // subroutine type declared with this name
};

const std::unordered_set<std::string>& qualifierKeywords() {
    static const std::unordered_set<std::string> s = {
        "const", "in", "out", "inout", "attribute", "varying", "uniform", "buffer", "shared", "centroid", "sample", "patch",
        "flat", "smooth", "noperspective", "invariant", "precise", "coherent", "volatile", "restrict", "readonly",
        "writeonly", "lowp", "mediump", "highp", "layout", "subroutine"};
    return s;
}

class Parser {
public:
    Parser(Shader& sh, const CompileOptions& opts, std::string& log) : sh_(sh), opts_(opts), log_(log) {}
    bool run(const PPResult& pp);

private:
    Shader& sh_;
    const CompileOptions& opts_;
    std::string& log_;
    const std::vector<PPToken>* toks_ = nullptr;
    size_t p_ = 0;
    bool preamble_ = false;
    std::vector<std::unordered_map<std::string, Sym>> scopes_;
    Function* curFn_ = nullptr;
    int loopDepth_ = 0, switchDepth_ = 0;
    int nextId_ = 1;
    int nonPrecisionQuals_ = 0;
    std::map<int, int> atomicNextOffset_;  // atomic counter binding -> next default offset
    std::map<int, std::vector<std::pair<int, int>>> atomicRanges_;  // binding -> [begin, end) byte ranges  // qualifiers other than precision seen by the last parseQualifiers()
    Qualifiers defaultUbo_, defaultSsbo_;
    std::unordered_map<const Builtin*, Function*> builtinFns_;
    std::unordered_map<Variable*, int> maxIndex_;
    std::map<std::pair<StructDef*, int>, int> maxMemberIndex_;  // implicitly sized block members

    // ---- token helpers --------------------------------------------------
    const PPToken& peek(int k = 0) const {
        size_t i = p_ + k;
        return i < toks_->size() ? (*toks_)[i] : toks_->back();
    }
    int line() const { return peek().line; }
    bool isP(const char* s, int k = 0) const { const auto& t = peek(k); return t.kind == PK::Punct && t.text == s; }
    bool isKw(const char* s, int k = 0) const { const auto& t = peek(k); return t.kind == PK::Ident && t.text == s; }
    bool accept(const char* s) { if (isP(s)) { p_++; return true; } return false; }
    bool acceptKw(const char* s) { if (isKw(s)) { p_++; return true; } return false; }
    [[noreturn]] void fail(const std::string& msg, int ln = -1) {
        log_ += "ERROR: 0:" + std::to_string(ln < 0 ? line() : ln) + ": " + msg + "\n";
        throw ParseError();
    }
    void warn(const std::string& msg) { log_ += "WARNING: 0:" + std::to_string(line()) + ": " + msg + "\n"; }
    void expect(const char* s) {
        if (!accept(s)) fail(std::string("syntax error: expected '") + s + "' but found '" + peek().text + "'");
    }
    std::string expectIdent() {
        const auto& t = peek();
        if (t.kind != PK::Ident) fail("syntax error: expected identifier but found '" + t.text + "'");
        p_++;
        return t.text;
    }

    // ---- allocation -------------------------------------------------------
    Expr* E(EK k, const Type& t, int ln = -1) {
        sh_.exprs.emplace_back();
        Expr* e = &sh_.exprs.back();
        e->kind = k;
        e->type = t;
        e->line = ln < 0 ? line() : ln;
        return e;
    }
    Stmt* S(SK k) {
        sh_.stmts.emplace_back();
        Stmt* s = &sh_.stmts.back();
        s->kind = k;
        s->line = line();
        return s;
    }
    Variable* newVar(const std::string& name, const Type& t, const Qualifiers& q) {
        sh_.vars.emplace_back();
        Variable* v = &sh_.vars.back();
        v->name = name;
        v->type = t;
        v->q = q;
        v->id = nextId_++;
        v->line = line();
        return v;
    }

    // ---- scopes -------------------------------------------------------------
    void push() { scopes_.emplace_back(); }
    void pop() { scopes_.pop_back(); }
    Sym* lookup(const std::string& n) {
        for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
            auto f = it->find(n);
            if (f != it->end()) return &f->second;
        }
        return nullptr;
    }
    StructDef* lookupType(const std::string& n) {
        for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
            auto f = it->find(n);
            if (f != it->end()) {
                if (f->second.type) return f->second.type;
                if (f->second.var) return nullptr;  // shadowed by a variable
            }
        }
        return nullptr;
    }
    bool isTypeName(const std::string& n) { return typeKeyword(n) || lookupType(n); }
    void declareVar(Variable* v) {
        auto& sc = scopes_.back();
        auto it = sc.find(v->name);
        if (it != sc.end() && (it->second.var || it->second.type) && !preamble_) {
            fail("redefinition of '" + v->name + "'");
        }
        Sym& s = sc[v->name];
        s.var = v;
        s.member = -1;
    }

    // ---- types / conversions ---------------------------------------------------
    static int baseConvCost(BT from, BT to) {
        if (from == to) return 0;
        switch (to) {
            case BT::UInt: return from == BT::Int ? 1 : -1;
            case BT::Float: return (from == BT::Int || from == BT::UInt) ? 2 : -1;
            case BT::Double: return from == BT::Float ? 3 : (from == BT::Int || from == BT::UInt) ? 4 : -1;
            default: return -1;
        }
    }
    static int convCost(const Type& from, const Type& to) {
        if (from == to) return 0;
        if (from.dims != to.dims || from.vec != to.vec || from.cols != to.cols) return -1;
        if (!from.isNumericBase() || !to.isNumericBase()) return -1;
        return baseConvCost(from.base, to.base);
    }
    Expr* convert(Expr* e, const Type& to, const char* what = "conversion") {
        if (e->type == to) return e;
        if (e->type.isError() || to.isError()) return e;
        if (convCost(e->type, to) < 0) fail(std::string("cannot convert from '") + e->type.str() + "' to '" + to.str() + "' in " + what, e->line);
        Expr* c = E(EK::Ctor, to, e->line);
        c->args.push_back(e);
        c->cv = foldCtor(to, c->args);
        return c;
    }
    BT commonBase(BT a, BT b, int ln) {
        if (a == b) return a;
        if (baseConvCost(a, b) >= 0) return b;
        if (baseConvCost(b, a) >= 0) return a;
        fail("type mismatch between operands", ln);
    }

    bool extEnabled(const char* name) const {
        auto it = sh_.extensions.find(name);
        return it != sh_.extensions.end() && it->second != "disable";
    }
    // binding + array size must stay within the implementation's binding points
    void checkBinding(const Type& t, int binding, const std::string& name) {
        BT b = t.stripArrays().base;
        int n = t.isArray() ? std::max(1, t.totalArrayElements()) : 1;
        int limit = 0;
        if (b == BT::Sampler) limit = kMaxCombinedTextureImageUnits;
        else if (b == BT::Image) limit = kMaxImageUnits;
        else if (b == BT::AtomicUint) { limit = kMaxAtomicCounterBufferBindings; n = 1; }
        else return;
        if (binding + n > limit) fail("'" + name + "' : binding " + std::to_string(binding) + " exceeds the maximum binding point");
    }
    void checkArrayOfArrays(const Type& t) {
        if (t.dims.size() > 1 && !preamble_ && sh_.version < (sh_.es ? 310 : 430) && !extEnabled("GL_ARB_arrays_of_arrays"))
            fail("arrays of arrays require GLSL 4.30 or GL_ARB_arrays_of_arrays");
    }

    // ---- constant expressions ------------------------------------------------------
    int64_t constInt(Expr* e, const char* what) {
        if (!e->cv || !e->type.isScalar() || !(e->type.isIntLike() || e->type.base == BT::Bool))
            fail(std::string(what) + " must be a constant integer expression", e->line);
        return e->type.base == BT::UInt ? (int64_t)e->cv->c[0].u : e->cv->c[0].i;
    }

    // ---- top level ---------------------------------------------------------------
    void parseTokens(const std::vector<PPToken>& toks) {
        toks_ = &toks;
        p_ = 0;
        while (peek().kind != PK::End) parseExternal();
    }
    void parseExternal();
    bool parseQualifiers(Qualifiers& q);
    bool isQualifierAt(int k);
    void parseLayout(Qualifiers& q);
    Type parseTypeSpecifier(bool allowStructDef = true);
    StructDef* parseStruct();
    void parseArrayDims(Type& t);
    void parseBlock(Qualifiers q, std::vector<Stmt*>* out);
    void parseFunction(const Qualifiers& q, const Type& ret, const std::string& name);
    std::vector<Variable*> parseParamList();
    Expr* mkSubroutineCall(Variable* v, Expr* index, std::vector<Expr*> args, int ln);
    void parseDeclarators(Qualifiers q, const Type& t, std::string name, bool global, std::vector<Variable*>* out);
    void checkWriteOnlyReads();
    Expr* parseInitializer(const Type& t);
    void applyStageLayout(const Qualifiers& q);
    bool redeclareBuiltin(const Qualifiers& q, const Type& t, const std::string& name);
    void finalize();

    // statements
    Stmt* parseStatement();
    Stmt* parseCompound(bool newScope);
    bool startsDeclaration();
    Stmt* parseLocalDeclaration();

    // expressions
    Expr* parseExpression();
    Expr* parseAssignment();
    Expr* parseConditional();
    Expr* parseBinary(int minPrec);
    Expr* parseUnary();
    Expr* parsePostfix(Expr* e);
    Expr* parsePrimary();
    std::vector<Expr*> parseArgs();

    Expr* mkBinary(Op op, Expr* a, Expr* b, int ln);
    Expr* mkUnary(Op op, Expr* a, int ln);
    Expr* mkAssign(Op op, Expr* l, Expr* r, int ln);
    Expr* mkTernary(Expr* c, Expr* a, Expr* b, int ln);
    Expr* mkIndex(Expr* b, Expr* i, int ln);
    Expr* mkField(Expr* b, const std::string& name, int ln);
    Expr* mkCall(const std::string& name, std::vector<Expr*> args, int ln);
    void checkImageDecl(const std::string& name, const Type& t, const Qualifiers& q);
    Expr* mkCtor(Type t, std::vector<Expr*> args, int ln);
    Expr* mkVarRef(const std::string& name, int ln);
    Expr* mkLiteral(const PPToken& t);
    void markWritten(Expr* e);
    Function* builtinFn(const Builtin* b);
};

bool Parser::run(const PPResult& pp) {
    sh_.version = pp.version;
    sh_.es = pp.es;
    sh_.compat = !pp.core;
    sh_.extensions = pp.extensions;
    defaultUbo_.layout.packing = Packing::Shared;
    defaultSsbo_.layout.packing = Packing::Shared;
    try {
        // builtin scope
        push();
        preamble_ = true;
        {
            CompileOptions o;
            PPResult pre = preprocess({builtinPreamble(sh_.stage)}, sh_.stage, o);
            parseTokens(pre.tokens);
        }
        preamble_ = false;
        // the preamble's globals are not emitted as user globals
        sh_.globals.clear();
        push();  // user global scope
        if (sh_.stage == Stage::Compute) {
            auto ext = sh_.extensions.find("GL_ARB_compute_shader");
            bool viaExt = ext != sh_.extensions.end() && ext->second != "disable";
            if (sh_.es ? sh_.version < 310 : (sh_.version < 430 && !viaExt))
                fail("compute shaders require GLSL 4.30 (or GL_ARB_compute_shader) / GLSL ES 3.10", 0);
        }
        parseTokens(pp.tokens);
        finalize();
    } catch (ParseError&) {
        return false;
    }
    return true;
}

void Parser::finalize() {
    // GLSL 4.60 4.3.4 / 4.3.6: arrayed interfaces of the geometry and tessellation stages
    if (!preamble_) {
        Stage st = sh_.stage;
        for (auto& v : sh_.vars) {
            if (!v.global || v.builtin != BuiltinVar::None || v.param) continue;
            if (v.q.storage != Storage::In && v.q.storage != Storage::Out) continue;
            bool in = v.q.storage == Storage::In;
            const std::string what = "'" + (v.block ? v.block->name : v.name) + "'";
            if (st == Stage::TessControl && in && v.q.patch) fail("tessellation control shader inputs cannot be per-patch: " + what, v.line);
            if (st == Stage::TessEval && !in && v.q.patch) fail("tessellation evaluation shader outputs cannot be per-patch: " + what, v.line);
            bool arrayed = (st == Stage::Geometry && in) || (st == Stage::TessControl && !v.q.patch) ||
                           (st == Stage::TessEval && in && !v.q.patch);
            if (!arrayed) continue;
            if (!v.type.isArray()) fail(what + " must be declared as an array", v.line);
            int n = v.type.dims[0];
            if (n == 0) continue;  // unsized: sized from the layout later
            if (st == Stage::Geometry && sh_.inLayout.primitive != Layout::kNoPrimitive && n != primitiveVertexCount(sh_.inLayout.primitive))
                fail("array size of geometry shader input " + what + " does not match the input primitive", v.line);
            if ((st == Stage::TessEval || st == Stage::TessControl) && in && n != 32)
                fail("array size of input " + what + " must be gl_MaxPatchVertices", v.line);
            if (st == Stage::TessControl && !in && sh_.outLayout.vertices > 0 && n != sh_.outLayout.vertices)
                fail("array size of output " + what + " does not match the output patch size", v.line);
        }
        // GLSL 4.60 4.3.4/4.3.6: fragment shader double inputs must be flat, outputs cannot be double
        if (st == Stage::Fragment)
            for (auto& v : sh_.vars) {
                if (!v.global || v.builtin != BuiltinVar::None || v.param) continue;
                if (v.q.storage == Storage::Out && v.type.stripArrays().base == BT::Double)
                    fail("fragment shader output '" + v.name + "' cannot be double-precision", v.line);
                if (v.q.storage != Storage::In) continue;
                if (v.block) {
                    for (auto& m : v.block->members)
                        if (m.type.stripArrays().base == BT::Double && m.q.interp != Interp::Flat && v.q.interp != Interp::Flat)
                            fail("double-precision fragment shader input '" + m.name + "' must be flat", v.line);
                } else if (v.type.stripArrays().base == BT::Double && v.q.interp != Interp::Flat) {
                    fail("double-precision fragment shader input '" + v.name + "' must be flat", v.line);
                }
            }
        if (st == Stage::TessControl && sh_.outLayout.vertices != -1 && (sh_.outLayout.vertices <= 0 || sh_.outLayout.vertices > 32))
            fail("invalid output patch vertex count", 0);
        if (st == Stage::Geometry && sh_.outLayout.maxVertices > 256) fail("max_vertices exceeds gl_MaxGeometryOutputVertices", 0);
        if (st == Stage::Geometry && sh_.inLayout.invocations > 32) fail("invocations exceeds gl_MaxGeometryShaderInvocations", 0);
    }
    {
        bool clipV = false, clipD = false;
        for (auto& v : sh_.vars) {
            if (v.builtin == BuiltinVar::ClipVertex && v.written) clipV = true;
            if (v.builtin == BuiltinVar::ClipDistance && v.written && v.q.storage == Storage::Out) clipD = true;
        }
        if (clipV && clipD) fail("a shader cannot statically write both gl_ClipVertex and gl_ClipDistance", 0);
    }
    // Implicitly sized arrays
    for (auto& [v, mx] : maxIndex_) {
        if (!v->type.dims.empty() && v->type.dims[0] == 0 && !(v->q.storage == Storage::Buffer)) v->type.dims[0] = mx + 1;
    }
    for (auto& v : sh_.vars) {
        if (v.global && !v.type.dims.empty() && v.type.dims[0] == 0 && v.q.storage != Storage::Buffer && !v.param) {
            bool bi = v.builtin == BuiltinVar::ClipDistance || v.builtin == BuiltinVar::CullDistance || v.builtin == BuiltinVar::TexCoord ||
                      v.builtin == BuiltinVar::FragData;
            // dynamically indexed builtin arrays get the implementation maximum; unused ones stay minimal
            v.type.dims[0] = bi ? (v.used || v.written ? 8 : 1) : 1;
        }
    }
    // Unsized arrays that are not the last member of a block are sized by the largest
    // constant index used (GLSL 4.60 §4.1.9); only the last member of a buffer block is runtime-sized.
    for (auto& sd : sh_.sdefs) {
        if (!sd.isBlock) continue;
        for (size_t i = 0; i < sd.members.size(); i++) {
            Type& mt = sd.members[i].type;
            bool runtimeSized = i + 1 == sd.members.size() && sd.blockQ.storage == Storage::Buffer;
            if (mt.dims.empty() || mt.dims[0] != 0 || runtimeSized) continue;
            auto it = maxMemberIndex_.find({&sd, (int)i});
            mt.dims[0] = it == maxMemberIndex_.end() ? 1 : it->second + 1;
        }
    }
    for (auto& e : sh_.exprs)
        if (e.kind == EK::Field && !e.type.dims.empty() && e.type.dims[0] == 0 && e.args[0]->type.sdef && e.args[0]->type.sdef->isBlock)
            e.type = e.args[0]->type.sdef->members[e.field].type;
    // GLSL 4.60 7.1: clip/cull distance array sizes are bounded by gl_MaxClipDistances,
    // gl_MaxCullDistances and (together) gl_MaxCombinedClipAndCullDistances.
    {
        int nclip = 0, ncull = 0;
        for (auto& v : sh_.vars) {
            if (!v.global || v.type.dims.empty() || !(v.used || v.written || v.redeclared)) continue;
            if (v.builtin == BuiltinVar::ClipDistance) nclip = std::max(nclip, v.type.dims[0]);
            if (v.builtin == BuiltinVar::CullDistance) ncull = std::max(ncull, v.type.dims[0]);
        }
        if (nclip > 8) fail("gl_ClipDistance array size exceeds gl_MaxClipDistances", 0);
        if (ncull > 8) fail("gl_CullDistance array size exceeds gl_MaxCullDistances", 0);
        if (nclip + ncull > 8) fail("combined gl_ClipDistance and gl_CullDistance sizes exceed gl_MaxCombinedClipAndCullDistances", 0);
    }
    // Fix up expression types that referenced implicitly sized arrays
    for (auto& e : sh_.exprs) {
        if (e.kind == EK::Var && e.var && e.type.base == e.var->type.base && !e.type.dims.empty() && e.type.dims[0] == 0)
            e.type = e.var->type;
    }
    for (auto* f : sh_.functions)
        if (f->name == "main" && f->body) sh_.hasMain = true;
    checkWriteOnlyReads();
}

// GLSL 4.60 §4.10: a writeonly buffer variable or member can't be read. Walks the real
// statement trees (not the expression arena, which holds speculative nodes too).
void Parser::checkWriteOnlyReads() {
    std::function<void(const Expr*, bool)> visitE;
    auto writeOnly = [](const Expr* e) {
        if (e->kind == EK::Var) return e->var->q.storage == Storage::Buffer && (e->var->q.memory & MemWriteOnly);
        if (e->kind == EK::Field) {
            const StructDef* sd = e->args[0]->type.sdef;
            return sd && sd->isBlock && e->field >= 0 && (sd->members[e->field].q.memory & MemWriteOnly);
        }
        return false;
    };
    visitE = [&](const Expr* e, bool target) {
        if (!e) return;
        if (!target && writeOnly(e)) fail("cannot read a writeonly buffer variable", e->line);
        switch (e->kind) {
            case EK::Assign:
                visitE(e->args[0], e->op == Op::Assign);
                visitE(e->args[1], false);
                return;
            case EK::Index:
                visitE(e->args[0], target);
                visitE(e->args[1], false);
                return;
            case EK::Field:
            case EK::Swizzle:
                visitE(e->args[0], target);
                return;
            case EK::Length:  // .length() doesn't read the array
                for (auto* a : e->args) visitE(a, true);
                return;
            case EK::Call:
                for (size_t i = 0; i < e->args.size(); i++) {
                    bool out = e->fn && i < e->fn->params.size() && e->fn->params[i]->q.storage == Storage::Out;
                    visitE(e->args[i], out);
                }
                return;
            default:
                for (auto* a : e->args) visitE(a, false);
                if (e->aux) visitE(e->aux, false);
                return;
        }
    };
    std::function<void(const Stmt*)> visitS = [&](const Stmt* st) {
        if (!st) return;
        for (auto* b : st->body) visitS(b);
        visitE(st->expr, false);
        visitS(st->init);
        visitE(st->cond, false);
        visitS(st->condDecl);
        visitE(st->incr, false);
        visitS(st->then);
        visitS(st->els);
        for (auto* v : st->vars) visitE(v->init, false);
    };
    for (auto* f : sh_.functions) visitS(f->body);
}

// ---------------------------------------------------------------------------
// Qualifiers & layout
// ---------------------------------------------------------------------------
bool Parser::isQualifierAt(int k) {
    const auto& t = peek(k);
    if (t.kind != PK::Ident || !qualifierKeywords().count(t.text)) return false;
    static const std::unordered_set<std::string> soft = {
        "sample", "patch", "buffer", "shared", "precise", "coherent", "volatile", "restrict", "readonly", "writeonly",
        "attribute", "varying", "smooth", "flat", "noperspective", "centroid", "invariant"};
    if (!soft.count(t.text)) return true;
    Sym* s = lookup(t.text);
    if (s && s->var) return false;
    // "<qualifier> <ident>", or a default-layout declaration such as "layout(std430) buffer;"
    return peek(k + 1).kind == PK::Ident || isP(";", k + 1);
}

bool Parser::parseQualifiers(Qualifiers& q) {
    bool any = false;
    nonPrecisionQuals_ = 0;
    for (;;) {
        const auto& t = peek();
        if (!isQualifierAt(0)) break;
        // 'in'/'out' etc must not be confused with identifiers (they are keywords)
        const std::string k = t.text;
        p_++;
        any = true;
        if (k != "lowp" && k != "mediump" && k != "highp") nonPrecisionQuals_++;
        if (k == "const") q.storage = Storage::Const;
        else if (k == "in") q.storage = (q.storage == Storage::Out) ? Storage::InOut : Storage::In;
        else if (k == "out") q.storage = (q.storage == Storage::In) ? Storage::InOut : Storage::Out;
        else if (k == "inout") q.storage = Storage::InOut;
        else if (k == "attribute") q.storage = Storage::In;
        else if (k == "varying") {
            q.storage = (sh_.stage == Stage::Fragment) ? Storage::In : Storage::Out;
            if (preamble_) q.storage = Storage::Out;
        } else if (k == "uniform") q.storage = Storage::Uniform;
        else if (k == "buffer") q.storage = Storage::Buffer;
        else if (k == "shared") q.storage = Storage::Shared;
        else if (k == "centroid") q.centroid = true;
        else if (k == "sample") q.sample = true;
        else if (k == "patch") q.patch = true;
        else if (k == "flat") { q.interp = Interp::Flat; q.interpExplicit = true; }
        else if (k == "smooth") { q.interp = Interp::Smooth; q.interpExplicit = true; }
        else if (k == "noperspective") { q.interp = Interp::NoPerspective; q.interpExplicit = true; }
        else if (k == "invariant") q.invariant = true;
        else if (k == "precise") q.precise = true;
        else if (k == "coherent") q.memory |= MemCoherent;
        else if (k == "volatile") q.memory |= MemVolatile;
        else if (k == "restrict") q.memory |= MemRestrict;
        else if (k == "readonly") q.memory |= MemReadOnly;
        else if (k == "writeonly") q.memory |= MemWriteOnly;
        else if (k == "lowp" || k == "mediump" || k == "highp") {}
        else if (k == "layout") parseLayout(q);
        else if (k == "subroutine") {
            q.subroutine = true;
            if (accept("(")) {
                do q.subroutineTypes.push_back(expectIdent());
                while (accept(","));
                expect(")");
            }
        }
    }
    return any;
}

void Parser::parseLayout(Qualifiers& q) {
    q.hasLayout = true;
    Layout& L = q.layout;
    expect("(");
    for (;;) {
        std::string id = expectIdent();
        std::string lid = id;
        std::transform(lid.begin(), lid.end(), lid.begin(), ::tolower);
        int64_t val = 0;
        bool hasVal = false;
        if (accept("=")) {
            // Before GLSL 4.40 (ARB_enhanced_layouts) layout values must be integer literals.
            if (!preamble_ && sh_.version < (sh_.es ? 310 : 440) && !extEnabled("GL_ARB_enhanced_layouts") &&
                !(peek().kind == PK::Number && (isP(",", 1) || isP(")", 1))))
                fail("layout qualifier '" + id + "' requires an integer literal before GLSL 4.40");
            Expr* e = parseConditional();
            val = constInt(e, "layout qualifier value");
            hasVal = true;
        }
        auto need = [&]() {
            if (!hasVal) fail("layout qualifier '" + id + "' requires a value");
            if (val < 0) fail("layout qualifier '" + id + "' cannot be negative");
            return (int)val;
        };
        if (lid == "location") L.location = need();
        else if (lid == "component") L.component = need();
        else if (lid == "binding") {
            L.binding = need();
            if (L.binding < 0) fail("'binding' : must be a non-negative integer");
        }
        else if (lid == "offset") L.offset = need();
        else if (lid == "align") L.align = need();
        else if (lid == "index") L.index = need();
        else if (lid == "set") L.set = need();
        else if (lid == "std140") L.packing = Packing::Std140;
        else if (lid == "std430") L.packing = Packing::Std430;
        else if (lid == "shared") L.packing = Packing::Shared;
        else if (lid == "packed") L.packing = Packing::Packed;
        else if (lid == "row_major") L.matrix = 2;
        else if (lid == "column_major") L.matrix = 1;
        else if (lid == "local_size_x") L.localSize[0] = need();
        else if (lid == "local_size_y") L.localSize[1] = need();
        else if (lid == "local_size_z") L.localSize[2] = need();
        else if (lid == "max_vertices") L.maxVertices = need();
        else if (lid == "invocations") L.invocations = need();
        else if (lid == "vertices") L.vertices = need();
        else if (lid == "stream") L.stream = need();
        else if (lid == "xfb_buffer") L.xfbBuffer = need();
        else if (lid == "xfb_offset") L.xfbOffset = need();
        else if (lid == "xfb_stride") L.xfbStride = need();
        else if (lid == "points") L.primitive = gl::POINTS;
        else if (lid == "lines") L.primitive = gl::LINES;
        else if (lid == "lines_adjacency") L.primitive = gl::LINES_ADJACENCY;
        else if (lid == "triangles") L.primitive = gl::TRIANGLES;
        else if (lid == "triangles_adjacency") L.primitive = gl::TRIANGLES_ADJACENCY;
        else if (lid == "line_strip") L.primitive = gl::LINE_STRIP;
        else if (lid == "triangle_strip") L.primitive = gl::TRIANGLE_STRIP;
        else if (lid == "quads") L.primitive = gl::QUADS;
        else if (lid == "isolines") L.primitive = gl::ISOLINES;
        else if (lid == "equal_spacing") L.spacing = gl::EQUAL;
        else if (lid == "fractional_even_spacing") L.spacing = gl::FRACTIONAL_EVEN;
        else if (lid == "fractional_odd_spacing") L.spacing = gl::FRACTIONAL_ODD;
        else if (lid == "cw") L.order = gl::CW;
        else if (lid == "ccw") L.order = gl::CCW;
        else if (lid == "point_mode") L.pointMode = true;
        else if (lid == "origin_upper_left") L.originUpperLeft = true;
        else if (lid == "pixel_center_integer") L.pixelCenterInteger = true;
        else if (lid == "early_fragment_tests") L.earlyFragmentTests = true;
        else if (lid == "depth_any") L.depthLayout = 1;
        else if (lid == "depth_greater") L.depthLayout = 2;
        else if (lid == "depth_less") L.depthLayout = 3;
        else if (lid == "depth_unchanged") L.depthLayout = 4;
        else if (lid == "blend_support_all_equations") L.blendSupportAll = true;
        else {
            static const std::unordered_set<std::string> formats = {
                "rgba32f", "rgba16f", "rg32f", "rg16f", "r11f_g11f_b10f", "r32f", "r16f", "rgba16", "rgb10_a2", "rgba8",
                "rg16", "rg8", "r16", "r8", "rgba16_snorm", "rgba8_snorm", "rg16_snorm", "rg8_snorm", "r16_snorm", "r8_snorm",
                "rgba32i", "rgba16i", "rgba8i", "rg32i", "rg16i", "rg8i", "r32i", "r16i", "r8i", "rgba32ui", "rgba16ui",
                "rgb10_a2ui", "rgba8ui", "rg32ui", "rg16ui", "rg8ui", "r32ui", "r16ui", "r8ui"};
            if (formats.count(lid)) L.format = lid;
            else warn("unknown layout qualifier '" + id + "' ignored");
        }
        if (accept(")")) break;
        expect(",");
    }
}

int primitiveVertexCount(uint32_t prim) {
    switch (prim) {
        case gl::POINTS: return 1;
        case gl::LINES: return 2;
        case gl::LINES_ADJACENCY: return 4;
        case gl::TRIANGLES: return 3;
        case gl::TRIANGLES_ADJACENCY: return 6;
        default: return 0;
    }
}

void Parser::applyStageLayout(const Qualifiers& q) {
    const Layout& L = q.layout;
    if (q.storage == Storage::Uniform) {
        if (L.packing != Packing::None) defaultUbo_.layout.packing = L.packing;
        if (L.matrix) defaultUbo_.layout.matrix = L.matrix;
        return;
    }
    if (q.storage == Storage::Buffer) {
        if (L.packing != Packing::None) defaultSsbo_.layout.packing = L.packing;
        if (L.matrix) defaultSsbo_.layout.matrix = L.matrix;
        return;
    }
    Layout& T = (q.storage == Storage::Out) ? sh_.outLayout : sh_.inLayout;
    if (q.storage == Storage::In && sh_.stage == Stage::Compute && (L.localSize[0] >= 0 || L.localSize[1] >= 0 || L.localSize[2] >= 0)) {
        // Every local size declaration in a shader must agree (unspecified components are 1).
        bool declared = T.localSize[0] >= 0 || T.localSize[1] >= 0 || T.localSize[2] >= 0;
        for (int i = 0; i < 3 && declared; i++)
            if (std::max(1, T.localSize[i]) != std::max(1, L.localSize[i])) fail("conflicting local_size declarations");
    }
    for (int i = 0; i < 3; i++) if (L.localSize[i] >= 0) T.localSize[i] = L.localSize[i];
    // GLSL 4.60 4.4: repeated stage layout declarations must agree
    if (L.maxVertices >= 0 && T.maxVertices >= 0 && L.maxVertices != T.maxVertices) fail("conflicting max_vertices declarations");
    if (L.invocations >= 0 && T.invocations >= 0 && L.invocations != T.invocations) fail("conflicting invocations declarations");
    if (L.vertices >= 0 && T.vertices >= 0 && L.vertices != T.vertices) fail("conflicting vertices declarations");
    if (L.primitive != Layout::kNoPrimitive && T.primitive != Layout::kNoPrimitive && L.primitive != T.primitive)
        fail("conflicting primitive type declarations");
    if (L.maxVertices >= 0) T.maxVertices = L.maxVertices;
    if (L.invocations >= 0) T.invocations = L.invocations;
    if (L.vertices >= 0) T.vertices = L.vertices;
    if (L.primitive != Layout::kNoPrimitive) T.primitive = L.primitive;
    if (L.spacing) T.spacing = L.spacing;
    if (L.order) T.order = L.order;
    if (L.pointMode) T.pointMode = true;
    if (L.earlyFragmentTests) T.earlyFragmentTests = true;
    if (L.stream >= 0) T.stream = L.stream;
    if (L.xfbBuffer >= 0) T.xfbBuffer = L.xfbBuffer;
    if (L.xfbStride >= 0) T.xfbStride = L.xfbStride;
    if (q.storage == Storage::In && sh_.stage == Stage::Compute && (L.localSize[0] >= 0 || L.localSize[1] >= 0 || L.localSize[2] >= 0)) {
        static const int kMaxSize[3] = {1024, 1024, 64};  // GL_MAX_COMPUTE_WORK_GROUP_SIZE
        int64_t total = 1;
        for (int i = 0; i < 3; i++) {
            if (L.localSize[i] == 0 || L.localSize[i] > kMaxSize[i]) fail("local_size exceeds GL_MAX_COMPUTE_WORK_GROUP_SIZE");
            total *= std::max(1, T.localSize[i]);
        }
        if (total > 1024) fail("local size exceeds GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS");
        // update gl_WorkGroupSize constant
        Sym* s = &scopes_[0]["gl_WorkGroupSize"];
        if (s->var) {
            auto cv = std::make_shared<ConstVal>();
            for (int i = 0; i < 3; i++) {
                Scalar sc;
                sc.u = (uint32_t)std::max(1, T.localSize[i]);
                cv->c.push_back(sc);
            }
            s->var->constVal = cv;
        }
    }
    if (q.storage == Storage::In && sh_.stage == Stage::Geometry && L.primitive != Layout::kNoPrimitive) {
        // size gl_in and unsized input arrays
        int n = primitiveVertexCount(L.primitive);
        for (auto& sc : scopes_)
            for (auto& [name, sym] : sc)
                if (sym.var && sym.var->global && sym.var->q.storage == Storage::In && !sym.var->type.dims.empty() &&
                    sym.var->type.dims[0] == 0)
                    sym.var->type.dims[0] = n;
    }
    if (q.storage == Storage::Out && sh_.stage == Stage::TessControl && L.vertices > 0) {
        for (auto& sc : scopes_)
            for (auto& [name, sym] : sc)
                if (sym.var && sym.var->global && sym.var->q.storage == Storage::Out && !sym.var->q.patch &&
                    !sym.var->type.dims.empty() && sym.var->type.dims[0] == 0)
                    sym.var->type.dims[0] = L.vertices;
    }
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
void Parser::parseArrayDims(Type& t) {
    std::vector<int> d;
    while (accept("[")) {
        if (accept("]")) { d.push_back(0); continue; }
        Expr* e = parseConditional();
        int64_t n = constInt(e, "array size");
        if (n <= 0) fail("array size must be greater than zero", e->line);
        d.push_back((int)n);
        expect("]");
    }
    // appended dims are innermost relative to existing name dims; caller handles order
    t.dims.insert(t.dims.end(), d.begin(), d.end());
    checkArrayOfArrays(t);
}

StructDef* Parser::parseStruct() {
    // 'struct' consumed
    std::string name;
    if (peek().kind == PK::Ident && !isP("{")) name = expectIdent();
    expect("{");
    sh_.sdefs.emplace_back();
    StructDef* sd = &sh_.sdefs.back();
    sd->name = name.empty() ? "anon_struct" : name;
    sd->id = nextId_++;
    while (!accept("}")) {
        Qualifiers mq;
        parseQualifiers(mq);
        if (nonPrecisionQuals_ && !preamble_) fail("structure members may only have precision qualifiers");
        Type mt = parseTypeSpecifier();
        do {
            Member m;
            m.line = line();
            m.name = expectIdent();
            m.type = mt;
            Type nd;
            parseArrayDims(nd);
            if (!nd.dims.empty()) { m.type.dims.insert(m.type.dims.begin(), nd.dims.begin(), nd.dims.end()); checkArrayOfArrays(m.type); }
            m.q = mq;
            if (sd->findMember(m.name) >= 0) fail("duplicate struct member '" + m.name + "'");
            sd->members.push_back(m);
        } while (accept(","));
        expect(";");
    }
    if (sd->members.empty()) fail("empty struct");
    sh_.structs.push_back(sd);
    if (!name.empty()) {
        auto& sc = scopes_.back();
        if (sc.count(name) && (sc[name].type || sc[name].var) && !preamble_) fail("redefinition of '" + name + "'");
        sc[name].type = sd;
    }
    if (scopes_.size() <= 2) {
        GlobalDecl g;
        g.kind = GlobalDecl::StructType;
        g.sdef = sd;
        sh_.globals.push_back(g);
    }
    return sd;
}

Type Parser::parseTypeSpecifier(bool allowStructDef) {
    Type t;
    if (acceptKw("struct")) {
        StructDef* sd = parseStruct();
        t.base = BT::Struct;
        t.sdef = sd;
    } else {
        const auto& tk = peek();
        if (tk.kind != PK::Ident) fail("syntax error: expected type but found '" + tk.text + "'");
        if (const Type* kw = typeKeyword(tk.text)) {
            t = *kw;
        } else if (StructDef* sd = lookupType(tk.text)) {
            t.base = BT::Struct;
            t.sdef = sd;
        } else {
            fail("'" + tk.text + "' : undeclared type");
        }
        p_++;
    }
    parseArrayDims(t);
    return t;
}

// ---------------------------------------------------------------------------
// Declarations
// ---------------------------------------------------------------------------
void Parser::parseExternal() {
    if (accept(";")) return;
    if (acceptKw("precision")) {
        while (!accept(";")) { if (peek().kind == PK::End) fail("unexpected end of input"); p_++; }
        return;
    }
    Qualifiers q;
    bool hasQ = parseQualifiers(q);
    if (q.subroutine) {
        if (q.storage == Storage::Uniform) {
            std::string tn = expectIdent();
            Sym* ts = lookup(tn);
            if (!ts || !ts->subrType) fail("'" + tn + "' is not a subroutine type");
            Type t;
            t.base = BT::Subroutine;
            t.subr = ts->subrType;
            std::string name = expectIdent();
            parseDeclarators(q, t, name, true, nullptr);
            return;
        }
        if (q.subroutineTypes.empty()) {
            // subroutine type declaration: subroutine <ret> <name>(<params>);
            Type ret = parseTypeSpecifier();
            std::string name = expectIdent();
            std::vector<Variable*> params = parseParamList();
            expect(";");
            sh_.funcs.emplace_back();
            Function* f = &sh_.funcs.back();
            f->name = name;
            f->ret = ret;
            f->params = params;
            f->line = line();
            Sym& sym = scopes_.back()[name];
            if (sym.var || sym.type || sym.subrType) fail("redefinition of '" + name + "'");
            sym.subrType = f;
            return;
        }
        Type t = parseTypeSpecifier();
        std::string name = expectIdent();
        if (!isP("(")) fail("expected a subroutine function definition");
        parseFunction(q, t, name);
        return;
    }
    if (hasQ && isP(";")) {
        p_++;
        if (q.storage == Storage::Uniform || q.storage == Storage::Buffer) {
            // default block layouts only take packing and matrix layout (GLSL 4.60 §4.4.5)
            if (q.layout.binding >= 0 || q.layout.offset >= 0 || q.layout.align >= 0 || q.layout.location >= 0)
                fail("only packing and matrix layout qualifiers are allowed in a default block layout");
            if (q.storage == Storage::Uniform && q.layout.packing == Packing::Std430)
                fail("'std430' : only allowed on shader storage blocks");
        }
        applyStageLayout(q);
        return;
    }
    if (hasQ && peek().kind == PK::Ident && !isTypeName(peek().text) && (isP(";", 1) || isP(",", 1))) {
        // e.g. "invariant gl_Position;" or "invariant varyingName;"
        do {
            std::string n = expectIdent();
            Sym* s = lookup(n);
            if (!s || !s->var) fail("'" + n + "' : undeclared identifier");
            if (q.invariant) s->var->q.invariant = true;
            if (q.precise) s->var->q.precise = true;
        } while (accept(","));
        expect(";");
        return;
    }
    if (hasQ && peek().kind == PK::Ident && peek().text != "struct" && !isTypeName(peek().text) && isP("{", 1)) {
        parseBlock(q, nullptr);
        return;
    }
    Type t = parseTypeSpecifier();
    if (t.base == BT::AtomicUint && isP(";")) {
        // "layout(binding = B, offset = O) uniform atomic_uint;" sets the next offset for binding B
        p_++;
        if (q.hasLayout && q.layout.offset >= 0) atomicNextOffset_[std::max(0, q.layout.binding)] = q.layout.offset;
        return;
    }
    if (accept(";")) return;  // struct declaration
    if (accept(",")) {
        // "type , a;" : a declaration list may start with a bare type
    }
    std::string name = expectIdent();
    if (isP("(")) {
        parseFunction(q, t, name);
        return;
    }
    parseDeclarators(q, t, name, true, nullptr);
}

void Parser::parseBlock(Qualifiers q, std::vector<Stmt*>* out) {
    std::string blockName = expectIdent();
    expect("{");
    sh_.sdefs.emplace_back();
    StructDef* sd = &sh_.sdefs.back();
    sd->name = blockName;
    sd->isBlock = true;
    sd->id = nextId_++;
    const Qualifiers& def = q.storage == Storage::Buffer ? defaultSsbo_ : defaultUbo_;
    if (q.storage == Storage::Uniform && q.layout.packing == Packing::Std430 && !preamble_)
        fail("'std430' : only allowed on shader storage blocks");
    if (q.storage == Storage::Uniform || q.storage == Storage::Buffer) {
        if (q.layout.packing == Packing::None) q.layout.packing = def.layout.packing;
        if (!q.layout.matrix) q.layout.matrix = def.layout.matrix;
    }
    sd->blockQ = q;
    while (!accept("}")) {
        Qualifiers mq;
        parseQualifiers(mq);
        if ((q.storage == Storage::Uniform || q.storage == Storage::Buffer) && !preamble_ &&
            (mq.layout.packing != Packing::None || mq.layout.binding >= 0))
            fail("packing and binding layout qualifiers are not allowed on block members");
        Type mt = parseTypeSpecifier(false);
        do {
            Member m;
            m.line = line();
            m.name = expectIdent();
            m.type = mt;
            Type nd;
            parseArrayDims(nd);
            if (!nd.dims.empty()) { m.type.dims.insert(m.type.dims.begin(), nd.dims.begin(), nd.dims.end()); checkArrayOfArrays(m.type); }
            m.q = mq;
            if (!m.q.interpExplicit) { m.q.interp = q.interp; m.q.interpExplicit = q.interpExplicit; }
            if (q.centroid) m.q.centroid = true;
            if (q.sample) m.q.sample = true;
            if (q.patch) m.q.patch = true;
            if (sd->findMember(m.name) >= 0) fail("duplicate block member '" + m.name + "'");
            sd->members.push_back(m);
        } while (accept(","));
        expect(";");
    }
    std::string inst;
    Type t;
    t.base = BT::Struct;
    t.sdef = sd;
    if (peek().kind == PK::Ident) {
        inst = expectIdent();
        Type nd;
        parseArrayDims(nd);
        t.dims = nd.dims;
    }
    expect(";");
    if (q.hasLayout && q.layout.binding >= 0 && !preamble_) {
        int n = t.isArray() ? std::max(1, t.totalArrayElements()) : 1;
        int limit = q.storage == Storage::Uniform ? kMaxUniformBufferBindings
                  : q.storage == Storage::Buffer ? kMaxShaderStorageBufferBindings : 0;
        if (!limit) fail("the binding layout qualifier only applies to uniform and buffer blocks");
        if (q.layout.binding + n > limit) fail("block '" + blockName + "' : binding exceeds the maximum binding point");
    }
    if (!t.dims.empty() && t.dims[0] == 0) {
        if (sh_.stage == Stage::Geometry && q.storage == Storage::In && sh_.inLayout.primitive != Layout::kNoPrimitive)
            t.dims[0] = primitiveVertexCount(sh_.inLayout.primitive);
        else if ((sh_.stage == Stage::TessControl || sh_.stage == Stage::TessEval) && q.storage == Storage::In && !q.patch)
            t.dims[0] = 32;
        else if (sh_.stage == Stage::TessControl && q.storage == Storage::Out && !q.patch && sh_.outLayout.vertices > 0)
            t.dims[0] = sh_.outLayout.vertices;
    }
    if (blockName == "gl_PerVertex" && !preamble_) {
        // Redeclaration of a builtin block: members map onto the existing builtins.
        if (!inst.empty() && inst != "gl_in" && inst != "gl_out") fail("invalid gl_PerVertex redeclaration");
        return;
    }
    sd->instanceName = inst;
    Variable* v = newVar(inst.empty() ? "_" + blockName : inst, t, q);
    v->block = sd;
    v->anonBlock = inst.empty();
    v->global = true;
    // GS outputs belong to the vertex stream of their layout(stream) or the current default
    if (sh_.stage == Stage::Geometry && q.storage == Storage::Out && v->q.layout.stream < 0) v->q.layout.stream = std::max(0, sh_.outLayout.stream);
    if (preamble_) {
        v->builtin = builtinVarId(inst, sh_.stage);
        if ((sh_.stage == Stage::TessControl || sh_.stage == Stage::TessEval) && inst == "gl_in") v->type.dims[0] = 32;
    }
    if (!inst.empty()) {
        declareVar(v);
    } else {
        for (size_t i = 0; i < sd->members.size(); i++) {
            auto& sc = scopes_.back();
            const std::string& mn = sd->members[i].name;
            if (sc.count(mn) && sc[mn].var && !preamble_) fail("redefinition of '" + mn + "'");
            sc[mn].var = v;
            sc[mn].member = (int)i;
        }
    }
    sh_.globalVars.push_back(v);
    if (!preamble_) {
        GlobalDecl g;
        g.kind = GlobalDecl::Var;
        g.var = v;
        sh_.globals.push_back(g);
    }
    (void)out;
}

bool Parser::redeclareBuiltin(const Qualifiers& q, const Type& t, const std::string& name) {
    auto it = scopes_[0].find(name);
    if (it == scopes_[0].end() || !it->second.var) return false;
    Variable* v = it->second.var;
    if (q.invariant) v->q.invariant = true;
    if (q.precise) v->q.precise = true;
    if (q.hasLayout) {
        if (q.layout.originUpperLeft) v->q.layout.originUpperLeft = true;
        if (q.layout.pixelCenterInteger) v->q.layout.pixelCenterInteger = true;
        if (q.layout.depthLayout) v->q.layout.depthLayout = q.layout.depthLayout;
        if (q.layout.location >= 0) v->q.layout.location = q.layout.location;
    }
    if (q.interpExplicit) { v->q.interp = q.interp; v->q.interpExplicit = true; }
    if (!t.dims.empty() && t.dims[0] > 0 && !v->type.dims.empty()) v->type.dims[0] = t.dims[0];
    v->redeclared = true;
    return true;
}

Expr* Parser::parseInitializer(const Type& t) {
    if (accept("{")) {
        // initializer list (GL 4.20)
        std::vector<Expr*> items;
        Type et;
        bool isArr = t.isArray(), isStruct = t.isStruct(), isMat = t.isMatrix(), isVec = t.isVector();
        int idx = 0;
        do {
            if (isP("}")) break;
            Type it;
            if (isArr) it = t.elem();
            else if (isStruct) {
                if (idx >= (int)t.sdef->members.size()) fail("too many initializers");
                it = t.sdef->members[idx].type;
            } else if (isMat) it = t.column();
            else if (isVec) it = t.scalarType();
            else it = t;
            Expr* item = parseInitializer(it);
            // initializer-list elements follow assignment conversion rules (no bool -> float etc.)
            if (!it.isArray() && !item->type.isArray()) item = convert(item, it, "initializer list");
            items.push_back(item);
            idx++;
        } while (accept(","));
        expect("}");
        Type tt = t;
        if (isArr && tt.dims[0] == 0) tt.dims[0] = (int)items.size();
        if ((isVec && (int)items.size() != t.vec) || (isMat && (int)items.size() != t.cols) ||
            (isStruct && items.size() != t.sdef->members.size()) || (isArr && (int)items.size() != tt.dims[0]))
            fail("initializer list has the wrong number of elements");
        if (!isArr && !isStruct && !isMat && !isVec) {
            if (items.size() != 1) fail("invalid initializer list");
            return convert(items[0], t, "initializer");
        }
        return mkCtor(tt, items, line());
    }
    Expr* e = parseAssignment();
    return e;
}

void Parser::parseDeclarators(Qualifiers q, const Type& baseT, std::string name, bool global, std::vector<Variable*>* out) {
    if (q.storage == Storage::Buffer && !preamble_) fail("'buffer' : only allowed on interface blocks");
    for (;;) {
        Type t = baseT;
        Type nd;
        parseArrayDims(nd);
        if (!nd.dims.empty()) { t.dims.insert(t.dims.begin(), nd.dims.begin(), nd.dims.end()); checkArrayOfArrays(t); }
        int ln = line();
        if (global && !preamble_ && name.compare(0, 3, "gl_") == 0) {
            if (redeclareBuiltin(q, t, name)) {
                if (accept("=")) fail("cannot initialize builtin redeclaration");
                if (!accept(",")) break;
                name = expectIdent();
                continue;
            }
            fail("'" + name + "' : identifiers starting with \"gl_\" are reserved");
        }
        if (t.base == BT::Void) fail("'" + name + "' : illegal use of type 'void'");
        if (q.hasLayout && q.layout.binding >= 0 && !preamble_) {
            if (!t.stripArrays().isOpaque()) fail("'" + name + "' : the binding layout qualifier only applies to opaque types and blocks");
            checkBinding(t, q.layout.binding, name);
        }
        if (t.stripArrays().base == BT::AtomicUint && global && !preamble_) {
            // atomic counter offsets are assigned in declaration order per binding
            if (q.layout.binding < 0) fail("'" + name + "' : atomic counters require a binding layout qualifier");
            int b = q.layout.binding;
            int off = q.layout.offset >= 0 ? q.layout.offset : atomicNextOffset_[b];
            if (off % 4) fail("'" + name + "' : atomic counter offset must be a multiple of 4");
            int size = 4 * (t.isArray() ? std::max(1, t.totalArrayElements()) : 1);
            for (auto& r : atomicRanges_[b])
                if (off < r.second && r.first < off + size) fail("'" + name + "' : atomic counter overlaps another counter");
            atomicRanges_[b].push_back({off, off + size});
            q.hasLayout = true;
            q.layout.offset = off;
            atomicNextOffset_[b] = off + size;
        }
        Variable* v = newVar(name, t, q);
        v->global = global;
        v->line = ln;
        if (global && sh_.stage == Stage::Geometry && q.storage == Storage::Out && v->q.layout.stream < 0)
            v->q.layout.stream = std::max(0, sh_.outLayout.stream);
        if (preamble_) v->builtin = builtinVarId(name, sh_.stage);
        if (accept("=")) {
            Expr* init = parseInitializer(t);
            if (v->type.isArray() && v->type.dims[0] == 0 && init->type.isArray()) v->type.dims[0] = init->type.dims[0];
            init = convert(init, v->type, "initializer");
            v->init = init;
            if ((q.storage == Storage::Const || q.storage == Storage::Uniform) && init->cv) v->constVal = init->cv;
            if (q.storage == Storage::Const && !init->cv && v->type.isNumeric() && !preamble_) {
                // non-constant initializer for const: allowed as a read-only variable (GLSL < 4.20 forbids it)
            }
        } else if (q.storage == Storage::Const && !preamble_) {
            fail("'" + name + "' : const variables must be initialized");
        }
        if (global && !preamble_ && sh_.stage == Stage::Compute && (q.storage == Storage::In || q.storage == Storage::Out))
            fail("'" + name + "' : compute shaders cannot declare input or output variables");
        if (v->type.stripArrays().base == BT::Image && !preamble_ && q.storage == Storage::Uniform) checkImageDecl(name, v->type, q);
        if (v->type.stripArrays().base == BT::AtomicUint && !preamble_) {
            if (!v->type.dims.empty() && v->type.dims[0] == 0) fail("'" + name + "' : atomic counter arrays must be explicitly sized");
            if (q.layout.offset >= 0) {
                if (q.layout.offset % 4) fail("'" + name + "' : atomic counter offset must be a multiple of 4");
                int64_t end = (int64_t)q.layout.offset + 4 * std::max(1, v->type.totalArrayElements());
                if (end > kMaxAtomicCounterBufferSize) fail("'" + name + "' : atomic counter offset exceeds GL_MAX_ATOMIC_COUNTER_BUFFER_SIZE");
            }
        }
        // implicit storage for shared/in/out arrays sized by layout
        if (global && sh_.stage == Stage::Geometry && q.storage == Storage::In && !t.dims.empty() && t.dims[0] == 0 &&
            sh_.inLayout.primitive != Layout::kNoPrimitive)
            v->type.dims[0] = primitiveVertexCount(sh_.inLayout.primitive);
        if (global && sh_.stage == Stage::TessControl && q.storage == Storage::Out && !q.patch && !t.dims.empty() &&
            t.dims[0] == 0 && sh_.outLayout.vertices > 0)
            v->type.dims[0] = sh_.outLayout.vertices;
        if (global && (sh_.stage == Stage::TessControl || sh_.stage == Stage::TessEval) && q.storage == Storage::In &&
            !q.patch && !t.dims.empty() && t.dims[0] == 0)
            v->type.dims[0] = 32;
        declareVar(v);
        if (global) {
            sh_.globalVars.push_back(v);
            if (!preamble_) {
                GlobalDecl g;
                g.kind = GlobalDecl::Var;
                g.var = v;
                sh_.globals.push_back(g);
            }
        }
        if (out) out->push_back(v);
        if (!accept(",")) break;
        name = expectIdent();
    }
    expect(";");
}

std::vector<Variable*> Parser::parseParamList() {
    expect("(");
    std::vector<Variable*> params;
    if (isKw("void") && isP(")", 1)) p_ += 1;
    if (!accept(")")) {
        for (;;) {
            Qualifiers pq;
            parseQualifiers(pq);
            if (pq.storage == Storage::Buffer || pq.storage == Storage::Uniform || pq.storage == Storage::Shared)
                fail("invalid storage qualifier on a function parameter");
            if (pq.storage == Storage::Temp) pq.storage = Storage::In;
            Type pt = parseTypeSpecifier(false);
            std::string pn;
            if (peek().kind == PK::Ident) {
                pn = expectIdent();
                Type nd;
                parseArrayDims(nd);
                if (!nd.dims.empty()) { pt.dims.insert(pt.dims.begin(), nd.dims.begin(), nd.dims.end()); checkArrayOfArrays(pt); }
            }
            Qualifiers vq = pq;
            if (vq.storage == Storage::Const) vq.storage = Storage::In;
            Variable* v = newVar(pn, pt, vq);
            v->param = true;
            if (pq.storage == Storage::Const) v->q.storage = Storage::Const;
            params.push_back(v);
            if (accept(")")) break;
            expect(",");
        }
    }
    return params;
}

void Parser::parseFunction(const Qualifiers& q, const Type& ret, const std::string& name) {
    std::vector<Variable*> params = parseParamList();
    std::string mangled = name + "(";
    for (auto* pv : params) {
        mangled += pv->type.str();
        mangled += pv->q.storage == Storage::Out ? "o" : pv->q.storage == Storage::InOut ? "io" : "";
        mangled += ",";
    }
    mangled += ")";
    // find existing declaration
    Function* f = nullptr;
    auto& gs = scopes_[1];
    Sym& sym = gs[name];
    if (sym.var) fail("'" + name + "' : redefinition (already a variable)");
    for (auto* ef : sym.funcs) if (ef->mangled == mangled) f = ef;
    bool hasBody = isP("{");
    if (f) {
        if (f->body && hasBody) fail("function '" + name + "' already has a body");
        if (!(f->ret == ret)) fail("function '" + name + "' redeclared with a different return type");
    } else {
        sh_.funcs.emplace_back();
        f = &sh_.funcs.back();
        f->name = name;
        f->ret = ret;
        f->mangled = mangled;
        f->line = line();
        f->params = params;
        sym.funcs.push_back(f);
        sh_.functions.push_back(f);
    }
    if (q.subroutine) {
        f->subroutineTypes = q.subroutineTypes;
        if (q.hasLayout && q.layout.index >= 0) f->subroutineIndex = q.layout.index;
        for (auto& tn : q.subroutineTypes) {
            Sym* ts = lookup(tn);
            if (!ts || !ts->subrType) fail("'" + tn + "' is not a subroutine type");
            const Function* proto = ts->subrType;
            bool match = proto->ret == ret && proto->params.size() == params.size();
            for (size_t i = 0; match && i < params.size(); i++)
                match = proto->params[i]->type == params[i]->type && proto->params[i]->q.storage == params[i]->q.storage;
            if (!match) fail("subroutine function '" + name + "' does not match subroutine type '" + tn + "'");
        }
    }
    if (!hasBody) {
        expect(";");
        return;
    }
    f->params = params;
    push();
    for (auto* pv : params) if (!pv->name.empty()) declareVar(pv);
    Function* saved = curFn_;
    curFn_ = f;
    f->body = parseCompound(false);
    curFn_ = saved;
    pop();
    GlobalDecl g;
    g.kind = GlobalDecl::FunctionDef;
    g.fn = f;
    sh_.globals.push_back(g);
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------
Stmt* Parser::parseCompound(bool newScope) {
    Stmt* s = S(SK::Block);
    s->newScope = newScope;
    expect("{");
    if (newScope) push();
    while (!accept("}")) {
        if (peek().kind == PK::End) fail("unexpected end of input, missing '}'");
        s->body.push_back(parseStatement());
    }
    if (newScope) pop();
    return s;
}

bool Parser::startsDeclaration() {
    const auto& t = peek();
    if (t.kind != PK::Ident) return false;
    if (t.text == "struct" || t.text == "precision") return true;
    if (isQualifierAt(0)) return true;
    if (isTypeName(t.text)) {
        if (isP("(", 1)) return false;
        if (isP("[", 1)) {
            // could be "float[3] x" or "float[3](...)"
            int depth = 0;
            for (int k = 1;; k++) {
                const auto& tk = peek(k);
                if (tk.kind == PK::End) return false;
                if (tk.kind == PK::Punct && tk.text == "[") depth++;
                else if (tk.kind == PK::Punct && tk.text == "]") {
                    depth--;
                    if (depth == 0 && !isP("[", k + 1)) return !isP("(", k + 1);
                }
            }
        }
        return true;
    }
    return false;
}

Stmt* Parser::parseLocalDeclaration() {
    if (acceptKw("precision")) {
        while (!accept(";")) p_++;
        return S(SK::Empty);
    }
    Qualifiers q;
    parseQualifiers(q);
    Type t = parseTypeSpecifier();
    Stmt* s = S(SK::Decl);
    if (accept(";")) return s;  // struct definition
    accept(",");  // "type , a;" : a declaration list may start with a bare type
    std::string name = expectIdent();
    parseDeclarators(q, t, name, false, &s->vars);
    return s;
}

Stmt* Parser::parseStatement() {
    const auto& t = peek();
    int ln = t.line;
    if (isP("{")) return parseCompound(true);
    if (accept(";")) return S(SK::Empty);
    if (t.kind == PK::Ident) {
        if (acceptKw("if")) {
            Stmt* s = S(SK::If);
            s->line = ln;
            expect("(");
            Expr* c = parseExpression();
            if (!(c->type.isScalar() && c->type.base == BT::Bool)) fail("if condition must be a scalar boolean", c->line);
            s->expr = c;
            expect(")");
            push();
            s->then = parseStatement();
            pop();
            if (acceptKw("else")) {
                push();
                s->els = parseStatement();
                pop();
            }
            return s;
        }
        if (acceptKw("for")) {
            Stmt* s = S(SK::For);
            s->line = ln;
            expect("(");
            push();
            if (accept(";")) s->init = S(SK::Empty);
            else if (startsDeclaration()) s->init = parseLocalDeclaration();
            else {
                Stmt* es = S(SK::Expr);
                es->expr = parseExpression();
                expect(";");
                s->init = es;
            }
            if (!isP(";")) {
                Expr* c = parseExpression();
                if (!(c->type.isScalar() && c->type.base == BT::Bool)) fail("for condition must be a scalar boolean", c->line);
                s->cond = c;
            }
            expect(";");
            if (!isP(")")) s->incr = parseExpression();
            expect(")");
            loopDepth_++;
            push();
            s->then = parseStatement();
            pop();
            loopDepth_--;
            pop();
            return s;
        }
        if (acceptKw("while")) {
            Stmt* s = S(SK::While);
            s->line = ln;
            expect("(");
            Expr* c = parseExpression();
            if (!(c->type.isScalar() && c->type.base == BT::Bool)) fail("while condition must be a scalar boolean", c->line);
            s->expr = c;
            expect(")");
            loopDepth_++;
            push();
            s->then = parseStatement();
            pop();
            loopDepth_--;
            return s;
        }
        if (acceptKw("do")) {
            Stmt* s = S(SK::DoWhile);
            s->line = ln;
            loopDepth_++;
            push();
            s->then = parseStatement();
            pop();
            loopDepth_--;
            if (!acceptKw("while")) fail("expected 'while' after do body");
            expect("(");
            Expr* c = parseExpression();
            if (!(c->type.isScalar() && c->type.base == BT::Bool)) fail("do-while condition must be a scalar boolean", c->line);
            s->expr = c;
            expect(")");
            expect(";");
            return s;
        }
        if (acceptKw("switch")) {
            Stmt* s = S(SK::Switch);
            s->line = ln;
            expect("(");
            Expr* c = parseExpression();
            if (!(c->type.isScalar() && c->type.isIntLike())) fail("switch expression must be a scalar integer", c->line);
            s->expr = c;
            expect(")");
            expect("{");
            switchDepth_++;
            push();
            while (!accept("}")) {
                if (peek().kind == PK::End) fail("unexpected end of input in switch");
                s->body.push_back(parseStatement());
            }
            pop();
            switchDepth_--;
            return s;
        }
        if (acceptKw("case")) {
            if (!switchDepth_) fail("'case' outside of switch");
            Stmt* s = S(SK::Case);
            s->line = ln;
            Expr* e = parseConditional();
            constInt(e, "case label");
            s->expr = e;
            expect(":");
            return s;
        }
        if (acceptKw("default")) {
            if (!switchDepth_) fail("'default' outside of switch");
            expect(":");
            Stmt* s = S(SK::Default);
            s->line = ln;
            return s;
        }
        if (acceptKw("break")) {
            if (!loopDepth_ && !switchDepth_) fail("'break' outside of loop or switch");
            expect(";");
            Stmt* s = S(SK::Break);
            s->line = ln;
            return s;
        }
        if (acceptKw("continue")) {
            if (!loopDepth_) fail("'continue' outside of loop");
            expect(";");
            Stmt* s = S(SK::Continue);
            s->line = ln;
            return s;
        }
        if (acceptKw("return")) {
            Stmt* s = S(SK::Return);
            s->line = ln;
            if (!accept(";")) {
                Expr* e = parseExpression();
                if (curFn_->ret.isVoid()) fail("void function cannot return a value", e->line);
                s->expr = convert(e, curFn_->ret, "return");
                expect(";");
            } else if (!curFn_->ret.isVoid()) {
                fail("non-void function must return a value", ln);
            }
            return s;
        }
        if (acceptKw("discard")) {
            if (sh_.stage != Stage::Fragment) fail("'discard' is only allowed in fragment shaders", ln);
            expect(";");
            Stmt* s = S(SK::Discard);
            s->line = ln;
            return s;
        }
        if (startsDeclaration()) return parseLocalDeclaration();
    }
    Stmt* s = S(SK::Expr);
    s->line = ln;
    s->expr = parseExpression();
    expect(";");
    return s;
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------
Expr* Parser::parseExpression() {
    Expr* e = parseAssignment();
    while (isP(",")) {
        int ln = line();
        p_++;
        Expr* r = parseAssignment();
        Expr* c = E(EK::Comma, r->type, ln);
        c->args = {e, r};
        e = c;
    }
    return e;
}

static Op assignOp(const std::string& s) {
    if (s == "=") return Op::Assign;
    if (s == "+=") return Op::AddAssign;
    if (s == "-=") return Op::SubAssign;
    if (s == "*=") return Op::MulAssign;
    if (s == "/=") return Op::DivAssign;
    if (s == "%=") return Op::ModAssign;
    if (s == "<<=") return Op::ShlAssign;
    if (s == ">>=") return Op::ShrAssign;
    if (s == "&=") return Op::AndAssign;
    if (s == "|=") return Op::OrAssign;
    if (s == "^=") return Op::XorAssign;
    return Op::None;
}

Expr* Parser::parseAssignment() {
    Expr* l = parseConditional();
    const auto& t = peek();
    if (t.kind == PK::Punct) {
        Op op = assignOp(t.text);
        if (op != Op::None) {
            int ln = t.line;
            p_++;
            Expr* r = parseAssignment();
            return mkAssign(op, l, r, ln);
        }
    }
    return l;
}

Expr* Parser::parseConditional() {
    Expr* c = parseBinary(1);
    if (isP("?")) {
        int ln = line();
        p_++;
        Expr* a = parseExpression();
        expect(":");
        Expr* b = parseAssignment();
        return mkTernary(c, a, b, ln);
    }
    return c;
}

static int binPrec(const PPToken& t, Op& op) {
    if (t.kind != PK::Punct) return -1;
    static const std::unordered_map<std::string, std::pair<int, Op>> m = {
        {"||", {1, Op::LogOr}},  {"^^", {2, Op::LogXor}}, {"&&", {3, Op::LogAnd}}, {"|", {4, Op::BitOr}},
        {"^", {5, Op::BitXor}},  {"&", {6, Op::BitAnd}},  {"==", {7, Op::Eq}},     {"!=", {7, Op::Ne}},
        {"<", {8, Op::Lt}},      {">", {8, Op::Gt}},      {"<=", {8, Op::Le}},     {">=", {8, Op::Ge}},
        {"<<", {9, Op::Shl}},    {">>", {9, Op::Shr}},    {"+", {10, Op::Add}},    {"-", {10, Op::Sub}},
        {"*", {11, Op::Mul}},    {"/", {11, Op::Div}},    {"%", {11, Op::Mod}}};
    auto it = m.find(t.text);
    if (it == m.end()) return -1;
    op = it->second.second;
    return it->second.first;
}

Expr* Parser::parseBinary(int minPrec) {
    Expr* l = parseUnary();
    for (;;) {
        Op op;
        int pr = binPrec(peek(), op);
        if (pr < 0 || pr < minPrec) break;
        int ln = line();
        p_++;
        Expr* r = parseBinary(pr + 1);
        l = mkBinary(op, l, r, ln);
    }
    return l;
}

Expr* Parser::parseUnary() {
    const auto& t = peek();
    int ln = t.line;
    if (t.kind == PK::Punct) {
        Op op = Op::None;
        if (t.text == "++") op = Op::PreInc;
        else if (t.text == "--") op = Op::PreDec;
        else if (t.text == "-") op = Op::Neg;
        else if (t.text == "+") op = Op::Plus;
        else if (t.text == "!") op = Op::Not;
        else if (t.text == "~") op = Op::BitNot;
        if (op != Op::None) {
            p_++;
            Expr* a = parseUnary();
            return mkUnary(op, a, ln);
        }
    }
    return parsePostfix(parsePrimary());
}

std::vector<Expr*> Parser::parseArgs() {
    std::vector<Expr*> args;
    expect("(");
    if (isKw("void") && isP(")", 1)) p_++;
    if (accept(")")) return args;
    for (;;) {
        args.push_back(parseAssignment());
        if (accept(")")) break;
        expect(",");
    }
    return args;
}

Expr* Parser::parsePrimary() {
    const auto& t = peek();
    int ln = t.line;
    if (t.kind == PK::Number) {
        p_++;
        return mkLiteral(t);
    }
    if (t.kind == PK::Punct && t.text == "(") {
        p_++;
        Expr* e = parseExpression();
        expect(")");
        return e;
    }
    if (t.kind != PK::Ident) fail("syntax error: unexpected '" + t.text + "'");
    if (t.text == "true" || t.text == "false") {
        p_++;
        Expr* e = E(EK::Const, Type::scalar(BT::Bool), ln);
        e->cv = std::make_shared<ConstVal>();
        Scalar s;
        s.b = t.text == "true";
        e->cv->c.push_back(s);
        return e;
    }
    // constructors of builtin / struct types
    if (isTypeName(t.text) && (isP("(", 1) || isP("[", 1))) {
        Type ct = parseTypeSpecifier(false);
        auto args = parseArgs();
        return mkCtor(ct, args, ln);
    }
    if (t.text == "struct") fail("syntax error: unexpected 'struct'");
    std::string name = t.text;
    p_++;
    if (isP("(")) {
        auto args = parseArgs();
        return mkCall(name, args, ln);
    }
    return mkVarRef(name, ln);
}

Expr* Parser::parsePostfix(Expr* e) {
    for (;;) {
        int ln = line();
        if (accept("[")) {
            Expr* i = parseExpression();
            expect("]");
            e = mkIndex(e, i, ln);
            if (e->type.base == BT::Subroutine && !e->type.isArray() && isP("(")) {
                // call through an element of a subroutine uniform array (of arrays): flatten the index
                std::vector<Expr*> idx;
                Expr* b = e;
                while (b->kind == EK::Index) { idx.insert(idx.begin(), b->args[1]); b = b->args[0]; }
                if (b->kind != EK::Var) fail("invalid subroutine call", ln);
                Expr* flat = nullptr;
                const std::vector<int>& dims = b->var->type.dims;
                for (size_t k = 0; k < idx.size(); k++) {
                    Expr* ik = idx[k]->type.base == BT::Int ? idx[k] : mkCtor(Type::scalar(BT::Int), {idx[k]}, ln);
                    if (!flat) { flat = ik; continue; }
                    Expr* d = E(EK::Const, Type::scalar(BT::Int), ln);
                    d->cv = std::make_shared<ConstVal>();
                    Scalar sc;
                    sc.i = dims[k];
                    d->cv->c.push_back(sc);
                    flat = mkBinary(Op::Add, mkBinary(Op::Mul, flat, d, ln), ik, ln);
                }
                auto args = parseArgs();
                e = mkSubroutineCall(b->var, flat, args, ln);
            }
        } else if (accept(".")) {
            std::string f = expectIdent();
            if (f == "length" && isP("(")) {
                expect("(");
                expect(")");
                Expr* L = E(EK::Length, Type::scalar(BT::Int), ln);
                L->args.push_back(e);
                const Type& bt = e->type;
                int n = -1;
                if (bt.isArray()) n = bt.dims[0];
                else if (bt.isMatrix()) n = bt.cols;
                else if (bt.isVector()) n = bt.vec;
                else fail("length() can only be applied to arrays, vectors and matrices", ln);
                if (n > 0) {
                    L->cv = std::make_shared<ConstVal>();
                    Scalar s;
                    s.i = n;
                    L->cv->c.push_back(s);
                }
                e = L;
            } else {
                e = mkField(e, f, ln);
            }
        } else if (isP("++") || isP("--")) {
            Op op = isP("++") ? Op::PostInc : Op::PostDec;
            p_++;
            e = mkUnary(op, e, ln);
        } else {
            break;
        }
    }
    return e;
}

Expr* Parser::mkLiteral(const PPToken& t) {
    std::string s = t.text;
    int ln = t.line;
    bool isHex = s.size() > 1 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X');
    bool isFloat = !isHex && (s.find('.') != std::string::npos || s.find('e') != std::string::npos || s.find('E') != std::string::npos);
    std::string suf;
    while (!s.empty() && std::isalpha((unsigned char)s.back()) && !(isHex && std::isxdigit((unsigned char)s.back()))) {
        suf.insert(suf.begin(), s.back());
        s.pop_back();
    }
    std::string lsuf = suf;
    std::transform(lsuf.begin(), lsuf.end(), lsuf.begin(), ::tolower);
    if (lsuf == "f" || lsuf == "lf" || lsuf == "hf") isFloat = true;
    auto cv = std::make_shared<ConstVal>();
    Scalar sc;
    Expr* e;
    if (isFloat) {
        bool dbl = lsuf == "lf";
        if (!lsuf.empty() && lsuf != "f" && lsuf != "lf" && lsuf != "hf") fail("invalid floating point suffix '" + suf + "'", ln);
        double v = std::strtod(s.c_str(), nullptr);
        sc.f = dbl ? v : (double)(float)v;
        e = E(EK::Const, Type::scalar(dbl ? BT::Double : BT::Float), ln);
    } else {
        bool u = lsuf == "u";
        if (!lsuf.empty() && !u) fail("invalid integer suffix '" + suf + "'", ln);
        unsigned long long v = 0;
        errno = 0;
        if (isHex) v = std::strtoull(s.c_str() + 2, nullptr, 16);
        else if (s.size() > 1 && s[0] == '0') v = std::strtoull(s.c_str(), nullptr, 8);
        else v = std::strtoull(s.c_str(), nullptr, 10);
        if (v > 0xffffffffull) fail("integer constant overflow", ln);
        if (u) sc.u = (uint32_t)v;
        else sc.i = (int32_t)(uint32_t)v;
        e = E(EK::Const, Type::scalar(u ? BT::UInt : BT::Int), ln);
    }
    cv->c.push_back(sc);
    e->cv = cv;
    return e;
}

Expr* Parser::mkVarRef(const std::string& name, int ln) {
    Sym* s = lookup(name);
    if (!s || !s->var) {
        if (s && !s->funcs.empty()) fail("'" + name + "' : function name used as a variable", ln);
        fail("'" + name + "' : undeclared identifier", ln);
    }
    Variable* v = s->var;
    Expr* e = E(EK::Var, v->type, ln);
    e->var = v;
    if (s->member >= 0) {
        Expr* f = E(EK::Field, v->block->members[s->member].type, ln);
        f->args.push_back(e);
        f->field = s->member;
        return f;
    }
    if (v->q.storage == Storage::Const && v->constVal) e->cv = v->constVal;
    return e;
}

void Parser::markWritten(Expr* e) {
    Expr* lastIndex = nullptr;
    while (e->kind == EK::Index || e->kind == EK::Field || e->kind == EK::Swizzle) {
        if (e->kind == EK::Index && e->args[0]->kind == EK::Var) lastIndex = e;
        e = e->args[0];
    }
    if (e->kind != EK::Var) return;
    e->var->written = true;
    // GLSL 4.60 4.3.6: per-vertex outputs of a tessellation control shader may only be written
    // at the invocation's own vertex (index gl_InvocationID).
    const Variable* v = e->var;
    if (!preamble_ && sh_.stage == Stage::TessControl && v->global && v->q.storage == Storage::Out && !v->q.patch && v->type.isArray() &&
        (v->builtin == BuiltinVar::None || v->builtin == BuiltinVar::Out)) {
        const Expr* idx = lastIndex ? lastIndex->args[1] : nullptr;
        if (!idx || idx->kind != EK::Var || idx->var->builtin != BuiltinVar::InvocationID)
            fail("tessellation control shader per-vertex output '" + v->name + "' must be indexed with gl_InvocationID when written");
    }
}

Expr* Parser::mkIndex(Expr* b, Expr* i, int ln) {
    if (!(i->type.isScalar() && i->type.isIntLike())) fail("array index must be a scalar integer", ln);
    Type rt;
    const Type& bt = b->type;
    int limit = -1;
    if (bt.isArray()) { rt = bt.elem(); limit = bt.dims[0]; }
    else if (bt.isMatrix()) { rt = bt.column(); limit = bt.cols; }
    else if (bt.isVector()) { rt = bt.scalarType(); limit = bt.vec; }
    else fail("'[' : left of '[' is not an array, matrix or vector", ln);
    if (sh_.es && sh_.version < 320 && !i->cv && bt.isArray()) {
        // GLSL ES 3.10 §4.3.9 / §4.1.7.1: arrays of uniform/buffer blocks and of opaque types
        // may only be indexed with constant integral expressions.
        bool blockArray = b->kind == EK::Var && b->var->block &&
                          (b->var->q.storage == Storage::Uniform || b->var->q.storage == Storage::Buffer);
        if (blockArray || rt.base == BT::Sampler || rt.base == BT::Image)
            fail("'[' : arrays of interface blocks and opaque types must be indexed with a constant expression in GLSL ES 3.10", ln);
    }
    Expr* e = E(EK::Index, rt, ln);
    e->args = {b, i};
    if (i->cv) {
        int64_t idx = i->type.base == BT::UInt ? (int64_t)i->cv->c[0].u : i->cv->c[0].i;
        if (idx < 0 || (limit > 0 && idx >= limit)) fail("array index out of range", ln);
        if (limit == 0 && b->kind == EK::Var) {
            int& m = maxIndex_[b->var];
            m = std::max(m, (int)idx);
        }
        if (limit == 0 && b->kind == EK::Field && b->args[0]->type.sdef && b->args[0]->type.sdef->isBlock) {
            auto key = std::make_pair((StructDef*)b->args[0]->type.sdef, b->field);
            auto it = maxMemberIndex_.find(key);
            maxMemberIndex_[key] = it == maxMemberIndex_.end() ? (int)idx : std::max(it->second, (int)idx);
        }
        if (b->cv) e->cv = foldIndex(b, (int)idx, rt);
    } else if (limit == 0 && b->kind == EK::Var && b->var->q.storage != Storage::Buffer) {
        // GLSL 4.60 7.1: gl_ClipDistance / gl_CullDistance must be sized by redeclaration or by
        // constant indexing only (other unsized arrays stay lenient for old shaders).
        if (!preamble_ && (b->var->builtin == BuiltinVar::ClipDistance || b->var->builtin == BuiltinVar::CullDistance))
            fail("'" + b->var->name + "' must be redeclared with an explicit size to be indexed with a non-constant expression", ln);
        b->var->used = true;  // dynamic indexing of implicitly-sized array: size to the maximum at finalize
    }
    return e;
}

Expr* Parser::mkField(Expr* b, const std::string& name, int ln) {
    const Type& bt = b->type;
    if (bt.base == BT::Struct && !bt.isArray()) {
        int idx = bt.sdef->findMember(name);
        if (idx < 0) fail("'" + name + "' : no such field in structure '" + bt.sdef->name + "'", ln);
        Expr* e = E(EK::Field, bt.sdef->members[idx].type, ln);
        e->args.push_back(b);
        e->field = idx;
        if (b->cv) e->cv = foldField(b, idx, e->type);
        return e;
    }
    if ((bt.isScalar() || bt.isVector())) {
        if (name.size() > 4) fail("illegal vector field selection '" + name + "'", ln);
        static const char* sets[] = {"xyzw", "rgba", "stpq"};
        int set = -1;
        uint8_t swz[4];
        for (size_t i = 0; i < name.size(); i++) {
            int found = -1, fs = -1;
            for (int s = 0; s < 3; s++) {
                const char* p = std::strchr(sets[s], name[i]);
                if (p && name[i]) { found = (int)(p - sets[s]); fs = s; break; }
            }
            if (found < 0 || (set >= 0 && fs != set)) fail("illegal vector field selection '" + name + "'", ln);
            set = fs;
            if (found >= bt.vec) fail("vector field selection out of range '" + name + "'", ln);
            swz[i] = (uint8_t)found;
        }
        int n = (int)name.size();
        Type rt = n == 1 ? bt.scalarType() : Type::vector(bt.base, n);
        Expr* e = E(EK::Swizzle, rt, ln);
        e->args.push_back(b);
        std::memcpy(e->swz, swz, n);
        e->nswz = (uint8_t)n;
        if (b->cv) e->cv = foldSwizzle(b, swz, n, rt);
        return e;
    }
    fail("'" + name + "' : field selection requires structure or vector on left hand side", ln);
}

Expr* Parser::mkUnary(Op op, Expr* a, int ln) {
    const Type& t = a->type;
    switch (op) {
        case Op::Neg:
        case Op::Plus:
            if (!t.isNumeric() || t.base == BT::Bool) fail("wrong operand type for unary minus/plus", ln);
            break;
        case Op::Not:
            if (!(t.isScalar() && t.base == BT::Bool)) fail("'!' requires a scalar boolean", ln);
            break;
        case Op::BitNot:
            if (!(t.isNumeric() && t.isIntLike() && !t.isMatrix())) fail("'~' requires an integer", ln);
            break;
        default:  // inc/dec
            if (!t.isNumeric() || t.base == BT::Bool) fail("wrong operand type for increment/decrement", ln);
            if (!exprIsLvalue(a)) fail("increment/decrement requires an l-value", ln);
            markWritten(a);
            break;
    }
    Expr* e = E(EK::Unary, t, ln);
    e->op = op;
    e->args.push_back(a);
    if (op == Op::Neg || op == Op::Plus || op == Op::Not || op == Op::BitNot) e->cv = foldUnary(op, t, a);
    return e;
}

Expr* Parser::mkBinary(Op op, Expr* a, Expr* b, int ln) {
    Type ta = a->type, tb = b->type;
    if (ta.isError() || tb.isError()) fail("invalid operand", ln);
    Type rt;
    switch (op) {
        case Op::Add:
        case Op::Sub:
        case Op::Mul:
        case Op::Div:
        case Op::Mod: {
            if (!ta.isNumeric() || !tb.isNumeric() || ta.base == BT::Bool || tb.base == BT::Bool)
                fail("wrong operand types for arithmetic operator ('" + ta.str() + "' and '" + tb.str() + "')", ln);
            if (op == Op::Mod && (!ta.isIntLike() || !tb.isIntLike()))
                fail("'%' requires integer operands", ln);
            BT cb = commonBase(ta.base, tb.base, ln);
            a = convert(a, ta.withBase(cb));
            b = convert(b, tb.withBase(cb));
            ta = a->type;
            tb = b->type;
            if (ta.isScalar()) rt = tb;
            else if (tb.isScalar()) rt = ta;
            else if (ta.isVector() && tb.isVector()) {
                if (ta.vec != tb.vec) fail("vector size mismatch", ln);
                rt = ta;
            } else if (ta.isMatrix() && tb.isMatrix()) {
                if (op == Op::Mul) {
                    if (ta.cols != tb.vec) fail("matrix size mismatch in multiplication", ln);
                    rt = Type::matrix(cb, tb.cols, ta.vec);
                } else {
                    if (ta.cols != tb.cols || ta.vec != tb.vec) fail("matrix size mismatch", ln);
                    rt = ta;
                }
            } else if (ta.isMatrix() && tb.isVector()) {
                if (op != Op::Mul || ta.cols != tb.vec) fail("matrix/vector size mismatch", ln);
                rt = Type::vector(cb, ta.vec);
            } else if (ta.isVector() && tb.isMatrix()) {
                if (op != Op::Mul || ta.vec != tb.vec) fail("vector/matrix size mismatch", ln);
                rt = Type::vector(cb, tb.cols);
            } else fail("invalid operands", ln);
            break;
        }
        case Op::Shl:
        case Op::Shr:
            if (!ta.isNumeric() || !tb.isNumeric() || !ta.isIntLike() || !tb.isIntLike() || ta.isMatrix() || tb.isMatrix())
                fail("shift requires integer operands", ln);
            if (tb.isVector() && (!ta.isVector() || ta.vec != tb.vec)) fail("shift operand size mismatch", ln);
            rt = ta;
            break;
        case Op::BitAnd:
        case Op::BitOr:
        case Op::BitXor: {
            if (!ta.isNumeric() || !tb.isNumeric() || !ta.isIntLike() || !tb.isIntLike() || ta.isMatrix() || tb.isMatrix())
                fail("bitwise operator requires integer operands", ln);
            BT cb = commonBase(ta.base, tb.base, ln);
            a = convert(a, ta.withBase(cb));
            b = convert(b, tb.withBase(cb));
            if (a->type.isVector() && b->type.isVector() && a->type.vec != b->type.vec) fail("vector size mismatch", ln);
            rt = a->type.isScalar() ? b->type : a->type;
            break;
        }
        case Op::LogAnd:
        case Op::LogOr:
        case Op::LogXor:
            if (!(ta.isScalar() && ta.base == BT::Bool && tb.isScalar() && tb.base == BT::Bool))
                fail("logical operator requires scalar boolean operands", ln);
            rt = Type::scalar(BT::Bool);
            break;
        case Op::Lt:
        case Op::Gt:
        case Op::Le:
        case Op::Ge: {
            if (!ta.isScalar() || !tb.isScalar() || ta.base == BT::Bool || tb.base == BT::Bool)
                fail("relational operator requires scalar numeric operands (use lessThan() etc for vectors)", ln);
            BT cb = commonBase(ta.base, tb.base, ln);
            a = convert(a, ta.withBase(cb));
            b = convert(b, tb.withBase(cb));
            rt = Type::scalar(BT::Bool);
            break;
        }
        case Op::Eq:
        case Op::Ne: {
            if (ta.containsOpaque() || tb.containsOpaque()) fail("cannot compare opaque types", ln);
            if (ta.isNumeric() && tb.isNumeric() && ta.base != tb.base) {
                BT cb = commonBase(ta.base, tb.base, ln);
                a = convert(a, ta.withBase(cb));
                b = convert(b, tb.withBase(cb));
            }
            if (a->type != b->type) fail("comparison of mismatched types '" + a->type.str() + "' and '" + b->type.str() + "'", ln);
            rt = Type::scalar(BT::Bool);
            break;
        }
        default:
            fail("internal: bad binary op", ln);
    }
    Expr* e = E(EK::Binary, rt, ln);
    e->op = op;
    e->args = {a, b};
    e->cv = foldBinary(op, rt, a, b);
    return e;
}

static Op compoundToBinary(Op op) {
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

Expr* Parser::mkAssign(Op op, Expr* l, Expr* r, int ln) {
    if (!exprIsLvalue(l)) fail("assignment to a non l-value", ln);
    if (l->type.containsOpaque()) fail("cannot assign to opaque type", ln);
    markWritten(l);
    if (op == Op::Assign) {
        r = convert(r, l->type, "assignment");
    } else {
        Op bop = compoundToBinary(op);
        // type check via the binary operator
        Expr* tmp = mkBinary(bop, l, r, ln);
        if (tmp->type != l->type) fail("compound assignment result type mismatch", ln);
        // keep the (possibly converted) rhs
        r = tmp->args[1];
        if (bop == Op::Shl || bop == Op::Shr) r = tmp->args[1];
    }
    Expr* e = E(EK::Assign, l->type, ln);
    e->op = op;
    e->args = {l, r};
    return e;
}

Expr* Parser::mkTernary(Expr* c, Expr* a, Expr* b, int ln) {
    if (!(c->type.isScalar() && c->type.base == BT::Bool)) fail("ternary condition must be a scalar boolean", ln);
    if (a->type != b->type) {
        if (convCost(a->type, b->type) >= 0) a = convert(a, b->type);
        else if (convCost(b->type, a->type) >= 0) b = convert(b, a->type);
        else fail("ternary operands have mismatched types '" + a->type.str() + "' and '" + b->type.str() + "'", ln);
    }
    Expr* e = E(EK::Ternary, a->type, ln);
    e->args = {c, a, b};
    if (c->cv && a->cv && b->cv) e->cv = c->cv->c[0].b ? a->cv : b->cv;
    return e;
}

Expr* Parser::mkCtor(Type t, std::vector<Expr*> args, int ln) {
    if (t.isOpaque()) fail("cannot construct opaque type", ln);
    if (args.empty()) fail("constructor requires arguments", ln);
    if (t.isArray()) {
        if (t.dims[0] == 0) t.dims[0] = (int)args.size();
        if ((int)args.size() != t.dims[0]) fail("wrong number of arguments for array constructor", ln);
        Type et = t.elem();
        for (auto& a : args) {
            if (et.isArray() && a->type.isArray() && et.dims[0] == 0) et = a->type;  // unsized inner dims
            a = convert(a, et, "array constructor");
        }
        t = et.arrayOf(t.dims[0]);
    } else if (t.base == BT::Struct) {
        if (args.size() != t.sdef->members.size()) fail("wrong number of arguments for struct constructor", ln);
        for (size_t i = 0; i < args.size(); i++) args[i] = convert(args[i], t.sdef->members[i].type, "struct constructor");
    } else {
        int need = t.components();
        int have = 0;
        for (auto* a : args) {
            if (!a->type.isNumeric()) fail("constructor argument must be a scalar, vector or matrix", ln);
            have += a->type.components();
        }
        if (t.isScalar()) {
            if (args.size() != 1) fail("too many arguments for scalar constructor", ln);
        } else if (!(args.size() == 1 && (args[0]->type.isScalar() || (args[0]->type.isMatrix() && t.isMatrix())))) {
            if (t.isMatrix()) for (auto* a : args) if (a->type.isMatrix() && args.size() > 1) fail("matrix argument in multi-argument matrix constructor", ln);
            if (have < need) fail("not enough data provided for construction", ln);
            if (have - args.back()->type.components() >= need) fail("too many arguments for constructor", ln);
        }
    }
    Expr* e = E(EK::Ctor, t, ln);
    e->args = args;
    e->cv = foldCtor(t, e->args);
    return e;
}

Expr* Parser::mkSubroutineCall(Variable* v, Expr* index, std::vector<Expr*> args, int ln) {
    const Function* proto = v->type.subr;
    if (args.size() != proto->params.size()) fail("wrong number of arguments for subroutine '" + v->name + "'", ln);
    for (size_t i = 0; i < args.size(); i++) {
        const Variable* p = proto->params[i];
        if (p->q.storage == Storage::Out || p->q.storage == Storage::InOut) {
            if (!exprIsLvalue(args[i])) fail("subroutine argument " + std::to_string(i + 1) + " must be an l-value", ln);
            markWritten(args[i]);
            if (args[i]->type != p->type) fail("subroutine out argument type mismatch", ln);
        } else {
            args[i] = convert(args[i], p->type, "subroutine argument");
        }
    }
    Expr* e = E(EK::SubroutineCall, proto->ret, ln);
    e->var = v;
    e->fn = const_cast<Function*>(proto);
    e->args = args;
    e->aux = index;
    return e;
}

Function* Parser::builtinFn(const Builtin* b) {
    auto it = builtinFns_.find(b);
    if (it != builtinFns_.end()) return it->second;
    sh_.funcs.emplace_back();
    Function* f = &sh_.funcs.back();
    f->name = b->name;
    f->ret = b->ret;
    f->builtin = b;
    builtinFns_[b] = f;
    return f;
}

// Format layout qualifiers of image uniforms (GLSL 4.60 §4.4.6.2, GLSL ES 3.10 §4.4.7).
void Parser::checkImageDecl(const std::string& name, const Type& t, const Qualifiers& q) {
    const std::string& f = q.layout.format;
    bool writeonly = q.memory & MemWriteOnly, readonly = q.memory & MemReadOnly;
    if (f.empty()) {
        if (!writeonly || sh_.es) fail("'" + name + "' : image variables not qualified writeonly must have a format layout qualifier");
        return;
    }
    BT fb = f.size() > 2 && f.compare(f.size() - 2, 2, "ui") == 0 ? BT::UInt : f.back() == 'i' ? BT::Int : BT::Float;
    const Type e = t.stripArrays();
    if (fb != e.sampled) fail("'" + name + "' : format layout qualifier '" + f + "' does not match the image type");
    if (sh_.es) {
        bool esType = !e.ms && (e.dim == Dim::D2 || ((e.dim == Dim::D3 || e.dim == Dim::Cube) && !e.arrayed));
        if (!esType) fail("'" + name + "' : image type '" + e.str() + "' is not available in GLSL ES 3.10");
        static const std::unordered_set<std::string> esFormats = {"rgba32f", "rgba16f", "r32f", "rgba8", "rgba8_snorm", "rgba32i", "rgba16i",
                                                                  "rgba8i", "r32i", "rgba32ui", "rgba16ui", "rgba8ui", "r32ui"};
        if (!esFormats.count(f)) fail("'" + name + "' : format layout qualifier '" + f + "' is not supported in GLSL ES");
        if (f != "r32f" && f != "r32i" && f != "r32ui" && !readonly && !writeonly)
            fail("'" + name + "' : image variables must be qualified readonly or writeonly unless their format is r32f, r32i or r32ui");
    }
}

// Image builtins respect the variable's memory qualifiers.
static const Variable* imageRoot(const Expr* e) {
    while (e && (e->kind == EK::Index || e->kind == EK::Field)) e = e->args[0];
    return e && e->kind == EK::Var ? e->var : nullptr;
}

Expr* Parser::mkCall(const std::string& name, std::vector<Expr*> args, int ln) {
    Sym* s = lookup(name);
    if (s && s->type) return mkCtor(Type{BT::Struct, 1, 0, Dim::None, false, false, false, BT::Float, s->type, {}}, args, ln);
    if (s && s->var) {
        if (s->var->type.base == BT::Subroutine && !s->var->type.isArray()) return mkSubroutineCall(s->var, nullptr, args, ln);
        fail("'" + name + "' : called object is not a function", ln);
    }
    // candidates
    struct Cand { Function* user; const Builtin* bi; int cost; };
    std::vector<Cand> cands;
    auto scoreParams = [&](const std::vector<Type>& ptypes, const std::vector<uint8_t>& outs) -> int {
        if (ptypes.size() != args.size()) return -1;
        int total = 0;
        for (size_t i = 0; i < args.size(); i++) {
            const Type& pt = ptypes[i];
            const Type& at = args[i]->type;
            if (outs[i]) {
                if (at == pt) continue;
                // out params: lenient - allow when convertible either way
                int c = convCost(pt, at);
                if (c < 0) c = convCost(at, pt);
                if (c < 0) return -1;
                total += c * 4;
                continue;
            }
            if (at == pt) continue;
            // unsized array params match any size
            if (pt.isArray() && at.isArray() && pt.dims.size() == at.dims.size() && pt.stripArrays() == at.stripArrays()) continue;
            int c = convCost(at, pt);
            if (c < 0) return -1;
            total += c;
        }
        return total;
    };
    if (s) {
        for (auto* f : s->funcs) {
            std::vector<Type> pts;
            std::vector<uint8_t> outs;
            for (auto* pv : f->params) {
                pts.push_back(pv->type);
                outs.push_back(pv->q.storage == Storage::Out ? 1 : pv->q.storage == Storage::InOut ? 2 : 0);
            }
            int c = scoreParams(pts, outs);
            if (c >= 0) cands.push_back({f, nullptr, c});
        }
    }
    if (auto* bl = findBuiltins(name)) {
        for (auto* b : *bl) {
            if (!(b->stageMask & (1u << (int)sh_.stage))) continue;
            int c = scoreParams(b->params, b->paramOut);
            if (c >= 0) cands.push_back({nullptr, b, c});
        }
    }
    if (cands.empty()) {
        if (!s && !findBuiltins(name)) fail("'" + name + "' : no matching overloaded function found (undeclared)", ln);
        std::string sig = name + "(";
        for (size_t i = 0; i < args.size(); i++) sig += (i ? ", " : "") + args[i]->type.str();
        fail("'" + sig + ")' : no matching overloaded function found", ln);
    }
    std::stable_sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
        if (a.cost != b.cost) return a.cost < b.cost;
        return a.user && !b.user;
    });
    const Cand& best = cands[0];
    std::vector<Type> ptypes;
    std::vector<uint8_t> outs;
    Type ret;
    if (best.user) {
        for (auto* pv : best.user->params) {
            ptypes.push_back(pv->type);
            outs.push_back(pv->q.storage == Storage::Out ? 1 : pv->q.storage == Storage::InOut ? 2 : 0);
        }
        ret = best.user->ret;
    } else {
        ptypes = best.bi->params;
        outs = best.bi->paramOut;
        ret = best.bi->ret;
    }
    if (best.bi && name.compare(0, 5, "image") == 0 && !args.empty() && args[0]->type.base == BT::Image && name != "imageSize" &&
        name != "imageSamples") {
        if (const Variable* iv = imageRoot(args[0])) {
            bool load = name == "imageLoad" || name.compare(0, 11, "imageAtomic") == 0;
            bool store = name == "imageStore" || name.compare(0, 11, "imageAtomic") == 0;
            if (load && (iv->q.memory & MemWriteOnly)) fail("'" + name + "' : cannot read from writeonly image '" + iv->name + "'", ln);
            if (store && (iv->q.memory & MemReadOnly)) fail("'" + name + "' : cannot write to readonly image '" + iv->name + "'", ln);
        }
    }
    for (size_t i = 0; i < args.size(); i++) {
        if (outs[i]) {
            if (!exprIsLvalue(args[i])) fail("'" + name + "' : argument " + std::to_string(i + 1) + " is an out parameter and must be an l-value", ln);
            markWritten(args[i]);
        } else if (!(ptypes[i].isArray() && args[i]->type.isArray())) {
            args[i] = convert(args[i], ptypes[i], "function argument");
        }
    }
    Expr* e = E(best.user ? EK::Call : EK::Builtin, ret, ln);
    e->args = args;
    e->fn = best.user ? best.user : builtinFn(best.bi);
    if (!best.user) {
        e->cv = foldBuiltin(name, ret, args);
        if (name == "mix" && !best.bi->params[0].isFloatLike() && sh_.version < (sh_.es ? 310 : 450) &&
            !extEnabled("GL_EXT_shader_integer_mix"))
            fail("mix() with integer or boolean operands requires GLSL 4.50 or GL_EXT_shader_integer_mix", ln);
        if (name == "ftransform" && sh_.stage != Stage::Vertex) fail("ftransform() is only available in vertex shaders", ln);
        if (name.compare(0, 6, "atomic") == 0 && name.compare(0, 13, "atomicCounter") != 0) {
            Expr* b = args[0];
            while (b->kind == EK::Index || b->kind == EK::Field || b->kind == EK::Swizzle) b = b->args[0];
            if (b->kind != EK::Var || (b->var->q.storage != Storage::Buffer && b->var->q.storage != Storage::Shared))
                fail("'" + name + "' : first argument must be a buffer or shared variable", ln);
        }
        if (name == "barrier" && sh_.stage != Stage::Compute && sh_.stage != Stage::TessControl)
            fail("barrier() is only available in compute and tessellation control shaders", ln);
        if ((name == "EmitVertex" || name == "EndPrimitive" || name == "EmitStreamVertex" || name == "EndStreamPrimitive") &&
            sh_.stage != Stage::Geometry)
            fail("'" + name + "' is only available in geometry shaders", ln);
    }
    return e;
}

}  // namespace

bool parseShader(Shader& sh, const PPResult& pp, const CompileOptions& opts, std::string& log) {
    Parser p(sh, opts, log);
    return p.run(pp);
}

}  // namespace glsl
