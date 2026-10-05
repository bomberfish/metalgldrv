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
        case EK::Index:
        case EK::Field:
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

struct ParseError {};

struct Sym {
    Variable* var = nullptr;
    int member = -1;  // anonymous block member index
    StructDef* type = nullptr;
    std::vector<Function*> funcs;
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
    Qualifiers defaultUbo_, defaultSsbo_;
    std::unordered_map<const Builtin*, Function*> builtinFns_;
    std::unordered_map<Variable*, int> maxIndex_;

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
    void parseDeclarators(Qualifiers q, const Type& t, std::string name, bool global, std::vector<Variable*>* out);
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
        parseTokens(pp.tokens);
        finalize();
    } catch (ParseError&) {
        return false;
    }
    return true;
}

void Parser::finalize() {
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
    // Fix up expression types that referenced implicitly sized arrays
    for (auto& e : sh_.exprs) {
        if (e.kind == EK::Var && e.var && e.type.base == e.var->type.base && !e.type.dims.empty() && e.type.dims[0] == 0)
            e.type = e.var->type;
    }
    for (auto* f : sh_.functions)
        if (f->name == "main" && f->body) sh_.hasMain = true;
}

// ---------------------------------------------------------------------------
// Qualifiers & layout
// ---------------------------------------------------------------------------
bool Parser::isQualifierAt(int k) {
    const auto& t = peek(k);
    if (t.kind != PK::Ident || !qualifierKeywords().count(t.text)) return false;
    static const std::unordered_set<std::string> soft = {
        "sample", "patch", "buffer", "shared", "precise", "coherent", "volatile", "restrict", "readonly", "writeonly",
        "subroutine", "attribute", "varying", "smooth", "flat", "noperspective", "centroid", "invariant"};
    if (!soft.count(t.text)) return true;
    Sym* s = lookup(t.text);
    if (s && s->var) return false;
    return peek(k + 1).kind == PK::Ident;
}

bool Parser::parseQualifiers(Qualifiers& q) {
    bool any = false;
    for (;;) {
        const auto& t = peek();
        if (!isQualifierAt(0)) break;
        // 'in'/'out' etc must not be confused with identifiers (they are keywords)
        const std::string k = t.text;
        p_++;
        any = true;
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
        else if (k == "subroutine") fail("subroutines are not supported");
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
            Expr* e = parseConditional();
            val = constInt(e, "layout qualifier value");
            hasVal = true;
        }
        auto need = [&]() { if (!hasVal) fail("layout qualifier '" + id + "' requires a value"); return (int)val; };
        if (lid == "location") L.location = need();
        else if (lid == "component") L.component = need();
        else if (lid == "binding") L.binding = need();
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

static int primitiveVertexCount(uint32_t prim) {
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
    for (int i = 0; i < 3; i++) if (L.localSize[i] >= 0) T.localSize[i] = L.localSize[i];
    if (L.maxVertices >= 0) T.maxVertices = L.maxVertices;
    if (L.invocations >= 0) T.invocations = L.invocations;
    if (L.vertices >= 0) T.vertices = L.vertices;
    if (L.primitive) T.primitive = L.primitive;
    if (L.spacing) T.spacing = L.spacing;
    if (L.order) T.order = L.order;
    if (L.pointMode) T.pointMode = true;
    if (L.earlyFragmentTests) T.earlyFragmentTests = true;
    if (L.stream >= 0) T.stream = L.stream;
    if (L.xfbBuffer >= 0) T.xfbBuffer = L.xfbBuffer;
    if (L.xfbStride >= 0) T.xfbStride = L.xfbStride;
    if (q.storage == Storage::In && sh_.stage == Stage::Compute && (L.localSize[0] >= 0 || L.localSize[1] >= 0 || L.localSize[2] >= 0)) {
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
    if (q.storage == Storage::In && sh_.stage == Stage::Geometry && L.primitive) {
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
        Type mt = parseTypeSpecifier();
        do {
            Member m;
            m.line = line();
            m.name = expectIdent();
            m.type = mt;
            Type nd;
            parseArrayDims(nd);
            if (!nd.dims.empty()) m.type.dims.insert(m.type.dims.begin(), nd.dims.begin(), nd.dims.end());
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
    if (hasQ && isP(";")) {
        p_++;
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
    if (hasQ && peek().kind == PK::Ident && !isTypeName(peek().text) && isP("{", 1)) {
        parseBlock(q, nullptr);
        return;
    }
    Type t = parseTypeSpecifier();
    if (accept(";")) return;  // struct declaration
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
    if (q.storage == Storage::Uniform || q.storage == Storage::Buffer) {
        if (q.layout.packing == Packing::None) q.layout.packing = def.layout.packing;
        if (!q.layout.matrix) q.layout.matrix = def.layout.matrix;
    }
    sd->blockQ = q;
    while (!accept("}")) {
        Qualifiers mq;
        parseQualifiers(mq);
        Type mt = parseTypeSpecifier(false);
        do {
            Member m;
            m.line = line();
            m.name = expectIdent();
            m.type = mt;
            Type nd;
            parseArrayDims(nd);
            if (!nd.dims.empty()) m.type.dims.insert(m.type.dims.begin(), nd.dims.begin(), nd.dims.end());
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
            items.push_back(parseInitializer(it));
            idx++;
        } while (accept(","));
        expect("}");
        Type tt = t;
        if (isArr && tt.dims[0] == 0) tt.dims[0] = (int)items.size();
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
    for (;;) {
        Type t = baseT;
        Type nd;
        parseArrayDims(nd);
        if (!nd.dims.empty()) t.dims.insert(t.dims.begin(), nd.dims.begin(), nd.dims.end());
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
        Variable* v = newVar(name, t, q);
        v->global = global;
        v->line = ln;
        if (preamble_) v->builtin = builtinVarId(name, sh_.stage);
        if (accept("=")) {
            Expr* init = parseInitializer(t);
            if (v->type.isArray() && v->type.dims[0] == 0 && init->type.isArray()) v->type.dims[0] = init->type.dims[0];
            init = convert(init, v->type, "initializer");
            v->init = init;
            if (q.storage == Storage::Const && init->cv) v->constVal = init->cv;
            if (q.storage == Storage::Const && !init->cv && v->type.isNumeric() && !preamble_) {
                // non-constant initializer for const: allowed as a read-only variable (GLSL < 4.20 forbids it)
            }
        } else if (q.storage == Storage::Const && !preamble_) {
            fail("'" + name + "' : const variables must be initialized");
        }
        // implicit storage for shared/in/out arrays sized by layout
        if (global && sh_.stage == Stage::Geometry && q.storage == Storage::In && !t.dims.empty() && t.dims[0] == 0 &&
            sh_.inLayout.primitive)
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

void Parser::parseFunction(const Qualifiers& q, const Type& ret, const std::string& name) {
    expect("(");
    std::vector<Variable*> params;
    if (isKw("void") && isP(")", 1)) p_ += 1;
    if (!accept(")")) {
        for (;;) {
            Qualifiers pq;
            parseQualifiers(pq);
            if (pq.storage == Storage::Temp) pq.storage = Storage::In;
            Type pt = parseTypeSpecifier(false);
            std::string pn;
            if (peek().kind == PK::Ident) {
                pn = expectIdent();
                Type nd;
                parseArrayDims(nd);
                if (!nd.dims.empty()) pt.dims.insert(pt.dims.begin(), nd.dims.begin(), nd.dims.end());
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
    (void)q;
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
    while (e->kind == EK::Index || e->kind == EK::Field || e->kind == EK::Swizzle) e = e->args[0];
    if (e->kind == EK::Var) e->var->written = true;
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
    Expr* e = E(EK::Index, rt, ln);
    e->args = {b, i};
    if (i->cv) {
        int64_t idx = i->type.base == BT::UInt ? (int64_t)i->cv->c[0].u : i->cv->c[0].i;
        if (idx < 0 || (limit > 0 && idx >= limit)) fail("array index out of range", ln);
        if (limit == 0 && b->kind == EK::Var) {
            int& m = maxIndex_[b->var];
            m = std::max(m, (int)idx);
        }
        if (b->cv) e->cv = foldIndex(b, (int)idx, rt);
    } else if (limit == 0 && b->kind == EK::Var && b->var->q.storage != Storage::Buffer) {
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

Expr* Parser::mkCall(const std::string& name, std::vector<Expr*> args, int ln) {
    Sym* s = lookup(name);
    if (s && s->type) return mkCtor(Type{BT::Struct, 1, 0, Dim::None, false, false, false, BT::Float, s->type, {}}, args, ln);
    if (s && s->var) fail("'" + name + "' : called object is not a function", ln);
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
