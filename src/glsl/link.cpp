// Program linking: cross-stage validation, resource layout, reflection.
#include <algorithm>
#include <cstring>
#include <functional>

#include "glsl/fold.h"
#include "glsl/gl_enums.h"
#include "glsl/program.h"
#include "glsl/types.h"

namespace glsl {

std::string fillMultipassRecords(const LinkedProgram& prog, MultiPassInfo& mp);  // msl.cpp
void reflectProgramInterface(const LinkedProgram& prog, Reflection& r);              // reflect_io.cpp
std::vector<bool> usedArrayElements(const StageCode& sc, const Variable* v, int n);   // reflect_io.cpp

// ---------------------------------------------------------------------------
// Layout engine
// ---------------------------------------------------------------------------
uint32_t scalarSize(BT b) { return b == BT::Double || b == BT::Int64 || b == BT::UInt64 ? 8 : 4; }

// Value of a GLSL built-in limit constant (src/glsl/limits.def).
static int limitValue(const char* name) {
    static const std::pair<const char*, int> kLimits[] = {
#define X(n, glenum, value) {#n, value},
#include "glsl/limits.def"
#undef X
    };
    for (auto& [n, v] : kLimits)
        if (!std::strcmp(n, name)) return v;
    return 1 << 30;
}

static uint32_t roundUp(uint32_t v, uint32_t a) { return a ? (v + a - 1) / a * a : v; }

static TL computeLayoutImpl(const Type& t, Packing packing, bool rowMajor, int blockAlign);

TL computeLayout(const Type& t, Packing packing, bool rowMajor) { return computeLayoutImpl(t, packing, rowMajor, -1); }

// blockAlign: layout(align = N) on an interface block applies to each member (GLSL 4.60 §4.4.5)
static TL computeLayoutImpl(const Type& t, Packing packing, bool rowMajor, int blockAlign) {
    bool s140 = packing != Packing::Std430;
    auto L = std::make_shared<TypeLayout>();
    if (!t.dims.empty()) {
        TL e = computeLayout(t.elem(), packing, rowMajor);
        uint32_t a = e->align;
        if (s140) a = roundUp(a, 16);
        L->align = a;
        L->arrayStride = roundUp(e->size, a);
        L->size = L->arrayStride * (uint32_t)std::max(t.dims[0], 0);
        L->elem = e;
        L->rowMajor = e->rowMajor;
        L->matrixStride = e->matrixStride;
        return L;
    }
    if (t.base == BT::Struct) {
        uint32_t off = 0, maxA = 1;
        for (auto& m : t.sdef->members) {
            bool rm = m.q.layout.matrix ? m.q.layout.matrix == 2 : rowMajor;
            TL ml = computeLayout(m.type, packing, rm);
            uint32_t a = ml->align;
            if (m.q.layout.align > 0) a = std::max(a, (uint32_t)m.q.layout.align);
            else if (blockAlign > 0) a = std::max(a, (uint32_t)blockAlign);
            if (m.q.layout.offset >= 0) off = (uint32_t)m.q.layout.offset;
            off = roundUp(off, a);
            L->offsets.push_back(off);
            L->members.push_back(ml);
            off += ml->size;
            maxA = std::max(maxA, a);
        }
        if (s140) maxA = roundUp(maxA, 16);
        L->align = maxA;
        L->size = roundUp(off, maxA);
        return L;
    }
    uint32_t N = scalarSize(t.base);
    if (t.cols) {
        // matrix = array of column (or row) vectors
        int vecLen = rowMajor ? t.cols : t.vec;
        int count = rowMajor ? t.vec : t.cols;
        uint32_t va = vecLen == 2 ? 2 * N : 4 * N;
        if (s140) va = roundUp(va, 16);
        L->align = va;
        L->matrixStride = va;
        L->rowMajor = rowMajor;
        L->size = va * count;
        return L;
    }
    if (t.isOpaque()) { L->size = 8; L->align = 8; return L; }
    int n = t.vec;
    L->align = n == 1 ? N : n == 2 ? 2 * N : 4 * N;
    L->size = n * N;
    return L;
}

TL computeBlockLayout(const StructDef& sd, Packing packing, bool rowMajor) {
    Type t;
    t.base = BT::Struct;
    t.sdef = const_cast<StructDef*>(&sd);
    return computeLayoutImpl(t, packing, rowMajor, sd.blockQ.layout.align);
}

XfbCapture xfbCaptureOf(const Type& t, int buffer, uint32_t offset) {
    XfbCapture c;
    c.buffer = buffer;
    c.offset = offset;
    uint32_t off = offset;
    bool first = true;
    // aggregates are flattened to components, each aligned to its own size
    std::function<void(const Type&)> walk = [&](const Type& ty) {
        if (ty.isArray()) {
            Type e = ty.elem();
            for (int i = 0; i < std::max(ty.dims[0], 1); i++) walk(e);
            return;
        }
        if (ty.base == BT::Struct) {
            for (auto& m : ty.sdef->members) walk(m.type);
            return;
        }
        uint32_t cs = scalarSize(ty.base);
        if (first) c.align = cs;
        first = false;
        if (cs == 8) c.dbl = true;
        for (int i = 0; i < ty.components(); i++) {
            off = roundUp(off, cs);
            off += cs;
        }
    };
    walk(t);
    if (c.dbl) {
        c.align = 8;
        off = roundUp(off, 8);
    }
    c.size = off - offset;
    return c;
}

std::string checkXfbLayout(const std::vector<XfbCapture>& caps, const int explicitStride[4]) {
    for (auto& c : caps)
        if (c.offset % c.align)
            return "'xfb_offset' : offset of '" + c.name + "' must be a multiple of " + std::to_string(c.align);
    for (int b = 0; b < 4; b++) {
        std::vector<const XfbCapture*> list;
        bool dbl = false;
        uint32_t end = 0;
        for (auto& c : caps)
            if (c.buffer == b) {
                list.push_back(&c);
                dbl |= c.dbl;
                end = std::max(end, c.offset + c.size);
            }
        std::sort(list.begin(), list.end(), [](const XfbCapture* x, const XfbCapture* y) { return x->offset < y->offset; });
        for (size_t i = 1; i < list.size(); i++)
            if (list[i]->offset < list[i - 1]->offset + list[i - 1]->size)
                return "transform feedback outputs '" + list[i - 1]->name + "' and '" + list[i]->name + "' overlap in buffer " +
                       std::to_string(b);
        uint32_t stride = roundUp(end, dbl ? 8 : 4);
        if (explicitStride[b] >= 0) {
            stride = (uint32_t)explicitStride[b];
            if (stride % (dbl ? 8 : 4))
                return "'xfb_stride' : stride of buffer " + std::to_string(b) + " must be a multiple of " + (dbl ? "8" : "4");
            if (end > stride)
                return "'xfb_stride' : captured outputs overflow the stride of buffer " + std::to_string(b);
        }
        if (stride > 4 * kMaxXfbInterleavedComponents)
            return "transform feedback buffer " + std::to_string(b) + " exceeds gl_MaxTransformFeedbackInterleavedComponents";
    }
    return "";
}

std::string varyingKey(const Variable* v, Stage stage) {
    switch (v->builtin) {
        case BuiltinVar::FrontColor:
        case BuiltinVar::FragColorIn: return "c0";
        case BuiltinVar::FrontSecondaryColor:
        case BuiltinVar::SecondaryColorIn: return "c1";
        case BuiltinVar::BackColor: return "bc0";
        case BuiltinVar::BackSecondaryColor: return "bc1";
        case BuiltinVar::TexCoord: return "tc";
        case BuiltinVar::FogFragCoord: return "fog";
        default: break;
    }
    if (v->block && !v->block->blockQ.hasLayout) return "b_" + v->block->name;
    if (v->block && v->block->blockQ.layout.location >= 0) return "l" + std::to_string(v->block->blockQ.layout.location);
    if (v->block) return "b_" + v->block->name;
    if (v->q.layout.location >= 0) {
        std::string k = "l" + std::to_string(v->q.layout.location);
        if (v->q.layout.component > 0) k += "c" + std::to_string(v->q.layout.component);
        return k;
    }
    return "n_" + v->name;
}

// ---------------------------------------------------------------------------
// AST walking
// ---------------------------------------------------------------------------
namespace {

struct Walker {
    std::function<void(const Expr*)> onExpr;
    std::function<void(const Stmt*)> onStmt;
    void expr(const Expr* e) {
        if (!e) return;
        onExpr(e);
        for (auto* a : e->args) expr(a);
        expr(e->aux);
    }
    void stmt(const Stmt* s) {
        if (!s) return;
        if (onStmt) onStmt(s);
        for (auto* b : s->body) stmt(b);
        expr(s->expr);
        stmt(s->init);
        expr(s->cond);
        stmt(s->condDecl);
        expr(s->incr);
        stmt(s->then);
        stmt(s->els);
        for (auto* v : s->vars) expr(v->init);
    }
};

const Variable* findBuiltinVar(const Shader& sh, const char* name) {
    for (auto& v : sh.vars)
        if (v.name == name && v.global) return &v;
    return nullptr;
}

class Linker {
public:
    explicit Linker(const LinkInput& in) : in_(in) {}
    LinkResult run();

private:
    const LinkInput& in_;
    std::shared_ptr<LinkedProgram> prog_;
    LinkResult res_;
    bool ok_ = true;
    std::map<std::string, int> explicitLoc_;  // uniform leaf path -> location derived from layout(location)

    void error(const std::string& m) {
        ok_ = false;
        res_.infoLog += "error: " + m + "\n";
    }
    void collectStage(Stage s);
    void assignAttributes();
    void assignFragOutputs();
    void assignUniforms();
    void addUniformEntries(const std::string& name, const Type& t, TL L, uint32_t off, int blockIndex, uint32_t stageMask,
                           bool topLevelSsbo, int topArraySize, uint32_t topArrayStride, int& location,
                           std::set<int>& usedLocs, const Qualifiers* q);
    void checkVaryings();
    void checkBlockInterfaces();
    void writeInit(const Type& t, const TL& L, const ConstVal& cv, size_t& ci, uint8_t* dst);
    void stageMetadata();
    bool tessLevelCode(const Expr* e, std::vector<TessLevelOp>& out);
    bool arrangementFree(Stage st, bool barriers) const;
    void tessFusedAnalysis();
    void tessLevelAnalysis();
    bool tesAffineAnalysis();
};

void Linker::collectStage(Stage s) {
    StageCode& sc = prog_->stages[(int)s];
    // definitions
    for (auto& sh : sc.shaders)
        for (auto* f : sh->functions)
            if (f->body) {
                if (sc.defs.count(f->mangled)) {
                    error("function '" + f->name + "' is defined multiple times");
                    continue;
                }
                sc.defs[f->mangled] = f;
                if (f->name == "main") {
                    if (sc.main) error("multiple definitions of main()");
                    sc.main = f;
                    sc.primary = sh.get();
                }
            }
    if (!sc.main) {
        error(std::string("missing main() in ") +
              (s == Stage::Vertex ? "vertex" : s == Stage::Fragment ? "fragment" : s == Stage::Compute ? "compute" : s == Stage::Geometry ? "geometry"
               : s == Stage::Task ? "task" : s == Stage::Mesh ? "mesh" : "tessellation") +
              " shader");
        return;
    }
    // stage layouts merged across objects
    for (auto& sh : sc.shaders) {
        auto merge = [&](Layout& d, const Layout& l) {
            // GLSL 4.60 4.4: all declarations of a layout across a stage's shaders must agree
            if (l.maxVertices >= 0 && d.maxVertices >= 0 && l.maxVertices != d.maxVertices) error("conflicting max_vertices declarations");
            if (l.maxPrimitives >= 0 && d.maxPrimitives >= 0 && l.maxPrimitives != d.maxPrimitives) error("conflicting max_primitives declarations");
            if (l.vertices >= 0 && d.vertices >= 0 && l.vertices != d.vertices) error("conflicting output patch vertex counts");
            if (l.invocations >= 0 && d.invocations >= 0 && l.invocations != d.invocations) error("conflicting invocations declarations");
            if (l.primitive != Layout::kNoPrimitive && d.primitive != Layout::kNoPrimitive && l.primitive != d.primitive)
                error("conflicting primitive type declarations");
            for (int i = 0; i < 3; i++) if (l.localSize[i] >= 0) d.localSize[i] = l.localSize[i];
            if (l.maxVertices >= 0) d.maxVertices = l.maxVertices;
            if (l.maxPrimitives >= 0) d.maxPrimitives = l.maxPrimitives;
            if (l.invocations >= 0) d.invocations = l.invocations;
            if (l.vertices >= 0) d.vertices = l.vertices;
            if (l.primitive != Layout::kNoPrimitive) d.primitive = l.primitive;
            if (l.spacing) d.spacing = l.spacing;
            if (l.order) d.order = l.order;
            if (l.pointMode) d.pointMode = true;
            if (l.earlyFragmentTests) d.earlyFragmentTests = true;
            if (l.postDepthCoverage) d.postDepthCoverage = true;
            d.blendSupport |= l.blendSupport;
        };
        const Layout& l = sh->inLayout;
        bool has = l.localSize[0] >= 0 || l.localSize[1] >= 0 || l.localSize[2] >= 0;
        bool had = sc.inLayout.localSize[0] >= 0 || sc.inLayout.localSize[1] >= 0 || sc.inLayout.localSize[2] >= 0;
        if ((s == Stage::Compute || s == Stage::Task || s == Stage::Mesh) && has && had)
            for (int i = 0; i < 3; i++)
                if (std::max(1, l.localSize[i]) != std::max(1, sc.inLayout.localSize[i])) {
                    error(std::string(s == Stage::Compute ? "compute" : s == Stage::Task ? "task" : "mesh") + " shaders declare conflicting local sizes");
                    break;
                }
        merge(sc.inLayout, sh->inLayout);
        merge(sc.outLayout, sh->outLayout);
    }
    // reachability (and the static call graph, which must not be recursive)
    std::vector<const Function*> work{sc.main};
    sc.reachable.insert(sc.main);
    const Function* curFn = nullptr;
    std::map<const Function*, std::set<const Function*>> calls;
    Walker w;
    w.onStmt = [&](const Stmt* st) {
        if (st->kind == SK::Discard) sc.usesDiscard = true;
    };
    w.onExpr = [&](const Expr* e) {
        if (e->kind == EK::Var && e->var->global) sc.used.insert(e->var);
        if (e->kind == EK::SubroutineCall) {
            sc.used.insert(e->var);
            for (auto& [mangled, f] : sc.defs) {
                if (std::find(f->subroutineTypes.begin(), f->subroutineTypes.end(), e->fn->name) == f->subroutineTypes.end()) continue;
                if (curFn) calls[curFn].insert(f);
                if (!sc.reachable.count(f)) {
                    sc.reachable.insert(f);
                    work.push_back(f);
                }
            }
        }
        if (e->kind == EK::Call) {
            auto it = sc.defs.find(e->fn->mangled);
            if (it == sc.defs.end()) {
                error("function '" + e->fn->name + "' is used but not defined");
                return;
            }
            if (curFn) calls[curFn].insert(it->second);
            if (!sc.reachable.count(it->second)) {
                sc.reachable.insert(it->second);
                work.push_back(it->second);
            }
        }
        if (e->kind == EK::Builtin && e->fn->name == "ftransform") {
            for (auto& sh : sc.shaders) {
                if (auto* v = findBuiltinVar(*sh, "gl_ModelViewProjectionMatrix")) sc.used.insert(v);
                if (auto* v = findBuiltinVar(*sh, "gl_Vertex")) sc.used.insert(v);
            }
        }
    };
    // global initializers run unconditionally
    std::set<std::string> seen;
    for (auto& sh : sc.shaders)
        for (auto& g : sh->globals)
            if (g.kind == GlobalDecl::Var && !seen.count(g.var->name)) {
                seen.insert(g.var->name);
                sc.globals.push_back(g.var);
                if (g.var->init) w.expr(g.var->init);
                // Uniform/storage blocks with a shared/std140/std430 layout are active even when not referenced.
                if (g.var->block && (g.var->q.storage == Storage::Uniform || g.var->q.storage == Storage::Buffer) &&
                    g.var->block->blockQ.layout.packing != Packing::Packed)
                    sc.activeBlocks.insert(g.var);
            }
    while (!work.empty()) {
        const Function* f = work.back();
        work.pop_back();
        curFn = f;
        w.stmt(f->body);
    }
    curFn = nullptr;
    {
        // GLSL forbids recursion, including through subroutine uniforms
        std::map<const Function*, int> state;  // 1 = on stack, 2 = done
        std::function<bool(const Function*)> dfs = [&](const Function* f) {
            state[f] = 1;
            for (auto* g : calls[f]) {
                int st = state[g];
                if (st == 1) { error("recursive call to function '" + g->name + "'"); return true; }
                if (st == 0 && dfs(g)) return true;
            }
            state[f] = 2;
            return false;
        };
        dfs(sc.main);
    }
    // subroutine functions must be defined, and subroutine uniforms need a compatible subroutine
    for (auto& sh : sc.shaders) {
        for (auto* f : sh->functions)
            if (!f->subroutineTypes.empty() && !sc.defs.count(f->mangled))
                error("subroutine function '" + f->name + "' is declared but not defined");
        for (auto& g : sh->globals) {
            if (g.kind != GlobalDecl::Var || g.var->type.base != BT::Subroutine || g.var->q.storage != Storage::Uniform) continue;
            bool any = false;
            for (auto& [mangled, f] : sc.defs)
                if (std::find(f->subroutineTypes.begin(), f->subroutineTypes.end(), g.var->type.subr->name) != f->subroutineTypes.end())
                    any = true;
            if (!any) error("subroutine uniform '" + g.var->name + "' has no compatible subroutine function");
        }
    }
    for (auto* v : sc.used) {
        sc.usedNames.insert(v->name);
        if (v->builtin == BuiltinVar::FragDepth && v->written) sc.writesFragDepth = true;
        if (v->builtin == BuiltinVar::ClipVertex)
            for (auto& sh : sc.shaders)
                if (auto* cp = findBuiltinVar(*sh, "gl_ClipPlane")) sc.usedNames.insert(cp->name);
    }
    // gl_ClipPlane referenced through gl_ClipVertex
    if (sc.usedNames.count("gl_ClipPlane"))
        for (auto& sh : sc.shaders)
            if (auto* cp = findBuiltinVar(*sh, "gl_ClipPlane")) sc.used.insert(cp);
}

static bool isCompatAttrib(BuiltinVar b) {
    return b == BuiltinVar::Vertex || b == BuiltinVar::Normal || b == BuiltinVar::Color || b == BuiltinVar::SecondaryColor ||
           b == BuiltinVar::MultiTexCoord || b == BuiltinVar::FogCoord;
}

// Vertex shader inputs: one location per scalar/vector (dvec3/dvec4 included, GLSL 4.60 4.4.1)
// and per matrix column.
static int locationsFor(const Type& t) {
    Type b = t.stripArrays();
    int per = b.cols ? b.cols : 1;
    return per * t.totalArrayElements();
}

void Linker::assignAttributes() {
    StageCode& sc = prog_->stages[(int)Stage::Vertex];
    if (!sc.main) return;
    std::vector<const Variable*> attrs;
    for (auto* v : sc.used)
        if (v->q.storage == Storage::In && v->global && (v->builtin == BuiltinVar::None || isCompatAttrib(v->builtin)))
            attrs.push_back(v);
    std::sort(attrs.begin(), attrs.end(), [](const Variable* a, const Variable* b) {
        // compat attributes after user attributes; otherwise declaration order
        bool ca = a->builtin != BuiltinVar::None, cb = b->builtin != BuiltinVar::None;
        if (ca != cb) return !ca;
        return a->id < b->id;
    });
    // taken: 0 free, 1 user attribute, 2 compatibility attribute (may alias user attributes, as in NVIDIA's GL)
    std::vector<int> taken(32, 0);
    std::vector<std::pair<const Variable*, int>> assigned;
    auto mark = [&](int loc, int n, const std::string& name, int kind) {
        for (int i = loc; i < loc + n; i++) {
            if (i >= 32) { error("attribute '" + name + "' location out of range"); return; }
            // desktop GLSL allows attribute aliasing (explicit locations / bindings, incl. component packing)
            if (taken[i] == 2 && kind == 2)
                error("attribute '" + name + "' location " + std::to_string(i) + " overlaps another attribute");
            if (!taken[i] || kind == 1) taken[i] = kind;
        }
    };
    auto compatLocation = [](const Variable* v) -> int {
        switch (v->builtin) {
            case BuiltinVar::Vertex: return 0;
            case BuiltinVar::Normal: return 2;
            case BuiltinVar::Color: return 3;
            case BuiltinVar::SecondaryColor: return 4;
            case BuiltinVar::FogCoord: return 5;
            case BuiltinVar::MultiTexCoord: return 8 + (v->name.back() - '0');
            default: return -1;
        }
    };
    for (auto* v : attrs) {
        int cl = compatLocation(v);
        if (cl < 0) continue;
        mark(cl, 1, v->name, 2);
        assigned.push_back({v, cl});
    }
    std::vector<const Variable*> pending;
    for (auto* v : attrs) {
        if (compatLocation(v) >= 0) continue;
        int n = locationsFor(v->type);
        int loc = -1;
        if (v->q.layout.location >= 0) loc = v->q.layout.location;
        else if (in_.attribBindings.count(v->name)) loc = in_.attribBindings.at(v->name);
        if (loc >= 0) {
            mark(loc, n, v->name, 1);
            assigned.push_back({v, loc});
        } else {
            pending.push_back(v);
        }
    }
    for (auto* v : pending) {
        int n = locationsFor(v->type);
        int loc = -1;
        for (int i = 0; i + n <= 32 && loc < 0; i++) {
            bool free = true;
            for (int k = i; k < i + n; k++) if (taken[k]) free = false;
            if (free) loc = i;
        }
        if (loc < 0) { error("too many vertex attributes"); return; }
        mark(loc, n, v->name, 1);
        assigned.push_back({v, loc});
    }
    for (auto& [v, loc] : assigned) {
        Attribute a;
        a.name = v->name;
        a.type = glTypeEnum(v->type);
        a.arraySize = v->type.isArray() ? v->type.dims[0] : 1;
        a.location = loc;
        a.locationsUsed = locationsFor(v->type);
        prog_->refl.attributes.push_back(a);
        prog_->attribLocations[v->name] = loc;
    }
}

void Linker::assignFragOutputs() {
    StageCode& sc = prog_->stages[(int)Stage::Fragment];
    if (!sc.main) return;
    std::vector<const Variable*> outs;
    for (auto* v : sc.globals)
        if (v->q.storage == Storage::Out && v->builtin == BuiltinVar::None) outs.push_back(v);
    // Outputs the shader writes claim their locations first: only active outputs may not share a
    // location (GL 4.6 §15.2.3). Engines such as Unigine declare several outputs, write one, and bind
    // them all to location 0 with BindFragDataLocation; an unwritten one that collides is inactive and dropped.
    std::vector<const Variable*> order = outs;
    std::stable_partition(order.begin(), order.end(), [&](const Variable* v) { return sc.used.count(v) > 0; });
    std::set<std::pair<int, int>> taken;
    std::set<const Variable*> unassigned;
    for (auto* v : order) {
        int loc = -1, idx = 0;
        bool bound = false;
        if (v->q.layout.location >= 0) { loc = v->q.layout.location; idx = std::max(0, v->q.layout.index); }
        else if (in_.fragDataBindings.count(v->name)) {
            loc = in_.fragDataBindings.at(v->name);
            if (in_.fragDataIndices.count(v->name)) idx = in_.fragDataIndices.at(v->name);
            bound = true;
        }
        if (loc < 0) { unassigned.insert(v); continue; }
        int n = v->type.isArray() ? v->type.dims[0] : 1;
        bool clash = false;
        for (int i = 0; i < n; i++) clash |= taken.count({loc + i, idx}) > 0;
        if (clash && bound && !sc.used.count(v)) continue;
        if (clash) error("fragment output '" + v->name + "' location overlaps another output");
        for (int i = 0; i < n; i++) taken.insert({loc + i, idx});
        prog_->fragOutputs[v->name] = {loc, idx};
    }
    std::vector<const Variable*> pending;  // declaration order
    for (auto* v : outs)
        if (unassigned.count(v)) pending.push_back(v);
    for (auto* v : pending) {
        int n = v->type.isArray() ? v->type.dims[0] : 1;
        int loc = 0;
        for (;; loc++) {
            bool free = true;
            for (int i = 0; i < n; i++) if (taken.count({loc + i, 0})) free = false;
            if (free) break;
        }
        for (int i = 0; i < n; i++) taken.insert({loc + i, 0});
        prog_->fragOutputs[v->name] = {loc, 0};
    }
    for (auto* v : outs) {
        auto it = prog_->fragOutputs.find(v->name);
        if (it == prog_->fragOutputs.end()) continue;  // dropped above
        auto [loc, idx] = it->second;
        FragOutput o;
        o.name = v->name;
        o.type = glTypeEnum(v->type);
        o.arraySize = v->type.isArray() ? v->type.dims[0] : 1;
        o.location = loc;
        o.index = idx;
        prog_->refl.fragOutputs.push_back(o);
    }
}

static const int kMaxUniformLocations = 16384;  // GL_MAX_UNIFORM_LOCATIONS

// Leaves of a uniform in GL location order: (path, number of locations).
static void uniformLeaves(const Type& t, const std::string& path, std::vector<std::pair<std::string, int>>& out) {
    if (t.base == BT::Struct && t.dims.empty()) {
        for (auto& m : t.sdef->members) uniformLeaves(m.type, path + "." + m.name, out);
        return;
    }
    Type et = t.elem();
    if (!t.dims.empty() && (et.base == BT::Struct || !et.dims.empty())) {
        for (int i = 0; i < t.dims[0]; i++) uniformLeaves(et, path + "[" + std::to_string(i) + "]", out);
        return;
    }
    out.push_back({path, t.dims.empty() ? 1 : std::max(1, t.dims[0])});
}

void Linker::addUniformEntries(const std::string& name, const Type& t, TL L, uint32_t off, int blockIndex,
                               uint32_t stageMask, bool ssbo, int topArraySize, uint32_t topArrayStride, int& location,
                               std::set<int>& usedLocs, const Qualifiers* q) {
    auto& refl = prog_->refl;
    if (t.base == BT::Struct && t.dims.empty()) {
        for (size_t i = 0; i < t.sdef->members.size(); i++)
            addUniformEntries(name + "." + t.sdef->members[i].name, t.sdef->members[i].type, L->members[i], off + L->offsets[i],
                              blockIndex, stageMask, ssbo, topArraySize, topArrayStride, location, usedLocs, nullptr);
        return;
    }
    if (t.stripArrays().isOpaque()) return;  // reported separately with their slots
    Type et = t.elem();
    bool elemIsAggregate = !t.dims.empty() && (et.base == BT::Struct || !et.dims.empty());
    if (!t.dims.empty() && elemIsAggregate) {
        int n = t.dims[0];
        if (ssbo && n == 0) n = 1;
        if (ssbo && topArraySize >= 0 && et.base == BT::Struct && topArrayStride == L->arrayStride) n = 1;  // top-level struct arrays: element 0 only
        for (int i = 0; i < n; i++)
            addUniformEntries(name + "[" + std::to_string(i) + "]", et, L->elem, off + i * L->arrayStride, blockIndex, stageMask,
                              ssbo, topArraySize, topArrayStride, location, usedLocs, nullptr);
        return;
    }
    if (ssbo) {
        BufferVariable bv;
        bv.name = t.dims.empty() ? name : name + "[0]";
        bv.type = glTypeEnum(t);
        bv.arraySize = t.dims.empty() ? 1 : t.dims[0];
        bv.blockIndex = blockIndex;
        bv.offset = (int)off;
        bv.arrayStride = t.dims.empty() ? 0 : (int)L->arrayStride;
        Type st = t.stripArrays();
        bv.matrixStride = st.cols ? (int)(t.dims.empty() ? L->matrixStride : L->elem->matrixStride) : 0;
        bv.rowMajor = st.cols && (t.dims.empty() ? L->rowMajor : L->elem->rowMajor);
        bv.topLevelArraySize = topArraySize < 0 ? 1 : topArraySize;
        bv.topLevelArrayStride = (int)topArrayStride;
        refl.storageBlocks[blockIndex].activeUniforms.push_back((int)refl.bufferVariables.size());
        refl.bufferVariables.push_back(bv);
        return;
    }
    Uniform u;
    u.name = t.dims.empty() ? name : name + "[0]";
    u.type = glTypeEnum(t);
    u.arraySize = t.dims.empty() ? 1 : t.dims[0];
    u.blockIndex = blockIndex;
    u.offset = (int)off;
    u.arrayStride = t.dims.empty() ? 0 : (int)L->arrayStride;
    Type st = t.stripArrays();
    if (st.cols) {
        u.matrixStride = (int)(t.dims.empty() ? L->matrixStride : L->elem->matrixStride);
        u.rowMajor = t.dims.empty() ? L->rowMajor : L->elem->rowMajor;
    }
    u.stageMask = stageMask;
    if (blockIndex < 0) {
        if (name.compare(0, 3, "gl_") == 0) {
            u.location = -1;
        } else {
            int n = u.arraySize;
            auto ex = explicitLoc_.find(name);
            (void)q;
            if (ex != explicitLoc_.end()) {
                u.location = ex->second;
            } else {
                int loc = location;
                for (;;) {
                    bool free = true;
                    for (int i = 0; i < n; i++) if (usedLocs.count(loc + i)) free = false;
                    if (free) break;
                    loc++;
                }
                u.location = loc;
                location = loc + n;
            }
            for (int i = 0; i < n; i++) usedLocs.insert(u.location + i);
        }
    } else {
        refl.uniformBlocks[blockIndex].activeUniforms.push_back((int)refl.uniforms.size());
    }
    refl.uniforms.push_back(u);
}

static void writeScalarBytes(BT b, const Scalar& sc, uint8_t* dst) {
    switch (b) {
        case BT::Float: { float f = (float)sc.f; std::memcpy(dst, &f, 4); break; }
        case BT::Double: std::memcpy(dst, &sc.f, 8); break;
        case BT::Int: { int32_t i = (int32_t)sc.i; std::memcpy(dst, &i, 4); break; }
        case BT::UInt: { uint32_t u = (uint32_t)sc.u; std::memcpy(dst, &u, 4); break; }
        case BT::Bool: { uint32_t u = sc.b ? 1u : 0u; std::memcpy(dst, &u, 4); break; }
        case BT::Int64: case BT::UInt64: std::memcpy(dst, &sc.u, 8); break;
        default: break;
    }
}

void Linker::writeInit(const Type& t, const TL& L, const ConstVal& cv, size_t& ci, uint8_t* dst) {
    if (!t.dims.empty()) {
        for (int i = 0; i < t.dims[0]; i++) writeInit(t.elem(), L->elem, cv, ci, dst + i * L->arrayStride);
        return;
    }
    if (t.base == BT::Struct) {
        for (size_t m = 0; m < t.sdef->members.size(); m++) writeInit(t.sdef->members[m].type, L->members[m], cv, ci, dst + L->offsets[m]);
        return;
    }
    uint32_t N = scalarSize(t.base);
    if (t.cols) {
        for (int c = 0; c < t.cols; c++)
            for (int r = 0; r < t.vec; r++) {
                if (ci >= cv.c.size()) return;
                uint32_t o = L->rowMajor ? r * L->matrixStride + c * N : c * L->matrixStride + r * N;
                writeScalarBytes(t.base, cv.c[ci++], dst + o);
            }
        return;
    }
    for (int k = 0; k < t.vec && ci < cv.c.size(); k++) writeScalarBytes(t.base, cv.c[ci++], dst + k * N);
}

void Linker::assignUniforms() {
    auto& refl = prog_->refl;
    // Gather used uniform-ish variables across stages, merged by name.
    struct Entry { const Variable* v; uint32_t mask; };
    std::vector<Entry> defaults, opaques;
    std::map<std::string, Entry> blockVars;  // key -> first variable
    std::map<std::string, std::vector<uint32_t>> blockElemMask;  // key -> per-element REFERENCED_BY stage mask
    std::vector<std::string> blockOrder;
    std::map<std::string, size_t> defIdx, opIdx;
    std::map<std::string, int> declBinding;  // uniform name -> layout(binding) from any stage that declares one
    for (int s = 0; s < (int)Stage::Count; s++) {
        StageCode& sc = prog_->stages[s];
        if (!sc.main) continue;
        for (auto* v : sc.globals) {
            (void)v;
        }
        std::vector<const Variable*> used(sc.used.begin(), sc.used.end());
        std::sort(used.begin(), used.end(), [](const Variable* a, const Variable* b) { return a->id < b->id; });
        // declared-but-unreferenced shared/std140/std430 blocks are active without being referenced by this stage
        std::vector<const Variable*> act(sc.activeBlocks.begin(), sc.activeBlocks.end());
        std::sort(act.begin(), act.end(), [](const Variable* a, const Variable* b) { return a->id < b->id; });
        for (auto* v : act) {
            if (sc.used.count(v)) continue;
            std::string key = std::string(v->q.storage == Storage::Buffer ? "B:" : "U:") + v->block->name;
            if (!blockVars.count(key)) { blockVars[key] = {v, 0u}; blockOrder.push_back(key); }
            if (v->block->blockQ.layout.binding >= 0) declBinding[key] = v->block->blockQ.layout.binding;
        }
        for (auto* v : used) {
            if (v->block && (v->q.storage == Storage::Uniform || v->q.storage == Storage::Buffer)) {
                std::string key = std::string(v->q.storage == Storage::Buffer ? "B:" : "U:") + v->block->name;
                if (!blockVars.count(key)) { blockVars[key] = {v, 0u}; blockOrder.push_back(key); }
                else if (blockVars[key].v->anonBlock != v->anonBlock)
                    // GLSL 4.60 §4.3.9: matched blocks must all have an instance name or all lack one
                    error("block '" + v->block->name + "' has an instance name in one stage but not in another");
                else if (blockVars[key].v->block != v->block) {
                    // Matched blocks need the same members: names, types and order (GLSL 4.60 §4.3.9).
                    const auto& m0 = blockVars[key].v->block->members;
                    const auto& m1 = v->block->members;
                    bool same = m0.size() == m1.size();
                    for (size_t i = 0; same && i < m0.size(); i++)
                        same = m0[i].name == m1[i].name && m0[i].type.str() == m1[i].type.str() &&
                               m0[i].type.dims == m1[i].type.dims;
                    if (!same) error("block '" + v->block->name + "' is declared with different members in different stages");
                }
                if (v->block->blockQ.layout.binding >= 0) declBinding[key] = v->block->blockQ.layout.binding;
                blockVars[key].mask |= 1u << s;
                if (v->type.isArray()) {  // arrays of blocks: which elements this stage references
                    std::vector<bool> els = usedArrayElements(sc, v, std::max(1, v->type.dims[0]));
                    auto& em = blockElemMask[key];
                    em.resize(std::max(em.size(), els.size()), 0u);
                    for (size_t i = 0; i < els.size(); i++)
                        if (els[i]) em[i] |= 1u << s;
                }
                continue;
            }
            if (v->q.storage != Storage::Uniform) continue;
            if (v->builtin == BuiltinVar::DepthRange || v->builtin == BuiltinVar::NumSamples) continue;  // system values
            if (v->type.base == BT::Subroutine) continue;         // subroutine uniforms have their own interface
            bool opq = v->type.containsOpaque();
            bool aggregate = v->type.stripArrays().base == BT::Struct;
            if (v->q.layout.binding >= 0) declBinding[v->name] = v->q.layout.binding;
            auto addTo = [&](std::map<std::string, size_t>& idx, std::vector<Entry>& vec) {
                auto it = idx.find(v->name);
                if (it == idx.end()) {
                    idx[v->name] = vec.size();
                    vec.push_back({v, 1u << s});
                } else {
                    Entry& e = vec[it->second];
                    if (e.v->type != v->type && e.v->type.str() != v->type.str())
                        error("uniform '" + v->name + "' declared with different types in different stages");
                    if (v->type.stripArrays().base == BT::Image && e.v->q.layout.format != v->q.layout.format)
                        error("image uniform '" + v->name + "' declared with different format qualifiers in different stages");
                    if (v->q.layout.binding >= 0 && e.v->q.layout.binding >= 0 && v->q.layout.binding != e.v->q.layout.binding)
                        error("uniform '" + v->name + "' declared with different bindings in different stages");
                    if (v->q.layout.location >= 0) {
                        if (e.v->q.layout.location >= 0 && e.v->q.layout.location != v->q.layout.location)
                            error("uniform '" + v->name + "' declared with different locations in different stages");
                        else if (e.v->q.layout.location < 0)
                            e.v = v;  // the explicit location applies to the whole program
                    }
                    e.mask |= 1u << s;
                }
            };
            // structs mixing opaque and plain members: plain members live in the default block
            if (!opq || aggregate) addTo(defIdx, defaults);
            if (opq) addTo(opIdx, opaques);
        }
    }
    int slot = 0;
    std::set<int> usedLocs;
    // Explicit locations reserve first. An aggregate with layout(location = L) assigns consecutive locations to its
    // leaves (struct members / array elements in declaration order), each leaf array taking one location per element.
    {
        std::set<std::string> seenVar;
        std::vector<const Variable*> vars;
        for (auto& e : defaults) if (seenVar.insert(e.v->name).second) vars.push_back(e.v);
        for (auto& e : opaques) if (seenVar.insert(e.v->name).second) vars.push_back(e.v);
        // An explicit location given in any shader applies to the program-wide uniform, and inactive uniforms
        // with explicit locations still reserve them.
        std::map<std::string, int> declLoc;
        for (int st = 0; st < (int)Stage::Count; st++)
            for (auto& sh : prog_->stages[st].shaders)
                for (auto& g : sh->globals) {
                    const Variable* gv = g.var;
                    if (g.kind != GlobalDecl::Var || gv->q.storage != Storage::Uniform || gv->block ||
                        gv->type.base == BT::Subroutine || gv->q.layout.location < 0)
                        continue;
                    auto it = declLoc.find(gv->name);
                    if (it != declLoc.end() && it->second != gv->q.layout.location)
                        error("uniform '" + gv->name + "' declared with different locations in different shaders");
                    declLoc[gv->name] = gv->q.layout.location;
                    if (seenVar.insert(gv->name).second) vars.push_back(gv);
                }
        for (auto* v : vars) {
            auto dl = declLoc.find(v->name);
            if (dl == declLoc.end()) continue;
            std::vector<std::pair<std::string, int>> leaves;
            uniformLeaves(v->type, v->name, leaves);
            int loc = dl->second;
            for (auto& [path, n] : leaves) {
                explicitLoc_[path] = loc;
                for (int i = 0; i < n; i++) {
                    if (loc + i >= kMaxUniformLocations) {
                        error("uniform '" + v->name + "' location exceeds GL_MAX_UNIFORM_LOCATIONS");
                        return;
                    }
                    if (!usedLocs.insert(loc + i).second) {
                        error("location " + std::to_string(loc + i) + " is assigned to more than one uniform");
                        return;
                    }
                }
                loc += n;
            }
        }
    }
    int location = 0;

    // --- default block ---------------------------------------------------
    uint32_t off = 0;
    if (!defaults.empty()) {
        prog_->defaultSlot = slot;
        refl.slots.push_back({SlotKind::DefaultUniforms, 0});
        slot++;
    }
    for (auto& e : defaults) {
        TL L = computeLayout(e.v->type, Packing::Std430, e.v->q.layout.matrix == 2);
        off = roundUp(off, L->align);
        prog_->defaultOffsets[e.v->name] = off;
        addUniformEntries(e.v->name, e.v->type, L, off, -1, e.mask, false, -1, 0, location, usedLocs, &e.v->q);
        off += L->size;
    }
    refl.defaultUniformSize = roundUp(off, 16);
    refl.defaultUniformInit.assign(refl.defaultUniformSize, 0);
    for (auto& e : defaults) {
        if (!e.v->constVal) continue;
        TL L = computeLayout(e.v->type, Packing::Std430, e.v->q.layout.matrix == 2);
        size_t ci = 0;
        writeInit(e.v->type, L, *e.v->constVal, ci, refl.defaultUniformInit.data() + prog_->defaultOffsets[e.v->name]);
    }

    // --- blocks ---------------------------------------------------------------
    std::vector<std::string> ubos, ssbos;
    for (auto& k : blockOrder) (k[0] == 'U' ? ubos : ssbos).push_back(k);
    auto doBlocks = [&](std::vector<std::string>& keys, bool storage) {
        for (auto& key : keys) {
            const Variable* v = blockVars[key].v;
            uint32_t mask = blockVars[key].mask;
            const StructDef& sd = *v->block;
            Packing pk = sd.blockQ.layout.packing == Packing::Std430 ? Packing::Std430 : Packing::Std140;
            bool rm = sd.blockQ.layout.matrix == 2;
            BlockBinding bb;
            bb.storage = storage;
            bb.packing = pk;
            bb.rowMajor = rm;
            bb.layout = computeBlockLayout(sd, pk, rm);
            int n = v->type.isArray() ? v->type.dims[0] : 1;
            bb.count = n;
            bb.isArray = v->type.isArray();
            auto& list = storage ? refl.storageBlocks : refl.uniformBlocks;
            bb.firstBlock = (int)list.size();
            bb.slot = slot;
            slot += n;
            for (int i = 0; i < n; i++) {
                Block b;
                b.name = v->type.isArray() ? sd.name + "[" + std::to_string(i) + "]" : sd.name;
                b.binding = (declBinding.count(key) ? declBinding[key] : 0) + i;
                b.dataSize = bb.layout->size;
                b.slot = bb.slot + i;
                b.stageMask = mask;
                if (auto em = blockElemMask.find(key); em != blockElemMask.end() && (size_t)i < em->second.size()) b.stageMask = em->second[i];
                list.push_back(b);
                refl.slots.push_back({storage ? SlotKind::Ssbo : SlotKind::Ubo, (uint32_t)(bb.firstBlock + i)});
            }
            // members
            std::string prefix = sd.instanceName.empty() ? "" : sd.name + ".";
            for (size_t m = 0; m < sd.members.size(); m++) {
                const Member& mem = sd.members[m];
                // TOP_LEVEL_ARRAY_SIZE/STRIDE describe arrays of aggregates; a member that is a plain
                // array is itself the active variable (size 1, stride 0), as in Mesa.
                bool topAggregate = mem.type.isArray() && (mem.type.elem().isArray() || mem.type.elem().base == BT::Struct);
                int top = topAggregate ? mem.type.dims[0] : -1;
                uint32_t topStride = topAggregate ? bb.layout->members[m]->arrayStride : 0;
                if (storage) {
                    addUniformEntries(prefix + mem.name, mem.type, bb.layout->members[m], bb.layout->offsets[m], bb.firstBlock, mask,
                                      true, top < 0 ? 1 : top, topStride, location, usedLocs, nullptr);
                } else {
                    addUniformEntries(prefix + mem.name, mem.type, bb.layout->members[m], bb.layout->offsets[m], bb.firstBlock, mask,
                                      false, -1, 0, location, usedLocs, nullptr);
                }
            }
            // remaining blocks of an array share the member list
            for (int i = 1; i < n; i++) list[bb.firstBlock + i].activeUniforms = list[bb.firstBlock].activeUniforms;
            if (storage) {
                // GL_BUFFER_DATA_SIZE for runtime arrays counts one element
                if (!sd.members.empty() && sd.members.back().type.isArray() && sd.members.back().type.dims[0] == 0) {
                    uint32_t extra = bb.layout->members.back()->arrayStride;
                    for (int i = 0; i < n; i++) list[bb.firstBlock + i].dataSize = roundUp(bb.layout->offsets.back() + extra, bb.layout->align);
                }
            }
            prog_->blocks[key] = bb;
        }
    };
    doBlocks(ubos, false);
    doBlocks(ssbos, true);
    // SSBO size slots
    for (auto& key : ssbos) {
        BlockBinding& bb = prog_->blocks[key];
        bb.sizeSlot = slot;
        for (int i = 0; i < bb.count; i++) {
            refl.storageBlocks[bb.firstBlock + i].sizeSlot = slot;
            refl.slots.push_back({SlotKind::BufferSize, (uint32_t)(bb.firstBlock + i)});
            slot++;
        }
    }

    // --- opaque uniforms --------------------------------------------------------
    std::map<int, int> acbByBinding;
    std::map<int, uint32_t> acNextOffset;
    std::map<int, std::vector<std::pair<uint32_t, uint32_t>>> acRanges;  // binding -> [offset, end) of each counter
    // expand opaque leaves (struct members / arrays of structs) into individual uniforms named by their path
    struct OpaqueLeaf { const Variable* v; std::string path; Type type; uint32_t mask; bool explicitLoc; };
    std::vector<OpaqueLeaf> leaves;
    std::function<void(const Entry&, const std::string&, const Type&)> expand = [&](const Entry& e, const std::string& path, const Type& t) {
        Type bt = t.stripArrays();
        if (bt.isOpaque()) {
            leaves.push_back({e.v, path, t, e.mask, path == e.v->name});
            return;
        }
        if (bt.base != BT::Struct) return;
        if (t.isArray()) {
            for (int i = 0; i < t.dims[0]; i++) expand(e, path + "[" + std::to_string(i) + "]", t.elem());
            return;
        }
        for (auto& m : t.sdef->members) expand(e, path + "." + m.name, m.type);
    };
    for (auto& e : opaques) expand(e, e.v->name, e.v->type);
    {
        // GL_MAX_<stage>_ATOMIC_COUNTERS
        static const int kMax[(int)Stage::Count] = {
            limitValue("gl_MaxVertexAtomicCounters"), limitValue("gl_MaxTessControlAtomicCounters"),
            limitValue("gl_MaxTessEvaluationAtomicCounters"), limitValue("gl_MaxGeometryAtomicCounters"),
            limitValue("gl_MaxFragmentAtomicCounters"), limitValue("gl_MaxComputeAtomicCounters"),
            limitValue("gl_MaxComputeAtomicCounters"), limitValue("gl_MaxComputeAtomicCounters")};  // task, mesh
        int counters[(int)Stage::Count] = {};
        for (auto& e : opaques)
            if (e.v->type.stripArrays().base == BT::AtomicUint)
                for (int s = 0; s < (int)Stage::Count; s++)
                    if (e.mask & (1u << s)) counters[s] += std::max(1, e.v->type.totalArrayElements());
        for (int s = 0; s < (int)Stage::Count; s++)
            if (counters[s] > kMax[s]) error("too many atomic counters in one shader stage");
    }
    for (auto& lf : leaves) {
        const Variable* v = lf.v;
        Type bt = lf.type.stripArrays();
        int n = std::max(1, lf.type.totalArrayElements());
        // Arrays of arrays of opaque types occupy one flattened run of slots (row-major); GL enumerates
        // one active uniform per innermost array: x[i][j][0] (GL 4.6 §7.3.1.1).
        int inner = lf.type.isArray() ? std::max(1, lf.type.dims.back()) : 1;
        int groups = n / inner;
        OpaqueBinding ob;
        ob.type = bt;
        ob.count = n;
        ob.isArray = lf.type.isArray();
        ob.memory = v->q.memory;
        int firstUidx = (int)refl.uniforms.size();
        ob.uniformIndex = firstUidx;
        std::vector<Uniform> us;
        for (int gi = 0; gi < groups; gi++) {
            Uniform u;
            std::string sub;
            for (size_t d = 0, rem = gi; d + 1 < lf.type.dims.size(); d++) {
                int stride = 1;
                for (size_t k = d + 1; k + 1 < lf.type.dims.size(); k++) stride *= std::max(1, lf.type.dims[k]);
                sub += "[" + std::to_string(rem / stride) + "]";
                rem %= stride;
            }
            u.name = lf.type.isArray() ? lf.path + sub + "[0]" : lf.path;
            u.type = glTypeEnum(bt);
            u.arraySize = lf.type.isArray() ? inner : 1;
            u.stageMask = lf.mask;
            u.binding = lf.explicitLoc && declBinding.count(v->name) ? declBinding[v->name] + gi * inner : 0;
            // Atomic counters have no uniform location (GL 4.6 §7.6.1).
            if (bt.base == BT::AtomicUint) u.location = -1;
            else if (explicitLoc_.count(lf.path)) u.location = explicitLoc_[lf.path] + gi * inner;
            else {
                int loc = location;
                for (;;) {
                    bool free = true;
                    for (int i = 0; i < inner; i++) if (usedLocs.count(loc + i)) free = false;
                    if (free) break;
                    loc++;
                }
                u.location = loc;
                location = loc + inner;
            }
            if (u.location >= 0)
                for (int i = 0; i < inner; i++) usedLocs.insert(u.location + i);
            us.push_back(u);
        }
        auto pushSlots = [&](SlotKind kind) {
            for (int gi = 0; gi < groups; gi++)
                for (int i = 0; i < inner; i++) refl.slots.push_back({kind, (uint32_t)(firstUidx + gi), (uint32_t)i, textureTargetFor(bt)});
            slot += n;
        };
        if (bt.base == BT::Sampler) {
            ob.kind = SlotKind::Texture;
            ob.slot = slot;
            for (int gi = 0; gi < groups; gi++) us[gi].slot = slot + gi * inner;
            pushSlots(SlotKind::Texture);
            ob.samplerSlot = slot;
            pushSlots(SlotKind::Sampler);
            ob.biasSlot = slot;
            pushSlots(SlotKind::LodBias);
            // Metal border colors are limited to black/white: integer samplers that filter
            // (not buffer / multisample) get their border color emulated in the shader.
            if ((bt.sampled == BT::Int || bt.sampled == BT::UInt) && !bt.shadow && !bt.ms && bt.dim != Dim::Buffer && bt.dim != Dim::Cube) {
                ob.borderSlot = slot;
                pushSlots(SlotKind::SamplerBorder);
            }
        } else if (bt.base == BT::Image) {
            ob.kind = SlotKind::Image;
            ob.slot = slot;
            for (int gi = 0; gi < groups; gi++) us[gi].slot = slot + gi * inner;
            pushSlots(SlotKind::Image);
            // Multisample images are per-sample 2D-array shadows (slice = layer * samples + sample),
            // so the shader needs the sample count.
            if (bt.ms) {
                ob.samplesSlot = slot;
                pushSlots(SlotKind::ImageSamples);
            }
        } else {  // atomic_uint
            ob.kind = SlotKind::AtomicCounter;
            int binding = declBinding.count(v->name) ? declBinding[v->name] : 0;
            uint32_t aoff = v->q.layout.offset >= 0 ? (uint32_t)v->q.layout.offset : acNextOffset[binding];
            acNextOffset[binding] = aoff + 4 * n;
            // Distinct counters may not share storage (GLSL 4.60 §4.4.6.3).
            for (auto [lo, hi] : acRanges[binding])
                if (aoff < hi && lo < aoff + 4 * n) {
                    error("atomic counter '" + lf.path + "' overlaps another counter at binding " + std::to_string(binding));
                    break;
                }
            acRanges[binding].push_back({aoff, aoff + 4 * n});
            if (!acbByBinding.count(binding)) {
                acbByBinding[binding] = (int)refl.atomicCounterBuffers.size();
                AtomicCounterBuffer acb;
                acb.binding = binding;
                acb.dataSize = 0;
                acb.slot = -1;
                acb.stageMask = 0;
                refl.atomicCounterBuffers.push_back(acb);
            }
            int ai = acbByBinding[binding];
            auto& acb = refl.atomicCounterBuffers[ai];
            acb.dataSize = std::max(acb.dataSize, aoff + 4 * n);
            acb.stageMask |= lf.mask;
            ob.acbIndex = ai;
            ob.acOffset = aoff;
            for (int gi = 0; gi < groups; gi++) {
                acb.activeUniforms.push_back(firstUidx + gi);
                us[gi].binding = binding;
                us[gi].offset = (int)(aoff + 4 * gi * inner);
                us[gi].arrayStride = lf.type.isArray() ? 4 : 0;
                us[gi].atomicCounterBufferIndex = ai;
            }
        }
        for (auto& u : us) refl.uniforms.push_back(u);
        prog_->opaque[lf.path] = ob;
    }
    // Image uniform limits per stage and combined (GL 4.6 §7.6.1 / table 23.62).
    {
        enum : int {
#define X(name, glenum, value) k_##name = value,
#include "glsl/limits.def"
#undef X
        };
        static const int stageMax[(int)Stage::Count] = {k_gl_MaxVertexImageUniforms, k_gl_MaxTessControlImageUniforms,
                                                        k_gl_MaxTessEvaluationImageUniforms, k_gl_MaxGeometryImageUniforms,
                                                        k_gl_MaxFragmentImageUniforms, k_gl_MaxComputeImageUniforms,
                                                        k_gl_MaxComputeImageUniforms, k_gl_MaxComputeImageUniforms};
        static const char* stageName[(int)Stage::Count] = {"vertex", "tessellation control", "tessellation evaluation",
                                                           "geometry", "fragment", "compute", "task", "mesh"};
        int perStage[(int)Stage::Count] = {}, combined = 0;
        for (auto& lf : leaves) {
            if (lf.type.stripArrays().base != BT::Image) continue;
            int n = std::max(1, lf.type.totalArrayElements());
            for (int s = 0; s < (int)Stage::Count; s++)
                if (lf.mask & (1u << s)) perStage[s] += n, combined += n;
        }
        for (int s = 0; s < (int)Stage::Count; s++)
            if (perStage[s] > stageMax[s])
                error(std::string("too many image uniforms in the ") + stageName[s] + " shader (" + std::to_string(perStage[s]) + " > " +
                      std::to_string(stageMax[s]) + ")");
        if (combined > k_gl_MaxCombinedImageUniforms)
            error("too many image uniforms in the program (" + std::to_string(combined) + " > " + std::to_string(k_gl_MaxCombinedImageUniforms) + ")");
    }
    for (auto& u : refl.uniforms)
        if (u.location >= 0 && u.location + u.arraySize > kMaxUniformLocations) {
            error("the program uses more than GL_MAX_UNIFORM_LOCATIONS uniform locations");
            return;
        }
    // subroutines (per stage)
    for (int s = 0; s < (int)Stage::Count; s++) {
        StageCode& sc = prog_->stages[s];
        if (!sc.main) continue;
        StageSubroutines& ss = refl.subroutines[s];
        std::vector<const Function*> fns;
        std::set<std::string> seenFn;
        for (auto& sh : sc.shaders)
            for (auto& g : sh->globals)
                if (g.kind == GlobalDecl::FunctionDef && !g.fn->subroutineTypes.empty() && seenFn.insert(g.fn->name).second) fns.push_back(g.fn);
        if (fns.empty()) continue;
        std::vector<int> index(fns.size(), -1);
        std::set<int> usedIdx;
        for (size_t i = 0; i < fns.size(); i++)
            if (fns[i]->subroutineIndex >= 0) {
                index[i] = fns[i]->subroutineIndex;
                if (!usedIdx.insert(index[i]).second) error("duplicate subroutine index " + std::to_string(index[i]));
            }
        int next = 0;
        for (size_t i = 0; i < fns.size(); i++)
            if (index[i] < 0) {
                while (usedIdx.count(next)) next++;
                index[i] = next;
                usedIdx.insert(next);
            }
        int maxIdx = 0;
        for (int i : index) maxIdx = std::max(maxIdx, i);
        if (maxIdx >= 256) error("subroutine index exceeds GL_MAX_SUBROUTINES");
        ss.functions.assign(maxIdx + 1, "");
        for (size_t i = 0; i < fns.size(); i++) ss.functions[index[i]] = fns[i]->name;
        {
            // every declared subroutine uniform (active or not) must fit GL_MAX_SUBROUTINE_UNIFORM_LOCATIONS
            std::set<std::string> seenU;
            int explicitEnd = 0, implicitCount = 0;
            std::set<int> explicitLocs;
            for (auto& sh : sc.shaders)
                for (auto& g : sh->globals)
                    if (g.kind == GlobalDecl::Var && g.var->type.base == BT::Subroutine && g.var->q.storage == Storage::Uniform &&
                        seenU.insert(g.var->name).second) {
                        int n = std::max(1, g.var->type.totalArrayElements());
                        if (g.var->q.layout.location >= 0) {
                            explicitEnd = std::max(explicitEnd, g.var->q.layout.location + n);
                            for (int i = 0; i < n; i++) explicitLocs.insert(g.var->q.layout.location + i);
                        } else {
                            implicitCount += n;
                        }
                    }
            if (explicitEnd > 1024 || (int)explicitLocs.size() + implicitCount > 1024)
                error("subroutine uniforms exceed GL_MAX_SUBROUTINE_UNIFORM_LOCATIONS");
        }
        std::vector<const Variable*> unis;
        for (auto* v : sc.used)
            if (v->type.base == BT::Subroutine && v->q.storage == Storage::Uniform) unis.push_back(v);
        std::sort(unis.begin(), unis.end(), [](const Variable* a, const Variable* b) { return a->id < b->id; });
        std::set<int> takenLoc;
        for (auto* v : unis)
            if (v->q.layout.location >= 0)
                for (int i = 0; i < std::max(1, v->type.totalArrayElements()); i++)
                    if (!takenLoc.insert(v->q.layout.location + i).second)
                        error("subroutine uniform location " + std::to_string(v->q.layout.location + i) + " is used more than once");
        int nextLoc = 0;
        for (auto* v : unis) {
            SubroutineUniform su;
            su.name = v->name;
            su.arraySize = v->type.isArray() ? std::max(1, v->type.totalArrayElements()) : 1;
            su.dims = v->type.dims;
            if (v->q.layout.location >= 0) su.location = v->q.layout.location;
            else {
                for (;;) {
                    bool free = true;
                    for (int i = 0; i < su.arraySize; i++) if (takenLoc.count(nextLoc + i)) free = false;
                    if (free) break;
                    nextLoc++;
                }
                su.location = nextLoc;
                for (int i = 0; i < su.arraySize; i++) takenLoc.insert(nextLoc + i);
                nextLoc += su.arraySize;
            }
            for (size_t i = 0; i < fns.size(); i++)
                if (std::find(fns[i]->subroutineTypes.begin(), fns[i]->subroutineTypes.end(), v->type.subr->name) != fns[i]->subroutineTypes.end())
                    su.compatible.push_back(index[i]);
            ss.numLocations = std::max(ss.numLocations, su.location + su.arraySize);
            if (ss.numLocations > 1024) error("subroutine uniform locations exceed GL_MAX_SUBROUTINE_UNIFORM_LOCATIONS");
            ss.uniforms.push_back(su);
            prog_->subroutineLocation[{s, v->name}] = su.location;
        }
        for (size_t i = 0; i < fns.size(); i++) prog_->subroutineIndex[{s, fns[i]->name}] = index[i];
        if (!unis.empty()) {
            ss.slot = slot++;
            refl.slots.push_back({SlotKind::SubroutineIndices, (uint32_t)s});
        }
    }
    // GL 4.6 7.6.2: per-stage and combined atomic counter / atomic counter buffer limits
    {
        static const int kMaxAC = 1024, kMaxACB = 8;
        int combined = 0;
        for (int st = 0; st < (int)Stage::Count; st++) {
            int counters = 0, buffers = 0;
            for (auto& u : refl.uniforms)
                if (u.atomicCounterBufferIndex >= 0 && (u.stageMask & (1u << st))) counters += std::max(1, u.arraySize);
            for (auto& acb : refl.atomicCounterBuffers)
                if (acb.stageMask & (1u << st)) buffers++;
            if (counters > kMaxAC) error("too many atomic counters in one shader stage");
            if (buffers > kMaxACB) error("too many atomic counter buffers in one shader stage");
            combined += counters;
        }
        if (combined > kMaxAC) error("too many atomic counters (MAX_COMBINED_ATOMIC_COUNTERS)");
        if ((int)refl.atomicCounterBuffers.size() > kMaxACB) error("too many atomic counter buffers (MAX_COMBINED_ATOMIC_COUNTER_BUFFERS)");
    }
    for (auto& acb : refl.atomicCounterBuffers) {
        acb.slot = slot++;
        refl.slots.push_back({SlotKind::AtomicCounter, (uint32_t)(&acb - &refl.atomicCounterBuffers[0])});
    }
    for (auto& [name, ob] : prog_->opaque)
        if (ob.kind == SlotKind::AtomicCounter) ob.slot = refl.atomicCounterBuffers[ob.acbIndex].slot;
}

void Linker::checkVaryings() {
    // Interfaces between consecutive pre-rasterization stages (GL 4.6 7.4.1): matching
    // variables must have matching types (per-vertex arrays compared by element).
    {
        std::vector<Stage> chain;
        for (Stage s : {Stage::Vertex, Stage::TessControl, Stage::TessEval, Stage::Geometry})
            if (prog_->stages[(int)s].main) chain.push_back(s);
        auto arrayedOut = [](Stage s, const Variable* v) { return s == Stage::TessControl && !v->q.patch; };
        auto arrayedIn = [](Stage s, const Variable* v) { return s != Stage::Vertex && s != Stage::Fragment && !v->q.patch; };
        for (size_t i = 0; i + 1 < chain.size(); i++) {
            Stage ps = chain[i], cs = chain[i + 1];
            // GLSL 4.60 7.1.1: a built-in block redeclared on both sides of an interface must be redeclared the same way
            {
                std::vector<std::string> po, ci;
                for (auto& sh : prog_->stages[(int)ps].shaders) if (!sh->perVertexOut.empty()) po = sh->perVertexOut;
                for (auto& sh : prog_->stages[(int)cs].shaders) if (!sh->perVertexIn.empty()) ci = sh->perVertexIn;
                if (!po.empty() && !ci.empty() && po != ci) error("gl_PerVertex is redeclared differently in consecutive shader stages");
            }
            std::map<std::string, const Variable*> outs;
            for (auto* v : prog_->stages[(int)ps].globals)
                if (v->q.storage == Storage::Out && v->builtin == BuiltinVar::None) outs[varyingKey(v, ps)] = v;
            for (auto* v : prog_->stages[(int)cs].globals) {
                if (v->q.storage != Storage::In || v->builtin != BuiltinVar::None) continue;
                auto it = outs.find(varyingKey(v, cs));
                if (it == outs.end()) continue;
                Type a = it->second->type, b = v->type;
                if (arrayedOut(ps, it->second) && a.isArray()) a = a.elem();
                if (arrayedIn(cs, v) && b.isArray()) b = b.elem();
                if (it->second->q.patch != v->q.patch) {
                    error("per-patch qualifier mismatch for varying '" + v->name + "'");
                    continue;
                }
                if (a.str() != b.str() && !(a.base == BT::Struct && b.base == BT::Struct && a.sdef->name == b.sdef->name))
                    error("type mismatch for varying '" + v->name + "' ('" + a.str() + "' vs '" + b.str() + "')");
            }
        }
    }
    // Task -> mesh: the taskPayloadSharedEXT declarations must match.
    if (prog_->stages[(int)Stage::Task].main && prog_->stages[(int)Stage::Mesh].main) {
        const Variable *tp = nullptr, *mp = nullptr;
        for (auto* v : prog_->stages[(int)Stage::Task].globals) if (v->q.storage == Storage::TaskPayload) tp = v;
        for (auto* v : prog_->stages[(int)Stage::Mesh].globals) if (v->q.storage == Storage::TaskPayload) mp = v;
        if (tp && mp && tp->type.str() != mp->type.str() &&
            !(tp->type.base == BT::Struct && mp->type.base == BT::Struct && tp->type.sdef->name == mp->type.sdef->name && tp->type.dims == mp->type.dims))
            error("taskPayloadSharedEXT declarations of the task and mesh shaders do not match");
    }
    // Find the last pre-rasterization stage and verify fragment inputs.
    Stage pre = Stage::Count;
    for (Stage s : {Stage::Mesh, Stage::Geometry, Stage::TessEval, Stage::Vertex})
        if (prog_->stages[(int)s].main) { pre = s; break; }
    const StageCode& fs = prog_->stages[(int)Stage::Fragment];
    if (pre == Stage::Count) return;
    const StageCode& ps = prog_->stages[(int)pre];
    std::map<std::string, const Variable*> outs;
    for (auto* v : ps.globals)
        if (v->q.storage == Storage::Out && v->builtin == BuiltinVar::None) outs[varyingKey(v, pre)] = v;
    if (!fs.main) return;
    for (auto* v : fs.globals) {
        if (v->q.storage != Storage::In || v->builtin != BuiltinVar::None) continue;
        if (!fs.used.count(v)) continue;
        auto it = outs.find(varyingKey(v, Stage::Fragment));
        if (it == outs.end()) {
            if (!in_.separable) res_.infoLog += "warning: fragment input '" + v->name + "' is not written by the previous stage\n";
            continue;
        }
        Type a = it->second->type, b = v->type;
        if (pre == Stage::Mesh) {
            if (a.isArray()) a = a.elem();  // mesh outputs are per-vertex / per-primitive arrays
            if (it->second->q.perprimitive != v->q.perprimitive) {
                error("perprimitiveEXT qualifier mismatch for varying '" + v->name + "'");
                continue;
            }
        } else if (v->q.perprimitive) {
            error("fragment input '" + v->name + "' is perprimitiveEXT but the program has no mesh shader");
            continue;
        }
        if (a.str() != b.str() && !(a.base == BT::Struct && b.base == BT::Struct && a.sdef->name == b.sdef->name))
            error("type mismatch for varying '" + v->name + "' ('" + a.str() + "' vs '" + b.str() + "')");
    }
}

void Linker::stageMetadata() {
    auto& r = prog_->refl;
    r.stageMask = prog_->stageMask;
    if (auto& cs = prog_->stages[(int)Stage::Compute]; cs.main) {
        for (int i = 0; i < 3; i++) r.localSize[i] = std::max(1, cs.inLayout.localSize[i]);
        if (cs.inLayout.localSize[0] < 0 && cs.inLayout.localSize[1] < 0 && cs.inLayout.localSize[2] < 0)
            error("compute shader does not declare a local size");
    }
    if (auto& gs = prog_->stages[(int)Stage::Geometry]; gs.main) {
        bool inSet = gs.inLayout.primitive != Layout::kNoPrimitive, outSet = gs.outLayout.primitive != Layout::kNoPrimitive;
        r.gsInputPrimitive = inSet ? gs.inLayout.primitive : 0;
        r.gsOutputPrimitive = outSet ? gs.outLayout.primitive : 0;
        r.gsMaxVertices = gs.outLayout.maxVertices;
        r.gsInvocations = std::max(1, gs.inLayout.invocations);
        if (!inSet) error("geometry shader does not declare an input primitive type");
        if (!outSet) error("geometry shader does not declare an output primitive type");
        if (r.gsMaxVertices < 0) error("geometry shader does not declare max_vertices");
    }
    if (auto& tc = prog_->stages[(int)Stage::TessControl]; tc.main) r.tcsOutputVertices = tc.outLayout.vertices;
    if (auto& te = prog_->stages[(int)Stage::TessEval]; te.main) {
        r.tesPrimitiveMode = te.inLayout.primitive != Layout::kNoPrimitive ? te.inLayout.primitive : 0;
        r.tesSpacing = te.inLayout.spacing ? te.inLayout.spacing : gl::EQUAL;
        r.tesOrder = te.inLayout.order ? te.inLayout.order : gl::CCW;
        r.tesPointMode = te.inLayout.pointMode;
        if (!r.tesPrimitiveMode) error("tessellation evaluation shader does not declare a primitive mode");
    }
    for (Stage st : {Stage::Task, Stage::Mesh}) {
        auto& sc = prog_->stages[(int)st];
        if (!sc.main) continue;
        int* ls = st == Stage::Task ? r.taskLocalSize : r.meshLocalSize;
        for (int i = 0; i < 3; i++) ls[i] = std::max(1, sc.inLayout.localSize[i]);
        if (sc.inLayout.localSize[0] < 0 && sc.inLayout.localSize[1] < 0 && sc.inLayout.localSize[2] < 0)
            error(std::string(st == Stage::Task ? "task" : "mesh") + " shader does not declare a local size");
    }
    if (auto& ms = prog_->stages[(int)Stage::Mesh]; ms.main) {
        r.meshMaxVertices = ms.outLayout.maxVertices;
        r.meshMaxPrimitives = ms.outLayout.maxPrimitives;
        r.meshOutputPrimitive = ms.outLayout.primitive != Layout::kNoPrimitive ? ms.outLayout.primitive : 0;
        if (ms.outLayout.primitive == Layout::kNoPrimitive) error("mesh shader does not declare an output primitive type");
        if (r.meshMaxVertices < 0) error("mesh shader does not declare max_vertices");
        if (r.meshMaxPrimitives < 0) error("mesh shader does not declare max_primitives");
        // per-primitive gl_Layer / gl_ViewportIndex written through gl_MeshPrimitivesEXT[]
        for (auto& sh : ms.shaders)
            for (auto& e : sh->exprs) {
                if (e.kind != EK::Field || e.args[0]->kind != EK::Index || e.args[0]->args[0]->kind != EK::Var) continue;
                const Variable* v = e.args[0]->args[0]->var;
                if (v->builtin != BuiltinVar::MeshPrimitives || !ms.used.count(v)) continue;
                const std::string& m = v->block->members[e.field].name;
                if (m == "gl_Layer") r.writesLayer = true;
                if (m == "gl_ViewportIndex") r.writesViewportIndex = true;
            }
    }
    if (auto& fs = prog_->stages[(int)Stage::Fragment]; fs.main) {
        // Metal passes at most 124 user scalars from a vertex function to a fragment function
        if (prog_->stages[(int)Stage::Vertex].main && !prog_->stages[(int)Stage::Geometry].main &&
            !prog_->stages[(int)Stage::TessEval].main) {
            int scalars = 0;
            for (auto* v : fs.used) {
                if (v->q.storage != Storage::In) continue;
                bool compat = v->builtin == BuiltinVar::FragColorIn || v->builtin == BuiltinVar::SecondaryColorIn ||
                              v->builtin == BuiltinVar::TexCoord || v->builtin == BuiltinVar::FogFragCoord ||
                              v->builtin == BuiltinVar::ClipDistance || v->builtin == BuiltinVar::CullDistance;
                if (v->builtin != BuiltinVar::None && !compat) continue;
                std::function<int(const Type&)> count = [&](const Type& t) -> int {
                    int n = std::max(1, t.totalArrayElements());
                    Type b = t.stripArrays();
                    if (b.base == BT::Struct) {
                        int c = 0;
                        for (auto& m : b.sdef->members) c += count(m.type);
                        return n * c;
                    }
                    return n * b.components() * (b.is64() ? 2 : 1);
                };
                scalars += count(v->type);
            }
            r.vsMeshPassthrough = scalars > 124;
        }
        r.usesFragDepth = fs.writesFragDepth;
        // ARB_post_depth_coverage implicitly enables early fragment tests
        r.earlyFragmentTests = fs.inLayout.earlyFragmentTests || fs.inLayout.postDepthCoverage;
        if (fs.inLayout.postDepthCoverage)
            for (auto* v : fs.used)
                if (v->builtin == BuiltinVar::SampleMaskIn) r.postDepthCoverage = true;
        r.blendSupport = fs.outLayout.blendSupport;
        r.usesDiscard = fs.usesDiscard;
        if (prog_->strict()) {
            // EXT_fragment_shading_rate: the effective rate must be 1x1 for shaders that write depth or the sample
            // mask (FRAGMENT_SHADING_RATE_WITH_SHADER_DEPTH_STENCIL_WRITES / _WITH_SAMPLE_MASK are FALSE), shade
            // per sample, or read gl_FragCoord without enabling the extension.
            bool enabled = false;
            for (auto& sh : fs.shaders) {
                auto it = sh->extensions.find("GL_EXT_fragment_shading_rate");
                if (it != sh->extensions.end() && it->second != "disable") enabled = true;
            }
            bool force = fs.writesFragDepth;
            for (auto* v : fs.used) {
                if ((v->builtin == BuiltinVar::SampleMask && v->written) || v->builtin == BuiltinVar::SampleID ||
                    v->builtin == BuiltinVar::SamplePosition || (v->builtin == BuiltinVar::FragCoord && !enabled))
                    force = true;
                if (v->q.storage != Storage::In) continue;
                bool sample = v->q.sample;
                if (v->block)
                    for (auto& m : v->block->members) sample |= m.q.sample;
                if (sample) force = true;
            }
            // ARB_post_depth_coverage: the coverage is read per pixel (VariantKey::pdc)
            r.fsrForce1x1 = force || fs.inLayout.postDepthCoverage;
        }
    }
    // fp64 usage
    for (auto& sc : prog_->stages)
        for (auto& sh : sc.shaders)
            for (auto& e : sh->exprs)
                if (e.type.base == BT::Double) r.usesFp64 = true;
    // transform feedback
    Stage pre = Stage::Count;
    for (Stage st : {Stage::Geometry, Stage::TessEval, Stage::Vertex})
        if (prog_->stages[(int)st].main) { pre = st; break; }
    auto& mp = r.multipass;
    auto compSize = [](const Type& t) {
        Type b = t.stripArrays();
        int comps = b.base == BT::Struct ? 0 : b.components();
        if (b.base == BT::Struct)
            for (auto& m : b.sdef->members) comps += m.type.stripArrays().components() * std::max(1, m.type.totalArrayElements());
        return comps * (b.is64() ? 8 : 4) * std::max(1, t.totalArrayElements());
    };
    if (pre != Stage::Count) {
        const StageCode& ps = prog_->stages[(int)pre];
        // enhanced layouts: xfb_offset qualifiers take precedence over glTransformFeedbackVaryings.
        // The parser resolved each output's xfb_buffer and the block members' xfb_offsets.
        bool qualified = false;
        int strides[4] = {-1, -1, -1, -1};
        for (auto& sh : ps.shaders) {
            if (!sh->xfbMode) continue;
            qualified = true;
            for (int b = 0; b < 4; b++) {
                if (sh->xfbStride[b] < 0) continue;
                if (strides[b] >= 0 && strides[b] != sh->xfbStride[b])
                    error("'xfb_stride' : different strides declared for transform feedback buffer " + std::to_string(b));
                strides[b] = sh->xfbStride[b];
            }
        }
        std::vector<XfbCapture> caps;
        for (auto* v : ps.globals) {
            if (!qualified || v->q.storage != Storage::Out) continue;
            int vb = std::max(0, v->q.layout.xfbBuffer);
            // A struct (or array of structs) is captured member by member ("s[1].m"), each member
            // aligned like a standalone output and placed right after the previous one.
            std::function<void(const std::string&, const Type&, uint32_t&)> addLeaves = [&](const std::string& name, const Type& t,
                                                                                          uint32_t& off) {
                if (t.stripArrays().base == BT::Struct) {
                    if (t.isArray()) {
                        for (int i = 0; i < std::max(t.dims[0], 1); i++) addLeaves(name + "[" + std::to_string(i) + "]", t.elem(), off);
                    } else {
                        for (auto& m : t.sdef->members) addLeaves(name + "." + m.name, m.type, off);
                    }
                    return;
                }
                if (t.stripArrays().is64()) off = roundUp(off, 8);
                XfbVarying x;
                x.name = name;
                x.type = glTypeEnum(t);
                x.arraySize = t.isArray() ? t.dims[0] : 1;
                x.buffer = vb;
                x.offset = (int)off;
                r.xfbVaryings.push_back(x);
                off += xfbCaptureOf(t, vb, off).size;
            };
            auto addQ = [&](const std::string& name, const Type& t, int offset) {
                if (pre == Stage::Geometry) mp.xfbBufferStream[vb] = (uint32_t)std::max(0, v->q.layout.stream);
                uint32_t off = (uint32_t)offset;
                addLeaves(name, t, off);
                caps.push_back(xfbCaptureOf(t, vb, (uint32_t)offset));
                caps.back().name = name;
            };
            if (v->block) {
                for (auto& m : v->block->members)
                    if (m.q.layout.xfbOffset >= 0) addQ(v->anonBlock ? m.name : v->block->name + "." + m.name, m.type, m.q.layout.xfbOffset);
            } else if (v->q.layout.xfbOffset >= 0) {
                addQ(v->name, v->type, v->q.layout.xfbOffset);
            }
        }
        // builtin outputs captured through a gl_PerVertex redeclaration with xfb_offset
        for (auto* v : ps.used) {
            if (v->builtin == BuiltinVar::None || v->q.storage != Storage::Out || v->q.layout.xfbOffset < 0) continue;
            int vb = std::max(0, v->q.layout.xfbBuffer);  // resolved by the parser
            if (vb > 3) { error("xfb_buffer out of range"); continue; }
            XfbVarying x;
            x.name = v->name;
            x.type = glTypeEnum(v->type);
            x.arraySize = v->type.isArray() ? v->type.dims[0] : 1;
            x.buffer = vb;
            x.offset = v->q.layout.xfbOffset;
            r.xfbVaryings.push_back(x);
            caps.push_back(xfbCaptureOf(v->type, vb, (uint32_t)x.offset));
            caps.back().name = v->name;
            qualified = true;
        }
        if (qualified) {
            std::string err = checkXfbLayout(caps, strides);
            if (!err.empty()) error(err);
            for (int b = 0; b < 4; b++) {
                uint32_t end = 0;
                bool dbl = false, any = false;
                for (auto& c : caps)
                    if (c.buffer == b) {
                        any = true;
                        dbl |= c.dbl;
                        end = std::max(end, c.offset + c.size);
                    }
                if (!any && strides[b] < 0) continue;
                // a buffer with a declared stride but no captured outputs is not written (and need not be bound)
                if (any) mp.xfbBufferMask |= 1u << b;
                mp.xfbStride[b] = strides[b] >= 0 ? (uint32_t)strides[b] : roundUp(end, dbl ? 8 : 4);
            }
        } else if (!in_.xfbVaryings.empty()) {
            int buffer = 0, offset = 0;
            auto closeBuffer = [&]() {
                if (offset > 0 || (in_.xfbInterleaved && buffer == 0)) {
                    mp.xfbBufferMask |= 1u << buffer;
                    mp.xfbStride[buffer] = std::max<uint32_t>(mp.xfbStride[buffer], (uint32_t)offset);
                }
            };
            std::set<std::string> seenXfb;
            for (auto& name : in_.xfbVaryings) {
                if (name != "gl_NextBuffer" && name.compare(0, 17, "gl_SkipComponents") != 0 && !seenXfb.insert(name).second) {
                    error("transform feedback varying '" + name + "' specified more than once");
                    continue;
                }
                if (name == "gl_NextBuffer") {
                    closeBuffer();
                    buffer++;
                    offset = 0;
                    if (buffer > 3) { error("too many transform feedback buffers"); break; }
                    r.xfbVaryings.push_back({name, 0, 0, buffer, -1});  // reported with type NONE, size 0
                    continue;
                }
                if (name.compare(0, 17, "gl_SkipComponents") == 0) {
                    int n = std::atoi(name.c_str() + 17);
                    r.xfbVaryings.push_back({name, 0, n, buffer, offset});  // type NONE, size = skipped components
                    offset += 4 * n;
                    continue;
                }
                std::string base = name;
                int elem = -1;
                if (auto lb = name.find('['); lb != std::string::npos) {
                    base = name.substr(0, lb);
                    elem = std::atoi(name.c_str() + lb + 1);
                }
                std::string member;
                if (auto dot = base.find('.'); dot != std::string::npos) {
                    member = base.substr(dot + 1);
                    base = base.substr(0, dot);
                }
                Type t;
                bool found = false;
                int foundStream = 0;
                for (auto& sh : ps.shaders)
                    for (auto& v : sh->vars) {
                        if (found || !v.global || v.q.storage != Storage::Out) continue;
                        foundStream = v.q.layout.stream;
                        if (member.empty() && v.name == base && !v.anonBlock) { t = v.type; found = true; }
                        else if (member.empty() && v.anonBlock && v.block->findMember(base) >= 0) {
                            t = v.block->members[v.block->findMember(base)].type;
                            found = true;
                        } else if (!member.empty() && v.block && v.block->name == base && v.block->findMember(member) >= 0) {
                            t = v.block->members[v.block->findMember(member)].type;
                            found = true;
                        }
                    }
                if (!found) { error("transform feedback varying '" + name + "' is not an output of the last vertex stage"); continue; }
                if (pre == Stage::Geometry) mp.xfbBufferStream[buffer] = (uint32_t)std::max(0, foundStream);
                if (elem >= 0) {
                    if (!t.isArray() || elem >= t.dims[0]) { error("transform feedback varying '" + name + "' index out of range"); continue; }
                    t = t.elem();
                }
                // glTransformFeedbackVaryings packs tightly, doubles included (only xfb_offset
                // qualifiers must be 8-byte aligned).
                XfbVarying x;
                x.name = name;
                x.type = glTypeEnum(t);
                x.arraySize = t.isArray() ? t.dims[0] : 1;
                x.buffer = buffer;
                x.offset = offset;
                r.xfbVaryings.push_back(x);
                offset += compSize(t);
                // GL 4.6 11.1.2.1 limits (MAX_TRANSFORM_FEEDBACK_INTERLEAVED/SEPARATE_COMPONENTS)
                if (in_.xfbInterleaved && offset / 4 > 128) error("too many transform feedback components in buffer " + std::to_string(buffer));
                if (!in_.xfbInterleaved && compSize(t) / 4 > 64) error("transform feedback varying '" + name + "' has too many components");
                if (!in_.xfbInterleaved) {
                    closeBuffer();
                    buffer++;
                    offset = 0;
                    if (buffer > 4) { error("too many transform feedback buffers"); break; }
                }
            }
            if (in_.xfbInterleaved || offset > 0) closeBuffer();
        }
        // flags
        const StageCode& fs = prog_->stages[(int)Stage::Fragment];
        for (auto* v : fs.used) {
            if (v->q.storage != Storage::In || v->builtin != BuiltinVar::None) continue;
            Type b = v->type.stripArrays();
            bool flat = v->q.interp == Interp::Flat || b.isIntLike() || b.base == BT::Double;
            if (v->block)
                for (auto& m : v->block->members) {
                    Type mb = m.type.stripArrays();
                    flat |= m.q.interp == Interp::Flat || mb.isIntLike() || mb.base == BT::Double;
                }
            if (flat) r.usesFlatShading = true;
        }
        for (auto* v : ps.used) {
            if (v->builtin == BuiltinVar::Layer && v->q.storage == Storage::Out && v->written) r.writesLayer = true;
            if (v->builtin == BuiltinVar::ViewportIndex && v->q.storage == Storage::Out && v->written) r.writesViewportIndex = true;
            // EXT_fragment_shading_rate: the primitive's rate, from its provoking vertex (VariantKey::fsr pipelines)
            if (prog_->strict() && v->builtin == BuiltinVar::PrimitiveShadingRate && v->q.storage == Storage::Out && v->written &&
                (pre == Stage::Vertex || pre == Stage::Geometry))
                r.writesPrimitiveShadingRate = true;
        }
    }
    if (ok_) {
        std::string err = fillMultipassRecords(*prog_, mp);
        if (!err.empty()) res_.infoLog += "warning: " + err + "\n";
    }
}

// ---------------------------------------------------------------------------
// Tessellation fast path (Reflection::tessFusable / tessLevels, VariantKey::tessFused)
// ---------------------------------------------------------------------------
// Compiles a float scalar expression made of constants, default-block uniforms (float scalars, single
// components of float vectors, integer scalars converted to float), negation, + - * /, min, max and
// clamp into a postfix program. Returns false for anything else.
bool Linker::tessLevelCode(const Expr* e, std::vector<TessLevelOp>& out) {
    auto uniformOffset = [&](const Variable* v, uint32_t& off) {
        if (!v || !v->global || v->q.storage != Storage::Uniform || v->block || v->type.isOpaque() || v->type.isArray()) return false;
        auto it = prog_->defaultOffsets.find(v->name);
        if (it == prog_->defaultOffsets.end()) return false;
        off = it->second;
        return true;
    };
    TessLevelOp op;
    if (!e->type.isScalar() || e->type.base != BT::Float) return false;
    if (e->cv && !e->cv->c.empty()) {
        op.kind = TessLevelOp::Const;
        op.value = (float)e->cv->c[0].f;
        out.push_back(op);
        return true;
    }
    switch (e->kind) {
        case EK::Var:
            if (!uniformOffset(e->var, op.offset)) return false;
            op.kind = TessLevelOp::Float;
            out.push_back(op);
            return true;
        case EK::Swizzle:
            if (e->nswz != 1 || e->args[0]->kind != EK::Var || !e->args[0]->type.isVector() || !uniformOffset(e->args[0]->var, op.offset))
                return false;
            op.kind = TessLevelOp::Float;
            op.offset += 4u * e->swz[0];
            out.push_back(op);
            return true;
        case EK::Ctor: {  // conversion to float
            if (e->args.size() != 1) return false;
            const Expr* a = e->args[0];
            if (a->type.isScalar() && a->type.base == BT::Float) return tessLevelCode(a, out);
            if (!a->type.isScalar() || (a->type.base != BT::Int && a->type.base != BT::UInt) || a->kind != EK::Var ||
                !uniformOffset(a->var, op.offset))
                return false;
            op.kind = a->type.base == BT::Int ? TessLevelOp::Int : TessLevelOp::UInt;
            out.push_back(op);
            return true;
        }
        case EK::Unary:
            if (e->op != Op::Neg && e->op != Op::Plus) return false;
            if (!tessLevelCode(e->args[0], out)) return false;
            if (e->op == Op::Neg) out.push_back({TessLevelOp::Neg});
            return true;
        case EK::Binary: {
            TessLevelOp::Kind k;
            switch (e->op) {
                case Op::Add: k = TessLevelOp::Add; break;
                case Op::Sub: k = TessLevelOp::Sub; break;
                case Op::Mul: k = TessLevelOp::Mul; break;
                case Op::Div: k = TessLevelOp::Div; break;
                default: return false;
            }
            if (!tessLevelCode(e->args[0], out) || !tessLevelCode(e->args[1], out)) return false;
            out.push_back({k});
            return true;
        }
        case EK::Builtin: {
            const std::string& n = e->fn->name;
            TessLevelOp::Kind k;
            if (n == "min" && e->args.size() == 2) k = TessLevelOp::Min;
            else if (n == "max" && e->args.size() == 2) k = TessLevelOp::Max;
            else if (n == "clamp" && e->args.size() == 3) k = TessLevelOp::Clamp;
            else return false;
            for (auto* a : e->args)
                if (!tessLevelCode(a, out)) return false;
            out.push_back({k});
            return true;
        }
        default: return false;
    }
}

// Reflection::tessLevels: how the TCS sets the levels the domain uses. The TCS must write each of them exactly
// once, in main, as `gl_TessLevelX[c] = <expr>;` (or a chain of such assignments), either unconditionally or
// inside `if (gl_InvocationID == c)` (no else) for an invocation that exists, with no return before it and
// nothing else touching the levels. Left empty otherwise.
void Linker::tessLevelAnalysis() {
    auto& r = prog_->refl;
    for (auto& l : r.tessLevels) l.clear();
    const StageCode& tc = prog_->stages[(int)Stage::TessControl];
    if (!tc.main) return;
    int outv = tc.outLayout.vertices;
    auto isLevel = [](const Expr* e) {
        return e->kind == EK::Var && (e->var->builtin == BuiltinVar::TessLevelOuter || e->var->builtin == BuiltinVar::TessLevelInner);
    };
    // references to the levels: in all reachable code, and in one statement
    auto countRefs = [&](const auto& visit) {
        int n = 0;
        Walker w;
        w.onExpr = [&](const Expr* e) { if (isLevel(e)) n++; };
        visit(w);
        return n;
    };
    int refs = countRefs([&](Walker& w) { for (auto* f : tc.reachable) w.stmt(f->body); });
    auto refsIn = [&](const Stmt* s) { return countRefs([&](Walker& w) { w.stmt(s); }); };
    bool hasReturn = false, ok = true;
    int matched = 0;
    int writes[6] = {};
    auto containsReturn = [&](const Stmt* s) {
        bool ret = false;
        Walker w;
        w.onExpr = [](const Expr*) {};
        w.onStmt = [&](const Stmt* t) { if (t->kind == SK::Return) ret = true; };
        w.stmt(s);
        return ret;
    };
    // `gl_TessLevelX[c] = (gl_TessLevelY[d] = ...) = rhs`
    auto assignment = [&](const Expr* e) {
        std::vector<int> targets;
        while (e->kind == EK::Assign && e->op == Op::Assign) {
            const Expr* l = e->args[0];
            if (l->kind != EK::Index || !isLevel(l->args[0]) || !l->args[1]->cv) return false;
            bool inner = l->args[0]->var->builtin == BuiltinVar::TessLevelInner;
            int64_t c = l->args[1]->cv->c[0].i;
            if (c < 0 || c >= (inner ? 2 : 4)) return false;
            targets.push_back((int)c + (inner ? 4 : 0));
            e = e->args[1];
        }
        if (targets.empty()) return false;
        std::vector<TessLevelOp> code;
        if (!tessLevelCode(e, code)) return false;
        for (int t : targets) {
            if (writes[t]++) return false;
            r.tessLevels[t] = code;
        }
        matched += (int)targets.size();
        return true;
    };
    std::function<void(const Stmt*, bool)> walk = [&](const Stmt* s, bool top) {
        if (!s || !ok) return;
        int n = refsIn(s);
        if (n == 0) {
            if (containsReturn(s)) hasReturn = true;
            return;
        }
        if (hasReturn) { ok = false; return; }
        switch (s->kind) {
            case SK::Block:
                for (auto* b : s->body) walk(b, top);
                return;
            case SK::Expr:
                if (!assignment(s->expr)) ok = false;
                return;
            case SK::If: {
                // only `if (gl_InvocationID == c)` for an existing invocation, levels in the then branch
                const Expr* c = s->expr;
                const Expr *a = c ? c->args.size() == 2 ? c->args[0] : nullptr : nullptr, *b = a ? c->args[1] : nullptr;
                if (a && b && b->kind == EK::Var) std::swap(a, b);
                bool inv = top && c && c->kind == EK::Binary && c->op == Op::Eq && a && b && a->kind == EK::Var &&
                           a->var->builtin == BuiltinVar::InvocationID && b->cv && b->cv->c[0].i >= 0 && b->cv->c[0].i < outv;
                if (!inv || refsIn(s->els) || refsIn(s->then) != n) { ok = false; return; }
                walk(s->then, false);
                if (containsReturn(s->then)) hasReturn = true;
                return;
            }
            default: ok = false; return;
        }
    };
    if (tc.main->body)
        for (auto* s : tc.main->body->body) walk(s, true);
    int nOuter = r.tesPrimitiveMode == gl::TRIANGLES ? 3 : r.tesPrimitiveMode == gl::QUADS ? 4 : 2;
    int nInner = r.tesPrimitiveMode == gl::TRIANGLES ? 1 : r.tesPrimitiveMode == gl::QUADS ? 2 : 0;
    for (int i = 0; i < nOuter; i++) ok = ok && writes[i];
    for (int i = 0; i < nInner; i++) ok = ok && writes[4 + i];
    if (!ok || matched != refs)
        for (auto& l : r.tessLevels) l.clear();
}

// Reflection::tesAffine: every TES output the rest of the pipeline uses is affine in gl_TessCoord (a sum of
// gl_TessCoord components times per-patch values, plus a per-patch value), and outputs the fragment shader
// doesn't interpolate perspective-correctly at the pixel centre (flat, integer, noperspective, centroid, sample)
// don't depend on gl_TessCoord at all. Then the tessellated patch, at any levels, covers the triangle(s) of its
// corners with the same interpolated values: perspective-correct interpolation of an attribute affine in the
// domain is exact on every sub-triangle (both it and the clip position are affine there), flat values are the
// same at every vertex, and clipping by affine clip distances cuts the same region. Classes per expression:
// constant per patch (inputs, uniforms, levels, gl_PrimitiveID, ...), affine, other. Straight-line code with
// conditions constant per patch; anything written in loops, under other conditions or by called functions is
// "other"; a return before the end of main, cull distances, compatibility color varyings fail the analysis.
bool Linker::tesAffineAnalysis() {
    const StageCode& te = prog_->stages[(int)Stage::TessEval];
    const StageCode& fs = prog_->stages[(int)Stage::Fragment];
    if (!te.main || !te.main->body) return false;
    enum { C = 0, A = 1, N = 2 };
    using Slot = std::pair<const Variable*, int>;  // variable, block member (-1: whole variable)
    std::map<Slot, int> cls;
    bool fail = false;
    auto isOut = [](const Variable* v) { return v->global && v->q.storage == Storage::Out; };
    // functions other than main must not write outputs or globals (their writes aren't tracked)
    for (auto* f : te.reachable) {
        if (f == te.main) continue;
        Walker w;
        w.onExpr = [&](const Expr* e) {
            const Expr* l = nullptr;
            if (e->kind == EK::Assign) l = e->args[0];
            else if (e->kind == EK::Unary && (e->op == Op::PreInc || e->op == Op::PreDec || e->op == Op::PostInc || e->op == Op::PostDec))
                l = e->args[0];
            while (l && (l->kind == EK::Index || l->kind == EK::Field || l->kind == EK::Swizzle)) l = l->args[0];
            if (l && l->kind == EK::Var && l->var->global) fail = true;
        };
        w.stmt(f->body);
    }
    if (fail) return false;
    // user functions that read gl_TessCoord, outputs or private globals (directly or through their callees)
    // aren't constant per patch even with constant arguments
    std::set<const Function*> impure;
    for (bool changed = true; changed;) {
        changed = false;
        for (auto* f : te.reachable) {
            if (impure.count(f)) continue;
            bool bad = false;
            Walker w;
            w.onExpr = [&](const Expr* e) {
                if (e->kind == EK::Var && (e->var->builtin == BuiltinVar::TessCoord ||
                                           (e->var->global && (e->var->q.storage == Storage::Out || e->var->q.storage == Storage::Temp))))
                    bad = true;
                if ((e->kind == EK::Call || e->kind == EK::SubroutineCall) && e->fn) {
                    auto it = te.defs.find(e->fn->mangled);
                    if (e->kind == EK::SubroutineCall || it == te.defs.end() || impure.count(it->second)) bad = true;
                }
            };
            w.stmt(f->body);
            if (bad) {
                impure.insert(f);
                changed = true;
            }
        }
    }
    auto slotOf =[&](const Expr* e, bool& whole) -> Slot {
        whole = true;
        const Expr* member = nullptr;
        while (e->kind == EK::Index || e->kind == EK::Field || e->kind == EK::Swizzle) {
            if (e->kind == EK::Field && e->args[0]->kind == EK::Var && e->args[0]->var->block) member = e;
            else whole = false;
            e = e->args[0];
        }
        if (e->kind != EK::Var) return {nullptr, -1};
        if (member && member != e) {
            // `block.member` written whole only when the member access is the outermost expression
            return {e->var, member->field};
        }
        return {e->var, -1};
    };
    std::function<int(const Expr*)> ex;
    bool taint = false;  // inside a loop or a condition that isn't constant per patch
    auto write = [&](const Expr* l, int c) {
        bool whole;
        Slot s = slotOf(l, whole);
        if (!s.first) { fail = true; return; }
        // a member of a block written whole: whole iff the expression is just `block.member`
        if (s.second >= 0) whole = l->kind == EK::Field && l->args[0]->kind == EK::Var;
        else whole = l->kind == EK::Var;
        if (taint) c = N;
        int& v = cls[s];
        v = whole ? c : std::max(v, c);
    };
    auto readVar = [&](const Variable* v, int member) {
        if (v->builtin == BuiltinVar::TessCoord) return (int)A;
        auto it = cls.find({v, member});
        if (it != cls.end()) return it->second;
        if (member >= 0) {
            auto w = cls.find({v, -1});
            if (w != cls.end()) return w->second;
        }
        return (int)C;  // inputs, uniforms, levels, built-ins, unwritten locals
    };
    ex = [&](const Expr* e) -> int {
        if (e->cv) return C;
        switch (e->kind) {
            case EK::Const: case EK::Length: return C;
            case EK::Var: return readVar(e->var, -1);
            case EK::Field:
                if (e->args[0]->kind == EK::Var && e->args[0]->var->block) return readVar(e->args[0]->var, e->field);
                return ex(e->args[0]);
            case EK::Swizzle: return ex(e->args[0]);
            case EK::Index: {
                int i = ex(e->args[1]), b = ex(e->args[0]);
                return i == C ? b : N;
            }
            case EK::Unary:
                if (e->op == Op::Neg || e->op == Op::Plus) return ex(e->args[0]);
                if (e->op == Op::PreInc || e->op == Op::PreDec || e->op == Op::PostInc || e->op == Op::PostDec) {
                    int c = ex(e->args[0]);
                    write(e->args[0], c);
                    return c;
                }
                return ex(e->args[0]) == C ? C : N;
            case EK::Binary: {
                int a = ex(e->args[0]), b = ex(e->args[1]);
                switch (e->op) {
                    case Op::Add: case Op::Sub: return std::max(a, b);
                    case Op::Mul: return a == C ? b : b == C ? a : N;
                    case Op::Div: return b == C ? a : N;
                    default: return a == C && b == C ? C : N;
                }
            }
            case EK::Assign: {
                int rc = ex(e->args[1]);
                if (e->op != Op::Assign) {
                    int lc = ex(e->args[0]);
                    switch (e->op) {
                        case Op::AddAssign: case Op::SubAssign: rc = std::max(lc, rc); break;
                        case Op::MulAssign: rc = lc == C ? rc : rc == C ? lc : N; break;
                        case Op::DivAssign: rc = rc == C ? lc : N; break;
                        default: rc = lc == C && rc == C ? C : N; break;
                    }
                }
                write(e->args[0], rc);
                return rc;
            }
            case EK::Ternary: {
                int c = ex(e->args[0]), a = ex(e->args[1]), b = ex(e->args[2]);
                return c == C ? std::max(a, b) : N;
            }
            case EK::Comma: {
                int c = C;
                for (auto* a : e->args) c = ex(a);
                return c;
            }
            case EK::Ctor: {
                int c = C;
                for (auto* a : e->args) c = std::max(c, ex(a));
                Type t = e->type.stripArrays();
                if (c != C && t.base != BT::Float && t.base != BT::Double) return N;  // e.g. truncation to int
                return c;
            }
            case EK::Builtin: case EK::Call: case EK::SubroutineCall: {
                std::vector<int> a;
                for (auto* x : e->args) a.push_back(ex(x));
                // out / inout arguments: written with an unknown value
                if (e->fn)
                    for (size_t i = 0; i < e->args.size() && i < e->fn->params.size(); i++) {
                        Storage s = e->fn->params[i]->q.storage;
                        if (s == Storage::Out || s == Storage::InOut) write(e->args[i], N);
                    }
                int all = C;
                for (int c : a) all = std::max(all, c);
                if (e->kind != EK::Builtin) {
                    auto it = e->fn ? te.defs.find(e->fn->mangled) : te.defs.end();
                    if (e->kind == EK::SubroutineCall || it == te.defs.end() || impure.count(it->second)) return N;
                }
                if (all == C) return C;
                if (e->kind != EK::Builtin || !e->fn) return N;
                const std::string& n = e->fn->name;
                if (n == "mix" && a.size() == 3) {
                    if (a[2] == C) return std::max(a[0], a[1]);
                    return a[0] == C && a[1] == C && a[2] == A ? A : N;
                }
                if ((n == "dot" || n == "cross" || n == "matrixCompMult" || n == "outerProduct") && a.size() == 2)
                    return a[0] == C ? a[1] : a[1] == C ? a[0] : N;
                return N;
            }
            default: return N;
        }
    };
    // statements of main; conditions constant per patch keep values affine (each patch takes one branch)
    std::function<void(const Stmt*)> st = [&](const Stmt* s) {
        if (!s || fail) return;
        switch (s->kind) {
            case SK::Block:
                for (auto* b : s->body) st(b);
                return;
            case SK::Expr: ex(s->expr); return;
            case SK::Decl:
                for (auto* v : s->vars) cls[{v, -1}] = taint ? (int)N : v->init ? ex(v->init) : (int)C;
                return;
            case SK::If: {
                bool saved = taint;
                if (ex(s->expr) != C) taint = true;
                auto before = cls;
                st(s->then);
                auto afterThen = cls;
                cls = before;
                st(s->els);
                for (auto& [k, v] : afterThen) cls[k] = std::max(cls.count(k) ? cls[k] : (int)C, v);
                taint = saved;
                return;
            }
            case SK::For: case SK::While: case SK::DoWhile: case SK::Switch: {
                bool saved = taint;
                taint = true;
                st(s->init);
                if (s->expr) ex(s->expr);
                if (s->cond) ex(s->cond);
                st(s->condDecl);
                st(s->then);
                for (auto* b : s->body) st(b);
                if (s->incr) ex(s->incr);
                taint = saved;
                return;
            }
            case SK::Return: case SK::Discard: fail = true; return;
            default: return;
        }
    };
    const auto& body = te.main->body->body;
    for (size_t i = 0; i < body.size() && !fail; i++) {
        if (body[i]->kind == SK::Return && i + 1 == body.size()) break;
        st(body[i]);
    }
    if (fail) return false;
    // fragment inputs by varying key: how each is interpolated
    std::map<std::string, const Variable*> fsIn;
    if (fs.main)
        for (auto* v : fs.used)
            if (v->q.storage == Storage::In && v->builtin == BuiltinVar::None) fsIn[varyingKey(v, Stage::Fragment)] = v;
    auto pixelCentreSmooth = [](const Qualifiers& q, const Type& t) {
        Type b = t.stripArrays();
        bool intLike = b.isIntLike() || b.base == BT::Double || b.base == BT::Bool;
        return q.interp == Interp::Smooth && !q.centroid && !q.sample && !intLike;
    };
    for (auto* v : te.globals) {
        if (!isOut(v)) continue;
        if (v->builtin == BuiltinVar::CullDistance && te.used.count(v)) return false;
        if (v->builtin != BuiltinVar::None && v->builtin != BuiltinVar::Position && v->builtin != BuiltinVar::ClipDistance &&
            v->builtin != BuiltinVar::PointSize && te.used.count(v))
            return false;  // compatibility varyings (flat shading is runtime state), gl_PerVertex redeclarations
    }
    for (auto& [slot, c] : cls) {
        const Variable* v = slot.first;
        if (!isOut(v) || c == C) continue;
        if (v->block && slot.second < 0) return false;  // a whole output block (its members' interpolation varies)
        if (c == N) {
            if (v->builtin == BuiltinVar::PointSize) continue;  // triangles: no point size
            if (v->builtin != BuiltinVar::None) return false;   // gl_Position, clip distances
            // a varying the fragment shader doesn't read doesn't matter
            auto it = fsIn.find(varyingKey(v, Stage::TessEval));
            if (it == fsIn.end()) continue;
            if (slot.second < 0 || !v->block) return false;
            const std::string& name = v->block->members[slot.second].name;
            int fm = it->second->block ? it->second->block->findMember(name) : -1;
            if (fm >= 0 || !it->second->block) return false;
            continue;
        }
        // affine: the fragment shader must interpolate it perspective-correctly at the pixel centre
        if (v->builtin != BuiltinVar::None) continue;  // gl_Position, clip distances
        auto it = fsIn.find(varyingKey(v, Stage::TessEval));
        if (it == fsIn.end()) continue;
        const Variable* f = it->second;
        if (slot.second >= 0 && v->block && f->block) {
            int fm = f->block->findMember(v->block->members[slot.second].name);
            if (fm < 0) continue;
            const Member& m = f->block->members[fm];
            Qualifiers q = m.q;
            if (f->q.interp != Interp::Smooth) q.interp = f->q.interp;
            q.centroid |= f->q.centroid;
            q.sample |= f->q.sample;
            if (!pixelCentreSmooth(q, m.type)) return false;
        } else if (!pixelCentreSmooth(f->q, f->type)) {
            return false;
        }
    }
    return true;
}

// Whether stage st runs correctly with its invocations in other arrangements (or not at all): no memory writes (they
// could happen more than once) and no subgroup operations; with `barriers`, no TCS barrier() either (invocations in
// sequence). A stage the program doesn't have is.
bool Linker::arrangementFree(Stage st, bool barriers) const {
    auto bufferRoot = [](const Expr* e) {
        while (e->kind == EK::Index || e->kind == EK::Field || e->kind == EK::Swizzle) e = e->args[0];
        return e->kind == EK::Var && e->var->q.storage == Storage::Buffer;
    };
    const StageCode& sc = prog_->stages[(int)st];
    if (!sc.main) return true;
    for (const std::string& n : sc.usedNames)
        if (n.compare(0, 11, "gl_Subgroup") == 0 || n.compare(0, 11, "gl_SubGroup") == 0) return false;
    bool bad = false;
    Walker w;
    w.onExpr = [&](const Expr* e) {
        switch (e->kind) {
            case EK::Assign: bad |= bufferRoot(e->args[0]); return;
            case EK::Unary:
                if (e->op == Op::PreInc || e->op == Op::PreDec || e->op == Op::PostInc || e->op == Op::PostDec) bad |= bufferRoot(e->args[0]);
                return;
            case EK::Call: case EK::Builtin: case EK::SubroutineCall: break;
            default: return;
        }
        if (!e->fn) return;
        for (size_t i = 0; i < e->args.size() && i < e->fn->params.size(); i++) {  // buffer lvalues as out / inout arguments
            Storage s = e->fn->params[i]->q.storage;
            if ((s == Storage::Out || s == Storage::InOut) && bufferRoot(e->args[i])) bad = true;
        }
        if (e->kind != EK::Builtin) return;
        const std::string& n = e->fn->name;
        if ((n.compare(0, 6, "atomic") == 0 && n != "atomicCounter") || n.compare(0, 10, "imageStore") == 0 ||
            n.compare(0, 11, "imageAtomic") == 0)
            bad = true;
        if ((n == "barrier" && st == Stage::TessControl && barriers) || n.compare(0, 8, "subgroup") == 0 || n.compare(0, 6, "ballot") == 0 ||
            n.compare(0, 14, "readInvocation") == 0 || n.compare(0, 19, "readFirstInvocation") == 0 ||
            n.compare(0, 13, "anyInvocation") == 0 || n.compare(0, 14, "allInvocations") == 0)
            bad = true;
    };
    for (auto* f : sc.reachable) w.stmt(f->body);
    return !bad;
}

// Decides Reflection::tessFusable, tessLevels and tesAffine (see glsl.h).
void Linker::tessFusedAnalysis() {
    auto& r = prog_->refl;
    const uint32_t m = prog_->stageMask;
    auto has = [&](Stage s) { return (m & (1u << (int)s)) != 0; };
    if (!has(Stage::TessEval)) return;
    tessLevelAnalysis();
    auto fusableStage = [&](Stage st, bool barriers) { return arrangementFree(st, barriers); };
    // Several patches per object threadgroup (tessellation emulation): VS and TCS run once per control point, but threads
    // past the draw's last patch repeat its work without storing it.
    r.tessBatchable = fusableStage(Stage::Vertex, false) && fusableStage(Stage::TessControl, false);
    if (!has(Stage::Vertex) || has(Stage::Geometry)) return;
    bool tri = r.tesPrimitiveMode == gl::TRIANGLES;
    if ((!tri && r.tesPrimitiveMode != gl::QUADS) || r.tesPointMode) return;
    if (r.writesLayer || r.writesViewportIndex) return;
    // Fragment shader: gl_PrimitiveID would count the rasterized triangles (not the patches) and the
    // inputs must fit a classic vertex function (124 scalars).
    if (const StageCode& fs = prog_->stages[(int)Stage::Fragment]; fs.main) {
        if (fs.usedNames.count("gl_PrimitiveID")) return;
        int scalars = 0;
        std::function<int(const Type&)> count = [&](const Type& t) -> int {
            int n = std::max(1, t.totalArrayElements());
            Type b = t.stripArrays();
            if (b.base == BT::Struct) {
                int c = 0;
                for (auto& mb : b.sdef->members) c += count(mb.type);
                return n * c;
            }
            return n * b.components() * (b.is64() ? 2 : 1);
        };
        for (auto* v : fs.used)
            if (v->q.storage == Storage::In && (v->builtin == BuiltinVar::None || v->builtin == BuiltinVar::ClipDistance ||
                                                v->builtin == BuiltinVar::CullDistance))
                scalars += count(v->type);
        if (scalars > 124) return;
    }
    // Metal's tessellator runs the TES as a vertex function, possibly more than once per domain point: no memory
    // writes or subgroup operations there (the VS and TCS run in the mgl_tcs kernel as in the emulation).
    r.tessMetal = fusableStage(Stage::TessEval, false);
    if (const StageCode& tc = prog_->stages[(int)Stage::TessControl]; tc.main && (tc.outLayout.vertices < 1 || tc.outLayout.vertices > 4)) return;
    // The fused vertex function runs the vertex shader once per patch vertex for every output vertex
    // and the TCS invocations one after another (no barrier()): memory writes would happen several
    // times and subgroup operations would see other invocations. Reading buffers and images is fine.
    for (Stage st : {Stage::Vertex, Stage::TessControl, Stage::TessEval})
        if (!fusableStage(st, true)) return;
    r.tessFusable = true;
    r.tesAffine = tesAffineAnalysis();
}

// Uniform/storage blocks with the same block name must match across the program (GLSL 4.60 4.3.9), and the
// global names exposed by anonymous uniform blocks must not collide with other uniforms.
void Linker::checkBlockInterfaces() {
    struct BlockInfo { const Variable* v; };
    std::map<std::string, BlockInfo> blocks;                 // "U:Name" / "B:Name"
    std::map<std::string, std::string> uniformNames;         // global uniform name -> owner ("" = plain uniform)
    for (int s = 0; s < (int)Stage::Count; s++) {
        for (auto& sh : prog_->stages[s].shaders) {
            for (auto& g : sh->globals) {
                if (g.kind != GlobalDecl::Var) continue;
                const Variable* v = g.var;
                if (v->q.storage != Storage::Uniform && v->q.storage != Storage::Buffer) continue;
                bool ubo = v->q.storage == Storage::Uniform;
                if (v->block) {
                    const StructDef& sd = *v->block;
                    std::string key = std::string(ubo ? "U:" : "B:") + sd.name;
                    auto it = blocks.find(key);
                    if (it == blocks.end()) {
                        blocks[key] = {v};
                    } else {
                        const Variable* o = it->second.v;
                        const StructDef& od = *o->block;
                        bool same = o->anonBlock == v->anonBlock && o->type.dims == v->type.dims &&
                                    od.members.size() == sd.members.size() &&
                                    od.blockQ.layout.packing == sd.blockQ.layout.packing &&
                                    od.blockQ.layout.matrix == sd.blockQ.layout.matrix &&
                                    !(od.blockQ.layout.binding >= 0 && sd.blockQ.layout.binding >= 0 &&
                                      od.blockQ.layout.binding != sd.blockQ.layout.binding);
                        for (size_t i = 0; same && i < sd.members.size(); i++) {
                            const Member& a = od.members[i];
                            const Member& b = sd.members[i];
                            same = a.name == b.name && a.type.str() == b.type.str() && a.q.layout.matrix == b.q.layout.matrix &&
                                   a.q.layout.offset == b.q.layout.offset && a.q.layout.align == b.q.layout.align;
                        }
                        if (!same) error(std::string(ubo ? "uniform" : "shader storage") + " block '" + sd.name +
                                         "' is declared differently in different shaders");
                    }
                    if (ubo && v->anonBlock) {
                        for (auto& m : sd.members) {
                            auto u = uniformNames.find(m.name);
                            if (u == uniformNames.end()) uniformNames[m.name] = sd.name;
                            else if (u->second != sd.name)
                                error("uniform name '" + m.name + "' is used by different uniform declarations");
                        }
                    }
                    continue;
                }
                if (!ubo) continue;
                auto u = uniformNames.find(v->name);
                if (u == uniformNames.end()) uniformNames[v->name] = "";
                else if (!u->second.empty()) error("uniform name '" + v->name + "' is also a member of uniform block '" + u->second + "'");
            }
        }
    }
}

LinkResult Linker::run() {
    prog_ = std::make_shared<LinkedProgram>();
    prog_->separable = in_.separable;
    prog_->strictLevel = in_.strictLevel;
    prog_->robustTexelFetch = in_.robustTexelFetch || in_.strict();
    for (auto& sh : in_.shaders) {
        if (!sh) { error("attempt to link an uncompiled shader"); continue; }
        prog_->stages[(int)sh->stage].shaders.push_back(sh);
        prog_->stageMask |= 1u << (int)sh->stage;
    }
    if (!prog_->stageMask) error("no shaders attached");
    if ((prog_->stageMask & (1u << (int)Stage::Compute)) && (prog_->stageMask & ~(1u << (int)Stage::Compute)))
        error("a compute shader cannot be linked with other shader stages");
    // GL 4.6 7.3: only non-separable programs need a TES alongside a TCS.
    if (!in_.separable && (prog_->stageMask & (1u << (int)Stage::TessEval)) == 0 && (prog_->stageMask & (1u << (int)Stage::TessControl)))
        error("tessellation control shader requires a tessellation evaluation shader");
    const uint32_t meshStages = (1u << (int)Stage::Task) | (1u << (int)Stage::Mesh);
    const uint32_t vertexStages = (1u << (int)Stage::Vertex) | (1u << (int)Stage::TessControl) | (1u << (int)Stage::TessEval) |
                                  (1u << (int)Stage::Geometry);
    if ((prog_->stageMask & meshStages) && (prog_->stageMask & vertexStages))
        error("mesh and task shaders cannot be linked with vertex, tessellation or geometry shaders");
    if (!in_.separable && (prog_->stageMask & (1u << (int)Stage::Task)) && !(prog_->stageMask & (1u << (int)Stage::Mesh)))
        error("a task shader requires a mesh shader");
    if (!in_.separable && (prog_->stageMask & ~(1u << (int)Stage::Compute)) && !(prog_->stageMask & (1u << (int)Stage::Vertex)) &&
        !(prog_->stageMask & (1u << (int)Stage::Mesh)))
        error("program has no vertex shader");
    for (int s = 0; s < (int)Stage::Count && ok_; s++)
        if (!prog_->stages[s].shaders.empty()) collectStage((Stage)s);
    if (ok_) {
        assignAttributes();
        assignFragOutputs();
        checkBlockInterfaces();
        if (ok_) assignUniforms();
        checkVaryings();
        stageMetadata();
        if (ok_) reflectProgramInterface(*prog_, prog_->refl);
        if (ok_) tessFusedAnalysis();
        if (ok_) {
            const StageCode& fs = prog_->stages[(int)Stage::Fragment];
            bool sampleMask = false;
            for (auto* v : fs.used)
                if (v->builtin == BuiltinVar::SampleMask) sampleMask = true;
            prog_->refl.fsOutputOnly = fs.main && !fs.usesDiscard && !fs.writesFragDepth && !sampleMask && arrangementFree(Stage::Fragment, false);
        }
    }
    // names produced by the stage feeding the rasterizer (used by the FS emitter to drop unmatched inputs)
    if (ok_) {
        res_.reflection = prog_->refl;
        res_.program = prog_;
    }
    return std::move(res_);
}

}  // namespace

LinkResult link(const LinkInput& input) {
    Linker l(input);
    return l.run();
}

}  // namespace glsl
