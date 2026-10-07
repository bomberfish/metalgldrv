// Program interface reflection for GL_PROGRAM_INPUT / GL_PROGRAM_OUTPUT (GL 4.6 §7.3.1):
// the active inputs of the first stage and the active outputs of the last stage of a
// program, including built-in variables, enumerated the way glGetProgramResource* reports them.
#include <functional>

#include "glsl/program.h"
#include "glsl/types.h"

namespace glsl {

namespace {

int locationsOf(const Type& t) {
    Type b = t.stripArrays();
    int per = 1;
    if (b.base == BT::Struct) {
        per = 0;
        for (auto& m : b.sdef->members) per += locationsOf(m.type);
    } else if (b.cols) {
        per = b.cols * ((b.is64() && b.vec > 2) ? 2 : 1);
    } else if (b.is64() && b.vec > 2) {
        per = 2;
    }
    return per * t.totalArrayElements();
}

bool perVertexArrayed(Stage s, Storage st, bool patch) {
    if (patch) return false;
    if (st == Storage::In) return s == Stage::TessControl || s == Stage::TessEval || s == Stage::Geometry;
    return s == Stage::TessControl;
}

struct Collector {
    Stage stage;
    std::vector<InterfaceVar>& out;

    // Adds `name` of type `t` (struct members and arrays of structs are enumerated individually).
    void add(const std::string& name, const Type& t, int location, int component, bool patch) {
        Type b = t.stripArrays();
        if (b.base == BT::Struct) {
            int n = t.isArray() ? std::max(1, t.totalArrayElements()) : 1;
            int loc = location;
            for (int i = 0; i < n; i++) {
                std::string base = t.isArray() ? name + "[" + std::to_string(i) + "]" : name;
                for (auto& m : b.sdef->members) {
                    add(base + "." + m.name, m.type, loc, 0, patch);
                    if (loc >= 0) loc += locationsOf(m.type);
                }
            }
            return;
        }
        InterfaceVar v;
        v.name = t.isArray() ? name + "[0]" : name;
        v.type = glTypeEnum(b);
        v.arraySize = t.isArray() ? std::max(1, t.dims[0]) : 1;
        v.location = location;
        v.component = std::max(0, component);
        v.perPatch = patch;
        v.stageMask = 1u << (int)stage;
        out.push_back(v);
    }
};

void walkExpr(const Expr* e, const std::function<void(const Expr*)>& f) {
    if (!e) return;
    f(e);
    for (auto* a : e->args) walkExpr(a, f);
    walkExpr(e->aux, f);
}

void walkStmt(const Stmt* s, const std::function<void(const Expr*)>& f) {
    if (!s) return;
    for (auto* b : s->body) walkStmt(b, f);
    walkExpr(s->expr, f);
    walkStmt(s->init, f);
    walkExpr(s->cond, f);
    walkStmt(s->condDecl, f);
    walkExpr(s->incr, f);
    walkStmt(s->then, f);
    walkStmt(s->els, f);
    for (auto* v : s->vars) walkExpr(v->init, f);
}

// Members of gl_in / gl_out referenced by the stage.
std::map<const Variable*, std::set<int>> usedBuiltinBlockMembers(const StageCode& sc) {
    std::map<const Variable*, std::set<int>> used;
    auto visit = [&](const Expr* e) {
        if (e->kind != EK::Field) return;
        const Expr* b = e->args[0];
        while (b->kind == EK::Index) b = b->args[0];
        if (b->kind == EK::Var && b->var->block && (b->var->builtin == BuiltinVar::In || b->var->builtin == BuiltinVar::Out))
            used[b->var].insert(e->field);
    };
    for (auto* f : sc.reachable) walkStmt(f->body, visit);
    for (auto* v : sc.globals) walkExpr(v->init, visit);
    return used;
}

bool isCompatAttrib(BuiltinVar b) {
    return b == BuiltinVar::Vertex || b == BuiltinVar::Normal || b == BuiltinVar::Color || b == BuiltinVar::SecondaryColor ||
           b == BuiltinVar::MultiTexCoord || b == BuiltinVar::FogCoord;
}

void collect(const LinkedProgram& P, Stage s, Storage storage, std::vector<InterfaceVar>& out) {
    const StageCode& sc = P.stages[(int)s];
    Collector c{s, out};
    // vertex inputs and fragment outputs carry the locations assigned by the linker
    auto findVar = [&](const std::string& name) -> const Variable* {
        for (auto* g : sc.used)
            if (g->global && g->name == name && g->q.storage == storage) return g;
        for (auto* g : sc.globals)
            if (g->name == name && g->q.storage == storage) return g;
        return nullptr;
    };
    if (s == Stage::Vertex && storage == Storage::In) {
        for (auto& a : P.refl.attributes) {
            const Variable* var = findVar(a.name);
            InterfaceVar v;
            v.name = var && var->type.isArray() ? a.name + "[0]" : a.name;
            v.type = a.type;
            v.arraySize = a.arraySize;
            v.location = a.location;
            v.component = var ? std::max(0, var->q.layout.component) : 0;
            v.stageMask = 1u << (int)s;
            out.push_back(v);
        }
    }
    if (s == Stage::Fragment && storage == Storage::Out) {
        for (auto& o : P.refl.fragOutputs) {
            const Variable* var = findVar(o.name);
            InterfaceVar v;
            v.name = var && var->type.isArray() ? o.name + "[0]" : o.name;
            v.type = o.type;
            v.arraySize = o.arraySize;
            v.location = o.location;
            v.locationIndex = o.index;
            v.component = var ? std::max(0, var->q.layout.component) : 0;
            v.stageMask = 1u << (int)s;
            out.push_back(v);
        }
    }
    auto members = usedBuiltinBlockMembers(sc);
    std::vector<const Variable*> vars(sc.used.begin(), sc.used.end());
    std::sort(vars.begin(), vars.end(), [](const Variable* a, const Variable* b) { return a->id < b->id; });
    for (auto* v : vars) {
        if (!v->global || v->q.storage != storage) continue;
        if (v->builtin == BuiltinVar::WorkGroupSize) continue;
        if (v->builtin >= BuiltinVar::SubgroupSize && v->builtin <= BuiltinVar::SubgroupID) continue;  // system values
        if (s == Stage::Vertex && storage == Storage::In && (v->builtin == BuiltinVar::None || isCompatAttrib(v->builtin))) continue;
        if (s == Stage::Fragment && storage == Storage::Out && v->builtin == BuiltinVar::None) continue;  // user outputs added above
        bool arrayed = perVertexArrayed(s, storage, v->q.patch);
        if (v->block) {
            const StructDef& sd = *v->block;
            if (v->builtin == BuiltinVar::In || v->builtin == BuiltinVar::Out) {
                auto it = members.find(v);
                if (it == members.end()) continue;
                for (int m : it->second) c.add(sd.name + "." + sd.members[m].name, sd.members[m].type, -1, 0, false);
                continue;
            }
            Type bt = v->type;
            if (arrayed && bt.isArray()) bt = bt.elem();
            int nBlocks = bt.isArray() ? std::max(1, bt.dims[0]) : 1;
            int loc = sd.blockQ.layout.location;
            bool patch = v->q.patch || sd.blockQ.patch;
            for (int bi = 0; bi < nBlocks; bi++) {
                std::string prefix = v->anonBlock ? "" : sd.name + (bt.isArray() ? "[" + std::to_string(bi) + "]" : "") + ".";
                for (auto& m : sd.members) {
                    if (m.q.layout.location >= 0) loc = m.q.layout.location;
                    c.add(prefix + m.name, m.type, loc, m.q.layout.component, patch || m.q.patch);
                    if (loc >= 0) loc += locationsOf(m.type);
                }
            }
            continue;
        }
        Type t = v->type;
        std::string name = v->name;
        if (arrayed && t.isArray()) {
            t = t.elem();
            // Structure members keep the per-vertex index in their names ("v[0].m", "v[0][1].m"),
            // enumerated for the first vertex only.
            if (t.stripArrays().base == BT::Struct) name += "[0]";
        }
        int loc = v->builtin == BuiltinVar::None ? v->q.layout.location : -1;
        c.add(name, t, loc, v->builtin == BuiltinVar::None ? v->q.layout.component : 0, v->q.patch);
    }
}

}  // namespace

// Elements of the array variable `v` (n elements) that reachable code in `sc` may access:
// constant indices mark one element, anything else marks them all.
std::vector<bool> usedArrayElements(const StageCode& sc, const Variable* v, int n) {
    std::vector<bool> used(n, false);
    bool all = false;
    std::function<void(const Expr*)> visit = [&](const Expr* e) {
        if (!e || all) return;
        if (e->kind == EK::Index && e->args.size() >= 2 && e->args[0]->kind == EK::Var && e->args[0]->var == v) {
            const Expr* ix = e->args[1];
            if (ix->cv && !ix->cv->c.empty() && ix->cv->c[0].i >= 0 && ix->cv->c[0].i < n) used[(size_t)ix->cv->c[0].i] = true;
            else all = true;
            visit(ix);
            return;
        }
        if (e->kind == EK::Var && e->var == v) all = true;
        for (auto* a : e->args) visit(a);
        visit(e->aux);
    };
    std::function<void(const Stmt*)> stmt = [&](const Stmt* s) {
        if (!s) return;
        for (auto* b : s->body) stmt(b);
        visit(s->expr);
        stmt(s->init);
        visit(s->cond);
        stmt(s->condDecl);
        visit(s->incr);
        stmt(s->then);
        stmt(s->els);
        for (auto* x : s->vars) visit(x->init);
    };
    for (auto* f : sc.reachable) stmt(f->body);
    for (auto* g : sc.globals) visit(g->init);
    if (all) used.assign(n, true);
    return used;
}

void reflectProgramInterface(const LinkedProgram& P, Reflection& r) {
    r.programInputs.clear();
    r.programOutputs.clear();
    static const Stage order[] = {Stage::Vertex, Stage::TessControl, Stage::TessEval, Stage::Geometry, Stage::Fragment, Stage::Compute};
    Stage first = Stage::Count, last = Stage::Count;
    for (Stage s : order) {
        if (!P.stages[(int)s].main) continue;
        if (first == Stage::Count) first = s;
        last = s;
    }
    if (first == Stage::Count) return;
    collect(P, first, Storage::In, r.programInputs);
    if (last != Stage::Compute) collect(P, last, Storage::Out, r.programOutputs);
}

}  // namespace glsl
