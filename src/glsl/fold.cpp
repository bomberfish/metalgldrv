#include "glsl/fold.h"

#include <cmath>
#include <cstring>

namespace glsl {

int flatCount(const Type& t) {
    if (!t.dims.empty()) return t.dims[0] * flatCount(t.elem());
    if (t.base == BT::Struct) {
        int n = 0;
        for (auto& m : t.sdef->members) n += flatCount(m.type);
        return n;
    }
    if (t.base == BT::Void || t.isOpaque()) return 0;
    return t.components();
}

double cvF(const Scalar& s, BT b) {
    switch (b) {
        case BT::Float: case BT::Double: return s.f;
        case BT::Int: return (double)(int32_t)s.i;
        case BT::UInt: return (double)(uint32_t)s.u;
        case BT::Bool: return s.b ? 1.0 : 0.0;
        default: return 0;
    }
}
int64_t cvI(const Scalar& s, BT b) {
    switch (b) {
        case BT::Float: case BT::Double: return (int64_t)(int32_t)(int64_t)s.f;
        case BT::Int: return (int32_t)s.i;
        case BT::UInt: return (int32_t)(uint32_t)s.u;
        case BT::Bool: return s.b;
        default: return 0;
    }
}
uint64_t cvU(const Scalar& s, BT b) {
    switch (b) {
        case BT::Float: case BT::Double: return s.f < 0 ? (uint32_t)(int32_t)(int64_t)s.f : (uint32_t)(uint64_t)s.f;
        case BT::Int: return (uint32_t)(int32_t)s.i;
        case BT::UInt: return (uint32_t)s.u;
        case BT::Bool: return s.b;
        default: return 0;
    }
}
bool cvB(const Scalar& s, BT b) {
    switch (b) {
        case BT::Float: case BT::Double: return s.f != 0;
        case BT::Int: return (int32_t)s.i != 0;
        case BT::UInt: return (uint32_t)s.u != 0;
        case BT::Bool: return s.b;
        default: return false;
    }
}

Scalar convertScalar(const Scalar& s, BT from, BT to) {
    Scalar r;
    switch (to) {
        case BT::Float: r.f = (double)(float)cvF(s, from); break;
        case BT::Double: r.f = cvF(s, from); break;
        case BT::Int: r.i = cvI(s, from); break;
        case BT::UInt: r.u = cvU(s, from); break;
        case BT::Bool: r.b = cvB(s, from); break;
        default: break;
    }
    return r;
}

static bool allConst(const std::vector<Expr*>& a) {
    for (auto* e : a) if (!e || !e->cv) return false;
    return true;
}

static Scalar fromF(BT b, double v) {
    Scalar s;
    if (b == BT::Float) s.f = (double)(float)v; else s.f = v;
    return s;
}
static Scalar fromI(int64_t v) { Scalar s; s.i = (int32_t)v; return s; }
static Scalar fromU(uint64_t v) { Scalar s; s.u = (uint32_t)v; return s; }
static Scalar fromB(bool v) { Scalar s; s.b = v; return s; }

CV foldBinary(Op op, const Type& res, const Expr* a, const Expr* b) {
    if (!a->cv || !b->cv) return nullptr;
    const Type& ta = a->type;
    const Type& tb = b->type;
    BT ba = ta.base;
    auto out = std::make_shared<ConstVal>();
    // equality on aggregates
    if (op == Op::Eq || op == Op::Ne) {
        if (a->cv->c.size() != b->cv->c.size()) return nullptr;
        bool eq = true;
        // compare by component as double/int depending on base (aggregates use first base)
        for (size_t i = 0; i < a->cv->c.size(); i++) {
            const Scalar& x = a->cv->c[i];
            const Scalar& y = b->cv->c[i];
            if (std::memcmp(&x, &y, sizeof(double)) != 0 || x.b != y.b) {
                if (ba == BT::Float || ba == BT::Double) { if (x.f != y.f) eq = false; }
                else eq = false;
            }
        }
        out->c.push_back(fromB(op == Op::Eq ? eq : !eq));
        return out;
    }
    if (op == Op::LogAnd || op == Op::LogOr || op == Op::LogXor) {
        bool x = a->cv->c[0].b, y = b->cv->c[0].b;
        out->c.push_back(fromB(op == Op::LogAnd ? (x && y) : op == Op::LogOr ? (x || y) : (x != y)));
        return out;
    }
    // linear algebra multiply: skip folding (rare in constant expressions)
    if (op == Op::Mul && (ta.isMatrix() || tb.isMatrix()) && !(ta.isScalar() || tb.isScalar())) return nullptr;
    int na = (int)a->cv->c.size(), nb = (int)b->cv->c.size();
    int n = std::max(na, nb);
    if (op == Op::Shl || op == Op::Shr) n = na;
    BT rb = res.base;
    BT ob = (op == Op::Lt || op == Op::Gt || op == Op::Le || op == Op::Ge) ? ba : rb;
    for (int i = 0; i < n; i++) {
        const Scalar& x = a->cv->c[na == 1 ? 0 : i];
        const Scalar& y = b->cv->c[nb == 1 ? 0 : i];
        BT bb = tb.base;
        if (ob == BT::Float || ob == BT::Double) {
            double X = cvF(x, ba), Y = cvF(y, bb), R = 0;
            switch (op) {
                case Op::Add: R = X + Y; break;
                case Op::Sub: R = X - Y; break;
                case Op::Mul: R = X * Y; break;
                case Op::Div: R = X / Y; break;
                case Op::Lt: out->c.push_back(fromB(X < Y)); continue;
                case Op::Gt: out->c.push_back(fromB(X > Y)); continue;
                case Op::Le: out->c.push_back(fromB(X <= Y)); continue;
                case Op::Ge: out->c.push_back(fromB(X >= Y)); continue;
                default: return nullptr;
            }
            out->c.push_back(fromF(rb, R));
        } else if (ob == BT::Int) {
            int32_t X = (int32_t)cvI(x, ba), Y = (int32_t)cvI(y, bb);
            int32_t R = 0;
            switch (op) {
                case Op::Add: R = (int32_t)((uint32_t)X + (uint32_t)Y); break;
                case Op::Sub: R = (int32_t)((uint32_t)X - (uint32_t)Y); break;
                case Op::Mul: R = (int32_t)((uint32_t)X * (uint32_t)Y); break;
                case Op::Div: R = Y ? X / Y : 0; break;
                case Op::Mod: R = Y ? X % Y : 0; break;
                case Op::Shl: R = (int32_t)((uint32_t)X << (cvU(y, bb) & 31)); break;
                case Op::Shr: R = X >> (cvU(y, bb) & 31); break;
                case Op::BitAnd: R = X & Y; break;
                case Op::BitOr: R = X | Y; break;
                case Op::BitXor: R = X ^ Y; break;
                case Op::Lt: out->c.push_back(fromB(X < Y)); continue;
                case Op::Gt: out->c.push_back(fromB(X > Y)); continue;
                case Op::Le: out->c.push_back(fromB(X <= Y)); continue;
                case Op::Ge: out->c.push_back(fromB(X >= Y)); continue;
                default: return nullptr;
            }
            out->c.push_back(fromI(R));
        } else if (ob == BT::UInt) {
            uint32_t X = (uint32_t)cvU(x, ba), Y = (uint32_t)cvU(y, bb), R = 0;
            switch (op) {
                case Op::Add: R = X + Y; break;
                case Op::Sub: R = X - Y; break;
                case Op::Mul: R = X * Y; break;
                case Op::Div: R = Y ? X / Y : 0; break;
                case Op::Mod: R = Y ? X % Y : 0; break;
                case Op::Shl: R = X << (Y & 31); break;
                case Op::Shr: R = X >> (Y & 31); break;
                case Op::BitAnd: R = X & Y; break;
                case Op::BitOr: R = X | Y; break;
                case Op::BitXor: R = X ^ Y; break;
                case Op::Lt: out->c.push_back(fromB(X < Y)); continue;
                case Op::Gt: out->c.push_back(fromB(X > Y)); continue;
                case Op::Le: out->c.push_back(fromB(X <= Y)); continue;
                case Op::Ge: out->c.push_back(fromB(X >= Y)); continue;
                default: return nullptr;
            }
            out->c.push_back(fromU(R));
        } else {
            return nullptr;
        }
    }
    return out;
}

CV foldUnary(Op op, const Type& res, const Expr* a) {
    if (!a->cv) return nullptr;
    auto out = std::make_shared<ConstVal>();
    BT b = a->type.base;
    for (auto& s : a->cv->c) {
        switch (op) {
            case Op::Plus: out->c.push_back(s); break;
            case Op::Neg:
                if (b == BT::Float || b == BT::Double) out->c.push_back(fromF(b, -s.f));
                else if (b == BT::Int) out->c.push_back(fromI(-(int64_t)(int32_t)s.i));
                else if (b == BT::UInt) out->c.push_back(fromU((uint32_t)(-(int64_t)s.u)));
                else return nullptr;
                break;
            case Op::Not: out->c.push_back(fromB(!s.b)); break;
            case Op::BitNot:
                if (b == BT::Int) out->c.push_back(fromI(~(int32_t)s.i));
                else out->c.push_back(fromU(~(uint32_t)s.u));
                break;
            default: return nullptr;
        }
    }
    return out;
}

CV foldCtor(const Type& to, const std::vector<Expr*>& args) {
    if (!allConst(args) || args.empty()) return nullptr;
    auto out = std::make_shared<ConstVal>();
    if (to.isArray() || to.base == BT::Struct) {
        for (auto* a : args) out->c.insert(out->c.end(), a->cv->c.begin(), a->cv->c.end());
        return out;
    }
    BT tb = to.base;
    int n = to.components();
    if (args.size() == 1 && args[0]->type.isScalar()) {
        Scalar s = convertScalar(args[0]->cv->c[0], args[0]->type.base, tb);
        if (to.isMatrix()) {
            Scalar zero = convertScalar(Scalar(), BT::Int, tb);
            for (int c = 0; c < to.cols; c++)
                for (int r = 0; r < to.vec; r++) out->c.push_back(c == r ? s : zero);
        } else {
            for (int i = 0; i < n; i++) out->c.push_back(s);
        }
        return out;
    }
    if (args.size() == 1 && args[0]->type.isMatrix() && to.isMatrix()) {
        const Type& ft = args[0]->type;
        for (int c = 0; c < to.cols; c++)
            for (int r = 0; r < to.vec; r++) {
                if (c < ft.cols && r < ft.vec) out->c.push_back(convertScalar(args[0]->cv->c[c * ft.vec + r], ft.base, tb));
                else out->c.push_back(convertScalar(fromI(c == r ? 1 : 0), BT::Int, tb));
            }
        return out;
    }
    for (auto* a : args) {
        for (auto& s : a->cv->c) {
            if ((int)out->c.size() >= n) break;
            out->c.push_back(convertScalar(s, a->type.base, tb));
        }
    }
    if ((int)out->c.size() < n) return nullptr;
    return out;
}

static int elemOffset(const Type& t, int idx) {
    if (t.isArray()) return idx * flatCount(t.elem());
    if (t.isMatrix()) return idx * t.vec;
    return idx;
}

CV foldIndex(const Expr* base, int idx, const Type& res) {
    if (!base->cv) return nullptr;
    int off = elemOffset(base->type, idx);
    int n = flatCount(res);
    if (off < 0 || off + n > (int)base->cv->c.size()) return nullptr;
    auto out = std::make_shared<ConstVal>();
    out->c.assign(base->cv->c.begin() + off, base->cv->c.begin() + off + n);
    return out;
}

CV foldSwizzle(const Expr* base, const uint8_t* swz, int n, const Type& res) {
    if (!base->cv) return nullptr;
    auto out = std::make_shared<ConstVal>();
    for (int i = 0; i < n; i++) {
        if (swz[i] >= base->cv->c.size()) return nullptr;
        out->c.push_back(base->cv->c[swz[i]]);
    }
    return out;
}

CV foldField(const Expr* base, int field, const Type& res) {
    if (!base->cv || !base->type.sdef) return nullptr;
    int off = 0;
    for (int i = 0; i < field; i++) off += flatCount(base->type.sdef->members[i].type);
    int n = flatCount(res);
    auto out = std::make_shared<ConstVal>();
    out->c.assign(base->cv->c.begin() + off, base->cv->c.begin() + off + n);
    return out;
}

CV foldBuiltin(const std::string& name, const Type& ret, const std::vector<Expr*>& args) {
    if (!allConst(args)) return nullptr;
    // Component-wise float functions
    using F1 = double (*)(double);
    static const std::pair<const char*, F1> unary[] = {
        {"radians", [](double x) { return x * M_PI / 180.0; }},
        {"degrees", [](double x) { return x * 180.0 / M_PI; }},
        {"sin", [](double x) { return std::sin(x); }},
        {"cos", [](double x) { return std::cos(x); }},
        {"tan", [](double x) { return std::tan(x); }},
        {"asin", [](double x) { return std::asin(x); }},
        {"acos", [](double x) { return std::acos(x); }},
        {"exp", [](double x) { return std::exp(x); }},
        {"log", [](double x) { return std::log(x); }},
        {"exp2", [](double x) { return std::exp2(x); }},
        {"log2", [](double x) { return std::log2(x); }},
        {"sqrt", [](double x) { return std::sqrt(x); }},
        {"inversesqrt", [](double x) { return 1.0 / std::sqrt(x); }},
        {"floor", [](double x) { return std::floor(x); }},
        {"ceil", [](double x) { return std::ceil(x); }},
        {"trunc", [](double x) { return std::trunc(x); }},
        {"round", [](double x) { return std::round(x); }},
        {"roundEven", [](double x) { return std::nearbyint(x); }},
        {"fract", [](double x) { return x - std::floor(x); }},
    };
    auto out = std::make_shared<ConstVal>();
    BT rb = ret.base;
    if (args.size() == 1 && (args[0]->type.base == BT::Float || args[0]->type.base == BT::Double)) {
        for (auto& u : unary) {
            if (name == u.first) {
                for (auto& s : args[0]->cv->c) out->c.push_back(fromF(rb, u.second(s.f)));
                return out;
            }
        }
    }
    auto comp = [&](int ai, int i) -> const Scalar& {
        auto& c = args[ai]->cv->c;
        return c[c.size() == 1 ? 0 : i];
    };
    int n = ret.isNumeric() ? ret.components() : 0;
    if (name == "abs" || name == "sign") {
        BT b = args[0]->type.base;
        for (auto& s : args[0]->cv->c) {
            if (b == BT::Int) {
                int32_t v = (int32_t)s.i;
                out->c.push_back(fromI(name == "abs" ? (v < 0 ? -v : v) : (v > 0) - (v < 0)));
            } else {
                out->c.push_back(fromF(rb, name == "abs" ? std::fabs(s.f) : (double)((s.f > 0) - (s.f < 0))));
            }
        }
        return out;
    }
    if (name == "min" || name == "max" || name == "pow" || name == "mod" || name == "atan" || name == "step") {
        if (args.size() != 2) return nullptr;
        for (int i = 0; i < n; i++) {
            const Scalar& x = comp(0, i);
            const Scalar& y = comp(1, i);
            BT bx = args[0]->type.base, by = args[1]->type.base;
            if (rb == BT::Int) {
                int64_t X = cvI(x, bx), Y = cvI(y, by);
                out->c.push_back(fromI(name == "min" ? std::min(X, Y) : std::max(X, Y)));
            } else if (rb == BT::UInt) {
                uint64_t X = cvU(x, bx), Y = cvU(y, by);
                out->c.push_back(fromU(name == "min" ? std::min(X, Y) : std::max(X, Y)));
            } else {
                double X = cvF(x, bx), Y = cvF(y, by), R;
                if (name == "min") R = std::min(X, Y);
                else if (name == "max") R = std::max(X, Y);
                else if (name == "pow") R = std::pow(X, Y);
                else if (name == "mod") R = X - Y * std::floor(X / Y);
                else if (name == "atan") R = std::atan2(X, Y);
                else R = Y < X ? 0.0 : 1.0;
                out->c.push_back(fromF(rb, R));
            }
        }
        return out;
    }
    if (name == "clamp" || name == "mix") {
        for (int i = 0; i < n; i++) {
            BT b0 = args[0]->type.base, b1 = args[1]->type.base, b2 = args[2]->type.base;
            if (name == "clamp") {
                if (rb == BT::Int) {
                    int64_t x = cvI(comp(0, i), b0), lo = cvI(comp(1, i), b1), hi = cvI(comp(2, i), b2);
                    out->c.push_back(fromI(std::min(std::max(x, lo), hi)));
                } else if (rb == BT::UInt) {
                    uint64_t x = cvU(comp(0, i), b0), lo = cvU(comp(1, i), b1), hi = cvU(comp(2, i), b2);
                    out->c.push_back(fromU(std::min(std::max(x, lo), hi)));
                } else {
                    double x = cvF(comp(0, i), b0), lo = cvF(comp(1, i), b1), hi = cvF(comp(2, i), b2);
                    out->c.push_back(fromF(rb, std::min(std::max(x, lo), hi)));
                }
            } else {
                if (b2 == BT::Bool) { out->c.push_back(comp(2, i).b ? comp(1, i) : comp(0, i)); continue; }
                double x = cvF(comp(0, i), b0), y = cvF(comp(1, i), b1), a = cvF(comp(2, i), b2);
                out->c.push_back(fromF(rb, x * (1 - a) + y * a));
            }
        }
        return out;
    }
    if (name == "dot" || name == "length") {
        double s = 0;
        auto& x = args[0]->cv->c;
        auto& y = args[name == "dot" ? 1 : 0]->cv->c;
        for (size_t i = 0; i < x.size(); i++) s += x[i].f * y[i].f;
        out->c.push_back(fromF(rb, name == "length" ? std::sqrt(s) : s));
        return out;
    }
    if (name == "normalize") {
        double s = 0;
        for (auto& v : args[0]->cv->c) s += v.f * v.f;
        s = std::sqrt(s);
        for (auto& v : args[0]->cv->c) out->c.push_back(fromF(rb, v.f / s));
        return out;
    }
    if (name == "smoothstep") {
        for (int i = 0; i < n; i++) {
            double e0 = comp(0, i).f, e1 = comp(1, i).f, x = comp(2, i).f;
            double t = std::min(std::max((x - e0) / (e1 - e0), 0.0), 1.0);
            out->c.push_back(fromF(rb, t * t * (3 - 2 * t)));
        }
        return out;
    }
    if (name == "floatBitsToInt" || name == "floatBitsToUint") {
        for (auto& s : args[0]->cv->c) {
            float f = (float)s.f;
            uint32_t u;
            std::memcpy(&u, &f, 4);
            out->c.push_back(name == "floatBitsToInt" ? fromI((int32_t)u) : fromU(u));
        }
        return out;
    }
    if (name == "intBitsToFloat" || name == "uintBitsToFloat") {
        for (auto& s : args[0]->cv->c) {
            uint32_t u = (uint32_t)s.u;
            float f;
            std::memcpy(&f, &u, 4);
            out->c.push_back(fromF(BT::Float, f));
        }
        return out;
    }
    return nullptr;
}

}  // namespace glsl
