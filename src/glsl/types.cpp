#include "glsl/types.h"

#include <unordered_map>

#include "glsl/gl_enums.h"

namespace glsl {

bool Type::operator==(const Type& o) const {
    if (base != o.base || vec != o.vec || cols != o.cols || dims != o.dims) return false;
    if (base == BT::Struct) return sdef == o.sdef;
    if (base == BT::Subroutine) return subr == o.subr;
    if (base == BT::Sampler || base == BT::Image)
        return dim == o.dim && arrayed == o.arrayed && shadow == o.shadow && ms == o.ms && sampled == o.sampled;
    return true;
}

bool Type::containsOpaque() const {
    if (isOpaque()) return true;
    if (base == BT::Struct && sdef)
        for (auto& m : sdef->members)
            if (m.type.containsOpaque()) return true;
    return false;
}

static const char* baseName(BT b) {
    switch (b) {
        case BT::Void: return "void";
        case BT::Bool: return "bool";
        case BT::Int: return "int";
        case BT::UInt: return "uint";
        case BT::Float: return "float";
        case BT::Double: return "double";
        case BT::AtomicUint: return "atomic_uint";
        case BT::Error: return "<error>";
        default: return "?";
    }
}

static const char* vecPrefix(BT b) {
    switch (b) {
        case BT::Bool: return "b";
        case BT::Int: return "i";
        case BT::UInt: return "u";
        case BT::Double: return "d";
        default: return "";
    }
}

static std::string dimSuffix(const Type& t) {
    std::string s;
    switch (t.dim) {
        case Dim::D1: s = "1D"; break;
        case Dim::D2: s = "2D"; break;
        case Dim::D3: s = "3D"; break;
        case Dim::Cube: s = "Cube"; break;
        case Dim::Rect: s = "2DRect"; break;
        case Dim::Buffer: s = "Buffer"; break;
        default: break;
    }
    if (t.ms) s += "MS";
    if (t.arrayed) s += "Array";
    if (t.shadow) s += "Shadow";
    return s;
}

std::string Type::str() const {
    std::string s;
    if (base == BT::Struct) s = sdef ? sdef->name : "struct";
    else if (base == BT::Subroutine) s = subr ? subr->name : "subroutine";
    else if (base == BT::Sampler) s = std::string(sampled == BT::Int ? "i" : sampled == BT::UInt ? "u" : "") + "sampler" + dimSuffix(*this);
    else if (base == BT::Image) s = std::string(sampled == BT::Int ? "i" : sampled == BT::UInt ? "u" : "") + "image" + dimSuffix(*this);
    else if (cols) {
        s = std::string(base == BT::Double ? "dmat" : "mat") + std::to_string(cols);
        if (cols != vec) s += "x" + std::to_string(vec);
    } else if (vec > 1) s = std::string(vecPrefix(base)) + "vec" + std::to_string(vec);
    else s = baseName(base);
    for (int d : dims) s += d ? "[" + std::to_string(d) + "]" : "[]";
    return s;
}

std::string typeMangle(const Type& t) { return t.str(); }

bool isIntegerType(const Type& t) { return t.base == BT::Int || t.base == BT::UInt; }

uint32_t glTypeEnum(const Type& t0) {
    Type t = t0.stripArrays();
    using namespace gl;
    switch (t.base) {
        case BT::Float:
        case BT::Double: {
            bool d = t.base == BT::Double;
            if (t.cols) {
                static const uint32_t fm[3][3] = {{FLOAT_MAT2, FLOAT_MAT2x3, FLOAT_MAT2x4},
                                                  {FLOAT_MAT3x2, FLOAT_MAT3, FLOAT_MAT3x4},
                                                  {FLOAT_MAT4x2, FLOAT_MAT4x3, FLOAT_MAT4}};
                static const uint32_t dm[3][3] = {{DOUBLE_MAT2, DOUBLE_MAT2x3, DOUBLE_MAT2x4},
                                                  {DOUBLE_MAT3x2, DOUBLE_MAT3, DOUBLE_MAT3x4},
                                                  {DOUBLE_MAT4x2, DOUBLE_MAT4x3, DOUBLE_MAT4}};
                return (d ? dm : fm)[t.cols - 2][t.vec - 2];
            }
            static const uint32_t fv[4] = {FLOAT, FLOAT_VEC2, FLOAT_VEC3, FLOAT_VEC4};
            static const uint32_t dv[4] = {DOUBLE, DOUBLE_VEC2, DOUBLE_VEC3, DOUBLE_VEC4};
            return (d ? dv : fv)[t.vec - 1];
        }
        case BT::Int: { static const uint32_t v[4] = {INT, INT_VEC2, INT_VEC3, INT_VEC4}; return v[t.vec - 1]; }
        case BT::UInt: { static const uint32_t v[4] = {UNSIGNED_INT, UNSIGNED_INT_VEC2, UNSIGNED_INT_VEC3, UNSIGNED_INT_VEC4}; return v[t.vec - 1]; }
        case BT::Bool: { static const uint32_t v[4] = {BOOL, BOOL_VEC2, BOOL_VEC3, BOOL_VEC4}; return v[t.vec - 1]; }
        case BT::AtomicUint: return UNSIGNED_INT_ATOMIC_COUNTER;
        case BT::Sampler: {
            int k = t.sampled == BT::Int ? 1 : t.sampled == BT::UInt ? 2 : 0;
            if (t.shadow) {
                switch (t.dim) {
                    case Dim::D1: return t.arrayed ? SAMPLER_1D_ARRAY_SHADOW : SAMPLER_1D_SHADOW;
                    case Dim::D2: return t.arrayed ? SAMPLER_2D_ARRAY_SHADOW : SAMPLER_2D_SHADOW;
                    case Dim::Cube: return t.arrayed ? SAMPLER_CUBE_MAP_ARRAY_SHADOW : SAMPLER_CUBE_SHADOW;
                    case Dim::Rect: return SAMPLER_2D_RECT_SHADOW;
                    default: return 0;
                }
            }
            static const uint32_t tbl[][3] = {
                {SAMPLER_1D, INT_SAMPLER_1D, UNSIGNED_INT_SAMPLER_1D},
                {SAMPLER_2D, INT_SAMPLER_2D, UNSIGNED_INT_SAMPLER_2D},
                {SAMPLER_3D, INT_SAMPLER_3D, UNSIGNED_INT_SAMPLER_3D},
                {SAMPLER_CUBE, INT_SAMPLER_CUBE, UNSIGNED_INT_SAMPLER_CUBE},
                {SAMPLER_2D_RECT, INT_SAMPLER_2D_RECT, UNSIGNED_INT_SAMPLER_2D_RECT},
                {SAMPLER_BUFFER, INT_SAMPLER_BUFFER, UNSIGNED_INT_SAMPLER_BUFFER},
                {SAMPLER_1D_ARRAY, INT_SAMPLER_1D_ARRAY, UNSIGNED_INT_SAMPLER_1D_ARRAY},
                {SAMPLER_2D_ARRAY, INT_SAMPLER_2D_ARRAY, UNSIGNED_INT_SAMPLER_2D_ARRAY},
                {SAMPLER_CUBE_MAP_ARRAY, INT_SAMPLER_CUBE_MAP_ARRAY, UNSIGNED_INT_SAMPLER_CUBE_MAP_ARRAY},
                {SAMPLER_2D_MULTISAMPLE, INT_SAMPLER_2D_MULTISAMPLE, UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE},
                {SAMPLER_2D_MULTISAMPLE_ARRAY, INT_SAMPLER_2D_MULTISAMPLE_ARRAY, UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY},
            };
            int row = 0;
            if (t.ms) row = t.arrayed ? 10 : 9;
            else switch (t.dim) {
                case Dim::D1: row = t.arrayed ? 6 : 0; break;
                case Dim::D2: row = t.arrayed ? 7 : 1; break;
                case Dim::D3: row = 2; break;
                case Dim::Cube: row = t.arrayed ? 8 : 3; break;
                case Dim::Rect: row = 4; break;
                case Dim::Buffer: row = 5; break;
                default: break;
            }
            return tbl[row][k];
        }
        case BT::Image: {
            int idx = 0;
            if (t.ms) idx = t.arrayed ? 10 : 9;
            else switch (t.dim) {
                case Dim::D1: idx = t.arrayed ? 6 : 0; break;
                case Dim::D2: idx = t.arrayed ? 7 : 1; break;
                case Dim::D3: idx = 2; break;
                case Dim::Rect: idx = 3; break;
                case Dim::Cube: idx = t.arrayed ? 8 : 4; break;
                case Dim::Buffer: idx = 5; break;
                default: break;
            }
            uint32_t base = t.sampled == BT::Int ? INT_IMAGE_1D : t.sampled == BT::UInt ? UNSIGNED_INT_IMAGE_1D : IMAGE_1D;
            return base + idx;
        }
        default: return 0;
    }
}

uint32_t textureTargetFor(const Type& t) {
    using namespace gl;
    if (t.ms) return t.arrayed ? TEXTURE_2D_MULTISAMPLE_ARRAY : TEXTURE_2D_MULTISAMPLE;
    switch (t.dim) {
        case Dim::D1: return t.arrayed ? TEXTURE_1D_ARRAY : TEXTURE_1D;
        case Dim::D2: return t.arrayed ? TEXTURE_2D_ARRAY : TEXTURE_2D;
        case Dim::D3: return TEXTURE_3D;
        case Dim::Cube: return t.arrayed ? TEXTURE_CUBE_MAP_ARRAY : TEXTURE_CUBE_MAP;
        case Dim::Rect: return TEXTURE_RECTANGLE;
        case Dim::Buffer: return TEXTURE_BUFFER;
        default: return 0;
    }
}

static std::unordered_map<std::string, Type> buildTypeKeywords() {
    std::unordered_map<std::string, Type> m;
    m["void"] = Type::voidT();
    m["bool"] = Type::scalar(BT::Bool);
    m["int"] = Type::scalar(BT::Int);
    m["uint"] = Type::scalar(BT::UInt);
    m["float"] = Type::scalar(BT::Float);
    m["double"] = Type::scalar(BT::Double);
    m["atomic_uint"] = Type::scalar(BT::AtomicUint);
    for (int n = 2; n <= 4; n++) {
        std::string s = std::to_string(n);
        m["vec" + s] = Type::vector(BT::Float, n);
        m["dvec" + s] = Type::vector(BT::Double, n);
        m["ivec" + s] = Type::vector(BT::Int, n);
        m["uvec" + s] = Type::vector(BT::UInt, n);
        m["bvec" + s] = Type::vector(BT::Bool, n);
        m["mat" + s] = Type::matrix(BT::Float, n, n);
        m["dmat" + s] = Type::matrix(BT::Double, n, n);
        for (int r = 2; r <= 4; r++) {
            m["mat" + s + "x" + std::to_string(r)] = Type::matrix(BT::Float, n, r);
            m["dmat" + s + "x" + std::to_string(r)] = Type::matrix(BT::Double, n, r);
        }
    }
    struct D { const char* n; Dim d; bool arr, ms; };
    const D dims[] = {{"1D", Dim::D1, false, false},      {"2D", Dim::D2, false, false},       {"3D", Dim::D3, false, false},
                      {"Cube", Dim::Cube, false, false},  {"2DRect", Dim::Rect, false, false}, {"Buffer", Dim::Buffer, false, false},
                      {"1DArray", Dim::D1, true, false},  {"2DArray", Dim::D2, true, false},   {"CubeArray", Dim::Cube, true, false},
                      {"2DMS", Dim::D2, false, true},     {"2DMSArray", Dim::D2, true, true}};
    const std::pair<const char*, BT> pfx[] = {{"", BT::Float}, {"i", BT::Int}, {"u", BT::UInt}};
    for (auto& p : pfx)
        for (auto& d : dims) {
            Type s;
            s.base = BT::Sampler;
            s.dim = d.d;
            s.arrayed = d.arr;
            s.ms = d.ms;
            s.sampled = p.second;
            m[std::string(p.first) + "sampler" + d.n] = s;
            Type i = s;
            i.base = BT::Image;
            m[std::string(p.first) + "image" + d.n] = i;
        }
    const D shadows[] = {{"1D", Dim::D1, false, false}, {"2D", Dim::D2, false, false}, {"Cube", Dim::Cube, false, false},
                         {"2DRect", Dim::Rect, false, false}, {"1DArray", Dim::D1, true, false}, {"2DArray", Dim::D2, true, false},
                         {"CubeArray", Dim::Cube, true, false}};
    for (auto& d : shadows) {
        Type s;
        s.base = BT::Sampler;
        s.dim = d.d;
        s.arrayed = d.arr;
        s.shadow = true;
        m[std::string("sampler") + d.n + "Shadow"] = s;
    }
    return m;
}

const Type* typeKeyword(const std::string& name) {
    static const auto m = buildTypeKeywords();
    auto it = m.find(name);
    return it == m.end() ? nullptr : &it->second;
}

}  // namespace glsl
