// MSL prelude shared by every generated shader.
#include <string>

namespace glsl {

extern const char* const kMSLPrelude;

const char* const kMSLPrelude = R"MSL(#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// fp64 stub: doubles are emulated with float precision until the soft-float
// library lands. Buffer layout stays IEEE binary64.
// ---------------------------------------------------------------------------
typedef float mgl_f64;
typedef float2 mgl_f64v2;
typedef float3 mgl_f64v3;
typedef float4 mgl_f64v4;
typedef float2x2 mgl_f64m2x2;
typedef float2x3 mgl_f64m2x3;
typedef float2x4 mgl_f64m2x4;
typedef float3x2 mgl_f64m3x2;
typedef float3x3 mgl_f64m3x3;
typedef float3x4 mgl_f64m3x4;
typedef float4x2 mgl_f64m4x2;
typedef float4x3 mgl_f64m4x3;
typedef float4x4 mgl_f64m4x4;

inline float mgl_f64_from_bits(uint2 b) {
    uint hi = b.y, lo = b.x;
    uint sign = hi & 0x80000000u;
    int e = int((hi >> 20) & 0x7ffu);
    uint mant = ((hi & 0xfffffu) << 3) | (lo >> 29);
    if (e == 0) return as_type<float>(sign);
    if (e == 0x7ff) return as_type<float>(sign | 0x7f800000u | (mant != 0u ? 0x400000u : 0u));
    int fe = e - 1023 + 127;
    if (fe >= 255) return as_type<float>(sign | 0x7f800000u);
    if (fe <= 0) return as_type<float>(sign);
    return as_type<float>(sign | (uint(fe) << 23) | mant);
}
inline uint2 mgl_f64_to_bits(float f) {
    uint u = as_type<uint>(f);
    uint sign = u & 0x80000000u;
    int e = int((u >> 23) & 0xffu);
    uint m = u & 0x7fffffu;
    if (e == 0) return uint2(0u, sign);
    if (e == 255) return uint2(0u, sign | 0x7ff00000u | (m != 0u ? 0x80000u : 0u));
    uint de = uint(e - 127 + 1023);
    return uint2(m << 29, sign | (de << 20) | (m >> 3));
}
inline mgl_f64 mgl_pack_f64(uint2 v) { return mgl_f64_from_bits(v); }
inline uint2 mgl_unpack_f64(mgl_f64 v) { return mgl_f64_to_bits(v); }

// ---------------------------------------------------------------------------
// GLSL arrays have value semantics
// ---------------------------------------------------------------------------
template <typename T, int N>
struct mgl_array {
    T e[N];
    thread T& operator[](int i) thread { return e[i]; }
    const thread T& operator[](int i) const thread { return e[i]; }
    threadgroup T& operator[](int i) threadgroup { return e[i]; }
    const threadgroup T& operator[](int i) const threadgroup { return e[i]; }
    device T& operator[](int i) device { return e[i]; }
    const device T& operator[](int i) const device { return e[i]; }
    const constant T& operator[](int i) const constant { return e[i]; }
};

// ---------------------------------------------------------------------------
// Byte-offset buffer access (std140 / std430 / default uniform block)
// ---------------------------------------------------------------------------
#define MGL_LDST(T, ST)                                                                                       \
    inline T mgl_ld_##T(constant uchar* p, uint o) { return T(*(constant ST*)(p + o)); }                      \
    inline T mgl_ld_##T(device const uchar* p, uint o) { return T(*(device const ST*)(p + o)); }               \
    inline T mgl_ld_##T(threadgroup const uchar* p, uint o) { return T(*(threadgroup const ST*)(p + o)); }     \
    inline void mgl_st_##T(device uchar* p, uint o, T v) { *(device ST*)(p + o) = ST(v); }
MGL_LDST(float, float)
MGL_LDST(float2, float2)
MGL_LDST(float3, packed_float3)
MGL_LDST(float4, float4)
MGL_LDST(int, int)
MGL_LDST(int2, int2)
MGL_LDST(int3, packed_int3)
MGL_LDST(int4, int4)
MGL_LDST(uint, uint)
MGL_LDST(uint2, uint2)
MGL_LDST(uint3, packed_uint3)
MGL_LDST(uint4, uint4)

#define MGL_LDST_BOOL(T, U, PU)                                                                                \
    inline T mgl_ld_##T(constant uchar* p, uint o) { return U(*(constant PU*)(p + o)) != U(0u); }               \
    inline T mgl_ld_##T(device const uchar* p, uint o) { return U(*(device const PU*)(p + o)) != U(0u); }        \
    inline void mgl_st_##T(device uchar* p, uint o, T v) { *(device PU*)(p + o) = PU(select(U(0u), U(1u), v)); }
MGL_LDST_BOOL(bool, uint, uint)
MGL_LDST_BOOL(bool2, uint2, uint2)
MGL_LDST_BOOL(bool3, uint3, packed_uint3)
MGL_LDST_BOOL(bool4, uint4, uint4)

#define MGL_LDST_F64(AS)                                                                                       \
    inline mgl_f64 mgl_ld_mgl_f64(AS const uchar* p, uint o) { return mgl_f64_from_bits(*(AS const uint2*)(p + o)); } \
    inline mgl_f64v2 mgl_ld_mgl_f64v2(AS const uchar* p, uint o) { return mgl_f64v2(mgl_ld_mgl_f64(p, o), mgl_ld_mgl_f64(p, o + 8)); } \
    inline mgl_f64v3 mgl_ld_mgl_f64v3(AS const uchar* p, uint o) { return mgl_f64v3(mgl_ld_mgl_f64(p, o), mgl_ld_mgl_f64(p, o + 8), mgl_ld_mgl_f64(p, o + 16)); } \
    inline mgl_f64v4 mgl_ld_mgl_f64v4(AS const uchar* p, uint o) { return mgl_f64v4(mgl_ld_mgl_f64(p, o), mgl_ld_mgl_f64(p, o + 8), mgl_ld_mgl_f64(p, o + 16), mgl_ld_mgl_f64(p, o + 24)); }
MGL_LDST_F64(constant)
MGL_LDST_F64(device)
inline void mgl_st_mgl_f64(device uchar* p, uint o, mgl_f64 v) { *(device uint2*)(p + o) = mgl_f64_to_bits(v); }
inline void mgl_st_mgl_f64v2(device uchar* p, uint o, mgl_f64v2 v) { for (int i = 0; i < 2; i++) mgl_st_mgl_f64(p, o + 8 * i, v[i]); }
inline void mgl_st_mgl_f64v3(device uchar* p, uint o, mgl_f64v3 v) { for (int i = 0; i < 3; i++) mgl_st_mgl_f64(p, o + 8 * i, v[i]); }
inline void mgl_st_mgl_f64v4(device uchar* p, uint o, mgl_f64v4 v) { for (int i = 0; i < 4; i++) mgl_st_mgl_f64(p, o + 8 * i, v[i]); }

// strided vectors (columns of row-major matrices)
#define MGL_LDST_STRIDED(S, V2, V3, V4)                                                                       \
    template <typename P> V2 mgl_ldcs_##V2(P p, uint o, uint s) { return V2(mgl_ld_##S(p, o), mgl_ld_##S(p, o + s)); } \
    template <typename P> V3 mgl_ldcs_##V3(P p, uint o, uint s) { return V3(mgl_ld_##S(p, o), mgl_ld_##S(p, o + s), mgl_ld_##S(p, o + 2 * s)); } \
    template <typename P> V4 mgl_ldcs_##V4(P p, uint o, uint s) { return V4(mgl_ld_##S(p, o), mgl_ld_##S(p, o + s), mgl_ld_##S(p, o + 2 * s), mgl_ld_##S(p, o + 3 * s)); } \
    inline void mgl_stcs_##V2(device uchar* p, uint o, uint s, V2 v) { for (int i = 0; i < 2; i++) mgl_st_##S(p, o + i * s, v[i]); } \
    inline void mgl_stcs_##V3(device uchar* p, uint o, uint s, V3 v) { for (int i = 0; i < 3; i++) mgl_st_##S(p, o + i * s, v[i]); } \
    inline void mgl_stcs_##V4(device uchar* p, uint o, uint s, V4 v) { for (int i = 0; i < 4; i++) mgl_st_##S(p, o + i * s, v[i]); }
MGL_LDST_STRIDED(float, float2, float3, float4)

// ---------------------------------------------------------------------------
// GLSL builtin helpers
// ---------------------------------------------------------------------------
template <typename T> inline T mgl_mod(T x, T y) { return x - y * floor(x / y); }

template <typename T, int C, int R> inline matrix<T, C, R> mgl_mcm(matrix<T, C, R> a, matrix<T, C, R> b) {
    matrix<T, C, R> r;
    for (int i = 0; i < C; i++) r[i] = a[i] * b[i];
    return r;
}
template <typename T, int C, int R> inline matrix<T, C, R> mgl_outer(vec<T, R> c, vec<T, C> r) {
    matrix<T, C, R> m;
    for (int i = 0; i < C; i++) m[i] = c * r[i];
    return m;
}
template <typename T, int C, int R> inline matrix<T, C, R> mgl_mdiag(T s) {
    matrix<T, C, R> m;
    for (int c = 0; c < C; c++)
        for (int r = 0; r < R; r++) m[c][r] = c == r ? s : T(0);
    return m;
}
template <typename T, int C, int R, int C2, int R2> inline matrix<T, C, R> mgl_mresize(matrix<T, C2, R2> s) {
    matrix<T, C, R> m;
    for (int c = 0; c < C; c++)
        for (int r = 0; r < R; r++) m[c][r] = (c < C2 && r < R2) ? s[c][r] : (c == r ? T(1) : T(0));
    return m;
}
template <typename T, int C, int R> inline matrix<T, C, R> mgl_madd(matrix<T, C, R> m, T s) { for (int i = 0; i < C; i++) m[i] += s; return m; }
template <typename T, int C, int R> inline matrix<T, C, R> mgl_madd(T s, matrix<T, C, R> m) { for (int i = 0; i < C; i++) m[i] += s; return m; }
template <typename T, int C, int R> inline matrix<T, C, R> mgl_msub(matrix<T, C, R> m, T s) { for (int i = 0; i < C; i++) m[i] -= s; return m; }
template <typename T, int C, int R> inline matrix<T, C, R> mgl_msub(T s, matrix<T, C, R> m) { for (int i = 0; i < C; i++) m[i] = s - m[i]; return m; }
template <typename T, int C, int R> inline matrix<T, C, R> mgl_mdiv(matrix<T, C, R> m, T s) { for (int i = 0; i < C; i++) m[i] /= s; return m; }
template <typename T, int C, int R> inline matrix<T, C, R> mgl_mdiv(T s, matrix<T, C, R> m) { for (int i = 0; i < C; i++) m[i] = s / m[i]; return m; }
template <typename T, int C, int R> inline matrix<T, C, R> mgl_mdiv(matrix<T, C, R> a, matrix<T, C, R> b) { for (int i = 0; i < C; i++) a[i] /= b[i]; return a; }
template <typename T, int C, int R> inline matrix<T, C, R> mgl_mneg(matrix<T, C, R> m) { for (int i = 0; i < C; i++) m[i] = -m[i]; return m; }

inline float2x2 mgl_inverse(float2x2 m) {
    float d = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    return float2x2(float2(m[1][1], -m[0][1]), float2(-m[1][0], m[0][0])) * (1.0f / d);
}
inline float3x3 mgl_inverse(float3x3 m) {
    float3 a = m[0], b = m[1], c = m[2];
    float3 r0 = cross(b, c), r1 = cross(c, a), r2 = cross(a, b);
    float d = dot(a, r0);
    return transpose(float3x3(r0, r1, r2)) * (1.0f / d);
}
inline float4x4 mgl_inverse(float4x4 m) {
    float a00 = m[0][0], a01 = m[0][1], a02 = m[0][2], a03 = m[0][3];
    float a10 = m[1][0], a11 = m[1][1], a12 = m[1][2], a13 = m[1][3];
    float a20 = m[2][0], a21 = m[2][1], a22 = m[2][2], a23 = m[2][3];
    float a30 = m[3][0], a31 = m[3][1], a32 = m[3][2], a33 = m[3][3];
    float b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10;
    float b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
    float b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30;
    float b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
    float det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    float id = 1.0f / det;
    return float4x4(
        float4(a11 * b11 - a12 * b10 + a13 * b09, a02 * b10 - a01 * b11 - a03 * b09, a31 * b05 - a32 * b04 + a33 * b03, a22 * b04 - a21 * b05 - a23 * b03) * id,
        float4(a12 * b08 - a10 * b11 - a13 * b07, a00 * b11 - a02 * b08 + a03 * b07, a32 * b02 - a30 * b05 - a33 * b01, a20 * b05 - a22 * b02 + a23 * b01) * id,
        float4(a10 * b10 - a11 * b08 + a13 * b06, a01 * b08 - a00 * b10 - a03 * b06, a30 * b04 - a31 * b02 + a33 * b00, a21 * b02 - a20 * b04 - a23 * b00) * id,
        float4(a11 * b07 - a10 * b09 - a12 * b06, a00 * b09 - a01 * b07 + a02 * b06, a31 * b01 - a30 * b03 - a32 * b00, a20 * b03 - a21 * b01 + a22 * b00) * id);
}

template <typename T> inline bool mgl_eq(T a, T b) { return a == b; }
template <typename T, int N> inline bool mgl_eq(vec<T, N> a, vec<T, N> b) { return all(a == b); }
template <typename T, int C, int R> inline bool mgl_eq(matrix<T, C, R> a, matrix<T, C, R> b) {
    for (int i = 0; i < C; i++)
        if (any(a[i] != b[i])) return false;
    return true;
}
template <typename T, int N> inline bool mgl_eq(mgl_array<T, N> a, mgl_array<T, N> b) {
    for (int i = 0; i < N; i++)
        if (!mgl_eq(a[i], b[i])) return false;
    return true;
}

inline int mgl_findLSB(uint x) { return x == 0u ? -1 : int(ctz(x)); }
inline int mgl_findLSB(int x) { return mgl_findLSB(uint(x)); }
inline int mgl_findMSB(uint x) { return x == 0u ? -1 : 31 - int(clz(x)); }
inline int mgl_findMSB(int x) { return mgl_findMSB(x < 0 ? ~uint(x) : uint(x)); }
template <typename T, int N> inline vec<int, N> mgl_findLSB(vec<T, N> v) { vec<int, N> r; for (int i = 0; i < N; i++) r[i] = mgl_findLSB(v[i]); return r; }
template <typename T, int N> inline vec<int, N> mgl_findMSB(vec<T, N> v) { vec<int, N> r; for (int i = 0; i < N; i++) r[i] = mgl_findMSB(v[i]); return r; }
template <typename T> inline T mgl_uaddCarry(T x, T y, thread T& c) { T r = x + y; c = select(T(0u), T(1u), r < x); return r; }
template <typename T> inline T mgl_usubBorrow(T x, T y, thread T& b) { b = select(T(0u), T(1u), x < y); return x - y; }
template <typename T> inline void mgl_mulExtended(T x, T y, thread T& msb, thread T& lsb) { msb = mulhi(x, y); lsb = x * y; }

#define MGL_ATOMICS(AS)                                                                                         \
    inline int mgl_atomic_cas(AS atomic_int* p, int c, int v) {                                                  \
        int e = c;                                                                                               \
        while (!atomic_compare_exchange_weak_explicit(p, &e, v, memory_order_relaxed, memory_order_relaxed)) {  \
            if (e != c) break;                                                                                   \
            e = c;                                                                                               \
        }                                                                                                        \
        return e;                                                                                                \
    }                                                                                                            \
    inline uint mgl_atomic_cas(AS atomic_uint* p, uint c, uint v) {                                              \
        uint e = c;                                                                                              \
        while (!atomic_compare_exchange_weak_explicit(p, &e, v, memory_order_relaxed, memory_order_relaxed)) {  \
            if (e != c) break;                                                                                   \
            e = c;                                                                                               \
        }                                                                                                        \
        return e;                                                                                                \
    }
MGL_ATOMICS(device)
MGL_ATOMICS(threadgroup)

template <typename Tex, typename C, typename T> inline T mgl_img_cas(Tex t, C coord, T cmp, T v) {
    vec<T, 4> e = vec<T, 4>(cmp);
    while (!t.atomic_compare_exchange_weak(coord, &e, vec<T, 4>(v))) {
        if (e.x != cmp) break;
        e = vec<T, 4>(cmp);
    }
    return e.x;
}
template <typename Tex, typename C, typename T> inline T mgl_img_cas_a(Tex t, C coord, uint layer, T cmp, T v) {
    vec<T, 4> e = vec<T, 4>(cmp);
    while (!t.atomic_compare_exchange_weak(coord, layer, &e, vec<T, 4>(v))) {
        if (e.x != cmp) break;
        e = vec<T, 4>(cmp);
    }
    return e.x;
}

inline bool mgl_alpha_pass(float a, uint func, float ref) {
    switch (func) {
        case 0x0200u: return false;
        case 0x0201u: return a < ref;
        case 0x0202u: return a == ref;
        case 0x0203u: return a <= ref;
        case 0x0204u: return a > ref;
        case 0x0205u: return a != ref;
        case 0x0206u: return a >= ref;
        default: return true;
    }
}

// Runtime system values (mirrors glsl::MGLSys)
struct MGLSys {
    float depthRangeNear, depthRangeFar;
    uint depthMode;
    uint clipDistanceMask;
    float4 viewport;
    float pointSize;
    uint flags;
    int baseVertex, baseInstance;
    int drawID, firstVertex;
    uint sampleMask;
    float alphaRef;
    uint alphaFunc;
    float minSampleShading;
    float renderTargetHeight;
    uint pad1;
    float4 currentAttrib[32];
};
)MSL";

}  // namespace glsl
