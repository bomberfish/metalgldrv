// MSL prelude, emitted on demand: each chunk is included only when the generated
// shader references one of the symbols it defines (Metal front-end time is
// dominated by parsing, so unused helpers cost real compile time).
#include <regex>
#include <set>
#include <string>
#include <vector>

namespace glsl {

namespace {

#if __has_include(<fp64_msl.inc>)
const char* const kFp64Library =
#include <fp64_msl.inc>
    ;
#else
const char* const kFp64Library = "// fp64 library (fp64_msl.inc) not available in this build\n";
#endif

const char* const kMultipassLibrary =
#include <multipass_msl.inc>
    ;

// Double matrices: C columns of vec<ulong, R>; all arithmetic via the mgl_f64_* library.
const char* const kDmatHelpers = R"DMAT(template <int C, int R>
struct mgl_dmat {
    vec<ulong, R> c[C];
    thread vec<ulong, R>& operator[](int i) thread { return c[i]; }
    const thread vec<ulong, R>& operator[](int i) const thread { return c[i]; }
    threadgroup vec<ulong, R>& operator[](int i) threadgroup { return c[i]; }
    const threadgroup vec<ulong, R>& operator[](int i) const threadgroup { return c[i]; }
    device vec<ulong, R>& operator[](int i) device { return c[i]; }
    const device vec<ulong, R>& operator[](int i) const device { return c[i]; }
    const constant vec<ulong, R>& operator[](int i) const constant { return c[i]; }
};
template <int C, int R> inline mgl_dmat<C, R> mgl_dmat_add(mgl_dmat<C, R> a, mgl_dmat<C, R> b) { for (int i = 0; i < C; i++) a[i] = mgl_f64_add(a[i], b[i]); return a; }
template <int C, int R> inline mgl_dmat<C, R> mgl_dmat_sub(mgl_dmat<C, R> a, mgl_dmat<C, R> b) { for (int i = 0; i < C; i++) a[i] = mgl_f64_sub(a[i], b[i]); return a; }
template <int C, int R> inline mgl_dmat<C, R> mgl_dmat_div(mgl_dmat<C, R> a, mgl_dmat<C, R> b) { for (int i = 0; i < C; i++) a[i] = mgl_f64_div(a[i], b[i]); return a; }
template <int C, int R> inline mgl_dmat<C, R> mgl_dmat_mcm(mgl_dmat<C, R> a, mgl_dmat<C, R> b) { for (int i = 0; i < C; i++) a[i] = mgl_f64_mul(a[i], b[i]); return a; }
template <int C, int R> inline mgl_dmat<C, R> mgl_dmat_adds(mgl_dmat<C, R> a, ulong s) { for (int i = 0; i < C; i++) a[i] = mgl_f64_add(a[i], vec<ulong, R>(s)); return a; }
template <int C, int R> inline mgl_dmat<C, R> mgl_dmat_adds(ulong s, mgl_dmat<C, R> a) { return mgl_dmat_adds(a, s); }
template <int C, int R> inline mgl_dmat<C, R> mgl_dmat_subs(mgl_dmat<C, R> a, ulong s) { for (int i = 0; i < C; i++) a[i] = mgl_f64_sub(a[i], vec<ulong, R>(s)); return a; }
template <int C, int R> inline mgl_dmat<C, R> mgl_dmat_subs(ulong s, mgl_dmat<C, R> a) { for (int i = 0; i < C; i++) a[i] = mgl_f64_sub(vec<ulong, R>(s), a[i]); return a; }
template <int C, int R> inline mgl_dmat<C, R> mgl_dmat_muls(mgl_dmat<C, R> a, ulong s) { for (int i = 0; i < C; i++) a[i] = mgl_f64_mul(a[i], vec<ulong, R>(s)); return a; }
template <int C, int R> inline mgl_dmat<C, R> mgl_dmat_muls(ulong s, mgl_dmat<C, R> a) { return mgl_dmat_muls(a, s); }
template <int C, int R> inline mgl_dmat<C, R> mgl_dmat_divs(mgl_dmat<C, R> a, ulong s) { for (int i = 0; i < C; i++) a[i] = mgl_f64_div(a[i], vec<ulong, R>(s)); return a; }
template <int C, int R> inline mgl_dmat<C, R> mgl_dmat_divs(ulong s, mgl_dmat<C, R> a) { for (int i = 0; i < C; i++) a[i] = mgl_f64_div(vec<ulong, R>(s), a[i]); return a; }
// ++/-- on double matrices are emitted as mgl_f64_add/sub(x, 1.0)
template <int C, int R> inline mgl_dmat<C, R> mgl_f64_add(mgl_dmat<C, R> a, ulong s) { return mgl_dmat_adds(a, s); }
template <int C, int R> inline mgl_dmat<C, R> mgl_f64_sub(mgl_dmat<C, R> a, ulong s) { return mgl_dmat_subs(a, s); }
template <int C, int R> inline mgl_dmat<C, R> mgl_dmat_neg(mgl_dmat<C, R> a) { for (int i = 0; i < C; i++) a[i] = mgl_f64_neg(a[i]); return a; }
template <int C, int R> inline vec<ulong, R> mgl_dmat_mv(mgl_dmat<C, R> m, vec<ulong, C> v) {
    vec<ulong, R> r = mgl_f64_mul(m[0], vec<ulong, R>(v[0]));
    for (int i = 1; i < C; i++) r = mgl_f64_fma(m[i], vec<ulong, R>(v[i]), r);
    return r;
}
template <int C, int R> inline vec<ulong, C> mgl_dmat_vm(vec<ulong, R> v, mgl_dmat<C, R> m) {
    vec<ulong, C> r;
    for (int i = 0; i < C; i++) r[i] = mgl_f64_dot(v, m[i]);
    return r;
}
template <int C, int R, int C2> inline mgl_dmat<C2, R> mgl_dmat_mm(mgl_dmat<C, R> a, mgl_dmat<C2, C> b) {
    mgl_dmat<C2, R> r;
    for (int j = 0; j < C2; j++) r[j] = mgl_dmat_mv(a, b[j]);
    return r;
}
template <int C, int R> inline mgl_dmat<R, C> mgl_dmat_transpose(mgl_dmat<C, R> m) {
    mgl_dmat<R, C> r;
    for (int c = 0; c < C; c++)
        for (int k = 0; k < R; k++) r[k][c] = m[c][k];
    return r;
}
template <int C, int R> inline mgl_dmat<C, R> mgl_dmat_outer(vec<ulong, R> c, vec<ulong, C> r) {
    mgl_dmat<C, R> m;
    for (int i = 0; i < C; i++) m[i] = mgl_f64_mul(c, vec<ulong, R>(r[i]));
    return m;
}
template <int C, int R> inline mgl_dmat<C, R> mgl_dmat_diag(ulong s) {
    mgl_dmat<C, R> m;
    for (int c = 0; c < C; c++)
        for (int k = 0; k < R; k++) m[c][k] = c == k ? s : 0ul;
    return m;
}
template <int C, int R, int C2, int R2> inline mgl_dmat<C, R> mgl_dmat_resize(mgl_dmat<C2, R2> s) {
    mgl_dmat<C, R> m;
    for (int c = 0; c < C; c++)
        for (int k = 0; k < R; k++) m[c][k] = (c < C2 && k < R2) ? s[c][k] : (c == k ? 0x3FF0000000000000ul : 0ul);
    return m;
}
template <int C, int R> inline mgl_dmat<C, R> mgl_dmat_from_float(matrix<float, C, R> f) {
    mgl_dmat<C, R> m;
    for (int c = 0; c < C; c++) m[c] = mgl_f64_from_float(f[c]);
    return m;
}
template <int C, int R> inline matrix<float, C, R> mgl_dmat_to_float(mgl_dmat<C, R> d) {
    matrix<float, C, R> m;
    for (int c = 0; c < C; c++) m[c] = mgl_f64_to_float(d[c]);
    return m;
}
template <int C, int R> inline bool mgl_eq(mgl_dmat<C, R> a, mgl_dmat<C, R> b) {
    for (int i = 0; i < C; i++)
        if (!all(mgl_f64_eq(a[i], b[i]))) return false;
    return true;
}
// LU decomposition with partial pivoting (determinant / inverse of square double matrices)
// Determinant and inverse by cofactor expansion (exact when every product/sum is
// representable, e.g. det == 0 for exactly singular matrices), in the usual
// det2/det3/det4 evaluation order. m[c][r] is column c, row r.
inline ulong mgl_f64_det2(ulong a00, ulong a10, ulong a01, ulong a11) {
    return mgl_f64_sub(mgl_f64_mul(a00, a11), mgl_f64_mul(a01, a10));
}
inline ulong mgl_f64_det3(ulong a00, ulong a10, ulong a20, ulong a01, ulong a11, ulong a21, ulong a02, ulong a12, ulong a22) {
    ulong t0 = mgl_f64_mul(a00, mgl_f64_det2(a11, a21, a12, a22));
    ulong t1 = mgl_f64_mul(a10, mgl_f64_det2(a01, a21, a02, a22));
    ulong t2 = mgl_f64_mul(a20, mgl_f64_det2(a01, a11, a02, a12));
    return mgl_f64_add(mgl_f64_sub(t0, t1), t2);
}
// E(r, c): row r, column c. Arguments are passed row-major as det2/det3/det4(mat(0,0), mat(0,1), ...).
template <int N> inline ulong mgl_dmat_det(mgl_dmat<N, N> m) {
#define E(r, c) m[(c) < N ? (c) : 0][(r) < N ? (r) : 0]
    if (N == 1) return E(0, 0);
    if (N == 2) return mgl_f64_det2(E(0, 0), E(0, 1), E(1, 0), E(1, 1));
    if (N == 3) return mgl_f64_det3(E(0, 0), E(0, 1), E(0, 2), E(1, 0), E(1, 1), E(1, 2), E(2, 0), E(2, 1), E(2, 2));
    // det4(_00.._33) = _00*det3(_11,_21,_31,_12,_22,_32,_13,_23,_33) - _10*det3(...) + _20*det3(...) - _30*det3(...)
    // where _ij is the i-th argument group element: _ij = E(j, i).
    ulong t[4];
    for (int k = 0; k < 4; k++) {
        ulong e[9];
        int n = 0;
        for (int j = 1; j < 4; j++)
            for (int i = 0; i < 4; i++)
                if (i != k) e[n++] = E(j, i);
        t[k] = mgl_f64_mul(E(0, k), mgl_f64_det3(e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8]));
    }
#undef E
    return mgl_f64_sub(mgl_f64_add(mgl_f64_sub(t[0], t[1]), t[2]), t[3]);
}
template <int N> inline mgl_dmat<N, N> mgl_dmat_inverse(mgl_dmat<N, N> m) {
    // inverse = transpose(cofactors) * (1 / det)
    ulong invDet = mgl_f64_div(0x3FF0000000000000ul, mgl_dmat_det(m));
    mgl_dmat<N, N> out;
    for (int c = 0; c < N; c++)
        for (int r = 0; r < N; r++) {
            ulong mi;
            if (N == 1) mi = 0x3FF0000000000000ul;
            else {
                // minor of element (row r, column c) = determinant without row r and column c
                mgl_dmat<N == 1 ? 1 : N - 1, N == 1 ? 1 : N - 1> sub;
                int sc = 0;
                for (int cc = 0; cc < N; cc++) {
                    if (cc == c) continue;
                    int sr = 0;
                    for (int rr = 0; rr < N; rr++) {
                        if (rr == r) continue;
                        sub[sc][sr++] = m[cc][rr];
                    }
                    sc++;
                }
                mi = mgl_dmat_det(sub);
            }
            ulong cof = ((r + c) & 1) ? mgl_f64_neg(mi) : mi;
            out[r][c] = mgl_f64_mul(cof, invDet);  // adjugate: transpose of the cofactors
        }
    return out;
}
)DMAT";

struct Chunk {
    std::vector<std::string> symbols;  // identifiers that pull this chunk in
    std::string code;
};

std::vector<Chunk> buildChunks() {
    std::vector<Chunk> c;
    // --- fp64 (soft-float library from the runtime + matrix helpers composed on it) ---------
    // The library is generated from src/glsl/runtime/fp64.metal by the build as a string literal.
    c.push_back({{"mgl_f64_lib"}, kFp64Library});
    // --- integer border color emulation (b[0] = color, b[1].x = clamp-to-border axis mask) ---
    c.push_back({{"mgl_int_border"}, R"(template <typename T, typename C>
inline T mgl_int_border(T v, C c, ulong addr) {
    constant int4* b = (constant int4*)addr;
    int m = b[1].x;
    if (m == 0) return v;
    bool o = false;
    for (int i = 0; i < int(sizeof(C) / sizeof(float)); i++)
        if (((m >> i) & 1) != 0 && (c[i] < 0.0f || c[i] >= 1.0f)) o = true;
    return o ? T(b[0]) : v;
}
)"});
    c.push_back({{"mgl_int_border_gather"}, R"(// Gathered texels in GL order: x = (i0,j1), y = (i1,j1), z = (i1,j0), w = (i0,j0).
inline bool mgl_border_out(int m, int2 p, int2 size) {
    return ((m & 1) != 0 && (p.x < 0 || p.x >= size.x)) || ((m & 2) != 0 && (p.y < 0 || p.y >= size.y));
}
template <typename T>
inline T mgl_int_border_gather(T v, float2 c, int2 size, int2 off, int comp, ulong addr) {
    constant int4* b = (constant int4*)addr;
    int m = b[1].x;
    if (m == 0) return v;
    int2 i0 = int2(floor(c * float2(size) - 0.5f)) + off, i1 = i0 + 1;
    auto bc = b[0][comp];
    if (mgl_border_out(m, int2(i0.x, i1.y), size)) v.x = bc;
    if (mgl_border_out(m, int2(i1.x, i1.y), size)) v.y = bc;
    if (mgl_border_out(m, int2(i1.x, i0.y), size)) v.z = bc;
    if (mgl_border_out(m, int2(i0.x, i0.y), size)) v.w = bc;
    return v;
}
)"});
    // --- cube maps without TEXTURE_CUBE_MAP_SEAMLESS: keep the filter footprint inside the face ---
    c.push_back({{"mgl_cube_noseam"}, R"(inline float3 mgl_cube_noseam(float3 d, float size, float on) {
    if (on == 0.0f) return d;
    float3 a = abs(d);
    float lim = max(a.x, max(a.y, a.z)) * max(1.0f - 1.0f / size, 0.0f);
    if (a.x >= a.y && a.x >= a.z) { d.y = clamp(d.y, -lim, lim); d.z = clamp(d.z, -lim, lim); }
    else if (a.y >= a.z) { d.x = clamp(d.x, -lim, lim); d.z = clamp(d.z, -lim, lim); }
    else { d.x = clamp(d.x, -lim, lim); d.y = clamp(d.y, -lim, lim); }
    return d;
}
)"});
    // --- arrays ---------------------------------------------------------------------
    c.push_back({{"mgl_array"}, R"(template <typename T, int N>
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
)"});
    // --- buffer access: all through uint storage (alias-safe across typed views) ----------
    struct LT { const char* T; const char* ST; const char* CV; };
    const LT lts[] = {{"float", "uint", "uint"},   {"float2", "uint2", "uint2"}, {"float3", "packed_uint3", "uint3"},
                      {"float4", "uint4", "uint4"}, {"int", "uint", "uint"},      {"int2", "uint2", "uint2"},
                      {"int3", "packed_uint3", "uint3"}, {"int4", "uint4", "uint4"}, {"uint", "uint", "uint"},
                      {"uint2", "uint2", "uint2"}, {"uint3", "packed_uint3", "uint3"}, {"uint4", "uint4", "uint4"}};
    for (auto& l : lts) {
        std::string T = l.T, ST = l.ST, CV = l.CV;
        std::string code;
        for (const char* as : {"constant", "device const", "threadgroup const"})
            code += "inline " + T + " mgl_ld_" + T + "(" + as + " uchar* p, uint o) { return as_type<" + T + ">(" + CV + "(*(" + as +
                    " " + ST + "*)(p + o))); }\n";
        c.push_back({{"mgl_ld_" + T}, code});
        c.push_back({{"mgl_st_" + T}, "inline void mgl_st_" + T + "(device uchar* p, uint o, " + T + " v) { *(device " + ST + "*)(p + o) = " +
                                          ST + "(as_type<" + CV + ">(v)); }\n"});
    }
    {
        std::string code;
        for (const char* as : {"constant", "device const", "threadgroup const"})
            code += std::string("inline ulong mgl_ld_ulong(") + as + " uchar* p, uint o) { return as_type<ulong>(*(" + as + " uint2*)(p + o)); }\n";
        c.push_back({{"mgl_ld_ulong"}, code});
        c.push_back({{"mgl_st_ulong"}, "inline void mgl_st_ulong(device uchar* p, uint o, ulong v) { *(device uint2*)(p + o) = as_type<uint2>(v); }\n"});
        for (int n = 2; n <= 4; n++) {
            std::string V = "ulong" + std::to_string(n), args, N = std::to_string(n);
            for (int i = 0; i < n; i++) args += std::string(i ? ", " : "") + "mgl_ld_ulong(p, o + " + std::to_string(8 * i) + "u)";
            c.push_back({{"mgl_ld_" + V}, "template <typename P> inline " + V + " mgl_ld_" + V + "(P p, uint o) { return " + V + "(" + args + "); }\n"});
            c.push_back({{"mgl_st_" + V}, "inline void mgl_st_" + V + "(device uchar* p, uint o, " + V + " v) { for (int i = 0; i < " + N +
                                              "; i++) mgl_st_ulong(p, o + 8u * uint(i), v[i]); }\n"});
            std::string sargs;
            for (int i = 0; i < n; i++) sargs += std::string(i ? ", " : "") + "mgl_ld_ulong(p, o + " + std::to_string(i) + " * s)";
            c.push_back({{"mgl_ldcs_" + V}, "template <typename P> inline " + V + " mgl_ldcs_" + V + "(P p, uint o, uint s) { return " + V + "(" + sargs + "); }\n"});
            c.push_back({{"mgl_stcs_" + V}, "inline void mgl_stcs_" + V + "(device uchar* p, uint o, uint s, " + V + " v) { for (int i = 0; i < " + N +
                                                "; i++) mgl_st_ulong(p, o + i * s, v[i]); }\n"});
        }
    }
    c.push_back({{"mgl_dmat"}, kDmatHelpers});
    c.push_back({{"mgl_eq_f64_overloads"}, R"(inline bool mgl_eq(ulong a, ulong b) { return mgl_f64_eq(a, b); }
inline bool mgl_eq(ulong2 a, ulong2 b) { return all(mgl_f64_eq(a, b)); }
inline bool mgl_eq(ulong3 a, ulong3 b) { return all(mgl_f64_eq(a, b)); }
inline bool mgl_eq(ulong4 a, ulong4 b) { return all(mgl_f64_eq(a, b)); }
)"});
    const LT bools[] = {{"bool", "uint", "uint"}, {"bool2", "uint2", "uint2"}, {"bool3", "packed_uint3", "uint3"}, {"bool4", "uint4", "uint4"}};
    for (auto& l : bools) {
        std::string T = l.T, ST = l.ST, U = l.CV;
        std::string code;
        for (const char* as : {"constant", "device const"})
            code += "inline " + T + " mgl_ld_" + T + "(" + as + " uchar* p, uint o) { return " + U + "(*(" + as + " " + ST + "*)(p + o)) != " + U + "(0u); }\n";
        c.push_back({{"mgl_ld_" + T}, code});
        c.push_back({{"mgl_st_" + T}, "inline void mgl_st_" + T + "(device uchar* p, uint o, " + T + " v) { *(device " + ST + "*)(p + o) = " + ST +
                                          "(select(" + U + "(0u), " + U + "(1u), v)); }\n"});
    }
    for (int n = 2; n <= 4; n++) {
        std::string V = "float" + std::to_string(n), N = std::to_string(n);
        std::string args;
        for (int i = 0; i < n; i++) args += std::string(i ? ", " : "") + "mgl_ld_float(p, o + " + std::to_string(i) + " * s)";
        c.push_back({{"mgl_ldcs_" + V}, "template <typename P> inline " + V + " mgl_ldcs_" + V + "(P p, uint o, uint s) { return " + V + "(" + args + "); }\n"});
        c.push_back({{"mgl_stcs_" + V}, "inline void mgl_stcs_" + V + "(device uchar* p, uint o, uint s, " + V + " v) { for (int i = 0; i < " + N +
                                            "; i++) mgl_st_float(p, o + i * s, v[i]); }\n"});
    }
    // --- math helpers -------------------------------------------------------------------
    c.push_back({{"mgl_mod"}, "template <typename T> inline T mgl_mod(T x, T y) { return x - y * floor(x / y); }\n"});
    c.push_back({{"mgl_isign"}, "template <typename T> inline T mgl_isign(T x) { return select(T(0), T(1), x > T(0)) - select(T(0), T(1), x < T(0)); }\n"});
    c.push_back({{"mgl_mcm"}, R"(template <typename T, int C, int R> inline matrix<T, C, R> mgl_mcm(matrix<T, C, R> a, matrix<T, C, R> b) {
    matrix<T, C, R> r;
    for (int i = 0; i < C; i++) r[i] = a[i] * b[i];
    return r;
}
)"});
    c.push_back({{"mgl_outer"}, R"(template <typename T, int C, int R> inline matrix<T, C, R> mgl_outer(vec<T, R> c, vec<T, C> r) {
    matrix<T, C, R> m;
    for (int i = 0; i < C; i++) m[i] = c * r[i];
    return m;
}
)"});
    c.push_back({{"mgl_mdiag"}, R"(template <typename T, int C, int R> inline matrix<T, C, R> mgl_mdiag(T s) {
    matrix<T, C, R> m;
    for (int c = 0; c < C; c++)
        for (int r = 0; r < R; r++) m[c][r] = c == r ? s : T(0);
    return m;
}
)"});
    c.push_back({{"mgl_mresize"}, R"(template <typename T, int C, int R, int C2, int R2> inline matrix<T, C, R> mgl_mresize(matrix<T, C2, R2> s) {
    matrix<T, C, R> m;
    for (int c = 0; c < C; c++)
        for (int r = 0; r < R; r++) m[c][r] = (c < C2 && r < R2) ? s[c][r] : (c == r ? T(1) : T(0));
    return m;
}
)"});
    c.push_back({{"mgl_madd", "mgl_msub", "mgl_mdiv", "mgl_mneg"}, R"(template <typename T, int C, int R> inline matrix<T, C, R> mgl_madd(matrix<T, C, R> m, T s) { for (int i = 0; i < C; i++) m[i] += s; return m; }
template <typename T, int C, int R> inline matrix<T, C, R> mgl_madd(T s, matrix<T, C, R> m) { for (int i = 0; i < C; i++) m[i] += s; return m; }
template <typename T, int C, int R> inline matrix<T, C, R> mgl_msub(matrix<T, C, R> m, T s) { for (int i = 0; i < C; i++) m[i] -= s; return m; }
template <typename T, int C, int R> inline matrix<T, C, R> mgl_msub(T s, matrix<T, C, R> m) { for (int i = 0; i < C; i++) m[i] = s - m[i]; return m; }
template <typename T, int C, int R> inline matrix<T, C, R> mgl_mdiv(matrix<T, C, R> m, T s) { for (int i = 0; i < C; i++) m[i] /= s; return m; }
template <typename T, int C, int R> inline matrix<T, C, R> mgl_mdiv(T s, matrix<T, C, R> m) { for (int i = 0; i < C; i++) m[i] = s / m[i]; return m; }
template <typename T, int C, int R> inline matrix<T, C, R> mgl_mdiv(matrix<T, C, R> a, matrix<T, C, R> b) { for (int i = 0; i < C; i++) a[i] /= b[i]; return a; }
template <typename T, int C, int R> inline matrix<T, C, R> mgl_mneg(matrix<T, C, R> m) { for (int i = 0; i < C; i++) m[i] = -m[i]; return m; }
)"});
    c.push_back({{"mgl_inverse"}, R"(inline float2x2 mgl_inverse(float2x2 m) {
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
)"});
    c.push_back({{"mgl_eq"}, R"(template <typename T> inline bool mgl_eq(T a, T b) { return a == b; }
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
)"});
    c.push_back({{"mgl_findLSB", "mgl_findMSB"}, R"(inline int mgl_findLSB(uint x) { return x == 0u ? -1 : int(ctz(x)); }
inline int mgl_findLSB(int x) { return mgl_findLSB(uint(x)); }
inline int mgl_findMSB(uint x) { return x == 0u ? -1 : 31 - int(clz(x)); }
inline int mgl_findMSB(int x) { return mgl_findMSB(x < 0 ? ~uint(x) : uint(x)); }
template <typename T, int N> inline vec<int, N> mgl_findLSB(vec<T, N> v) { vec<int, N> r; for (int i = 0; i < N; i++) r[i] = mgl_findLSB(v[i]); return r; }
template <typename T, int N> inline vec<int, N> mgl_findMSB(vec<T, N> v) { vec<int, N> r; for (int i = 0; i < N; i++) r[i] = mgl_findMSB(v[i]); return r; }
)"});
    c.push_back({{"mgl_uaddCarry", "mgl_usubBorrow", "mgl_mulExtended"}, R"(template <typename T> inline T mgl_uaddCarry(T x, T y, thread T& c) { T r = x + y; c = select(T(0u), T(1u), r < x); return r; }
template <typename T> inline T mgl_usubBorrow(T x, T y, thread T& b) { b = select(T(0u), T(1u), x < y); return x - y; }
template <typename T> inline void mgl_mulExtended(T x, T y, thread T& msb, thread T& lsb) { msb = mulhi(x, y); lsb = x * y; }
)"});
    std::string cas;
    for (const char* as : {"device", "threadgroup"})
        for (const char* t : {"int", "uint"})
            cas += std::string("inline ") + t + " mgl_atomic_cas(" + as + " atomic_" + t + "* p, " + t + " c, " + t + " v) {\n    " + t +
                   " e = c;\n    while (!atomic_compare_exchange_weak_explicit(p, &e, v, memory_order_relaxed, memory_order_relaxed)) {\n"
                   "        if (e != c) break;\n        e = c;\n    }\n    return e;\n}\n";
    c.push_back({{"mgl_atomic_cas"}, cas});
    c.push_back({{"mgl_img_cas", "mgl_img_cas_a"}, R"(template <typename Tex, typename C, typename T> inline T mgl_img_cas(Tex t, C coord, T cmp, T v) {
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
)"});
    // GL_SAMPLE_COVERAGE: round(value * samples) samples covered (optionally inverted).
    c.push_back({{"mgl_coverage_mask"}, R"(inline uint mgl_coverage_mask(float v, bool inv) {
    uint n = uint(rint(saturate(v) * float(get_num_samples())));
    uint m = n >= 32u ? 0xffffffffu : (1u << n) - 1u;
    return inv ? ~m : m;
}
)"});
    c.push_back({{"mgl_alpha_pass"}, R"(inline bool mgl_alpha_pass(float a, uint func, float ref) {
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
)"});
    // multi-pass helpers: split the embedded file at "//@chunk sym..." markers
    {
        std::string all = kMultipassLibrary;
        size_t pos = all.find("//@chunk ");
        while (pos != std::string::npos) {
            size_t eol = all.find('\n', pos);
            std::string header = all.substr(pos + 9, eol - pos - 9);
            size_t next = all.find("//@chunk ", eol);
            std::string body = all.substr(eol + 1, next == std::string::npos ? std::string::npos : next - eol - 1);
            Chunk ch;
            size_t a = 0;
            while (a < header.size()) {
                size_t b = header.find(' ', a);
                if (b == std::string::npos) b = header.size();
                if (b > a) ch.symbols.push_back(header.substr(a, b - a));
                a = b + 1;
            }
            ch.code = body;
            c.push_back(ch);
            pos = next;
        }
    }
    return c;
}

const char* kSysStruct = R"(// Runtime system values (mirrors glsl::MGLSys)
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
    float sampleCoverage;
    float tessDefaultOuter[4];
    float tessDefaultInner[2];
    uint pad2[2];
    float4 currentAttrib[32];
    uint4 currentAttribD[32];
};
)";

void collectIdents(const std::string& s, std::set<std::string>& out) {
    size_t i = 0, n = s.size();
    while (i < n) {
        char ch = s[i];
        if (((ch == 'm' && s.compare(i, 4, "mgl_") == 0) || (ch == 'M' && s.compare(i, 3, "MGL") == 0)) && (i == 0 || !(std::isalnum((unsigned char)s[i - 1]) || s[i - 1] == '_'))) {
            size_t j = i;
            while (j < n && (std::isalnum((unsigned char)s[j]) || s[j] == '_')) j++;
            out.insert(s.substr(i, j - i));
            i = j;
        } else {
            i++;
        }
    }
}

}  // namespace

std::string buildMSLPrelude(const std::string& body) {
    static const std::vector<Chunk> chunks = buildChunks();
    std::set<std::string> idents;
    auto addFp64Lib = [&]() {
        for (auto& id : std::set<std::string>(idents))
            if (id.compare(0, 8, "mgl_f64_") == 0 || id.compare(0, 8, "mgl_dmat") == 0) idents.insert("mgl_f64_lib");
    };
    collectIdents(body, idents);
    addFp64Lib();
    std::vector<bool> use(chunks.size(), false);
    // fixpoint: chunks can reference other chunks' symbols
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t i = 0; i < chunks.size(); i++) {
            if (use[i]) continue;
            for (auto& sym : chunks[i].symbols)
                if (idents.count(sym)) {
                    use[i] = true;
                    changed = true;
                    collectIdents(chunks[i].code, idents);
                    addFp64Lib();  // e.g. the mgl_eq double overloads call mgl_f64_eq
                    break;
                }
        }
    }
    std::string out = "#include <metal_stdlib>\nusing namespace metal;\n\n";
    for (size_t i = 0; i < chunks.size(); i++)
        if (use[i]) out += chunks[i].code;
    out += kSysStruct;
    return out;
}

}  // namespace glsl
