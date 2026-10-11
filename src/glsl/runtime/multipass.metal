// Runtime helpers for multi-pass pipelines (geometry via mesh shaders, tessellation,
// transform feedback). Embedded into generated MSL on demand; each section starts
// with a "//@chunk <symbols...>" line naming the identifiers that pull it in.

//@chunk MGLDrawInfo mgl_vptr
// Mirrors glsl::MGLSubDraw / glsl::MGLDrawInfo (glsl.h); GPU addresses are typed pointers here.
struct MGLSubDraw {
    uint firstPatch;
    uint drawID;
    int baseVertex;
    uint first;
    device const uchar* indices;
    uint pad0, pad1;
};
struct MGLDrawInfo {
    device const uchar* vertexBuffer[16];
    uint vertexStride[16];
    uint vertexDivisor[16];
    device const uint* indexBuffer;
    uint indexType;
    uint count;
    int baseVertex;
    uint baseInstance;
    uint firstVertex;
    uint instanceCount;
    uint primitiveCount;
    uint primitiveMode;
    uint verticesPerPrimitive;
    uint patchVertices;
    device uchar* xfbBuffer[4];
    uint xfbStride[4];
    uint xfbCapacity[4];
    uint xfbVertexBase;
    uint pad0;
    device uchar* tcsVertexOut;
    device uchar* tcsPatchOut;
    device atomic_uint* counters;
    device uint4* itemCounts;
    device const uint4* itemOffsets;
    device uchar* tesVertexOut;
    device uint* tesPrimOut;
    uint itemCount;
    uint tesPrimCount;
    uint xfbPrimLimit[4];
    uint xfbStreamBase[4];
    uint xfbBufferStream[4];
    device uint* gsRecord;
    device atomic_uint* stats;
    device uint4* spillBuffer;
    device atomic_uint* spillCounter;
    uint spillCapacity;
    uint gridBase;
    device const MGLSubDraw* subDraws;
    uint subDrawCount;
    uint subDrawIndexType;
    uint vertexBufferSize[16];
    device half* tessFactors;
};
// Address of the attribute data of vertex `vid` / instance `inst` in binding b, or null (the attribute's
// default value) when nothing is bound or the vertex lies outside the buffer (garbage indices must not read
// unmapped memory).
inline device const uchar* mgl_vptr(constant MGLDrawInfo& D, uint b, uint off, uint vid, uint inst) {
    device const uchar* p = D.vertexBuffer[b];
    if (p == nullptr) return nullptr;
    uint d = D.vertexDivisor[b];
    uint idx = d != 0u ? inst / d + D.baseInstance : vid;
    ulong at = ulong(idx) * ulong(D.vertexStride[b]) + ulong(off);
    if (at >= ulong(D.vertexBufferSize[b])) return nullptr;
    return p + at;
}

//@chunk mgl_sd_find mgl_sd_vertex
// Tessellation draws (one or all sub-draws of a multi-draw): the sub-draw holding patch `gp`
// (patch index counted over all sub-draws) and the vertex id of list position i of a sub-draw.
inline uint mgl_sd_find(constant MGLDrawInfo& D, uint gp) {
    uint lo = 0u, hi = D.subDrawCount;
    while (hi - lo > 1u) {
        uint mid = (lo + hi) / 2u;
        if (D.subDraws[mid].firstPatch <= gp) lo = mid;
        else hi = mid;
    }
    return lo;
}
inline uint mgl_sd_vertex(constant MGLDrawInfo& D, device const MGLSubDraw& s, uint i) {
    switch (D.subDrawIndexType) {
        case 1u: return s.first + i;  // non-indexed
        case 2u: return uint(((device const ushort*)s.indices)[i]) + uint(s.baseVertex);
        case 4u: return ((device const uint*)s.indices)[i] + uint(s.baseVertex);
        default: return D.indexBuffer[s.first + i] + uint(s.baseVertex);  // decomposed list
    }
}

//@chunk mgl_fetch_f mgl_fetch_i mgl_fetch_d
// Vertex attribute decoding for every GL vertex format (manual vertex pulling).
inline uint mgl_rd8(device const uchar* p, uint o) { return uint(p[o]); }
inline uint mgl_rd16(device const uchar* p, uint o) { return uint(p[o]) | (uint(p[o + 1]) << 8); }
inline uint mgl_rd32(device const uchar* p, uint o) {
    return uint(p[o]) | (uint(p[o + 1]) << 8) | (uint(p[o + 2]) << 16) | (uint(p[o + 3]) << 24);
}
inline float mgl_dbits_to_float(uint lo, uint hi) {
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
inline float mgl_uf11(uint v, uint mbits) {  // unsigned small float (10/11-bit, 5-bit exponent)
    uint e = (v >> mbits) & 31u, m = v & ((1u << mbits) - 1u);
    if (e == 0u) return float(m) / float(1u << mbits) * exp2(-14.0f);
    if (e == 31u) return m != 0u ? NAN : INFINITY;
    return (1.0f + float(m) / float(1u << mbits)) * exp2(float(int(e) - 15));
}
inline float4 mgl_fetch_f(device const uchar* p, uint type, uint size, bool norm) {
    float4 r = float4(0.0f, 0.0f, 0.0f, 1.0f);
    if (p == nullptr) return r;
    uint n = size == 5u ? 4u : size;
    switch (type) {
        case 0x1406u:  // FLOAT
            for (uint k = 0; k < n; k++) r[k] = as_type<float>(mgl_rd32(p, 4u * k));
            break;
        case 0x140Bu:  // HALF_FLOAT
            for (uint k = 0; k < n; k++) r[k] = float(as_type<half>(ushort(mgl_rd16(p, 2u * k))));
            break;
        case 0x140Au:  // DOUBLE
            for (uint k = 0; k < n; k++) r[k] = mgl_dbits_to_float(mgl_rd32(p, 8u * k), mgl_rd32(p, 8u * k + 4u));
            break;
        case 0x140Cu:  // FIXED
            for (uint k = 0; k < n; k++) r[k] = float(int(mgl_rd32(p, 4u * k))) / 65536.0f;
            break;
        case 0x1400u:  // BYTE
            for (uint k = 0; k < n; k++) {
                float v = float(int(mgl_rd8(p, k) << 24) >> 24);
                r[k] = norm ? max(v / 127.0f, -1.0f) : v;
            }
            break;
        case 0x1401u:  // UNSIGNED_BYTE
            for (uint k = 0; k < n; k++) {
                float v = float(mgl_rd8(p, k));
                r[k] = norm ? v / 255.0f : v;
            }
            break;
        case 0x1402u:  // SHORT
            for (uint k = 0; k < n; k++) {
                float v = float(int(mgl_rd16(p, 2u * k) << 16) >> 16);
                r[k] = norm ? max(v / 32767.0f, -1.0f) : v;
            }
            break;
        case 0x1403u:  // UNSIGNED_SHORT
            for (uint k = 0; k < n; k++) {
                float v = float(mgl_rd16(p, 2u * k));
                r[k] = norm ? v / 65535.0f : v;
            }
            break;
        case 0x1404u:  // INT
            for (uint k = 0; k < n; k++) {
                float v = float(int(mgl_rd32(p, 4u * k)));
                r[k] = norm ? max(v / 2147483647.0f, -1.0f) : v;
            }
            break;
        case 0x1405u:  // UNSIGNED_INT
            for (uint k = 0; k < n; k++) {
                float v = float(mgl_rd32(p, 4u * k));
                r[k] = norm ? v / 4294967295.0f : v;
            }
            break;
        case 0x8D9Fu: {  // INT_2_10_10_10_REV
            uint w = mgl_rd32(p, 0);
            int4 c = int4(int(w << 22) >> 22, int(w << 12) >> 22, int(w << 2) >> 22, int(w) >> 30);
            r = norm ? max(float4(c) / float4(511.0f, 511.0f, 511.0f, 1.0f), float4(-1.0f)) : float4(c);
            break;
        }
        case 0x8368u: {  // UNSIGNED_INT_2_10_10_10_REV
            uint w = mgl_rd32(p, 0);
            uint4 c = uint4(w & 1023u, (w >> 10) & 1023u, (w >> 20) & 1023u, w >> 30);
            r = norm ? float4(c) / float4(1023.0f, 1023.0f, 1023.0f, 3.0f) : float4(c);
            break;
        }
        case 0x8C3Bu: {  // UNSIGNED_INT_10F_11F_11F_REV
            uint w = mgl_rd32(p, 0);
            r = float4(mgl_uf11(w & 2047u, 6u), mgl_uf11((w >> 11) & 2047u, 6u), mgl_uf11(w >> 22, 5u), 1.0f);
            break;
        }
        default: break;
    }
    if (size == 5u) r = r.zyxw;  // GL_BGRA
    return r;
}
inline int4 mgl_fetch_i(device const uchar* p, uint type, uint size) {
    int4 r = int4(0, 0, 0, 1);
    if (p == nullptr) return r;
    uint n = size == 5u ? 4u : size;
    for (uint k = 0; k < n; k++) {
        switch (type) {
            case 0x1400u: r[k] = int(mgl_rd8(p, k) << 24) >> 24; break;
            case 0x1401u: r[k] = int(mgl_rd8(p, k)); break;
            case 0x1402u: r[k] = int(mgl_rd16(p, 2u * k) << 16) >> 16; break;
            case 0x1403u: r[k] = int(mgl_rd16(p, 2u * k)); break;
            default: r[k] = int(mgl_rd32(p, 4u * k)); break;
        }
    }
    if (size == 5u) r = r.zyxw;
    return r;
}
inline ulong4 mgl_fetch_d(device const uchar* p, uint size) {
    ulong4 r = ulong4(0ul, 0ul, 0ul, 0x3FF0000000000000ul);
    if (p == nullptr) return r;
    for (uint k = 0; k < size && k < 4u; k++) r[k] = ulong(mgl_rd32(p, 8u * k)) | (ulong(mgl_rd32(p, 8u * k + 4u)) << 32);
    return r;
}

//@chunk mgl_pk mgl_upk_float mgl_upk_float2 mgl_upk_float3 mgl_upk_float4 mgl_upk_int mgl_upk_int2 mgl_upk_int3 mgl_upk_int4 mgl_upk_uint mgl_upk_uint2 mgl_upk_uint3 mgl_upk_uint4 mgl_upk_ulong
// Varying records: every leaf occupies one 16-byte slot.
inline uint4 mgl_pk(float v) { return uint4(as_type<uint>(v), 0u, 0u, 0u); }
inline uint4 mgl_pk(float2 v) { return uint4(as_type<uint2>(v), 0u, 0u); }
inline uint4 mgl_pk(float3 v) { return uint4(as_type<uint3>(v), 0u); }
inline uint4 mgl_pk(float4 v) { return as_type<uint4>(v); }
inline uint4 mgl_pk(int v) { return uint4(uint(v), 0u, 0u, 0u); }
inline uint4 mgl_pk(int2 v) { return uint4(uint2(v), 0u, 0u); }
inline uint4 mgl_pk(int3 v) { return uint4(uint3(v), 0u); }
inline uint4 mgl_pk(int4 v) { return uint4(v); }
inline uint4 mgl_pk(uint v) { return uint4(v, 0u, 0u, 0u); }
inline uint4 mgl_pk(uint2 v) { return uint4(v, 0u, 0u); }
inline uint4 mgl_pk(uint3 v) { return uint4(v, 0u); }
inline uint4 mgl_pk(uint4 v) { return v; }
inline uint4 mgl_pk(ulong v) { return uint4(as_type<uint2>(v), 0u, 0u); }
inline float mgl_upk_float(uint4 u) { return as_type<float>(u.x); }
inline float2 mgl_upk_float2(uint4 u) { return as_type<float2>(u.xy); }
inline float3 mgl_upk_float3(uint4 u) { return as_type<float3>(u.xyz); }
inline float4 mgl_upk_float4(uint4 u) { return as_type<float4>(u); }
inline int mgl_upk_int(uint4 u) { return int(u.x); }
inline int2 mgl_upk_int2(uint4 u) { return int2(u.xy); }
inline int3 mgl_upk_int3(uint4 u) { return int3(u.xyz); }
inline int4 mgl_upk_int4(uint4 u) { return int4(u); }
inline uint mgl_upk_uint(uint4 u) { return u.x; }
inline uint2 mgl_upk_uint2(uint4 u) { return u.xy; }
inline uint3 mgl_upk_uint3(uint4 u) { return u.xyz; }
inline uint4 mgl_upk_uint4(uint4 u) { return u; }
inline ulong mgl_upk_ulong(uint4 u) { return as_type<ulong>(u.xy); }

//@chunk mgl_tess mgl_tess_setup mgl_tess_unit mgl_tess_unit_size mgl_tess_unit_vertex mgl_tess_unit_prim mgl_tess_unit_pieces mgl_tess_piece_size mgl_tess_piece_bounds mgl_tess_piece_vertex mgl_tess_piece_prim
// GL tessellation primitive generator (GL 4.6 section 11.2.2). The work for one
// patch is split into "units" (one ring-side strip, the center, or one isoline),
// each small enough for one mesh threadgroup (<= 130 vertices, <= 128 primitives).
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
struct mgl_tess {
    uint domain;     // 0 triangles, 1 quads, 2 isolines
    uint spacing;    // 0 equal, 1 fractional_even, 2 fractional_odd
    uint points;     // point_mode
    uint ccw;
    float xo[4];
    uint no[4];
    float xi[2];
    uint ni[2];
    uint units;
    uint rings;      // strip rings (triangles/quads) or rings with points
    uint special;    // 0 none, 1 single triangle, 2 single quad
};

inline void mgl_tess_level(float x, uint spacing, thread float& xe, thread uint& n) {
    if (spacing == 0u) {
        x = clamp(x, 1.0f, 64.0f);
        n = uint(ceil(x));
        xe = float(n);
    } else if (spacing == 2u) {
        x = clamp(x, 1.0f, 63.0f);
        n = uint(ceil(x));
        if ((n & 1u) == 0u) n++;
        xe = x;
    } else {
        x = clamp(x, 2.0f, 64.0f);
        n = uint(ceil(x));
        if ((n & 1u) != 0u) n++;
        xe = x;
    }
}

// Position of point i (0..n) on an edge subdivided for level x into n segments:
// n-2 segments of length 1/x and two shorter equal segments placed symmetrically around the center.
inline float mgl_tess_pos(float x, uint n, uint i) {
    if (n == 0u) return 0.0f;
    if (float(n) == x || n < 2u) return float(i) / float(n);
    float L = 1.0f / x, s = (1.0f - float(n - 2u) * L) * 0.5f;
    uint a = (n & 1u) != 0u ? (n - 3u) / 2u : n / 2u - 1u;
    uint b = (n & 1u) != 0u ? (n + 1u) / 2u : n / 2u;
    if (i >= n) return 1.0f;
    uint shorts = (i > a ? 1u : 0u) + (i > b ? 1u : 0u);
    return float(i - shorts) * L + float(shorts) * s;
}

inline mgl_tess mgl_tess_setup(uint domain, uint spacing, uint points, uint ccw, float4 outer, float2 inner) {
    mgl_tess t;
    t.domain = domain;
    t.spacing = spacing;
    t.points = points;
    t.ccw = ccw;
    t.units = 0u;
    t.rings = 0u;
    t.special = 0u;
    for (int k = 0; k < 4; k++) { t.xo[k] = 1.0f; t.no[k] = 1u; }
    for (int k = 0; k < 2; k++) { t.xi[k] = 1.0f; t.ni[k] = 1u; }
    uint nOuter = domain == 0u ? 3u : domain == 1u ? 4u : 2u;
    for (uint k = 0; k < nOuter; k++)
        if (!(outer[k] > 0.0f)) return t;  // culled (also catches NaN)
    if (domain == 2u) {
        uint lines = uint(ceil(clamp(outer[0], 1.0f, 64.0f)));
        t.xo[0] = float(lines);
        t.no[0] = lines;
        mgl_tess_level(outer[1], spacing, t.xo[1], t.no[1]);
        t.units = lines;
        return t;
    }
    for (uint k = 0; k < nOuter; k++) mgl_tess_level(outer[k], spacing, t.xo[k], t.no[k]);
    uint nInner = domain == 0u ? 1u : 2u;
    for (uint k = 0; k < nInner; k++) mgl_tess_level(inner[k], spacing, t.xi[k], t.ni[k]);
    bool allOne = true;
    for (uint k = 0; k < nOuter; k++) allOne = allOne && t.no[k] == 1u;
    for (uint k = 0; k < nInner; k++) allOne = allOne && t.ni[k] == 1u;
    if (allOne) {
        t.special = domain == 0u ? 1u : 2u;
        t.units = 1u;
        return t;
    }
    for (uint k = 0; k < nInner; k++)
        if (t.ni[k] == 1u) {  // treated as 1 + epsilon
            // fractional_odd: three segments whose two short ones have length epsilon/2 -> 0, so the
            // inner ring sits exactly on the outer edges (CTS looks for triangles spanning u/v 0..1)
            if (spacing == 2u) { t.ni[k] = 3u; t.xi[k] = 1.0f; }
            else { t.ni[k] = 2u; t.xi[k] = 2.0f; }
        }
    if (domain == 0u) {
        uint n = t.ni[0];
        if (points != 0u) {
            t.rings = (n + 1u) / 2u;  // rings with >= 1 segment
            t.units = 3u * t.rings + ((n & 1u) == 0u ? 1u : 0u);
        } else {
            t.rings = n / 2u;
            t.units = 3u * t.rings + ((n & 1u) != 0u ? 1u : 0u);
        }
    } else {
        uint nu = t.ni[0], nv = t.ni[1], m = min(nu, nv);
        if (points != 0u) {
            t.rings = m / 2u + 1u;
            uint u = 0u;
            for (uint r = 0; r < t.rings; r++) u += (nu - 2u * r >= 1u && nv - 2u * r >= 1u && 2u * r < m) ? 4u : 1u;
            t.units = u;
        } else {
            t.rings = m / 2u;
            t.units = 4u * t.rings + ((m & 1u) != 0u ? 1u : 0u);
        }
    }
    return t;
}

// T.xo[s] / T.no[s] without indexing by a variable, which would keep the struct in (slow) stack memory
inline float mgl_txo(thread const mgl_tess& T, uint s) { return s == 0u ? T.xo[0] : s == 1u ? T.xo[1] : s == 2u ? T.xo[2] : T.xo[3]; }
inline uint mgl_tno(thread const mgl_tess& T, uint s) { return s == 0u ? T.no[0] : s == 1u ? T.no[1] : s == 2u ? T.no[2] : T.no[3]; }
// corner s of the triangle domain's sides (side s runs from corner s to corner s + 1)
inline float3 mgl_tcorner(uint s) { return s == 0u ? float3(0, 1, 0) : s == 1u ? float3(0, 0, 1) : float3(1, 0, 0); }

// lerp that is exact at t == 0 and t == 1 (shared vertices must be bit-identical)
inline float3 mgl_tlerp(float3 a, float3 b, float t) { return a * (1.0f - t) + b * t; }

// Point j of ring r, side s. Returns the domain coordinate; *t = normalized parameter along the row.
inline float3 mgl_tess_rowpt(thread const mgl_tess& T, uint r, uint s, uint j, thread float& tparam) {
    if (T.domain == 0u) {
        const float3 G = float3(1.0f / 3.0f);
        uint n = T.ni[0];
        float3 a = mgl_tcorner(s), b = mgl_tcorner((s + 1u) % 3u);
        if (r == 0u) {
            tparam = mgl_tess_pos(mgl_txo(T, s), mgl_tno(T, s), j);
            return mgl_tlerp(a, b, tparam);
        }
        float pr = mgl_tess_pos(T.xi[0], n, r), pe = mgl_tess_pos(T.xi[0], n, n - r);
        float3 ca = mgl_tlerp(a, G, 2.0f * pr), cb = mgl_tlerp(b, G, 2.0f * pr);
        if (n - 2u * r == 0u) { tparam = 0.0f; return G; }
        tparam = (mgl_tess_pos(T.xi[0], n, r + j) - pr) / (pe - pr);
        if (j == n - 2u * r) tparam = 1.0f;
        return mgl_tlerp(ca, cb, tparam);
    }
    // quads: (u, v, 0)
    uint nu = T.ni[0], nv = T.ni[1];
    if (r == 0u) {
        float p = mgl_tess_pos(mgl_txo(T, s), mgl_tno(T, s), j);
        tparam = p;
        // edges walked from 1 to 0 use the mirrored point index (not 1 - p) so every edge's
        // coordinates come from the same set of values (GL 4.6 11.2.2 invariance rule 4)
        float q = mgl_tess_pos(mgl_txo(T, s), mgl_tno(T, s), mgl_tno(T, s) - j);
        switch (s) {
            case 0u: return float3(0.0f, q, 0.0f);
            case 1u: return float3(p, 0.0f, 0.0f);
            case 2u: return float3(1.0f, p, 0.0f);
            default: return float3(q, 1.0f, 0.0f);
        }
    }
    uint du = nu - 2u * r, dv = nv - 2u * r;
    float ulo = mgl_tess_pos(T.xi[0], nu, r), uhi = mgl_tess_pos(T.xi[0], nu, nu - r);
    float vlo = mgl_tess_pos(T.xi[1], nv, r), vhi = mgl_tess_pos(T.xi[1], nv, nv - r);
    switch (s) {
        case 0u: { float v = mgl_tess_pos(T.xi[1], nv, nv - r - j); tparam = dv ? float(j) / float(dv) : 0.0f; return float3(ulo, v, 0.0f); }
        case 1u: { float u = mgl_tess_pos(T.xi[0], nu, r + j); tparam = du ? float(j) / float(du) : 0.0f; return float3(u, vlo, 0.0f); }
        case 2u: { float v = mgl_tess_pos(T.xi[1], nv, r + j); tparam = dv ? float(j) / float(dv) : 0.0f; return float3(uhi, v, 0.0f); }
        default: { float u = mgl_tess_pos(T.xi[0], nu, nu - r - j); tparam = du ? float(j) / float(du) : 0.0f; return float3(u, vhi, 0.0f); }
    }
    (void)vhi;
    (void)uhi;
}

inline uint mgl_tess_rowsegs(thread const mgl_tess& T, uint r, uint s) {
    if (r == 0u) return mgl_tno(T, s);
    if (T.domain == 0u) return T.ni[0] - 2u * r;
    return (s & 1u) == 0u ? T.ni[1] - 2u * r : T.ni[0] - 2u * r;
}

inline void mgl_tess_tri(thread const mgl_tess& T, threadgroup float3* co, threadgroup ushort* idx, thread uint& np, uint a, uint b, uint c) {
    float2 pa = co[a].xy, pb = co[b].xy, pc = co[c].xy;
    float area = (pb.x - pa.x) * (pc.y - pa.y) - (pc.x - pa.x) * (pb.y - pa.y);
    if ((T.ccw != 0u && area < 0.0f) || (T.ccw == 0u && area > 0.0f)) { uint t = b; b = c; c = t; }
    idx[3u * np] = ushort(a);
    idx[3u * np + 1u] = ushort(b);
    idx[3u * np + 2u] = ushort(c);
    np++;
}

// Generates the vertices (domain coordinates) and primitives of one unit.
// Primitive indices: 3 per triangle, 2 per line, 1 per point.
inline void mgl_tess_unit(thread const mgl_tess& T, uint unit, threadgroup float3* co, threadgroup float* tp, threadgroup ushort* idx,
                          thread uint& nv, thread uint& np) {
    nv = 0u;
    np = 0u;
    if (T.domain == 2u) {  // isolines: one line per unit
        uint n = T.no[1];
        float v = float(unit) / T.xo[0];
        for (uint i = 0; i <= n; i++) co[nv++] = float3(mgl_tess_pos(T.xo[1], n, i), v, 0.0f);
        if (T.points != 0u) {
            for (uint i = 0; i < nv; i++) idx[np++] = ushort(i);
        } else {
            for (uint i = 0; i < n; i++) { idx[2u * np] = ushort(i); idx[2u * np + 1u] = ushort(i + 1u); np++; }
        }
        return;
    }
    if (T.special == 1u) {
        co[0] = float3(0, 0, 1); co[1] = float3(1, 0, 0); co[2] = float3(0, 1, 0);
        nv = 3u;
        if (T.points != 0u) { for (uint i = 0; i < 3u; i++) idx[np++] = ushort(i); }
        else mgl_tess_tri(T, co, idx, np, 0u, 1u, 2u);
        return;
    }
    if (T.special == 2u) {
        co[0] = float3(0, 0, 0); co[1] = float3(1, 0, 0); co[2] = float3(1, 1, 0); co[3] = float3(0, 1, 0);
        nv = 4u;
        if (T.points != 0u) { for (uint i = 0; i < 4u; i++) idx[np++] = ushort(i); }
        else { mgl_tess_tri(T, co, idx, np, 0u, 1u, 2u); mgl_tess_tri(T, co, idx, np, 0u, 2u, 3u); }
        return;
    }
    uint sides = T.domain == 0u ? 3u : 4u;
    if (T.points != 0u) {
        // each distinct vertex once: ring sides without their last point; degenerate rings whole
        uint u = unit;
        for (uint r = 0; r < T.rings; r++) {
            bool full;
            if (T.domain == 0u) full = T.ni[0] >= 2u * r + 1u;
            else full = T.ni[0] >= 2u * r + 1u && T.ni[1] >= 2u * r + 1u && 2u * r < min(T.ni[0], T.ni[1]);
            uint cnt = full ? sides : 1u;
            if (u < cnt) {
                float tt;
                if (full) {
                    uint segs = mgl_tess_rowsegs(T, r, u);
                    for (uint j = 0; j < segs; j++) co[nv++] = mgl_tess_rowpt(T, r, u, j, tt);
                } else if (T.domain == 0u) {
                    co[nv++] = float3(1.0f / 3.0f);
                } else {
                    uint s = T.ni[0] == 2u * r ? 2u : 1u;  // degenerate in u: line along v (side 2)
                    uint segs = mgl_tess_rowsegs(T, r, s);
                    for (uint j = 0; j <= segs; j++) co[nv++] = mgl_tess_rowpt(T, r, s, j, tt);
                }
                for (uint i = 0; i < nv; i++) idx[np++] = ushort(i);
                return;
            }
            u -= cnt;
        }
        if (T.domain == 0u && u == 0u && (T.ni[0] & 1u) == 0u) {  // even inner level: the innermost ring is the center point
            co[0] = float3(1.0f / 3.0f);
            nv = 1u;
            idx[np++] = 0;
        }
        return;
    }
    if (unit < sides * T.rings) {
        // strip between ring r side s (outer row P) and ring r+1 side s (inner row Q)
        uint r = unit / sides, s = unit % sides;
        uint a = mgl_tess_rowsegs(T, r, s), b = mgl_tess_rowsegs(T, r + 1u, s);
        float tt;
        for (uint j = 0; j <= a; j++) { co[nv] = mgl_tess_rowpt(T, r, s, j, tt); tp[nv] = tt; nv++; }
        uint q0 = nv;
        for (uint j = 0; j <= b; j++) { co[nv] = mgl_tess_rowpt(T, r + 1u, s, j, tt); tp[nv] = b ? tt : 0.5f; nv++; }
        uint i = 0, j = 0;
        while (i < a || j < b) {
            bool advP;
            if (i == a) advP = false;
            else if (j == b) advP = true;
            else advP = tp[i + 1u] <= tp[q0 + j + 1u];
            if (advP) { mgl_tess_tri(T, co, idx, np, i, i + 1u, q0 + j); i++; }
            else { mgl_tess_tri(T, co, idx, np, i, q0 + j + 1u, q0 + j); j++; }
        }
        return;
    }
    // center
    uint R = T.rings;
    if (T.domain == 0u) {  // innermost triangle (odd inner level)
        float tt;
        for (uint s = 0; s < 3u; s++) co[nv++] = mgl_tess_rowpt(T, R, s, 0u, tt);
        mgl_tess_tri(T, co, idx, np, 0u, 1u, 2u);
        return;
    }
    // quads: 1 x k grid inside the innermost ring
    uint nu = T.ni[0], nv2 = T.ni[1];
    uint du = nu - 2u * R, dv = nv2 - 2u * R;
    for (uint jv = 0; jv <= dv; jv++)
        for (uint iu = 0; iu <= du; iu++)
            co[nv++] = float3(mgl_tess_pos(T.xi[0], nu, R + iu), mgl_tess_pos(T.xi[1], nv2, R + jv), 0.0f);
    for (uint jv = 0; jv < dv; jv++)
        for (uint iu = 0; iu < du; iu++) {
            uint p00 = jv * (du + 1u) + iu, p10 = p00 + 1u, p01 = p00 + du + 1u, p11 = p01 + 1u;
            mgl_tess_tri(T, co, idx, np, p00, p10, p11);
            mgl_tess_tri(T, co, idx, np, p00, p11, p01);
        }
}

// (vertices, primitives) mgl_tess_unit generates for `unit`, without generating them.
inline uint2 mgl_tess_unit_size(thread const mgl_tess& T, uint unit) {
    if (T.domain == 2u) {
        uint n = T.no[1];
        return uint2(n + 1u, T.points != 0u ? n + 1u : n);
    }
    if (T.special == 1u) return uint2(3u, T.points != 0u ? 3u : 1u);
    if (T.special == 2u) return uint2(4u, T.points != 0u ? 4u : 2u);
    uint sides = T.domain == 0u ? 3u : 4u;
    if (T.points != 0u) {
        uint u = unit;
        for (uint r = 0; r < T.rings; r++) {
            bool full;
            if (T.domain == 0u) full = T.ni[0] >= 2u * r + 1u;
            else full = T.ni[0] >= 2u * r + 1u && T.ni[1] >= 2u * r + 1u && 2u * r < min(T.ni[0], T.ni[1]);
            uint cnt = full ? sides : 1u;
            if (u < cnt) {
                uint n;
                if (full) n = mgl_tess_rowsegs(T, r, u);
                else if (T.domain == 0u) n = 1u;
                else n = mgl_tess_rowsegs(T, r, T.ni[0] == 2u * r ? 2u : 1u) + 1u;
                return uint2(n, n);
            }
            u -= cnt;
        }
        if (T.domain == 0u && u == 0u && (T.ni[0] & 1u) == 0u) return uint2(1u, 1u);
        return uint2(0u);
    }
    if (unit < sides * T.rings) {
        uint r = unit / sides, s = unit % sides;
        uint a = mgl_tess_rowsegs(T, r, s), b = mgl_tess_rowsegs(T, r + 1u, s);
        return uint2(a + b + 2u, a + b);
    }
    if (T.domain == 0u) return uint2(3u, 1u);
    uint du = T.ni[0] - 2u * T.rings, dv = T.ni[1] - 2u * T.rings;
    return uint2((du + 1u) * (dv + 1u), 2u * du * dv);
}

// One element at a time, for threads that generate a unit in parallel: vertex k and primitive m of `unit`
// exactly as mgl_tess_unit produces them (bit-identical coordinates, same primitives in the same order).
// mgl_tess_unit_vertex also returns the value mgl_tess_unit stores in tp[k]; mgl_tess_unit_prim reads the
// unit's vertices (co) and those values (tp) and returns its unit-local indices (ipp of them used).
inline float3 mgl_tess_unit_vertex(thread const mgl_tess& T, uint unit, uint k, thread float& tparam) {
    tparam = 0.0f;
    if (T.domain == 2u) return float3(mgl_tess_pos(T.xo[1], T.no[1], k), float(unit) / T.xo[0], 0.0f);
    if (T.special == 1u) return k == 0u ? float3(0, 0, 1) : k == 1u ? float3(1, 0, 0) : float3(0, 1, 0);
    if (T.special == 2u) return k == 0u ? float3(0, 0, 0) : k == 1u ? float3(1, 0, 0) : k == 2u ? float3(1, 1, 0) : float3(0, 1, 0);
    uint sides = T.domain == 0u ? 3u : 4u;
    float tt;
    if (T.points != 0u) {
        uint u = unit;
        for (uint r = 0; r < T.rings; r++) {
            bool full;
            if (T.domain == 0u) full = T.ni[0] >= 2u * r + 1u;
            else full = T.ni[0] >= 2u * r + 1u && T.ni[1] >= 2u * r + 1u && 2u * r < min(T.ni[0], T.ni[1]);
            uint cnt = full ? sides : 1u;
            if (u < cnt) {
                if (full) return mgl_tess_rowpt(T, r, u, k, tt);
                if (T.domain == 0u) return float3(1.0f / 3.0f);
                return mgl_tess_rowpt(T, r, T.ni[0] == 2u * r ? 2u : 1u, k, tt);
            }
            u -= cnt;
        }
        return float3(1.0f / 3.0f);
    }
    if (unit < sides * T.rings) {
        uint r = unit / sides, s = unit % sides, a = mgl_tess_rowsegs(T, r, s);
        if (k <= a) {
            float3 c = mgl_tess_rowpt(T, r, s, k, tt);
            tparam = tt;
            return c;
        }
        uint b = mgl_tess_rowsegs(T, r + 1u, s);
        float3 c = mgl_tess_rowpt(T, r + 1u, s, k - a - 1u, tt);
        tparam = b ? tt : 0.5f;
        return c;
    }
    uint R = T.rings;
    if (T.domain == 0u) return mgl_tess_rowpt(T, R, k, 0u, tt);
    uint nu = T.ni[0], nv2 = T.ni[1], du = nu - 2u * R;
    uint iu = k % (du + 1u), jv = k / (du + 1u);
    return float3(mgl_tess_pos(T.xi[0], nu, R + iu), mgl_tess_pos(T.xi[1], nv2, R + jv), 0.0f);
}

inline uint3 mgl_tess_wind(thread const mgl_tess& T, threadgroup const float3* co, uint a, uint b, uint c) {
    float2 pa = co[a].xy, pb = co[b].xy, pc = co[c].xy;
    float area = (pb.x - pa.x) * (pc.y - pa.y) - (pc.x - pa.x) * (pb.y - pa.y);
    if ((T.ccw != 0u && area < 0.0f) || (T.ccw == 0u && area > 0.0f)) return uint3(a, c, b);
    return uint3(a, b, c);
}

inline uint3 mgl_tess_unit_prim(thread const mgl_tess& T, uint unit, uint m, threadgroup const float3* co, threadgroup const float* tp) {
    if (T.domain == 2u) return T.points != 0u ? uint3(m) : uint3(m, m + 1u, 0u);
    if (T.points != 0u) return uint3(m);
    if (T.special == 1u) return mgl_tess_wind(T, co, 0u, 1u, 2u);
    if (T.special == 2u) return m == 0u ? mgl_tess_wind(T, co, 0u, 1u, 2u) : mgl_tess_wind(T, co, 0u, 2u, 3u);
    uint sides = T.domain == 0u ? 3u : 4u;
    if (unit < sides * T.rings) {
        // mgl_tess_unit merges the outer row P (keys tp[1..a]) and the inner row Q (keys tp[q0+1..q0+b]),
        // taking P on ties: a stable merge. Before step m it has taken i points of P, i = the number of P
        // keys whose merged position (k - 1 + Q keys below it) is < m.
        uint r = unit / sides, s = unit % sides;
        uint a = mgl_tess_rowsegs(T, r, s), b = mgl_tess_rowsegs(T, r + 1u, s), q0 = a + 1u;
        uint lo = 0u, hi = min(m, a);  // largest i in [0, hi] with position(i) < m (position(0) = -1)
        while (lo < hi) {
            uint k = (lo + hi + 1u) / 2u;
            float x = tp[k];
            uint qlo = 0u, qhi = b;  // Q keys < x
            while (qlo < qhi) {
                uint q = (qlo + qhi + 1u) / 2u;
                if (tp[q0 + q] < x) qlo = q;
                else qhi = q - 1u;
            }
            if (k - 1u + qlo < m) lo = k;
            else hi = k - 1u;
        }
        uint i = lo, j = m - i;
        bool advP = i == a ? false : j == b ? true : tp[i + 1u] <= tp[q0 + j + 1u];
        return advP ? mgl_tess_wind(T, co, i, i + 1u, q0 + j) : mgl_tess_wind(T, co, i, q0 + j + 1u, q0 + j);
    }
    if (T.domain == 0u) return mgl_tess_wind(T, co, 0u, 1u, 2u);
    uint du = T.ni[0] - 2u * T.rings, cell = m / 2u, iu = cell % du, jv = cell / du;
    uint p00 = jv * (du + 1u) + iu, p10 = p00 + 1u, p01 = p00 + du + 1u, p11 = p01 + 1u;
    return (m & 1u) == 0u ? mgl_tess_wind(T, co, p00, p10, p11) : mgl_tess_wind(T, co, p00, p11, p01);
}

// Pieces: units split into runs of at most MGL_TESS_PIECE primitives, so that a mesh threadgroup needs room for
// only MGL_TESS_PIECE + 2 vertices (fewer outputs per threadgroup, more threadgroups in flight). Vertices on the
// boundary between two pieces are generated by both, with the same values. The pieces of a unit produce its
// primitives in order:
// * strip: merge steps m0 .. m1 - 1 (m0 = piece * MGL_TESS_PIECE); vertices: the outer row's points i0 .. i1,
//   then the inner row's j0 .. j1, where (i0, j0) / (i1, j1) are the merge's positions before steps m0 / m1
//   (mgl_tess_piece_bounds)
// * isoline: segments m0 .. m1 - 1, vertices m0 .. m1; points: points m0 .. m1 - 1
// * quad centre (a 1 x k or k x 1 grid of cells, 2 triangles each): cells c0 .. c1 - 1
#define MGL_TESS_PIECE 32u

inline uint mgl_tess_unit_pieces(thread const mgl_tess& T, uint unit) {
    return max(1u, (mgl_tess_unit_size(T, unit).y + MGL_TESS_PIECE - 1u) / MGL_TESS_PIECE);
}

inline bool mgl_tess_is_strip(thread const mgl_tess& T, uint unit) {
    return T.domain != 2u && T.special == 0u && T.points == 0u && unit < (T.domain == 0u ? 3u : 4u) * T.rings;
}

// Outer-row points the strip's merge has taken before step m (see mgl_tess_unit_prim), from the rows' parameters.
inline uint mgl_tess_merge_pos(thread const mgl_tess& T, uint r, uint s, uint a, uint b, uint m) {
    float tt;
    uint lo = 0u, hi = min(m, a);
    while (lo < hi) {
        uint k = (lo + hi + 1u) / 2u;
        mgl_tess_rowpt(T, r, s, k, tt);
        float x = tt;
        uint qlo = 0u, qhi = b;
        while (qlo < qhi) {
            uint q = (qlo + qhi + 1u) / 2u;
            mgl_tess_rowpt(T, r + 1u, s, q, tt);
            if (tt < x) qlo = q;
            else qhi = q - 1u;
        }
        if (k - 1u + qlo < m) lo = k;
        else hi = k - 1u;
    }
    return lo;
}

inline bool mgl_tess_is_qcentre(thread const mgl_tess& T, uint unit) {
    return T.domain == 1u && T.special == 0u && T.points == 0u && unit == 4u * T.rings;
}

// Strips: (i0, j0, i1, j1); quad centre: cells (c0, c1); isolines / points: (first unit vertex m0); others 0.
inline uint4 mgl_tess_piece_bounds(thread const mgl_tess& T, uint unit, uint piece) {
    uint m0 = piece * MGL_TESS_PIECE;
    uint np = mgl_tess_unit_size(T, unit).y, m1 = min(m0 + MGL_TESS_PIECE, np);
    if (mgl_tess_is_strip(T, unit)) {
        uint sides = T.domain == 0u ? 3u : 4u, r = unit / sides, s = unit % sides;
        uint a = mgl_tess_rowsegs(T, r, s), b = mgl_tess_rowsegs(T, r + 1u, s);
        uint i0 = m0 == 0u ? 0u : mgl_tess_merge_pos(T, r, s, a, b, m0);
        uint i1 = m1 == a + b ? a : mgl_tess_merge_pos(T, r, s, a, b, m1);
        return uint4(i0, m0 - i0, i1, m1 - i1);
    }
    if (mgl_tess_is_qcentre(T, unit)) return uint4(m0 / 2u, m1 / 2u, 0u, 0u);
    if (T.special != 0u || (T.domain == 0u && T.points == 0u)) return uint4(0u);  // one triangle / quad, triangle centre
    return uint4(m0, 0u, 0u, 0u);
}

// (vertices, primitives) of the piece; u = mgl_tess_unit_size(T, unit)
inline uint2 mgl_tess_piece_size(thread const mgl_tess& T, uint unit, uint piece, uint2 u) {
    uint m0 = piece * MGL_TESS_PIECE, np = min(u.y - min(u.y, m0), MGL_TESS_PIECE);
    if (mgl_tess_is_strip(T, unit)) return uint2(np + 2u, np);
    if (mgl_tess_is_qcentre(T, unit)) return uint2(2u * (np / 2u + 1u), np);
    if (T.points != 0u) return uint2(np, np);
    if (T.domain == 2u) return uint2(np + 1u, np);
    return u;  // one triangle / quad, triangle centre
}

inline uint2 mgl_tess_piece_size(thread const mgl_tess& T, uint unit, uint piece) {
    return mgl_tess_piece_size(T, unit, piece, mgl_tess_unit_size(T, unit));
}

inline float3 mgl_tess_piece_vertex(thread const mgl_tess& T, uint unit, uint4 pb, uint k, thread float& tparam) {
    uint uk = pb.x + k;
    if (mgl_tess_is_strip(T, unit)) {  // outer points i0 .. i1, then inner points j0 .. j1 (unit vertex a + 1 + j)
        uint sides = T.domain == 0u ? 3u : 4u, a = mgl_tess_rowsegs(T, unit / sides, unit % sides);
        uk = k <= pb.z - pb.x ? pb.x + k : a + 1u + pb.y + (k - (pb.z - pb.x + 1u));
    } else if (mgl_tess_is_qcentre(T, unit)) {
        uint du = T.ni[0] - 2u * T.rings, cnt = pb.y - pb.x + 1u;  // points per row of the piece
        if (du == 1u) uk = 2u * pb.x + k;                                       // 1 x k: rows c0 .. c1, 2 points each
        else uk = k < cnt ? pb.x + k : du + 1u + pb.x + (k - cnt);              // k x 1: bottom row, then top row
    } else if (T.special != 0u || (T.domain == 0u && T.points == 0u)) {
        uk = k;
    }
    return mgl_tess_unit_vertex(T, unit, uk, tparam);
}

// Primitive m of the piece, in piece-local vertex indices (co / tp: the piece's vertices and their parameters).
inline uint3 mgl_tess_piece_prim(thread const mgl_tess& T, uint unit, uint4 pb, uint m, threadgroup const float3* co, threadgroup const float* tp) {
    if (mgl_tess_is_strip(T, unit)) {
        // the piece's part of the strip's merge (mgl_tess_unit_prim), with rows i0 .. i1 and j0 .. j1
        uint a = pb.z - pb.x, b = pb.w - pb.y, q0 = a + 1u;
        uint lo = 0u, hi = min(m, a);
        while (lo < hi) {
            uint k = (lo + hi + 1u) / 2u;
            float x = tp[k];
            uint qlo = 0u, qhi = b;
            while (qlo < qhi) {
                uint q = (qlo + qhi + 1u) / 2u;
                if (tp[q0 + q] < x) qlo = q;
                else qhi = q - 1u;
            }
            if (k - 1u + qlo < m) lo = k;
            else hi = k - 1u;
        }
        uint i = lo, j = m - i;
        bool advP = i == a ? false : j == b ? true : tp[i + 1u] <= tp[q0 + j + 1u];
        return advP ? mgl_tess_wind(T, co, i, i + 1u, q0 + j) : mgl_tess_wind(T, co, i, q0 + j + 1u, q0 + j);
    }
    if (mgl_tess_is_qcentre(T, unit)) {
        uint du = T.ni[0] - 2u * T.rings, w = du == 1u ? 2u : pb.y - pb.x + 1u, cl = m / 2u;
        uint p00 = du == 1u ? 2u * cl : cl, p10 = p00 + 1u, p01 = p00 + w, p11 = p01 + 1u;
        return (m & 1u) == 0u ? mgl_tess_wind(T, co, p00, p10, p11) : mgl_tess_wind(T, co, p00, p11, p01);
    }
    return mgl_tess_unit_prim(T, unit, m, co, tp);  // points, isolines (local from m0), one triangle / quad, triangle centre
}

//@chunk mgl_tess_metal_factors
// Metal's tessellator (VariantKey::tessMetal): the GL levels of grid patch gp as MTLTriangleTessellationFactorsHalf /
// MTLQuadTessellationFactorsHalf. An outer level <= 0 or NaN culls the patch (factors 0); otherwise levels are
// raised to 1 (inner levels NaN or below 1 too), which changes nothing as both GL and Metal clamp them to the
// spacing's minimum, and keeps small positive levels from rounding to a half-precision 0. Levels round up to a
// half: 35.011 must still make 36 segments (half(35.011) is 35).
inline half mgl_tess_metal_level(float x) {
    half h = half(x);
    return float(h) < x ? as_type<half>(ushort(as_type<ushort>(h) + 1u)) : h;
}
template <typename P> inline void mgl_tess_metal_factors(device half* f, uint gp, P lv, bool quads) {
    uint n = quads ? 4u : 3u;
    bool cull = false;
    for (uint i = 0; i < n; i++) cull = cull || !(lv[i] > 0.0f);
    device half* o = f + gp * (quads ? 6u : 4u);
    for (uint i = 0; i < n; i++) o[i] = cull ? 0.0h : mgl_tess_metal_level(clamp(lv[i], 1.0f, 64.0f));
    for (uint i = 0; i < (quads ? 2u : 1u); i++) o[n + i] = cull ? 0.0h : mgl_tess_metal_level(lv[4u + i] > 1.0f ? min(lv[4u + i], 64.0f) : 1.0f);
}
