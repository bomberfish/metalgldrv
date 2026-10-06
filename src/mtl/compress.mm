// Simple, fast block encoders (bounding-box endpoints + nearest-palette indices).
// Quality is that of a basic real-time encoder; they exist so that compressed
// internal formats specified with uncompressed data are stored as GL requires.
#import "mtl/compress.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace mgl {
namespace {

struct Block {
    float p[16][4];  // texel (x, y) at p[y * 4 + x]
};

inline int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }
inline float clampf(float v, float lo, float hi) { return v < lo ? lo : v > hi ? hi : v; }
inline int toU8(float v) { return clampi((int)lrintf(clampf(v, 0, 1) * 255.0f), 0, 255); }

void put64le(uint8_t* o, uint64_t v) {
    for (int i = 0; i < 8; i++) o[i] = (uint8_t)(v >> (8 * i));
}
void put64be(uint8_t* o, uint64_t v) {
    for (int i = 0; i < 8; i++) o[i] = (uint8_t)(v >> (56 - 8 * i));
}

// --- BC1 color ----------------------------------------------------------------
inline uint16_t pack565(const int c[3]) {
    return (uint16_t)(((c[0] * 31 + 127) / 255) << 11 | ((c[1] * 63 + 127) / 255) << 5 | ((c[2] * 31 + 127) / 255));
}
inline void unpack565(uint16_t v, int c[3]) {
    int r = v >> 11 & 31, g = v >> 5 & 63, b = v & 31;
    c[0] = r << 3 | r >> 2;
    c[1] = g << 2 | g >> 4;
    c[2] = b << 3 | b >> 2;
}

// The used points with the smallest / largest projection on their principal axis
// (covariance power iteration), dims = 3 or 4 leading components.
void principalExtremes(const float (*pt)[4], const bool* use, int dims, int* imin, int* imax) {
    float mean[4] = {}, n = 0;
    for (int i = 0; i < 16; i++) {
        if (use && !use[i]) continue;
        for (int k = 0; k < dims; k++) mean[k] += pt[i][k];
        n++;
    }
    *imin = *imax = -1;
    if (n == 0) return;
    for (int k = 0; k < dims; k++) mean[k] /= n;
    float cov[4][4] = {};
    for (int i = 0; i < 16; i++) {
        if (use && !use[i]) continue;
        for (int a = 0; a < dims; a++)
            for (int c = 0; c < dims; c++) cov[a][c] += (pt[i][a] - mean[a]) * (pt[i][c] - mean[c]);
    }
    float ax[4] = {1, 1, 1, 1};
    for (int it = 0; it < 8; it++) {
        float v[4] = {}, len = 0;
        for (int a = 0; a < dims; a++) {
            for (int c = 0; c < dims; c++) v[a] += cov[a][c] * ax[c];
            len = std::max(len, std::fabs(v[a]));
        }
        if (len < 1e-12f) break;
        for (int a = 0; a < dims; a++) ax[a] = v[a] / len;
    }
    float pmin = 1e30f, pmax = -1e30f;
    for (int i = 0; i < 16; i++) {
        if (use && !use[i]) continue;
        float d = 0;
        for (int k = 0; k < dims; k++) d += pt[i][k] * ax[k];
        if (d < pmin) pmin = d, *imin = i;
        if (d > pmax) pmax = d, *imax = i;
    }
}

void colorEndpoints(const Block& b, const bool* use, int lo[3], int hi[3]) {
    int imin, imax;
    principalExtremes(b.p, use, 3, &imin, &imax);
    for (int k = 0; k < 3; k++) {
        lo[k] = imin < 0 ? 0 : toU8(b.p[imin][k]);
        hi[k] = imax < 0 ? 0 : toU8(b.p[imax][k]);
    }
}

// alphaMode: true for BC1 with 1-bit alpha (texels with alpha < 0.5 become transparent).
// fourColor: BC2/BC3 color blocks always decode in four-color mode.
void encodeBC1Color(const Block& b, uint8_t* out, bool alphaMode, bool fourColor) {
    bool use[16], anyTransparent = false, anyOpaque = false;
    for (int i = 0; i < 16; i++) {
        use[i] = !alphaMode || b.p[i][3] >= 0.5f;
        anyTransparent |= !use[i];
        anyOpaque |= use[i];
    }
    int lo[3], hi[3];
    colorEndpoints(b, use, lo, hi);
    uint16_t c0 = pack565(hi), c1 = pack565(lo);
    bool threeColor = anyTransparent && !fourColor;
    if (threeColor) {
        if (c0 > c1) std::swap(c0, c1);
    } else if (!fourColor) {
        if (c0 < c1) std::swap(c0, c1);
        if (c0 == c1) threeColor = true;  // c0 == c1 decodes in three-color mode; index 0 is exact
    }
    int e0[3], e1[3], pal[4][3];
    unpack565(c0, e0);
    unpack565(c1, e1);
    int npal;
    for (int k = 0; k < 3; k++) {
        pal[0][k] = e0[k];
        pal[1][k] = e1[k];
        if (threeColor) {
            pal[2][k] = (e0[k] + e1[k]) / 2;
            pal[3][k] = 0;
        } else {
            pal[2][k] = (2 * e0[k] + e1[k]) / 3;
            pal[3][k] = (e0[k] + 2 * e1[k]) / 3;
        }
    }
    npal = threeColor ? 3 : 4;
    uint32_t idx = 0;
    for (int i = 0; i < 16; i++) {
        int sel;
        if (!use[i] && threeColor) {
            sel = 3;
        } else {
            int c[3] = {toU8(b.p[i][0]), toU8(b.p[i][1]), toU8(b.p[i][2])};
            int best = 1 << 30;
            sel = 0;
            for (int j = 0; j < npal; j++) {
                int e = 0;
                for (int k = 0; k < 3; k++) e += (c[k] - pal[j][k]) * (c[k] - pal[j][k]);
                if (e < best) best = e, sel = j;
            }
        }
        idx |= (uint32_t)sel << (2 * i);
    }
    (void)anyOpaque;
    out[0] = c0 & 0xff;
    out[1] = c0 >> 8;
    out[2] = c1 & 0xff;
    out[3] = c1 >> 8;
    for (int i = 0; i < 4; i++) out[4 + i] = (uint8_t)(idx >> (8 * i));
}

// --- BC4 (one channel) -----------------------------------------------------------
void encodeBC4(const Block& b, int ch, bool snorm, uint8_t* out) {
    int v[16], mn = 1 << 20, mx = -(1 << 20);
    for (int i = 0; i < 16; i++) {
        float f = b.p[i][ch];
        v[i] = snorm ? clampi((int)lrintf(clampf(f, -1, 1) * 127.0f), -127, 127) : toU8(f);
        mn = std::min(mn, v[i]);
        mx = std::max(mx, v[i]);
    }
    // Eight-value mode needs e0 > e1.
    int e0 = mx, e1 = mn;
    int pal[8];
    pal[0] = e0;
    pal[1] = e1;
    for (int j = 2; j < 8; j++) pal[j] = ((8 - j) * e0 + (j - 1) * e1) / 7;
    uint64_t bits = 0;
    for (int i = 0; i < 16; i++) {
        int sel = 0, best = 1 << 30;
        if (e0 != e1)
            for (int j = 0; j < 8; j++) {
                int e = std::abs(v[i] - pal[j]);
                if (e < best) best = e, sel = j;
            }
        bits |= (uint64_t)sel << (3 * i);
    }
    out[0] = (uint8_t)(snorm ? (int8_t)e0 : e0);
    out[1] = (uint8_t)(snorm ? (int8_t)e1 : e1);
    for (int i = 0; i < 6; i++) out[2 + i] = (uint8_t)(bits >> (8 * i));
}

void encodeBC2Alpha(const Block& b, uint8_t* out) {
    uint64_t bits = 0;
    for (int i = 0; i < 16; i++) bits |= (uint64_t)clampi((int)lrintf(clampf(b.p[i][3], 0, 1) * 15.0f), 0, 15) << (4 * i);
    put64le(out, bits);
}

// --- BC7 mode 6 (one subset, RGBA 7.7.7.7 + p-bit endpoints, 4-bit indices) ---------
struct BitWriter {
    uint8_t* o;
    int pos = 0;
    void put(uint32_t v, int n) {
        for (int i = 0; i < n; i++, pos++)
            if (v >> i & 1) o[pos >> 3] |= (uint8_t)(1 << (pos & 7));
    }
};

const int kW4[16] = {0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};

void encodeBC7(const Block& b, uint8_t* out) {
    int imin, imax;
    principalExtremes(b.p, nullptr, 4, &imin, &imax);
    int ep[2][4], q[2][4], pb[2];
    for (int k = 0; k < 4; k++) {
        ep[0][k] = toU8(b.p[imin][k]);
        ep[1][k] = toU8(b.p[imax][k]);
    }
    // Quantize to 7 bits + shared p-bit, choosing the p-bit with the least error.
    int deq[2][4];
    for (int e = 0; e < 2; e++) {
        int bestErr = 1 << 30;
        for (int p = 0; p < 2; p++) {
            int err = 0, qq[4], dd[4];
            for (int k = 0; k < 4; k++) {
                qq[k] = clampi((ep[e][k] - p + 1) >> 1, 0, 127);
                dd[k] = qq[k] << 1 | p;
                err += std::abs(dd[k] - ep[e][k]);
            }
            if (err < bestErr) {
                bestErr = err;
                pb[e] = p;
                memcpy(q[e], qq, sizeof qq);
                memcpy(deq[e], dd, sizeof dd);
            }
        }
    }
    int idx[16];
    for (int i = 0; i < 16; i++) {
        int c[4] = {toU8(b.p[i][0]), toU8(b.p[i][1]), toU8(b.p[i][2]), toU8(b.p[i][3])};
        int best = 1 << 30;
        idx[i] = 0;
        for (int j = 0; j < 16; j++) {
            int e = 0;
            for (int k = 0; k < 4; k++) {
                int v = ((64 - kW4[j]) * deq[0][k] + kW4[j] * deq[1][k] + 32) >> 6;
                e += (v - c[k]) * (v - c[k]);
            }
            if (e < best) best = e, idx[i] = j;
        }
    }
    if (idx[0] >= 8) {  // the anchor index's top bit is implicit 0
        std::swap(q[0], q[1]);
        std::swap(pb[0], pb[1]);
        for (int i = 0; i < 16; i++) idx[i] = 15 - idx[i];
    }
    memset(out, 0, 16);
    BitWriter w{out};
    w.put(1u << 6, 7);
    for (int k = 0; k < 4; k++) {
        w.put(q[0][k], 7);
        w.put(q[1][k], 7);
    }
    w.put(pb[0], 1);
    w.put(pb[1], 1);
    for (int i = 0; i < 16; i++) w.put(idx[i], i == 0 ? 3 : 4);
}

// --- BC6H mode 11 (one region, 10-bit endpoints, 4-bit indices) -------------------
uint16_t floatToHalf(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = x >> 16 & 0x8000;
    int e = (int)(x >> 23 & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffff;
    if ((x & 0x7fffffff) >= 0x7f800000) return (uint16_t)(sign | 0x7bff);  // inf/nan -> max finite
    if (e >= 31) return (uint16_t)(sign | 0x7bff);
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        m |= 0x800000;
        uint32_t s = (uint32_t)(14 - e);
        uint32_t h = m >> s;
        if ((m >> (s - 1)) & 1) h++;
        return (uint16_t)(sign | h);
    }
    uint32_t h = (uint32_t)e << 10 | m >> 13;
    if (m & 0x1000) h++;
    return (uint16_t)(sign | std::min<uint32_t>(h, 0x7bff));
}

// Signed-magnitude half bit pattern (as an int) -> float.
float halfBitsToFloat(int v) {
    bool neg = v < 0;
    int m = neg ? -v : v;
    int e = m >> 10, f = m & 0x3ff;
    float r = e == 0 ? std::ldexp((float)f, -24) : std::ldexp((float)(f | 0x400), e - 25);
    return neg ? -r : r;
}

int bc6Unquantize(int q, bool sgn) {
    if (!sgn) {
        if (q == 0) return 0;
        if (q == 1023) return 0xffff;
        return ((q << 16) + 0x8000) >> 10;
    }
    bool neg = q < 0;
    int c = neg ? -q : q, u;
    if (c == 0) u = 0;
    else if (c >= 511) u = 0x7fff;
    else u = ((c << 15) + 0x4000) >> 9;
    return neg ? -u : u;
}
int bc6Finish(int v, bool sgn) {
    if (!sgn) return (v * 31) >> 6;
    return v < 0 ? -(((-v) * 31) >> 5) : (v * 31) >> 5;
}

void encodeBC6H(const Block& b, bool sgn, uint8_t* out) {
    // Work on half-float bit patterns (signed formats: sign-magnitude -> int).
    int h[16][3];
    for (int i = 0; i < 16; i++)
        for (int k = 0; k < 3; k++) {
            float f = b.p[i][k];
            if (!sgn) f = std::max(f, 0.0f);
            uint16_t hv = floatToHalf(f);
            h[i][k] = (hv & 0x8000) ? -(int)(hv & 0x7fff) : (int)(hv & 0x7fff);
        }
    // Endpoints: the extremes along the principal axis in half-bit space (where BC6H interpolates).
    float hp[16][4];
    for (int i = 0; i < 16; i++)
        for (int k = 0; k < 4; k++) hp[i][k] = k < 3 ? (float)h[i][k] : 0;
    int imin, imax;
    principalExtremes(hp, nullptr, 3, &imin, &imax);
    int lo[3], hi[3];
    for (int k = 0; k < 3; k++) {
        double s = sgn ? 62.0 : 31.0;
        lo[k] = sgn ? clampi((int)lround(h[imin][k] / s), -511, 511) : clampi((int)lround(h[imin][k] / s), 0, 1023);
        hi[k] = sgn ? clampi((int)lround(h[imax][k] / s), -511, 511) : clampi((int)lround(h[imax][k] / s), 0, 1023);
    }
    int pal[16][3];
    for (int j = 0; j < 16; j++)
        for (int k = 0; k < 3; k++) {
            int a = bc6Unquantize(lo[k], sgn), c = bc6Unquantize(hi[k], sgn);
            pal[j][k] = bc6Finish(((64 - kW4[j]) * a + kW4[j] * c + 32) >> 6, sgn);
        }
    // Pick indices by error in value space (the palette is not linear in value).
    float palf[16][3];
    for (int j = 0; j < 16; j++)
        for (int k = 0; k < 3; k++) palf[j][k] = halfBitsToFloat(pal[j][k]);
    int idx[16];
    for (int i = 0; i < 16; i++) {
        float best = -1;
        idx[i] = 0;
        for (int j = 0; j < 16; j++) {
            float e = 0;
            for (int k = 0; k < 3; k++) {
                float d = palf[j][k] - b.p[i][k];
                e += d * d;
            }
            if (best < 0 || e < best) best = e, idx[i] = j;
        }
    }
    if (idx[0] >= 8) {
        for (int k = 0; k < 3; k++) std::swap(lo[k], hi[k]);
        for (int i = 0; i < 16; i++) idx[i] = 15 - idx[i];
    }
    memset(out, 0, 16);
    BitWriter w{out};
    w.put(0x03, 5);  // mode 11
    for (int k = 0; k < 3; k++) w.put((uint32_t)lo[k] & 0x3ff, 10);
    for (int k = 0; k < 3; k++) w.put((uint32_t)hi[k] & 0x3ff, 10);
    for (int i = 0; i < 16; i++) w.put(idx[i], i == 0 ? 3 : 4);
}

// --- ETC2 / EAC ------------------------------------------------------------------
const int kEtcMod[8][4] = {{2, 8, -2, -8},      {5, 17, -5, -17},    {9, 29, -9, -29},    {13, 42, -13, -42},
                           {18, 60, -18, -60},  {24, 80, -24, -80},  {33, 106, -33, -106}, {47, 183, -47, -183}};

// Pixel index bit position of texel (x, y): ETC orders texels column-major.
inline int etcPos(int x, int y) { return x * 4 + y; }

// Best table + per-texel modifier for one 2x4 subblock around `base` (8-bit RGB).
// allowed: bitmask of modifier indices usable for opaque texels; skip: texels not encoded here.
int etcSubblock(const Block& b, int sx, const int base[3], uint8_t allowed, const bool* transparent, int* table, int sel[16]) {
    int bestErr = 1 << 30;
    for (int t = 0; t < 8; t++) {
        int err = 0, s[16];
        for (int x = sx; x < sx + 2; x++)
            for (int y = 0; y < 4; y++) {
                int i = y * 4 + x;
                if (transparent && transparent[i]) continue;
                int c[3] = {toU8(b.p[i][0]), toU8(b.p[i][1]), toU8(b.p[i][2])};
                int be = 1 << 30;
                for (int m = 0; m < 4; m++) {
                    if (!(allowed >> m & 1)) continue;
                    int mod = (allowed == 0xf || m == 1 || m == 3) ? kEtcMod[t][m] : 0;
                    int e = 0;
                    for (int k = 0; k < 3; k++) {
                        int v = clampi(base[k] + mod, 0, 255);
                        e += (v - c[k]) * (v - c[k]);
                    }
                    if (e < be) be = e, s[i] = m;
                }
                err += be;
            }
        if (err < bestErr) {
            bestErr = err;
            *table = t;
            for (int x = sx; x < sx + 2; x++)
                for (int y = 0; y < 4; y++) sel[y * 4 + x] = s[y * 4 + x];
        }
    }
    return bestErr;
}

void avgColor(const Block& b, int sx, const bool* transparent, float out[3]) {
    float s[3] = {0, 0, 0};
    int n = 0;
    for (int x = sx; x < sx + 2; x++)
        for (int y = 0; y < 4; y++) {
            int i = y * 4 + x;
            if (transparent && transparent[i]) continue;
            for (int k = 0; k < 3; k++) s[k] += clampf(b.p[i][k], 0, 1);
            n++;
        }
    for (int k = 0; k < 3; k++) out[k] = n ? s[k] / n : 0;
}

// ETC2 RGB8 (individual mode, no flip) or, with punchthrough, the RGB8A1 differential block.
void encodeETC2(const Block& b, bool punchthrough, uint8_t* out) {
    bool transparent[16] = {};
    bool anyTransparent = false;
    if (punchthrough)
        for (int i = 0; i < 16; i++) anyTransparent |= transparent[i] = b.p[i][3] < 0.5f;
    int sel[16] = {}, t1 = 0, t2 = 0;
    uint64_t v = 0;
    if (!punchthrough) {
        int b4[2][3], base[2][3];
        for (int s = 0; s < 2; s++) {
            float a[3];
            avgColor(b, s * 2, nullptr, a);
            for (int k = 0; k < 3; k++) {
                b4[s][k] = clampi((int)lrintf(a[k] * 15.0f), 0, 15);
                base[s][k] = b4[s][k] << 4 | b4[s][k];
            }
        }
        etcSubblock(b, 0, base[0], 0xf, nullptr, &t1, sel);
        etcSubblock(b, 2, base[1], 0xf, nullptr, &t2, sel);
        v = (uint64_t)b4[0][0] << 60 | (uint64_t)b4[1][0] << 56 | (uint64_t)b4[0][1] << 52 | (uint64_t)b4[1][1] << 48 |
            (uint64_t)b4[0][2] << 44 | (uint64_t)b4[1][2] << 40;
    } else {
        int b5[3], d[3], base[2][3];
        float a0[3], a1[3];
        avgColor(b, 0, transparent, a0);
        avgColor(b, 2, transparent, a1);
        for (int k = 0; k < 3; k++) {
            b5[k] = clampi((int)lrintf(a0[k] * 31.0f), 0, 31);
            int c2 = clampi((int)lrintf(a1[k] * 31.0f), 0, 31);
            d[k] = clampi(c2 - b5[k], -4, 3);  // b5 + d stays in [0, 31]: no T/H/planar mode
            int e2 = b5[k] + d[k];
            base[0][k] = b5[k] << 3 | b5[k] >> 2;
            base[1][k] = e2 << 3 | e2 >> 2;
        }
        // Opaque blocks use the regular modifiers; otherwise index 2 is transparent and
        // indices 0/2 carry no modifier.
        uint8_t allowed = anyTransparent ? 0xb : 0xf;
        etcSubblock(b, 0, base[0], allowed, transparent, &t1, sel);
        etcSubblock(b, 2, base[1], allowed, transparent, &t2, sel);
        for (int i = 0; i < 16; i++)
            if (transparent[i]) sel[i] = 2;
        v = (uint64_t)b5[0] << 59 | (uint64_t)(d[0] & 7) << 56 | (uint64_t)b5[1] << 51 | (uint64_t)(d[1] & 7) << 48 |
            (uint64_t)b5[2] << 43 | (uint64_t)(d[2] & 7) << 40;
        if (!anyTransparent) v |= 1ull << 33;  // opaque bit
    }
    if (!punchthrough) v |= 0ull << 33;  // individual mode
    v |= (uint64_t)t1 << 37 | (uint64_t)t2 << 34;
    // Selector m: 0 = +a, 1 = +b, 2 = -a, 3 = -b -> (msb, lsb) = (m >> 1, m & 1).
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++) {
            int m = sel[y * 4 + x], p = etcPos(x, y);
            v |= (uint64_t)(m >> 1) << (16 + p) | (uint64_t)(m & 1) << p;
        }
    put64be(out, v);
}

const int kEacMod[16][8] = {
    {-3, -6, -9, -15, 2, 5, 8, 14}, {-3, -7, -10, -13, 2, 6, 9, 12}, {-2, -5, -8, -13, 1, 4, 7, 12}, {-2, -4, -6, -13, 1, 3, 5, 12},
    {-3, -6, -8, -12, 2, 5, 7, 11}, {-3, -7, -9, -11, 2, 6, 8, 10},  {-4, -7, -8, -11, 3, 6, 7, 10}, {-3, -5, -8, -11, 2, 4, 7, 10},
    {-2, -6, -8, -10, 1, 5, 7, 9},  {-2, -5, -8, -10, 1, 4, 7, 9},   {-2, -4, -8, -10, 1, 3, 7, 9},  {-2, -5, -7, -10, 1, 4, 6, 9},
    {-3, -4, -7, -10, 2, 3, 6, 9},  {-1, -2, -3, -10, 0, 1, 2, 9},   {-4, -6, -8, -9, 3, 5, 7, 8},   {-3, -5, -7, -9, 2, 4, 6, 8}};

// EAC: mode 0 = 8-bit alpha, 1 = R11 unsigned, 2 = R11 signed.
void encodeEAC(const Block& b, int ch, int mode, uint8_t* out) {
    int tgt[16], mn = 1 << 30, mx = -(1 << 30);
    for (int i = 0; i < 16; i++) {
        float f = b.p[i][ch];
        tgt[i] = mode == 0 ? toU8(f) : mode == 1 ? (int)lrintf(clampf(f, 0, 1) * 2047.0f) : (int)lrintf(clampf(f, -1, 1) * 1023.0f);
        mn = std::min(mn, tgt[i]);
        mx = std::max(mx, tgt[i]);
    }
    auto decode = [&](int base, int mul, int mod) {
        if (mode == 0) return clampi(base + mod * mul, 0, 255);
        int m = mul ? mul * 8 : 1;
        if (mode == 1) return clampi(base * 8 + 4 + mod * m, 0, 2047);
        return clampi(base * 8 + mod * m, -1023, 1023);
    };
    int scale = mode == 0 ? 1 : 8;
    long long bestErr = -1;
    int bb = 0, bm = 1, bt = 0, bsel[16] = {};
    int mid = (mn + mx) / 2;
    int baseC = mode == 0 ? clampi(mid, 0, 255) : mode == 1 ? clampi((mid - 4) / 8, 0, 255) : clampi(mid / 8, -127, 127);
    for (int t = 0; t < 16; t++) {
        int span = kEacMod[t][7] - kEacMod[t][3];
        int m0 = clampi((int)lrintf((float)(mx - mn) / (span * scale)), 1, 15);
        for (int mul = std::max(1, m0 - 1); mul <= std::min(15, m0 + 1); mul++)
            for (int db = -1; db <= 1; db++) {
                int base = mode == 2 ? clampi(baseC + db, -127, 127) : clampi(baseC + db, 0, 255);
                long long err = 0;
                int sel[16];
                for (int i = 0; i < 16; i++) {
                    int be = 1 << 30;
                    for (int j = 0; j < 8; j++) {
                        int e = std::abs(decode(base, mul, kEacMod[t][j]) - tgt[i]);
                        if (e < be) be = e, sel[i] = j;
                    }
                    err += (long long)be * be;
                }
                if (bestErr < 0 || err < bestErr) {
                    bestErr = err;
                    bb = base, bm = mul, bt = t;
                    memcpy(bsel, sel, sizeof sel);
                }
            }
    }
    uint64_t v = (uint64_t)(uint8_t)bb << 56 | (uint64_t)bm << 52 | (uint64_t)bt << 48;
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++) v |= (uint64_t)bsel[y * 4 + x] << (45 - 3 * etcPos(x, y));
    put64be(out, v);
}

}  // namespace

bool cpuCompressible(const FormatInfo* f) {
    if (!f || f->comp != Comp::Compressed) return false;
    switch (f->internalFormat) {
    case GL_COMPRESSED_RGB_S3TC_DXT1_EXT: case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT: case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT:
    case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT: case GL_COMPRESSED_SRGB_S3TC_DXT1_EXT: case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT:
    case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT: case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT:
    case GL_COMPRESSED_RED_RGTC1: case GL_COMPRESSED_SIGNED_RED_RGTC1: case GL_COMPRESSED_RG_RGTC2: case GL_COMPRESSED_SIGNED_RG_RGTC2:
    case GL_COMPRESSED_RGBA_BPTC_UNORM: case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM: case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT:
    case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:
    case GL_COMPRESSED_RGB8_ETC2: case GL_COMPRESSED_SRGB8_ETC2: case GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2:
    case GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2: case GL_COMPRESSED_RGBA8_ETC2_EAC: case GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:
    case GL_COMPRESSED_R11_EAC: case GL_COMPRESSED_SIGNED_R11_EAC: case GL_COMPRESSED_RG11_EAC: case GL_COMPRESSED_SIGNED_RG11_EAC:
        return true;
    default: return false;  // ASTC
    }
}

void compressImage(const FormatInfo* f, const float* rgba, int w, int h, uint8_t* out) {
    int bx = (w + 3) / 4, by = (h + 3) / 4;
    size_t bs = f->bytesPerPixel;
    for (int j = 0; j < by; j++)
        for (int i = 0; i < bx; i++) {
            Block b;
            // Texels past the image edge replicate the last row/column.
            for (int y = 0; y < 4; y++)
                for (int x = 0; x < 4; x++) {
                    int sx = std::min(i * 4 + x, w - 1), sy = std::min(j * 4 + y, h - 1);
                    memcpy(b.p[y * 4 + x], rgba + ((size_t)sy * w + sx) * 4, 16);
                }
            uint8_t* o = out + ((size_t)j * bx + i) * bs;
            switch (f->internalFormat) {
            case GL_COMPRESSED_RGB_S3TC_DXT1_EXT: case GL_COMPRESSED_SRGB_S3TC_DXT1_EXT: encodeBC1Color(b, o, false, false); break;
            case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT: case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT: encodeBC1Color(b, o, true, false); break;
            case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT: case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT:
                encodeBC2Alpha(b, o);
                encodeBC1Color(b, o + 8, false, true);
                break;
            case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT: case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT:
                encodeBC4(b, 3, false, o);
                encodeBC1Color(b, o + 8, false, true);
                break;
            case GL_COMPRESSED_RED_RGTC1: encodeBC4(b, 0, false, o); break;
            case GL_COMPRESSED_SIGNED_RED_RGTC1: encodeBC4(b, 0, true, o); break;
            case GL_COMPRESSED_RG_RGTC2:
                encodeBC4(b, 0, false, o);
                encodeBC4(b, 1, false, o + 8);
                break;
            case GL_COMPRESSED_SIGNED_RG_RGTC2:
                encodeBC4(b, 0, true, o);
                encodeBC4(b, 1, true, o + 8);
                break;
            case GL_COMPRESSED_RGBA_BPTC_UNORM: case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM: encodeBC7(b, o); break;
            case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT: encodeBC6H(b, false, o); break;
            case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT: encodeBC6H(b, true, o); break;
            case GL_COMPRESSED_RGB8_ETC2: case GL_COMPRESSED_SRGB8_ETC2: encodeETC2(b, false, o); break;
            case GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2: case GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2: encodeETC2(b, true, o); break;
            case GL_COMPRESSED_RGBA8_ETC2_EAC: case GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:
                encodeEAC(b, 3, 0, o);
                encodeETC2(b, false, o + 8);
                break;
            case GL_COMPRESSED_R11_EAC: encodeEAC(b, 0, 1, o); break;
            case GL_COMPRESSED_SIGNED_R11_EAC: encodeEAC(b, 0, 2, o); break;
            case GL_COMPRESSED_RG11_EAC:
                encodeEAC(b, 0, 1, o);
                encodeEAC(b, 1, 1, o + 8);
                break;
            case GL_COMPRESSED_SIGNED_RG11_EAC:
                encodeEAC(b, 0, 2, o);
                encodeEAC(b, 1, 2, o + 8);
                break;
            default: memset(o, 0, bs); break;
            }
        }
}

}  // namespace mgl
