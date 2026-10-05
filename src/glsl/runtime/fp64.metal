// metalgldrv soft double precision for MSL.
//
// GLSL `double` is represented as `ulong` holding the IEEE-754 binary64 bit
// pattern; dvecN as vec<ulong,N>. All arithmetic goes through mgl_f64_*.
// add/sub/mul/div/sqrt/fma are correctly rounded (round-to-nearest-even);
// subnormals are supported. Approach inspired by philipturner/metal-float64
// (software IEEE emulation on integer ALUs); implementation is original.
//
// Internal significand convention ("m form"): value = m * 2^(e - 1023 - 62),
// a normalized m has its leading 1 at bit 62 and 10 rounding bits (9..0).

#define MGL_F64_SIGN 0x8000000000000000ul
#define MGL_F64_EXP 0x7FF0000000000000ul
#define MGL_F64_FRAC 0x000FFFFFFFFFFFFFul
#define MGL_F64_HIDDEN 0x0010000000000000ul
#define MGL_F64_QNAN 0x7FF8000000000000ul
#define MGL_F64_INF 0x7FF0000000000000ul
#define MGL_F64_ONE 0x3FF0000000000000ul

inline int mgl_f64_expf(ulong a) { return int((a >> 52) & 0x7FFul); }
inline bool mgl_f64_isnan(ulong a) { return (a & ~MGL_F64_SIGN) > MGL_F64_INF; }
inline bool mgl_f64_isinf(ulong a) { return (a & ~MGL_F64_SIGN) == MGL_F64_INF; }
inline bool mgl_f64_iszero(ulong a) { return (a & ~MGL_F64_SIGN) == 0; }

inline ulong mgl_f64_srj(ulong a, uint dist) {  // shift right, jamming lost bits into bit 0
    if (dist == 0) return a;
    if (dist >= 64) return a != 0 ? 1ul : 0ul;
    return (a >> dist) | ((a << (64 - dist)) != 0 ? 1ul : 0ul);
}

inline ulong mgl_f64_roundpack(ulong sign, int e, ulong m) {
    if (e <= 0) {
        m = mgl_f64_srj(m, uint(1 - e));
        e = 1;
    } else if (e >= 0x7FF) {
        return sign | MGL_F64_INF;
    }
    uint rb = uint(m & 0x3FFul);
    m = (m + 0x200ul) >> 10;
    if (rb == 0x200u) m &= ~1ul;
    if (m == 0) return sign;
    return sign + (ulong(e - 1) << 52) + m;  // a rounding carry propagates into the exponent
}

inline ulong mgl_f64_normroundpack(ulong sign, int e, ulong m) {
    if (m == 0) return sign;
    int s = int(clz(m)) - 1;
    if (s > 0) {
        m <<= uint(s);
        e -= s;
    } else if (s < 0) {
        m = mgl_f64_srj(m, 1);
        e += 1;
    }
    return mgl_f64_roundpack(sign, e, m);
}

inline void mgl_f64_unpack(ulong a, thread int& e, thread ulong& m) {
    int E = mgl_f64_expf(a);
    ulong f = a & MGL_F64_FRAC;
    if (E == 0) {
        e = 1;
        m = f << 10;
    } else {
        e = E;
        m = (f | MGL_F64_HIDDEN) << 10;
    }
}

inline ulong mgl_f64_propagate_nan(ulong a, ulong b) {
    if (mgl_f64_isnan(a)) return a | 0x0008000000000000ul;
    return b | 0x0008000000000000ul;
}

inline ulong mgl_f64_addmags(ulong a, ulong b, ulong sign) {
    int ea = mgl_f64_expf(a), eb = mgl_f64_expf(b);
    if (ea == 0x7FF) {
        if (a & MGL_F64_FRAC) return mgl_f64_propagate_nan(a, b);
        if (eb == 0x7FF && (b & MGL_F64_FRAC)) return mgl_f64_propagate_nan(b, a);
        return sign | MGL_F64_INF;
    }
    if (eb == 0x7FF) {
        if (b & MGL_F64_FRAC) return mgl_f64_propagate_nan(b, a);
        return sign | MGL_F64_INF;
    }
    int e1, e2;
    ulong m1, m2;
    mgl_f64_unpack(a, e1, m1);
    mgl_f64_unpack(b, e2, m2);
    if (e1 < e2) {
        int te = e1; e1 = e2; e2 = te;
        ulong tm = m1; m1 = m2; m2 = tm;
    }
    m2 = mgl_f64_srj(m2, uint(e1 - e2));
    ulong m = m1 + m2;
    int e = e1;
    if (m >= 0x8000000000000000ul) {
        m = mgl_f64_srj(m, 1);
        e += 1;
    }
    return mgl_f64_normroundpack(sign, e, m);
}

inline ulong mgl_f64_submags(ulong a, ulong b, ulong signA) {
    int ea = mgl_f64_expf(a), eb = mgl_f64_expf(b);
    if (ea == 0x7FF) {
        if (a & MGL_F64_FRAC) return mgl_f64_propagate_nan(a, b);
        if (eb == 0x7FF) {
            if (b & MGL_F64_FRAC) return mgl_f64_propagate_nan(b, a);
            return MGL_F64_QNAN | MGL_F64_SIGN;  // inf - inf
        }
        return signA | MGL_F64_INF;
    }
    if (eb == 0x7FF) {
        if (b & MGL_F64_FRAC) return mgl_f64_propagate_nan(b, a);
        return (signA ^ MGL_F64_SIGN) | MGL_F64_INF;
    }
    int e1, e2;
    ulong m1, m2;
    mgl_f64_unpack(a, e1, m1);
    mgl_f64_unpack(b, e2, m2);
    ulong sign = signA;
    if (e1 < e2 || (e1 == e2 && m1 < m2)) {
        int te = e1; e1 = e2; e2 = te;
        ulong tm = m1; m1 = m2; m2 = tm;
        sign ^= MGL_F64_SIGN;
    }
    // Keep one extra guard bit for the borrow when exponents differ.
    ulong m = m1 - mgl_f64_srj(m2, uint(e1 - e2));
    if (m == 0) return 0ul;  // exact cancellation -> +0 in round-to-nearest
    return mgl_f64_normroundpack(sign, e1, m);
}

inline ulong mgl_f64_add(ulong a, ulong b) {
    ulong sa = a & MGL_F64_SIGN, sb = b & MGL_F64_SIGN;
    ulong ma = a & ~MGL_F64_SIGN, mb = b & ~MGL_F64_SIGN;
    if (sa == sb) return mgl_f64_addmags(ma, mb, sa);
    return mgl_f64_submags(ma, mb, sa);
}

inline ulong mgl_f64_neg(ulong a) { return a ^ MGL_F64_SIGN; }
inline ulong mgl_f64_sub(ulong a, ulong b) { return mgl_f64_add(a, b ^ MGL_F64_SIGN); }

inline void mgl_f64_mul64(ulong a, ulong b, thread ulong& hi, thread ulong& lo) {
    ulong a0 = a & 0xFFFFFFFFul, a1 = a >> 32, b0 = b & 0xFFFFFFFFul, b1 = b >> 32;
    ulong p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    ulong mid = (p00 >> 32) + (p01 & 0xFFFFFFFFul) + (p10 & 0xFFFFFFFFul);
    hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    lo = (mid << 32) | (p00 & 0xFFFFFFFFul);
}

// Normalizes a subnormal significand (hidden-bit-at-52 form) in place.
inline void mgl_f64_normsub(thread int& e, thread ulong& sig) {
    int s = int(clz(sig)) - 11;
    sig <<= uint(s);
    e = 1 - s;
}

inline ulong mgl_f64_mul(ulong a, ulong b) {
    ulong sign = (a ^ b) & MGL_F64_SIGN;
    int ea = mgl_f64_expf(a), eb = mgl_f64_expf(b);
    ulong fa = a & MGL_F64_FRAC, fb = b & MGL_F64_FRAC;
    if (ea == 0x7FF) {
        if (fa || (eb == 0x7FF && fb)) return mgl_f64_propagate_nan(a, b);
        if (mgl_f64_iszero(b)) return MGL_F64_QNAN | MGL_F64_SIGN;
        return sign | MGL_F64_INF;
    }
    if (eb == 0x7FF) {
        if (fb) return mgl_f64_propagate_nan(b, a);
        if (mgl_f64_iszero(a)) return MGL_F64_QNAN | MGL_F64_SIGN;
        return sign | MGL_F64_INF;
    }
    if (mgl_f64_iszero(a) || mgl_f64_iszero(b)) return sign;
    ulong sa, sb;
    if (ea == 0) { sa = fa; mgl_f64_normsub(ea, sa); } else sa = fa | MGL_F64_HIDDEN;
    if (eb == 0) { sb = fb; mgl_f64_normsub(eb, sb); } else sb = fb | MGL_F64_HIDDEN;
    ulong hi, lo;
    mgl_f64_mul64(sa << 10, sb << 11, hi, lo);
    if (lo != 0) hi |= 1;
    return mgl_f64_normroundpack(sign, ea + eb - 1022, hi);
}

inline ulong mgl_f64_div(ulong a, ulong b) {
    ulong sign = (a ^ b) & MGL_F64_SIGN;
    int ea = mgl_f64_expf(a), eb = mgl_f64_expf(b);
    ulong fa = a & MGL_F64_FRAC, fb = b & MGL_F64_FRAC;
    if (ea == 0x7FF) {
        if (fa) return mgl_f64_propagate_nan(a, b);
        if (eb == 0x7FF) return fb ? mgl_f64_propagate_nan(b, a) : (MGL_F64_QNAN | MGL_F64_SIGN);
        return sign | MGL_F64_INF;
    }
    if (eb == 0x7FF) {
        if (fb) return mgl_f64_propagate_nan(b, a);
        return sign;
    }
    if (mgl_f64_iszero(b)) {
        if (mgl_f64_iszero(a)) return MGL_F64_QNAN | MGL_F64_SIGN;
        return sign | MGL_F64_INF;
    }
    if (mgl_f64_iszero(a)) return sign;
    ulong sa, sb;
    if (ea == 0) { sa = fa; mgl_f64_normsub(ea, sa); } else sa = fa | MGL_F64_HIDDEN;
    if (eb == 0) { sb = fb; mgl_f64_normsub(eb, sb); } else sb = fb | MGL_F64_HIDDEN;
    int e = ea - eb + 1023;
    ulong r = sa;
    if (r < sb) {
        r <<= 1;
        e -= 1;
    }
    // Restoring division: 63 quotient bits (leading 1 at bit 62).
    ulong q = 0;
    for (int i = 0; i < 63; i++) {
        q <<= 1;
        if (r >= sb) {
            r -= sb;
            q |= 1;
        }
        r <<= 1;
    }
    if (r != 0) q |= 1;
    return mgl_f64_roundpack(sign, e, q);
}

inline ulong mgl_f64_sqrt(ulong a) {
    ulong sign = a & MGL_F64_SIGN;
    int ea = mgl_f64_expf(a);
    ulong fa = a & MGL_F64_FRAC;
    if (ea == 0x7FF) {
        if (fa) return a | 0x0008000000000000ul;
        if (sign) return MGL_F64_QNAN | MGL_F64_SIGN;
        return a;
    }
    if (mgl_f64_iszero(a)) return a;
    if (sign) return MGL_F64_QNAN | MGL_F64_SIGN;
    ulong s;
    if (ea == 0) { s = fa; mgl_f64_normsub(ea, s); } else s = fa | MGL_F64_HIDDEN;
    // value = s * 2^(ea - 1075). Make the exponent (ea - 1023) even.
    int k = ea - 1023;
    if (k & 1) {
        s <<= 1;
        k -= 1;
    }
    // radicand R = s << 72 (128-bit, hi:lo). sqrt(R) has 63 significant bits.
    ulong rhi = s << 8, rlo = 0;
    ulong res = 0, remhi = 0, remlo = 0;
    for (int i = 0; i < 64; i++) {
        // rem = (rem << 2) | top two bits of R
        remhi = (remhi << 2) | (remlo >> 62);
        remlo = (remlo << 2) | (rhi >> 62);
        rhi = (rhi << 2) | (rlo >> 62);
        rlo <<= 2;
        // trial = (res << 2) | 1   (as 128-bit)
        ulong thi = res >> 62, tlo = (res << 2) | 1;
        bool ge = remhi > thi || (remhi == thi && remlo >= tlo);
        if (ge) {
            ulong borrow = remlo < tlo ? 1ul : 0ul;
            remlo -= tlo;
            remhi -= thi + borrow;
            res = (res << 1) | 1;
        } else {
            res <<= 1;
        }
    }
    // res ~ sqrt(s * 2^72) = sqrt(s) * 2^36, s in [2^52, 2^54) -> res in [2^62, 2^63)
    if (remhi != 0 || remlo != 0) res |= 1;
    // value = res * 2^-36 * 2^((k - 52)/2)  ->  e = (k - 52)/2 - 36 + 1023 + 62
    int e = (k - 52) / 2 + 1049;
    return mgl_f64_roundpack(0ul, e, res);
}

// 128-bit helpers (hi:lo)
inline void mgl_f64_srj128(thread ulong& hi, thread ulong& lo, uint dist) {
    if (dist == 0) return;
    if (dist >= 128) {
        lo = (hi | lo) != 0 ? 1ul : 0ul;
        hi = 0;
        return;
    }
    if (dist >= 64) {
        ulong sticky = (lo != 0 || (dist > 64 && (hi << (128 - dist)) != 0)) ? 1ul : 0ul;
        lo = (dist == 64 ? hi : hi >> (dist - 64)) | sticky;
        hi = 0;
        return;
    }
    ulong sticky = (lo << (64 - dist)) != 0 ? 1ul : 0ul;
    lo = (lo >> dist) | (hi << (64 - dist)) | sticky;
    hi >>= dist;
}

inline ulong mgl_f64_fma(ulong a, ulong b, ulong c) {
    ulong sp = (a ^ b) & MGL_F64_SIGN, sc = c & MGL_F64_SIGN;
    if (mgl_f64_isnan(a) || mgl_f64_isnan(b)) return mgl_f64_propagate_nan(mgl_f64_isnan(a) ? a : b, c);
    if (mgl_f64_isnan(c)) return c | 0x0008000000000000ul;
    bool ainf = mgl_f64_isinf(a), binf = mgl_f64_isinf(b);
    if (ainf || binf) {
        if (mgl_f64_iszero(a) || mgl_f64_iszero(b)) return MGL_F64_QNAN | MGL_F64_SIGN;
        if (mgl_f64_isinf(c) && sc != sp) return MGL_F64_QNAN | MGL_F64_SIGN;
        return sp | MGL_F64_INF;
    }
    if (mgl_f64_isinf(c)) return c;
    if (mgl_f64_iszero(a) || mgl_f64_iszero(b)) return mgl_f64_add(sp, c);  // exact signed zero product
    int ea = mgl_f64_expf(a), eb = mgl_f64_expf(b), ec = mgl_f64_expf(c);
    ulong sa = a & MGL_F64_FRAC, sb = b & MGL_F64_FRAC;
    if (ea == 0) mgl_f64_normsub(ea, sa); else sa |= MGL_F64_HIDDEN;
    if (eb == 0) mgl_f64_normsub(eb, sb); else sb |= MGL_F64_HIDDEN;
    // P = sa*sb in [2^104, 2^106), value P * 2^(ea+eb-2150); shifted left 20 -> < 2^126
    ulong phi, plo;
    mgl_f64_mul64(sa, sb, phi, plo);
    phi = (phi << 20) | (plo >> 44);
    plo <<= 20;
    int pexp = ea + eb - 2150;
    if (mgl_f64_iszero(c)) {
        // Exact product (value P * 2^(pexp-20)) rounded once; c == +-0 only affects the sign of zero.
        int L = 127 - int(clz(phi));  // phi != 0 since P >= 2^124 after the shift
        ulong mhi = phi, mlo = plo;
        mgl_f64_srj128(mhi, mlo, uint(L - 62));
        return mgl_f64_roundpack(sp, L + pexp + 1003, mlo);
    }
    ulong scs = c & MGL_F64_FRAC;
    if (ec == 0) mgl_f64_normsub(ec, scs); else scs |= MGL_F64_HIDDEN;
    // C = scs << 52 (value scs * 2^(ec-1075) = (scs<<52) * 2^(ec-1127)), then << 20 like P.
    ulong chi = scs << 8, clo = 0;  // (scs << 72) as 128-bit
    int cexp = ec - 1127;
    int base;
    if (pexp >= cexp) {
        mgl_f64_srj128(chi, clo, uint(min(pexp - cexp, 200)));
        base = pexp;
    } else {
        mgl_f64_srj128(phi, plo, uint(min(cexp - pexp, 200)));
        base = cexp;
    }
    ulong rhi, rlo, sign;
    if (sp == sc) {
        rlo = plo + clo;
        rhi = phi + chi + (rlo < plo ? 1ul : 0ul);
        sign = sp;
    } else {
        bool pge = phi > chi || (phi == chi && plo >= clo);
        if (pge) {
            rlo = plo - clo;
            rhi = phi - chi - (plo < clo ? 1ul : 0ul);
            sign = sp;
        } else {
            rlo = clo - plo;
            rhi = chi - phi - (clo < plo ? 1ul : 0ul);
            sign = sc;
        }
        if (rhi == 0 && rlo == 0) return 0ul;
    }
    int L = rhi != 0 ? 127 - int(clz(rhi)) : 63 - int(clz(rlo));
    ulong m;
    if (L >= 62) {
        ulong h = rhi, l = rlo;
        mgl_f64_srj128(h, l, uint(L - 62));
        m = l;
    } else {
        m = rlo << uint(62 - L);
    }
    return mgl_f64_roundpack(sign, L + base + 1003, m);
}

// --- comparisons ---
inline bool mgl_f64_eq(ulong a, ulong b) {
    if (mgl_f64_isnan(a) || mgl_f64_isnan(b)) return false;
    return a == b || ((a | b) & ~MGL_F64_SIGN) == 0;
}
inline bool mgl_f64_ne(ulong a, ulong b) { return !mgl_f64_eq(a, b); }
inline bool mgl_f64_lt(ulong a, ulong b) {
    if (mgl_f64_isnan(a) || mgl_f64_isnan(b)) return false;
    bool sa = (a & MGL_F64_SIGN) != 0, sb = (b & MGL_F64_SIGN) != 0;
    if (sa != sb) return sa && ((a | b) & ~MGL_F64_SIGN) != 0;
    return a != b && (sa ? a > b : a < b);
}
inline bool mgl_f64_le(ulong a, ulong b) {
    if (mgl_f64_isnan(a) || mgl_f64_isnan(b)) return false;
    return mgl_f64_lt(a, b) || mgl_f64_eq(a, b);
}
inline bool mgl_f64_gt(ulong a, ulong b) { return mgl_f64_lt(b, a); }
inline bool mgl_f64_ge(ulong a, ulong b) { return mgl_f64_le(b, a); }

// --- conversions ---
inline ulong mgl_f64_from_float(float f) {
    uint u = as_type<uint>(f);
    ulong sign = ulong(u & 0x80000000u) << 32;
    int e = int((u >> 23) & 0xFFu);
    uint frac = u & 0x7FFFFFu;
    if (e == 0xFF) return sign | MGL_F64_INF | (ulong(frac) << 29) | (frac ? 0x0008000000000000ul : 0ul);
    if (e == 0) {
        if (frac == 0) return sign;
        // subnormal float -> normal double
        int s = int(clz(frac)) - 8;
        frac = (frac << uint(s)) & 0x7FFFFFu;
        e = 1 - s;
    }
    return sign | (ulong(e - 127 + 1023) << 52) | (ulong(frac) << 29);
}

inline ulong mgl_f64_from_int(int i) {
    if (i == 0) return 0ul;
    ulong sign = i < 0 ? MGL_F64_SIGN : 0ul;
    ulong m = i < 0 ? ulong(-long(i)) : ulong(i);
    return mgl_f64_normroundpack(sign, 1023 + 62, m);
}

inline ulong mgl_f64_from_uint(uint u) {
    if (u == 0) return 0ul;
    return mgl_f64_normroundpack(0ul, 1023 + 62, ulong(u));
}

inline ulong mgl_f64_from_bool(bool b) { return b ? MGL_F64_ONE : 0ul; }
inline bool mgl_f64_to_bool(ulong a) { return !mgl_f64_iszero(a); }

inline float mgl_f64_to_float(ulong a) {
    uint sign = uint(a >> 32) & 0x80000000u;
    int e = mgl_f64_expf(a);
    ulong f = a & MGL_F64_FRAC;
    if (e == 0x7FF) return as_type<float>(sign | 0x7F800000u | (f ? (0x400000u | uint(f >> 29)) : 0u));
    if (e == 0 && f == 0) return as_type<float>(sign);
    if (e == 0) return as_type<float>(sign);  // double subnormals underflow to 0 in float
    ulong m = f | MGL_F64_HIDDEN;  // 53 bits
    int fe = e - 1023 + 127;
    // m form for float: keep 24 significant bits + 30 round bits
    if (fe >= 0xFF) return as_type<float>(sign | 0x7F800000u);
    uint mant;
    if (fe <= 0) {
        // subnormal float result
        uint shift = uint(29 + 1 - fe);
        ulong mm = shift < 64 ? m : 0;
        ulong rem = shift < 64 ? (m & ((1ul << shift) - 1)) : m;
        mant = shift < 64 ? uint(mm >> shift) : 0u;
        ulong hv = shift < 64 ? (1ul << (shift - 1)) : 0;
        if (shift < 64 && (rem > hv || (rem == hv && (mant & 1)))) mant++;
        return as_type<float>(sign | mant);
    }
    ulong rem = m & 0x1FFFFFFFul;  // low 29 bits
    mant = uint(m >> 29);           // 24 bits incl. hidden
    if (rem > 0x10000000ul || (rem == 0x10000000ul && (mant & 1))) mant++;
    uint bits = (uint(fe - 1) << 23) + mant;  // carry handles mantissa overflow
    if (bits >= 0x7F800000u) bits = 0x7F800000u;
    return as_type<float>(sign | bits);
}

inline int mgl_f64_to_int(ulong a) {
    int e = mgl_f64_expf(a);
    if (e == 0x7FF && (a & MGL_F64_FRAC)) return 0;
    int k = e - 1023;
    if (k < 0) return 0;
    ulong m = (a & MGL_F64_FRAC) | MGL_F64_HIDDEN;
    if (k > 30) return (a & MGL_F64_SIGN) ? int(0x80000000u) : 0x7FFFFFFF;
    long v = long(k >= 52 ? m << uint(k - 52) : m >> uint(52 - k));
    return (a & MGL_F64_SIGN) ? int(-v) : int(v);
}

inline uint mgl_f64_to_uint(ulong a) {
    if (a & MGL_F64_SIGN) return 0u;
    int e = mgl_f64_expf(a);
    if (e == 0x7FF) return (a & MGL_F64_FRAC) ? 0u : 0xFFFFFFFFu;
    int k = e - 1023;
    if (k < 0) return 0u;
    if (k > 31) return 0xFFFFFFFFu;
    ulong m = (a & MGL_F64_FRAC) | MGL_F64_HIDDEN;
    return uint(k >= 52 ? m << uint(k - 52) : m >> uint(52 - k));
}

// --- rounding & misc ---
inline ulong mgl_f64_trunc(ulong a) {
    int e = mgl_f64_expf(a);
    if (e >= 1075) return a;  // integral, inf or nan
    if (e < 1023) return a & MGL_F64_SIGN;
    ulong mask = MGL_F64_FRAC >> uint(e - 1023);
    return a & ~mask;
}

inline ulong mgl_f64_floor(ulong a) {
    ulong t = mgl_f64_trunc(a);
    if ((a & MGL_F64_SIGN) && t != a && !mgl_f64_isnan(a)) return mgl_f64_sub(t, MGL_F64_ONE);
    return t;
}

inline ulong mgl_f64_ceil(ulong a) {
    ulong t = mgl_f64_trunc(a);
    if (!(a & MGL_F64_SIGN) && t != a && !mgl_f64_isnan(a)) return mgl_f64_add(t, MGL_F64_ONE);
    return t;
}

inline ulong mgl_f64_roundEven(ulong a) {
    int e = mgl_f64_expf(a);
    if (e >= 1075) return a;
    ulong sign = a & MGL_F64_SIGN;
    if (e < 1022) return sign;  // |a| < 0.5
    int ea;
    ulong m;
    mgl_f64_unpack(a, ea, m);
    // integer part bits: shift so the binary point is at bit 10 (round bits of roundpack)
    // value = m * 2^(e-1085); integer = m >> (1085 - e)
    uint sh = uint(1085 - e);
    ulong ip = m >> sh;
    ulong rem = m & ((1ul << sh) - 1);
    ulong hv = 1ul << (sh - 1);
    if (rem > hv || (rem == hv && (ip & 1))) ip++;
    if (ip == 0) return sign;
    return mgl_f64_normroundpack(sign, 1023 + 62, ip);
}

inline ulong mgl_f64_round(ulong a) {
    // GLSL round(): implementation-defined direction for .5; use round-hv-away-from-zero.
    int e = mgl_f64_expf(a);
    if (e >= 1075) return a;
    ulong sign = a & MGL_F64_SIGN;
    ulong hv = 0x3FE0000000000000ul;
    return mgl_f64_trunc(mgl_f64_add(a & ~MGL_F64_SIGN, hv)) | sign;
}

inline ulong mgl_f64_fract(ulong a) { return mgl_f64_sub(a, mgl_f64_floor(a)); }
inline ulong mgl_f64_abs(ulong a) { return a & ~MGL_F64_SIGN; }
inline ulong mgl_f64_sign(ulong a) {
    if (mgl_f64_isnan(a) || mgl_f64_iszero(a)) return mgl_f64_iszero(a) ? 0ul : a;
    return (a & MGL_F64_SIGN) | MGL_F64_ONE;
}
inline ulong mgl_f64_mod(ulong a, ulong b) { return mgl_f64_sub(a, mgl_f64_mul(b, mgl_f64_floor(mgl_f64_div(a, b)))); }
inline ulong mgl_f64_min(ulong a, ulong b) { return mgl_f64_lt(b, a) ? b : a; }
inline ulong mgl_f64_max(ulong a, ulong b) { return mgl_f64_lt(a, b) ? b : a; }
inline ulong mgl_f64_clamp(ulong x, ulong lo, ulong hi) { return mgl_f64_min(mgl_f64_max(x, lo), hi); }
inline ulong mgl_f64_mix(ulong a, ulong b, ulong t) { return mgl_f64_add(mgl_f64_mul(a, mgl_f64_sub(MGL_F64_ONE, t)), mgl_f64_mul(b, t)); }
inline ulong mgl_f64_mix(ulong a, ulong b, bool t) { return t ? b : a; }
inline ulong mgl_f64_step(ulong edge, ulong x) { return mgl_f64_lt(x, edge) ? 0ul : MGL_F64_ONE; }
inline ulong mgl_f64_smoothstep(ulong e0, ulong e1, ulong x) {
    ulong t = mgl_f64_clamp(mgl_f64_div(mgl_f64_sub(x, e0), mgl_f64_sub(e1, e0)), 0ul, MGL_F64_ONE);
    ulong three = 0x4008000000000000ul, two = 0x4000000000000000ul;
    return mgl_f64_mul(mgl_f64_mul(t, t), mgl_f64_sub(three, mgl_f64_mul(two, t)));
}
inline ulong mgl_f64_inversesqrt(ulong a) { return mgl_f64_div(MGL_F64_ONE, mgl_f64_sqrt(a)); }

inline ulong mgl_f64_ldexp(ulong a, int k) {
    if (mgl_f64_isnan(a) || mgl_f64_isinf(a) || mgl_f64_iszero(a)) return a;
    int e;
    ulong m;
    mgl_f64_unpack(a, e, m);
    k = clamp(k, -4096, 4096);
    return mgl_f64_normroundpack(a & MGL_F64_SIGN, e + k, m);
}

inline ulong mgl_f64_frexp(ulong a, thread int& exp) {
    if (mgl_f64_isnan(a) || mgl_f64_isinf(a) || mgl_f64_iszero(a)) {
        exp = 0;
        return a;
    }
    int e = mgl_f64_expf(a);
    ulong sig = a & MGL_F64_FRAC;
    if (e == 0) mgl_f64_normsub(e, sig);
    else sig |= MGL_F64_HIDDEN;
    exp = e - 1022;
    return (a & MGL_F64_SIGN) | (0x3FEul << 52) | (sig & MGL_F64_FRAC);
}

inline ulong mgl_f64_modf(ulong a, thread ulong& ip) {
    ip = mgl_f64_trunc(a);
    if (mgl_f64_isinf(a)) return a & MGL_F64_SIGN;
    return mgl_f64_sub(a, ip);
}

inline ulong mgl_f64_pack2x32(uint2 v) { return ulong(v.x) | (ulong(v.y) << 32); }
inline uint2 mgl_f64_unpack2x32(ulong v) { return uint2(uint(v & 0xFFFFFFFFul), uint(v >> 32)); }

// --- vector overloads (component-wise) ---
#define MGL_F64_V1(NAME)                                                       \
    template <int N> inline vec<ulong, N> NAME(vec<ulong, N> a) {             \
        vec<ulong, N> r;                                                       \
        for (int i = 0; i < N; i++) r[i] = NAME(a[i]);                         \
        return r;                                                              \
    }
#define MGL_F64_V2(NAME)                                                       \
    template <int N> inline vec<ulong, N> NAME(vec<ulong, N> a, vec<ulong, N> b) { \
        vec<ulong, N> r;                                                       \
        for (int i = 0; i < N; i++) r[i] = NAME(a[i], b[i]);                   \
        return r;                                                              \
    }                                                                          \
    template <int N> inline vec<ulong, N> NAME(vec<ulong, N> a, ulong b) {    \
        vec<ulong, N> r;                                                       \
        for (int i = 0; i < N; i++) r[i] = NAME(a[i], b);                      \
        return r;                                                              \
    }
#define MGL_F64_VCMP(NAME)                                                     \
    template <int N> inline vec<bool, N> NAME(vec<ulong, N> a, vec<ulong, N> b) { \
        vec<bool, N> r;                                                        \
        for (int i = 0; i < N; i++) r[i] = NAME(a[i], b[i]);                   \
        return r;                                                              \
    }
MGL_F64_V2(mgl_f64_add)
MGL_F64_V2(mgl_f64_sub)
MGL_F64_V2(mgl_f64_mul)
MGL_F64_V2(mgl_f64_div)
MGL_F64_V2(mgl_f64_mod)
MGL_F64_V2(mgl_f64_min)
MGL_F64_V2(mgl_f64_max)
MGL_F64_V2(mgl_f64_step)
MGL_F64_V1(mgl_f64_neg)
MGL_F64_V1(mgl_f64_abs)
MGL_F64_V1(mgl_f64_sign)
MGL_F64_V1(mgl_f64_floor)
MGL_F64_V1(mgl_f64_ceil)
MGL_F64_V1(mgl_f64_trunc)
MGL_F64_V1(mgl_f64_round)
MGL_F64_V1(mgl_f64_roundEven)
MGL_F64_V1(mgl_f64_fract)
MGL_F64_V1(mgl_f64_sqrt)
MGL_F64_V1(mgl_f64_inversesqrt)
MGL_F64_VCMP(mgl_f64_eq)
MGL_F64_VCMP(mgl_f64_ne)
MGL_F64_VCMP(mgl_f64_lt)
MGL_F64_VCMP(mgl_f64_le)
MGL_F64_VCMP(mgl_f64_gt)
MGL_F64_VCMP(mgl_f64_ge)

template <int N> inline vec<ulong, N> mgl_f64_step(ulong edge, vec<ulong, N> x) {
    vec<ulong, N> r;
    for (int i = 0; i < N; i++) r[i] = mgl_f64_step(edge, x[i]);
    return r;
}
template <int N> inline vec<ulong, N> mgl_f64_fma(vec<ulong, N> a, vec<ulong, N> b, vec<ulong, N> c) {
    vec<ulong, N> r;
    for (int i = 0; i < N; i++) r[i] = mgl_f64_fma(a[i], b[i], c[i]);
    return r;
}
template <int N> inline vec<ulong, N> mgl_f64_clamp(vec<ulong, N> x, vec<ulong, N> lo, vec<ulong, N> hi) {
    vec<ulong, N> r;
    for (int i = 0; i < N; i++) r[i] = mgl_f64_clamp(x[i], lo[i], hi[i]);
    return r;
}
template <int N> inline vec<ulong, N> mgl_f64_clamp(vec<ulong, N> x, ulong lo, ulong hi) {
    vec<ulong, N> r;
    for (int i = 0; i < N; i++) r[i] = mgl_f64_clamp(x[i], lo, hi);
    return r;
}
template <int N> inline vec<ulong, N> mgl_f64_mix(vec<ulong, N> a, vec<ulong, N> b, vec<ulong, N> t) {
    vec<ulong, N> r;
    for (int i = 0; i < N; i++) r[i] = mgl_f64_mix(a[i], b[i], t[i]);
    return r;
}
template <int N> inline vec<ulong, N> mgl_f64_mix(vec<ulong, N> a, vec<ulong, N> b, ulong t) {
    vec<ulong, N> r;
    for (int i = 0; i < N; i++) r[i] = mgl_f64_mix(a[i], b[i], t);
    return r;
}
template <int N> inline vec<ulong, N> mgl_f64_mix(vec<ulong, N> a, vec<ulong, N> b, vec<bool, N> t) {
    vec<ulong, N> r;
    for (int i = 0; i < N; i++) r[i] = t[i] ? b[i] : a[i];
    return r;
}
template <int N> inline vec<ulong, N> mgl_f64_smoothstep(vec<ulong, N> e0, vec<ulong, N> e1, vec<ulong, N> x) {
    vec<ulong, N> r;
    for (int i = 0; i < N; i++) r[i] = mgl_f64_smoothstep(e0[i], e1[i], x[i]);
    return r;
}
template <int N> inline vec<ulong, N> mgl_f64_smoothstep(ulong e0, ulong e1, vec<ulong, N> x) {
    vec<ulong, N> r;
    for (int i = 0; i < N; i++) r[i] = mgl_f64_smoothstep(e0, e1, x[i]);
    return r;
}
template <int N> inline vec<bool, N> mgl_f64_isnan(vec<ulong, N> a) {
    vec<bool, N> r;
    for (int i = 0; i < N; i++) r[i] = mgl_f64_isnan(a[i]);
    return r;
}
template <int N> inline vec<bool, N> mgl_f64_isinf(vec<ulong, N> a) {
    vec<bool, N> r;
    for (int i = 0; i < N; i++) r[i] = mgl_f64_isinf(a[i]);
    return r;
}
template <int N> inline vec<ulong, N> mgl_f64_ldexp(vec<ulong, N> a, vec<int, N> k) {
    vec<ulong, N> r;
    for (int i = 0; i < N; i++) r[i] = mgl_f64_ldexp(a[i], k[i]);
    return r;
}
template <int N> inline vec<ulong, N> mgl_f64_frexp(vec<ulong, N> a, thread vec<int, N>& e) {
    vec<ulong, N> r;
    for (int i = 0; i < N; i++) {
        int ei;
        r[i] = mgl_f64_frexp(a[i], ei);
        e[i] = ei;
    }
    return r;
}
template <int N> inline vec<ulong, N> mgl_f64_modf(vec<ulong, N> a, thread vec<ulong, N>& ip) {
    vec<ulong, N> r;
    for (int i = 0; i < N; i++) {
        ulong p;
        r[i] = mgl_f64_modf(a[i], p);
        ip[i] = p;
    }
    return r;
}
template <int N> inline vec<ulong, N> mgl_f64_from_float(vec<float, N> a) {
    vec<ulong, N> r;
    for (int i = 0; i < N; i++) r[i] = mgl_f64_from_float(a[i]);
    return r;
}
template <int N> inline vec<ulong, N> mgl_f64_from_int(vec<int, N> a) {
    vec<ulong, N> r;
    for (int i = 0; i < N; i++) r[i] = mgl_f64_from_int(a[i]);
    return r;
}
template <int N> inline vec<ulong, N> mgl_f64_from_uint(vec<uint, N> a) {
    vec<ulong, N> r;
    for (int i = 0; i < N; i++) r[i] = mgl_f64_from_uint(a[i]);
    return r;
}
template <int N> inline vec<ulong, N> mgl_f64_from_bool(vec<bool, N> a) {
    vec<ulong, N> r;
    for (int i = 0; i < N; i++) r[i] = mgl_f64_from_bool(a[i]);
    return r;
}
template <int N> inline vec<float, N> mgl_f64_to_float(vec<ulong, N> a) {
    vec<float, N> r;
    for (int i = 0; i < N; i++) r[i] = mgl_f64_to_float(a[i]);
    return r;
}
template <int N> inline vec<int, N> mgl_f64_to_int(vec<ulong, N> a) {
    vec<int, N> r;
    for (int i = 0; i < N; i++) r[i] = mgl_f64_to_int(a[i]);
    return r;
}
template <int N> inline vec<uint, N> mgl_f64_to_uint(vec<ulong, N> a) {
    vec<uint, N> r;
    for (int i = 0; i < N; i++) r[i] = mgl_f64_to_uint(a[i]);
    return r;
}
template <int N> inline vec<bool, N> mgl_f64_to_bool(vec<ulong, N> a) {
    vec<bool, N> r;
    for (int i = 0; i < N; i++) r[i] = mgl_f64_to_bool(a[i]);
    return r;
}

// --- geometric ---
template <int N> inline ulong mgl_f64_dot(vec<ulong, N> a, vec<ulong, N> b) {
    ulong s = mgl_f64_mul(a[0], b[0]);
    for (int i = 1; i < N; i++) s = mgl_f64_add(s, mgl_f64_mul(a[i], b[i]));
    return s;
}
inline ulong mgl_f64_dot(ulong a, ulong b) { return mgl_f64_mul(a, b); }
template <int N> inline ulong mgl_f64_length(vec<ulong, N> a) { return mgl_f64_sqrt(mgl_f64_dot(a, a)); }
inline ulong mgl_f64_length(ulong a) { return mgl_f64_abs(a); }
template <int N> inline ulong mgl_f64_distance(vec<ulong, N> a, vec<ulong, N> b) { return mgl_f64_length(mgl_f64_sub(a, b)); }
inline ulong mgl_f64_distance(ulong a, ulong b) { return mgl_f64_abs(mgl_f64_sub(a, b)); }
template <int N> inline vec<ulong, N> mgl_f64_normalize(vec<ulong, N> a) { return mgl_f64_div(a, mgl_f64_length(a)); }
inline ulong mgl_f64_normalize(ulong a) { return mgl_f64_sign(a); }
inline ulong3 mgl_f64_cross(ulong3 a, ulong3 b) {
    return ulong3(mgl_f64_sub(mgl_f64_mul(a.y, b.z), mgl_f64_mul(b.y, a.z)), mgl_f64_sub(mgl_f64_mul(a.z, b.x), mgl_f64_mul(b.z, a.x)),
                  mgl_f64_sub(mgl_f64_mul(a.x, b.y), mgl_f64_mul(b.x, a.y)));
}
template <int N> inline vec<ulong, N> mgl_f64_faceforward(vec<ulong, N> n, vec<ulong, N> i, vec<ulong, N> nref) {
    return mgl_f64_lt(mgl_f64_dot(nref, i), 0ul) ? n : mgl_f64_neg(n);
}
template <int N> inline vec<ulong, N> mgl_f64_reflect(vec<ulong, N> i, vec<ulong, N> n) {
    ulong d = mgl_f64_mul(0x4000000000000000ul, mgl_f64_dot(n, i));
    return mgl_f64_sub(i, mgl_f64_mul(n, d));
}
template <int N> inline vec<ulong, N> mgl_f64_refract(vec<ulong, N> i, vec<ulong, N> n, ulong eta) {
    ulong d = mgl_f64_dot(n, i);
    ulong k = mgl_f64_sub(MGL_F64_ONE, mgl_f64_mul(mgl_f64_mul(eta, eta), mgl_f64_sub(MGL_F64_ONE, mgl_f64_mul(d, d))));
    if (mgl_f64_lt(k, 0ul)) return vec<ulong, N>(0ul);
    return mgl_f64_sub(mgl_f64_mul(i, eta), mgl_f64_mul(n, mgl_f64_add(mgl_f64_mul(eta, d), mgl_f64_sqrt(k))));
}
