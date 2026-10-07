#include "glsl/builtins.h"

#include <sstream>
#include <unordered_map>

#include "glsl/types.h"

namespace glsl {

namespace {

// Signature mini-language. Placeholders:
//   FD -> float/double (line expanded twice), IU -> int/uint (expanded twice)
//   genX: scalar..vec4 of X; vecX: vec2..vec4; matX: every matrix shape
//   'out '/'inout ' prefix on a parameter.
const char* kSigs[] = {
    // angle & trig
    "genF radians(genF)", "genF degrees(genF)", "genF sin(genF)", "genF cos(genF)", "genF tan(genF)", "genF asin(genF)",
    "genF acos(genF)", "genF atan(genF,genF)", "genF atan(genF)", "genF sinh(genF)", "genF cosh(genF)", "genF tanh(genF)",
    "genF asinh(genF)", "genF acosh(genF)", "genF atanh(genF)",
    // exponential
    "genF pow(genF,genF)", "genF exp(genF)", "genF log(genF)", "genF exp2(genF)", "genF log2(genF)", "genFD sqrt(genFD)",
    "genFD inversesqrt(genFD)",
    // common
    "genFD abs(genFD)", "genI abs(genI)", "genFD sign(genFD)", "genI sign(genI)", "genFD floor(genFD)", "genFD trunc(genFD)",
    "genFD round(genFD)", "genFD roundEven(genFD)", "genFD ceil(genFD)", "genFD fract(genFD)", "genFD mod(genFD,FD)",
    "genFD mod(genFD,genFD)", "genFD modf(genFD,out genFD)", "genFD min(genFD,genFD)", "genFD min(genFD,FD)",
    "genIU min(genIU,genIU)", "genIU min(genIU,IU)", "genFD max(genFD,genFD)", "genFD max(genFD,FD)", "genIU max(genIU,genIU)",
    "genIU max(genIU,IU)", "genFD clamp(genFD,genFD,genFD)", "genFD clamp(genFD,FD,FD)", "genIU clamp(genIU,genIU,genIU)",
    "genIU clamp(genIU,IU,IU)", "genFD mix(genFD,genFD,genFD)", "genFD mix(genFD,genFD,FD)", "genFD mix(genFD,genFD,genB)",
    "genIU mix(genIU,genIU,genB)", "genB mix(genB,genB,genB)", "genFD step(genFD,genFD)", "genFD step(FD,genFD)",
    "genFD smoothstep(genFD,genFD,genFD)", "genFD smoothstep(FD,FD,genFD)", "genB isnan(genFD)", "genB isinf(genFD)",
    "genI floatBitsToInt(genF)", "genU floatBitsToUint(genF)", "genF intBitsToFloat(genI)", "genF uintBitsToFloat(genU)",
    "genFD fma(genFD,genFD,genFD)", "genFD frexp(genFD,out genI)", "genFD ldexp(genFD,genI)",
    // packing
    "uint packUnorm2x16(vec2)", "uint packSnorm2x16(vec2)", "uint packUnorm4x8(vec4)", "uint packSnorm4x8(vec4)",
    "vec2 unpackUnorm2x16(uint)", "vec2 unpackSnorm2x16(uint)", "vec4 unpackUnorm4x8(uint)", "vec4 unpackSnorm4x8(uint)",
    "uint packHalf2x16(vec2)", "vec2 unpackHalf2x16(uint)", "double packDouble2x32(uvec2)", "uvec2 unpackDouble2x32(double)",
    // geometric
    "FD length(genFD)", "FD distance(genFD,genFD)", "FD dot(genFD,genFD)", "vec3 cross(vec3,vec3)", "dvec3 cross(dvec3,dvec3)",
    "genFD normalize(genFD)", "genFD faceforward(genFD,genFD,genFD)", "genFD reflect(genFD,genFD)",
    "genFD refract(genFD,genFD,FD)",
    // matrix
    "matFD matrixCompMult(matFD,matFD)",
    "float determinant(mat2)", "float determinant(mat3)", "float determinant(mat4)", "double determinant(dmat2)",
    "double determinant(dmat3)", "double determinant(dmat4)", "mat2 inverse(mat2)", "mat3 inverse(mat3)", "mat4 inverse(mat4)",
    "dmat2 inverse(dmat2)", "dmat3 inverse(dmat3)", "dmat4 inverse(dmat4)",
    // vector relational
    "vecB lessThan(vecFD,vecFD)", "vecB lessThan(vecIU,vecIU)", "vecB lessThanEqual(vecFD,vecFD)",
    "vecB lessThanEqual(vecIU,vecIU)", "vecB greaterThan(vecFD,vecFD)", "vecB greaterThan(vecIU,vecIU)",
    "vecB greaterThanEqual(vecFD,vecFD)", "vecB greaterThanEqual(vecIU,vecIU)", "vecB equal(vecFD,vecFD)",
    "vecB equal(vecIU,vecIU)", "vecB equal(vecB,vecB)", "vecB notEqual(vecFD,vecFD)", "vecB notEqual(vecIU,vecIU)",
    "vecB notEqual(vecB,vecB)", "bool any(vecB)", "bool all(vecB)", "vecB not(vecB)",
    // integer
    "genU uaddCarry(genU,genU,out genU)", "genU usubBorrow(genU,genU,out genU)",
    "void umulExtended(genU,genU,out genU,out genU)", "void imulExtended(genI,genI,out genI,out genI)",
    "genIU bitfieldExtract(genIU,int,int)", "genIU bitfieldInsert(genIU,genIU,int,int)", "genIU bitfieldReverse(genIU)",
    "genI bitCount(genIU)", "genI findLSB(genIU)", "genI findMSB(genIU)",
    // derivatives / interpolation
    "genF dFdx(genF)", "genF dFdy(genF)", "genF dFdxFine(genF)", "genF dFdyFine(genF)", "genF dFdxCoarse(genF)",
    "genF dFdyCoarse(genF)", "genF fwidth(genF)", "genF fwidthFine(genF)", "genF fwidthCoarse(genF)",
    "genF interpolateAtCentroid(genF)", "genF interpolateAtSample(genF,int)", "genF interpolateAtOffset(genF,vec2)",
    // noise
    "float noise1(genF)", "vec2 noise2(genF)", "vec3 noise3(genF)", "vec4 noise4(genF)",
    // atomics on buffer/shared memory
    "IU atomicAdd(inout IU,IU)", "IU atomicMin(inout IU,IU)", "IU atomicMax(inout IU,IU)", "IU atomicAnd(inout IU,IU)",
    "IU atomicOr(inout IU,IU)", "IU atomicXor(inout IU,IU)", "IU atomicExchange(inout IU,IU)",
    "IU atomicCompSwap(inout IU,IU,IU)",
    "uint atomicCounterIncrement(atomic_uint)", "uint atomicCounterDecrement(atomic_uint)", "uint atomicCounter(atomic_uint)",
    "uint atomicCounterAdd(atomic_uint,uint)", "uint atomicCounterSubtract(atomic_uint,uint)",
    "uint atomicCounterMin(atomic_uint,uint)", "uint atomicCounterMax(atomic_uint,uint)", "uint atomicCounterAnd(atomic_uint,uint)",
    "uint atomicCounterOr(atomic_uint,uint)", "uint atomicCounterXor(atomic_uint,uint)",
    "uint atomicCounterExchange(atomic_uint,uint)", "uint atomicCounterCompSwap(atomic_uint,uint,uint)",
    // barriers
    "void barrier()", "void memoryBarrier()", "void memoryBarrierAtomicCounter()", "void memoryBarrierBuffer()",
    "void memoryBarrierShared()", "void memoryBarrierImage()", "void groupMemoryBarrier()",
    // geometry
    "void EmitVertex()", "void EndPrimitive()", "void EmitStreamVertex(int)", "void EndStreamPrimitive(int)",
    // vote
    "bool anyInvocation(bool)", "bool allInvocations(bool)", "bool allInvocationsEqual(bool)", "bool anyInvocationARB(bool)",
    "bool allInvocationsARB(bool)", "bool allInvocationsEqualARB(bool)",
    // compat
    "vec4 ftransform()",
};

// ARB_gpu_shader_int64
const char* kInt64Sigs[] = {
    "genI64 abs(genI64)", "genI64 sign(genI64)", "genI64 min(genI64,genI64)", "genI64 min(genI64,int64_t)",
    "genU64 min(genU64,genU64)", "genU64 min(genU64,uint64_t)", "genI64 max(genI64,genI64)", "genI64 max(genI64,int64_t)",
    "genU64 max(genU64,genU64)", "genU64 max(genU64,uint64_t)", "genI64 clamp(genI64,genI64,genI64)",
    "genI64 clamp(genI64,int64_t,int64_t)", "genU64 clamp(genU64,genU64,genU64)", "genU64 clamp(genU64,uint64_t,uint64_t)",
    "genI64 mix(genI64,genI64,genB)", "genU64 mix(genU64,genU64,genB)", "genI64 doubleBitsToInt64(genD)",
    "genU64 doubleBitsToUint64(genD)", "genD int64BitsToDouble(genI64)", "genD uint64BitsToDouble(genU64)",
    "int64_t packInt2x32(ivec2)", "uint64_t packUint2x32(uvec2)", "ivec2 unpackInt2x32(int64_t)", "uvec2 unpackUint2x32(uint64_t)",
    "vecB lessThan(vecI64,vecI64)", "vecB lessThan(vecU64,vecU64)", "vecB lessThanEqual(vecI64,vecI64)",
    "vecB lessThanEqual(vecU64,vecU64)", "vecB greaterThan(vecI64,vecI64)", "vecB greaterThan(vecU64,vecU64)",
    "vecB greaterThanEqual(vecI64,vecI64)", "vecB greaterThanEqual(vecU64,vecU64)", "vecB equal(vecI64,vecI64)",
    "vecB equal(vecU64,vecU64)", "vecB notEqual(vecI64,vecI64)", "vecB notEqual(vecU64,vecU64)",
    // ARB_shader_ballot
    "uint64_t ballotARB(bool)", "genF readInvocationARB(genF,uint)", "genI readInvocationARB(genI,uint)",
    "genU readInvocationARB(genU,uint)", "genF readFirstInvocationARB(genF)", "genI readFirstInvocationARB(genI)",
    "genU readFirstInvocationARB(genU)",
};

// KHR_shader_subgroup. Extra placeholders, expanded before addSig: genT = genF/genD/genI/genU/genB,
// genN = numeric (genF/genD/genI/genU), genL = logical (genI/genU/genB).
const char* kSubgroupSigs[] = {
    // basic
    "void subgroupBarrier()", "void subgroupMemoryBarrier()", "void subgroupMemoryBarrierBuffer()",
    "void subgroupMemoryBarrierImage()", "bool subgroupElect()",
    // vote
    "bool subgroupAll(bool)", "bool subgroupAny(bool)", "bool subgroupAllEqual(genT)",
    // arithmetic
    "genN subgroupAdd(genN)", "genN subgroupMul(genN)", "genN subgroupMin(genN)", "genN subgroupMax(genN)",
    "genL subgroupAnd(genL)", "genL subgroupOr(genL)", "genL subgroupXor(genL)",
    "genN subgroupInclusiveAdd(genN)", "genN subgroupInclusiveMul(genN)", "genN subgroupInclusiveMin(genN)",
    "genN subgroupInclusiveMax(genN)", "genL subgroupInclusiveAnd(genL)", "genL subgroupInclusiveOr(genL)",
    "genL subgroupInclusiveXor(genL)", "genN subgroupExclusiveAdd(genN)", "genN subgroupExclusiveMul(genN)",
    "genN subgroupExclusiveMin(genN)", "genN subgroupExclusiveMax(genN)", "genL subgroupExclusiveAnd(genL)",
    "genL subgroupExclusiveOr(genL)", "genL subgroupExclusiveXor(genL)",
    // ballot
    "genT subgroupBroadcast(genT,uint)", "genT subgroupBroadcastFirst(genT)", "uvec4 subgroupBallot(bool)",
    "bool subgroupInverseBallot(uvec4)", "bool subgroupBallotBitExtract(uvec4,uint)", "uint subgroupBallotBitCount(uvec4)",
    "uint subgroupBallotInclusiveBitCount(uvec4)", "uint subgroupBallotExclusiveBitCount(uvec4)",
    "uint subgroupBallotFindLSB(uvec4)", "uint subgroupBallotFindMSB(uvec4)",
    // shuffle, shuffle_relative
    "genT subgroupShuffle(genT,uint)", "genT subgroupShuffleXor(genT,uint)", "genT subgroupShuffleUp(genT,uint)",
    "genT subgroupShuffleDown(genT,uint)",
    // clustered
    "genN subgroupClusteredAdd(genN,uint)", "genN subgroupClusteredMul(genN,uint)", "genN subgroupClusteredMin(genN,uint)",
    "genN subgroupClusteredMax(genN,uint)", "genL subgroupClusteredAnd(genL,uint)", "genL subgroupClusteredOr(genL,uint)",
    "genL subgroupClusteredXor(genL,uint)",
    // quad
    "genT subgroupQuadBroadcast(genT,uint)", "genT subgroupQuadSwapHorizontal(genT)", "genT subgroupQuadSwapVertical(genT)",
    "genT subgroupQuadSwapDiagonal(genT)",
    // NV_shader_subgroup_partitioned
    "uvec4 subgroupPartitionNV(genT)",
    "genN subgroupPartitionedAddNV(genN,uvec4)", "genN subgroupPartitionedMulNV(genN,uvec4)", "genN subgroupPartitionedMinNV(genN,uvec4)",
    "genN subgroupPartitionedMaxNV(genN,uvec4)", "genL subgroupPartitionedAndNV(genL,uvec4)", "genL subgroupPartitionedOrNV(genL,uvec4)",
    "genL subgroupPartitionedXorNV(genL,uvec4)", "genN subgroupPartitionedInclusiveAddNV(genN,uvec4)",
    "genN subgroupPartitionedInclusiveMulNV(genN,uvec4)", "genN subgroupPartitionedInclusiveMinNV(genN,uvec4)",
    "genN subgroupPartitionedInclusiveMaxNV(genN,uvec4)", "genL subgroupPartitionedInclusiveAndNV(genL,uvec4)",
    "genL subgroupPartitionedInclusiveOrNV(genL,uvec4)", "genL subgroupPartitionedInclusiveXorNV(genL,uvec4)",
    "genN subgroupPartitionedExclusiveAddNV(genN,uvec4)", "genN subgroupPartitionedExclusiveMulNV(genN,uvec4)",
    "genN subgroupPartitionedExclusiveMinNV(genN,uvec4)", "genN subgroupPartitionedExclusiveMaxNV(genN,uvec4)",
    "genL subgroupPartitionedExclusiveAndNV(genL,uvec4)", "genL subgroupPartitionedExclusiveOrNV(genL,uvec4)",
    "genL subgroupPartitionedExclusiveXorNV(genL,uvec4)",
};

struct Ctx {
    std::unordered_map<std::string, std::vector<const Builtin*>> byName;
    std::vector<std::unique_ptr<Builtin>> all;
    Builtin* add(const std::string& name, const Type& ret, std::vector<Type> params, std::vector<uint8_t> outs = {}) {
        auto b = std::make_unique<Builtin>();
        b->name = name;
        b->op = BIOp::Generic;
        b->ret = ret;
        b->params = std::move(params);
        b->paramOut = outs.empty() ? std::vector<uint8_t>(b->params.size(), 0) : std::move(outs);
        Builtin* r = b.get();
        byName[name].push_back(r);
        all.push_back(std::move(b));
        return r;
    }
    // Lookups with an explicit LOD bias need implicit derivatives: fragment shaders only.
    void addBias(const std::string& name, const Type& ret, std::vector<Type> params) {
        add(name, ret, std::move(params))->stageMask = 1u << (int)Stage::Fragment;
    }
};

std::string replaceAll(std::string s, const std::string& a, const std::string& b) {
    size_t p = 0;
    while ((p = s.find(a, p)) != std::string::npos) { s.replace(p, a.size(), b); p += b.size(); }
    return s;
}

// Resolve a placeholder type word given the current generic size / matrix shape.
bool resolveWord(const std::string& w, int n, int mc, int mr, Type& out) {
    auto vecOf = [&](BT b, int k) { return k == 1 ? Type::scalar(b) : Type::vector(b, k); };
    static const std::pair<const char*, BT> bases[] = {{"F", BT::Float}, {"D", BT::Double}, {"I", BT::Int}, {"U", BT::UInt}, {"B", BT::Bool},
                                                       {"I64", BT::Int64}, {"U64", BT::UInt64}};
    for (auto& b : bases) {
        if (w == std::string("gen") + b.first) { out = vecOf(b.second, n); return true; }
        if (w == std::string("vec") + b.first) { out = vecOf(b.second, n); return true; }
        if (w == std::string("mat") + b.first) { out = Type::matrix(b.second, mc, mr); return true; }
    }
    const Type* t = typeKeyword(w);
    if (!t) return false;
    out = *t;
    return true;
}

void addSig(Ctx& c, std::string sig) {
    // expansions of FD / IU
    if (sig.find("FD") != std::string::npos) {
        addSig(c, replaceAll(replaceAll(replaceAll(replaceAll(sig, "genFD", "genF"), "vecFD", "vecF"), "matFD", "matF"), "FD", "float"));
        addSig(c, replaceAll(replaceAll(replaceAll(replaceAll(sig, "genFD", "genD"), "vecFD", "vecD"), "matFD", "matD"), "FD", "double"));
        return;
    }
    if (sig.find("IU") != std::string::npos) {
        addSig(c, replaceAll(replaceAll(replaceAll(sig, "genIU", "genI"), "vecIU", "vecI"), "IU", "int"));
        addSig(c, replaceAll(replaceAll(replaceAll(sig, "genIU", "genU"), "vecIU", "vecU"), "IU", "uint"));
        return;
    }
    // parse "RET name(args)"
    size_t sp = sig.find(' ');
    size_t lp = sig.find('(');
    std::string retW = sig.substr(0, sp);
    std::string name = sig.substr(sp + 1, lp - sp - 1);
    std::string argstr = sig.substr(lp + 1, sig.size() - lp - 2);
    std::vector<std::string> args;
    std::vector<uint8_t> outs;
    std::stringstream ss(argstr);
    std::string a;
    while (std::getline(ss, a, ',')) {
        uint8_t o = 0;
        if (a.compare(0, 4, "out ") == 0) { o = 1; a = a.substr(4); }
        else if (a.compare(0, 6, "inout ") == 0) { o = 2; a = a.substr(6); }
        args.push_back(a);
        outs.push_back(o);
    }
    bool hasGen = sig.find("gen") != std::string::npos;
    bool hasVec = sig.find("vecF") != std::string::npos || sig.find("vecD") != std::string::npos ||
                  sig.find("vecI") != std::string::npos || sig.find("vecU") != std::string::npos || sig.find("vecB") != std::string::npos;
    bool hasMat = sig.find("matF") != std::string::npos || sig.find("matD") != std::string::npos;
    std::vector<std::tuple<int, int, int>> shapes;
    if (hasMat) {
        for (int cc = 2; cc <= 4; cc++) for (int r = 2; r <= 4; r++) shapes.push_back({1, cc, r});
    } else if (hasGen) {
        for (int n = 1; n <= 4; n++) shapes.push_back({n, 0, 0});
    } else if (hasVec) {
        for (int n = 2; n <= 4; n++) shapes.push_back({n, 0, 0});
    } else {
        shapes.push_back({1, 0, 0});
    }
    for (auto [n, mc, mr] : shapes) {
        Type ret;
        if (retW == "void") ret = Type::voidT();
        else if (!resolveWord(retW, n, mc, mr, ret)) continue;
        std::vector<Type> ps;
        bool ok = true;
        for (auto& w : args) {
            Type t;
            if (!resolveWord(w, n, mc, mr, t)) { ok = false; break; }
            ps.push_back(t);
        }
        if (ok) c.add(name, ret, ps, outs);
    }
}

Type vecT(BT b, int n) { return n == 1 ? Type::scalar(b) : Type::vector(b, n); }

int coordDim(const Type& s) {
    switch (s.dim) {
        case Dim::D1: return 1;
        case Dim::D2: return 2;
        case Dim::D3: return 3;
        case Dim::Cube: return 3;
        case Dim::Rect: return 2;
        case Dim::Buffer: return 1;
        default: return 0;
    }
}
int sizeDim(const Type& s) {
    int n = s.dim == Dim::Cube ? 2 : coordDim(s);
    if (s.arrayed) n++;
    return n;
}

void addTextureFunctions(Ctx& c) {
    static const char* samplerNames[] = {
        "sampler1D", "sampler2D", "sampler3D", "samplerCube", "sampler2DRect", "samplerBuffer", "sampler1DArray", "sampler2DArray",
        "samplerCubeArray", "sampler2DMS", "sampler2DMSArray", "sampler1DShadow", "sampler2DShadow", "samplerCubeShadow",
        "sampler2DRectShadow", "sampler1DArrayShadow", "sampler2DArrayShadow", "samplerCubeArrayShadow"};
    const Type F = Type::scalar(BT::Float), I = Type::scalar(BT::Int);
    for (const char* pfx : {"", "i", "u"}) {
        for (const char* sn : samplerNames) {
            std::string name = std::string(pfx) + sn;
            const Type* tp = typeKeyword(name);
            if (!tp) continue;  // e.g. isampler2DShadow
            const Type T = *tp;
            const Type G4 = Type::vector(T.sampled, 4);
            const int cd = coordDim(T);
            const bool rect = T.dim == Dim::Rect, buf = T.dim == Dim::Buffer, cube = T.dim == Dim::Cube;
            const bool ms = T.ms, sh = T.shadow, arr = T.arrayed;
            const Type ret = sh ? F : G4;
            int pn = cd + (arr ? 1 : 0) + (sh ? 1 : 0);
            if (sh && T.dim == Dim::D1) pn = 3;
            bool cubeArrShadow = sh && cube && arr;
            if (cubeArrShadow) pn = 4;
            const Type P = vecT(BT::Float, pn);
            const Type gradT = vecT(BT::Float, cd);
            const Type offT = vecT(BT::Int, cube ? 3 : cd);
            const Type fetchP = vecT(BT::Int, cd + (arr ? 1 : 0));

            // textureSize
            if (rect || buf || ms) c.add("textureSize", vecT(BT::Int, sizeDim(T)), {T});
            else c.add("textureSize", vecT(BT::Int, sizeDim(T)), {T, I});
            if (ms) {
                c.add("textureSamples", I, {T});
                c.add("texelFetch", G4, {T, fetchP, I});
                continue;
            }
            if (buf) {
                c.add("texelFetch", G4, {T, I});
                continue;
            }
            if (!rect) {
                c.add("textureQueryLod", Type::vector(BT::Float, 2), {T, vecT(BT::Float, cd)});
                c.add("textureQueryLOD", Type::vector(BT::Float, 2), {T, vecT(BT::Float, cd)});  // ARB_texture_query_lod spelling
                c.add("textureQueryLevels", I, {T});
            }
            // texture
            if (cubeArrShadow) {
                c.add("texture", F, {T, P, F});
                c.add("textureGrad", F, {T, P, F, gradT, gradT});  // nonstandard but harmless
            } else {
                c.add("texture", ret, {T, P});
                bool bias = !rect && !(sh && arr && T.dim == Dim::D2);
                if (bias) c.addBias("texture", ret, {T, P, F});
                // textureLod
                if (!rect) c.add("textureLod", ret, {T, P, F});
                // textureGrad
                c.add("textureGrad", ret, {T, P, gradT, gradT});
                if (!cube) {
                    c.add("textureOffset", ret, {T, P, offT});
                    if (bias) c.addBias("textureOffset", ret, {T, P, offT, F});
                    if (!rect) c.add("textureLodOffset", ret, {T, P, F, offT});
                    c.add("textureGradOffset", ret, {T, P, gradT, gradT, offT});
                }
            }
            // texelFetch
            if (!sh && !cube) {
                if (rect) {
                    c.add("texelFetch", G4, {T, fetchP});
                    c.add("texelFetchOffset", G4, {T, fetchP, offT});
                } else {
                    c.add("texelFetch", G4, {T, fetchP, I});
                    c.add("texelFetchOffset", G4, {T, fetchP, I, offT});
                }
            }
            // projective
            if (!arr && !cube) {
                std::vector<Type> projP;
                if (sh) projP.push_back(Type::vector(BT::Float, 4));
                else {
                    if (cd < 3) projP.push_back(vecT(BT::Float, cd + 1));
                    projP.push_back(Type::vector(BT::Float, 4));
                }
                for (auto& pp : projP) {
                    c.add("textureProj", ret, {T, pp});
                    if (!rect) c.addBias("textureProj", ret, {T, pp, F});
                    if (!rect) c.add("textureProjLod", ret, {T, pp, F});
                    c.add("textureProjGrad", ret, {T, pp, gradT, gradT});
                    c.add("textureProjOffset", ret, {T, pp, offT});
                    if (!rect) c.addBias("textureProjOffset", ret, {T, pp, offT, F});
                    if (!rect) c.add("textureProjLodOffset", ret, {T, pp, F, offT});
                    c.add("textureProjGradOffset", ret, {T, pp, gradT, gradT, offT});
                }
            }
            // gather
            if ((T.dim == Dim::D2 || cube || rect)) {
                const Type gP = vecT(BT::Float, cd + (arr ? 1 : 0));
                if (sh) {
                    c.add("textureGather", Type::vector(BT::Float, 4), {T, gP, F});
                    if (!cube) {
                        c.add("textureGatherOffset", Type::vector(BT::Float, 4), {T, gP, F, Type::vector(BT::Int, 2)});
                        c.add("textureGatherOffsets", Type::vector(BT::Float, 4), {T, gP, F, Type::vector(BT::Int, 2).arrayOf(4)});
                    }
                } else {
                    c.add("textureGather", G4, {T, gP});
                    c.add("textureGather", G4, {T, gP, I});
                    if (!cube) {
                        c.add("textureGatherOffset", G4, {T, gP, Type::vector(BT::Int, 2)});
                        c.add("textureGatherOffset", G4, {T, gP, Type::vector(BT::Int, 2), I});
                        c.add("textureGatherOffsets", G4, {T, gP, Type::vector(BT::Int, 2).arrayOf(4)});
                        c.add("textureGatherOffsets", G4, {T, gP, Type::vector(BT::Int, 2).arrayOf(4), I});
                    }
                }
            }
        }
    }
    // Legacy (GLSL <= 1.20 / compatibility) and extension spellings.
    auto S = [](const char* n) { return *typeKeyword(n); };
    const Type V2 = Type::vector(BT::Float, 2), V3 = Type::vector(BT::Float, 3), V4 = Type::vector(BT::Float, 4);
    struct L { const char* name; const char* sampler; std::vector<Type> coords; bool bias, lod, grad; Type ret; };
    const std::vector<L> legacy = {
        {"texture1D", "sampler1D", {F}, true, false, false, V4},
        {"texture1DProj", "sampler1D", {V2, V4}, true, false, false, V4},
        {"texture2D", "sampler2D", {V2}, true, false, false, V4},
        {"texture2DProj", "sampler2D", {V3, V4}, true, false, false, V4},
        {"texture3D", "sampler3D", {V3}, true, false, false, V4},
        {"texture3DProj", "sampler3D", {V4}, true, false, false, V4},
        {"textureCube", "samplerCube", {V3}, true, false, false, V4},
        {"shadow1D", "sampler1DShadow", {V3}, true, false, false, V4},
        {"shadow2D", "sampler2DShadow", {V3}, true, false, false, V4},
        {"shadow1DProj", "sampler1DShadow", {V4}, true, false, false, V4},
        {"shadow2DProj", "sampler2DShadow", {V4}, true, false, false, V4},
        {"texture2DRect", "sampler2DRect", {V2}, false, false, false, V4},
        {"texture2DRectProj", "sampler2DRect", {V3, V4}, false, false, false, V4},
        {"shadow2DRect", "sampler2DRectShadow", {V3}, false, false, false, V4},
        {"shadow2DRectProj", "sampler2DRectShadow", {V4}, false, false, false, V4},
        {"texture1DArray", "sampler1DArray", {V2}, true, false, false, V4},
        {"texture2DArray", "sampler2DArray", {V3}, true, false, false, V4},
        {"shadow1DArray", "sampler1DArrayShadow", {V3}, true, false, false, V4},
        {"shadow2DArray", "sampler2DArrayShadow", {V4}, false, false, false, V4},
    };
    for (auto& l : legacy) {
        Type T = S(l.sampler);
        for (auto& p : l.coords) {
            c.add(l.name, l.ret, {T, p});
            if (l.bias) c.addBias(l.name, l.ret, {T, p, F});
        }
    }
    const std::vector<L> lods = {
        {"texture1DLod", "sampler1D", {F}, 0, 1, 0, V4},          {"texture1DProjLod", "sampler1D", {V2, V4}, 0, 1, 0, V4},
        {"texture2DLod", "sampler2D", {V2}, 0, 1, 0, V4},         {"texture2DProjLod", "sampler2D", {V3, V4}, 0, 1, 0, V4},
        {"texture3DLod", "sampler3D", {V3}, 0, 1, 0, V4},         {"texture3DProjLod", "sampler3D", {V4}, 0, 1, 0, V4},
        {"textureCubeLod", "samplerCube", {V3}, 0, 1, 0, V4},     {"shadow1DLod", "sampler1DShadow", {V3}, 0, 1, 0, V4},
        {"shadow2DLod", "sampler2DShadow", {V3}, 0, 1, 0, V4},    {"shadow1DProjLod", "sampler1DShadow", {V4}, 0, 1, 0, V4},
        {"shadow2DProjLod", "sampler2DShadow", {V4}, 0, 1, 0, V4}, {"texture2DArrayLod", "sampler2DArray", {V3}, 0, 1, 0, V4},
        {"texture1DArrayLod", "sampler1DArray", {V2}, 0, 1, 0, V4}, {"texture2DLodEXT", "sampler2D", {V2}, 0, 1, 0, V4},
        {"texture2DProjLodEXT", "sampler2D", {V3, V4}, 0, 1, 0, V4}, {"textureCubeLodEXT", "samplerCube", {V3}, 0, 1, 0, V4},
        {"texture2DLodARB", "sampler2D", {V2}, 0, 1, 0, V4},
    };
    for (auto& l : lods) {
        Type T = S(l.sampler);
        for (auto& p : l.coords) c.add(l.name, l.ret, {T, p, F});
    }
    struct Gd { const char* name; const char* sampler; std::vector<Type> coords; Type grad; };
    const std::vector<Gd> grads = {
        {"texture1DGradARB", "sampler1D", {F}, F},           {"texture1DProjGradARB", "sampler1D", {V2, V4}, F},
        {"texture2DGradARB", "sampler2D", {V2}, V2},         {"texture2DProjGradARB", "sampler2D", {V3, V4}, V2},
        {"texture3DGradARB", "sampler3D", {V3}, V3},         {"texture3DProjGradARB", "sampler3D", {V4}, V3},
        {"textureCubeGradARB", "samplerCube", {V3}, V3},     {"shadow1DGradARB", "sampler1DShadow", {V3}, F},
        {"shadow2DGradARB", "sampler2DShadow", {V3}, V2},    {"shadow2DProjGradARB", "sampler2DShadow", {V4}, V2},
        {"texture2DRectGradARB", "sampler2DRect", {V2}, V2}, {"texture2DGradEXT", "sampler2D", {V2}, V2},
        {"texture2DProjGradEXT", "sampler2D", {V3, V4}, V2}, {"textureCubeGradEXT", "samplerCube", {V3}, V3},
    };
    for (auto& g : grads) {
        Type T = S(g.sampler);
        for (auto& p : g.coords) c.add(g.name, V4, {T, p, g.grad, g.grad});
    }
    // EXT_gpu_shader4 spellings
    for (const char* pfx : {"", "i", "u"}) {
        auto st = [&](const char* n) { return S((std::string(pfx) + n).c_str()); };
        Type G4 = Type::vector(st("sampler2D").sampled, 4);
        c.add("texelFetch1D", G4, {st("sampler1D"), I, I});
        c.add("texelFetch2D", G4, {st("sampler2D"), Type::vector(BT::Int, 2), I});
        c.add("texelFetch3D", G4, {st("sampler3D"), Type::vector(BT::Int, 3), I});
        c.add("texelFetch2DRect", G4, {st("sampler2DRect"), Type::vector(BT::Int, 2)});
        c.add("texelFetch1DArray", G4, {st("sampler1DArray"), Type::vector(BT::Int, 2), I});
        c.add("texelFetch2DArray", G4, {st("sampler2DArray"), Type::vector(BT::Int, 3), I});
        c.add("texelFetchBuffer", G4, {st("samplerBuffer"), I});
        c.add("textureSize1D", I, {st("sampler1D"), I});
        c.add("textureSize2D", Type::vector(BT::Int, 2), {st("sampler2D"), I});
        c.add("textureSize3D", Type::vector(BT::Int, 3), {st("sampler3D"), I});
        c.add("textureSizeCube", Type::vector(BT::Int, 2), {st("samplerCube"), I});
        c.add("textureSize2DRect", Type::vector(BT::Int, 2), {st("sampler2DRect")});
        c.add("textureSize2DArray", Type::vector(BT::Int, 3), {st("sampler2DArray"), I});
        c.add("textureSizeBuffer", I, {st("samplerBuffer")});
    }
}

void addImageFunctions(Ctx& c) {
    static const char* imgNames[] = {"image1D", "image2D", "image3D", "image2DRect", "imageCube", "imageBuffer", "image1DArray",
                                     "image2DArray", "imageCubeArray", "image2DMS", "image2DMSArray"};
    const Type I = Type::scalar(BT::Int);
    for (const char* pfx : {"", "i", "u"}) {
        for (const char* in : imgNames) {
            const Type T = *typeKeyword(std::string(pfx) + in);
            const Type G4 = Type::vector(T.sampled, 4);
            int cd = coordDim(T);
            int pn = cd + (T.arrayed ? 1 : 0);
            if (T.dim == Dim::Cube) pn = 3;
            const Type P = vecT(BT::Int, pn);
            std::vector<Type> base = {T, P};
            if (T.ms) base.push_back(I);
            auto with = [&](std::vector<Type> extra) { auto v = base; v.insert(v.end(), extra.begin(), extra.end()); return v; };
            c.add("imageLoad", G4, base);
            c.add("imageStore", Type::voidT(), with({G4}));
            c.add("imageSize", vecT(BT::Int, sizeDim(T)), {T});
            if (T.ms) c.add("imageSamples", I, {T});
            if (T.sampled != BT::Float) {
                Type S = Type::scalar(T.sampled);
                for (const char* op : {"imageAtomicAdd", "imageAtomicMin", "imageAtomicMax", "imageAtomicAnd", "imageAtomicOr",
                                       "imageAtomicXor", "imageAtomicExchange"})
                    c.add(op, S, with({S}));
                c.add("imageAtomicCompSwap", S, with({S, S}));
            } else {
                c.add("imageAtomicExchange", Type::scalar(BT::Float), with({Type::scalar(BT::Float)}));
            }
        }
    }
}

void addMatrixFunctions(Ctx& c) {
    for (BT b : {BT::Float, BT::Double}) {
        for (int cc = 2; cc <= 4; cc++)
            for (int r = 2; r <= 4; r++) {
                c.add("outerProduct", Type::matrix(b, cc, r), {Type::vector(b, r), Type::vector(b, cc)});
                c.add("transpose", Type::matrix(b, r, cc), {Type::matrix(b, cc, r)});
            }
    }
}

Ctx& ctx() {
    static Ctx* c = [] {
        Ctx* c = new Ctx;
        for (const char* s : kSigs) addSig(*c, s);
        for (const char* s : kSubgroupSigs) {
            static const std::pair<const char*, const char*> groups[] = {{"genT", "FDIUB"}, {"genN", "FDIU"}, {"genL", "IUB"}};
            std::string sig = s;
            bool expanded = false;
            for (auto& [ph, bases] : groups) {
                if (sig.find(ph) == std::string::npos) continue;
                for (const char* b = bases; *b; b++) addSig(*c, replaceAll(sig, ph, std::string("gen") + *b));
                expanded = true;
            }
            if (!expanded) addSig(*c, sig);
        }
        c->add("subgroupMemoryBarrierShared", Type::voidT(), {})->stageMask = 1u << (int)Stage::Compute;
        for (const char* s : kInt64Sigs) addSig(*c, s);
        addTextureFunctions(*c);
        addImageFunctions(*c);
        addMatrixFunctions(*c);
        return c;
    }();
    return *c;
}

// ---------------------------------------------------------------------------
// Builtin variable preambles
// ---------------------------------------------------------------------------
const char* kCommonPreamble = R"(
const ivec3 gl_MaxComputeWorkGroupCount = ivec3(65535, 65535, 65535);
const ivec3 gl_MaxComputeWorkGroupSize = ivec3(1024, 1024, 64);
struct gl_DepthRangeParameters { float near; float far; float diff; };
uniform gl_DepthRangeParameters gl_DepthRange;
uniform mat4 gl_ModelViewMatrix;
uniform mat4 gl_ProjectionMatrix;
uniform mat4 gl_ModelViewProjectionMatrix;
uniform mat4 gl_TextureMatrix[8];
uniform mat3 gl_NormalMatrix;
uniform mat4 gl_ModelViewMatrixInverse;
uniform mat4 gl_ProjectionMatrixInverse;
uniform mat4 gl_ModelViewProjectionMatrixInverse;
uniform mat4 gl_TextureMatrixInverse[8];
uniform mat4 gl_ModelViewMatrixTranspose;
uniform mat4 gl_ProjectionMatrixTranspose;
uniform mat4 gl_ModelViewProjectionMatrixTranspose;
uniform mat4 gl_TextureMatrixTranspose[8];
uniform mat4 gl_ModelViewMatrixInverseTranspose;
uniform mat4 gl_ProjectionMatrixInverseTranspose;
uniform mat4 gl_ModelViewProjectionMatrixInverseTranspose;
uniform mat4 gl_TextureMatrixInverseTranspose[8];
uniform float gl_NormalScale;
uniform vec4 gl_ClipPlane[8];
struct gl_PointParameters { float size; float sizeMin; float sizeMax; float fadeThresholdSize; float distanceConstantAttenuation; float distanceLinearAttenuation; float distanceQuadraticAttenuation; };
uniform gl_PointParameters gl_Point;
struct gl_MaterialParameters { vec4 emission; vec4 ambient; vec4 diffuse; vec4 specular; float shininess; };
uniform gl_MaterialParameters gl_FrontMaterial;
uniform gl_MaterialParameters gl_BackMaterial;
struct gl_LightSourceParameters { vec4 ambient; vec4 diffuse; vec4 specular; vec4 position; vec4 halfVector; vec3 spotDirection; float spotExponent; float spotCutoff; float spotCosCutoff; float constantAttenuation; float linearAttenuation; float quadraticAttenuation; };
uniform gl_LightSourceParameters gl_LightSource[8];
struct gl_LightModelParameters { vec4 ambient; };
uniform gl_LightModelParameters gl_LightModel;
struct gl_LightModelProducts { vec4 sceneColor; };
uniform gl_LightModelProducts gl_FrontLightModelProduct;
uniform gl_LightModelProducts gl_BackLightModelProduct;
struct gl_LightProducts { vec4 ambient; vec4 diffuse; vec4 specular; };
uniform gl_LightProducts gl_FrontLightProduct[8];
uniform gl_LightProducts gl_BackLightProduct[8];
uniform vec4 gl_TextureEnvColor[8];
uniform vec4 gl_EyePlaneS[8];
uniform vec4 gl_EyePlaneT[8];
uniform vec4 gl_EyePlaneR[8];
uniform vec4 gl_EyePlaneQ[8];
uniform vec4 gl_ObjectPlaneS[8];
uniform vec4 gl_ObjectPlaneT[8];
uniform vec4 gl_ObjectPlaneR[8];
uniform vec4 gl_ObjectPlaneQ[8];
struct gl_FogParameters { vec4 color; float density; float start; float end; float scale; };
uniform gl_FogParameters gl_Fog;
)";

const char* kVertexPreamble = R"(
in int gl_VertexID;
in int gl_InstanceID;
in int gl_BaseVertex;
in int gl_BaseInstance;
in int gl_DrawID;
in int gl_BaseVertexARB;
in int gl_BaseInstanceARB;
in int gl_DrawIDARB;
in int gl_VertexIndex;
in int gl_InstanceIndex;
in vec4 gl_Vertex;
in vec3 gl_Normal;
in vec4 gl_Color;
in vec4 gl_SecondaryColor;
in vec4 gl_MultiTexCoord0;
in vec4 gl_MultiTexCoord1;
in vec4 gl_MultiTexCoord2;
in vec4 gl_MultiTexCoord3;
in vec4 gl_MultiTexCoord4;
in vec4 gl_MultiTexCoord5;
in vec4 gl_MultiTexCoord6;
in vec4 gl_MultiTexCoord7;
in float gl_FogCoord;
out vec4 gl_Position;
out float gl_PointSize;
out float gl_ClipDistance[];
out float gl_CullDistance[];
out vec4 gl_ClipVertex;
out int gl_Layer;
out int gl_ViewportIndex;
out vec4 gl_FrontColor;
out vec4 gl_BackColor;
out vec4 gl_FrontSecondaryColor;
out vec4 gl_BackSecondaryColor;
out vec4 gl_TexCoord[];
out float gl_FogFragCoord;
)";

const char* kPerVertexIn = R"(
in gl_PerVertex { vec4 gl_Position; float gl_PointSize; float gl_ClipDistance[8]; float gl_CullDistance[8]; vec4 gl_ClipVertex;
  vec4 gl_FrontColor; vec4 gl_BackColor; vec4 gl_FrontSecondaryColor; vec4 gl_BackSecondaryColor; vec4 gl_TexCoord[8]; float gl_FogFragCoord; } gl_in[];
)";

const char* kGeometryPreamble = R"(
in int gl_PrimitiveIDIn;
in int gl_InvocationID;
out vec4 gl_Position;
out float gl_PointSize;
out float gl_ClipDistance[];
out float gl_CullDistance[];
out vec4 gl_ClipVertex;
out int gl_PrimitiveID;
out int gl_Layer;
out int gl_ViewportIndex;
out vec4 gl_FrontColor;
out vec4 gl_BackColor;
out vec4 gl_FrontSecondaryColor;
out vec4 gl_BackSecondaryColor;
out vec4 gl_TexCoord[];
out float gl_FogFragCoord;
)";

const char* kPerVertexInTess = R"(
in gl_PerVertex { vec4 gl_Position; float gl_PointSize; float gl_ClipDistance[8]; float gl_CullDistance[8]; } gl_in[];
)";

const char* kTessControlPreamble = R"(
in int gl_PatchVerticesIn;
in int gl_PrimitiveID;
in int gl_InvocationID;
out gl_PerVertex { vec4 gl_Position; float gl_PointSize; float gl_ClipDistance[8]; float gl_CullDistance[8]; } gl_out[];
patch out float gl_TessLevelOuter[4];
patch out float gl_TessLevelInner[2];
)";

const char* kTessEvalPreamble = R"(
in int gl_PatchVerticesIn;
in int gl_PrimitiveID;
in vec3 gl_TessCoord;
patch in float gl_TessLevelOuter[4];
patch in float gl_TessLevelInner[2];
out vec4 gl_Position;
out float gl_PointSize;
out float gl_ClipDistance[];
out float gl_CullDistance[];
out int gl_Layer;
out int gl_ViewportIndex;
)";

const char* kFragmentPreamble = R"(
in vec4 gl_FragCoord;
in bool gl_FrontFacing;
in vec2 gl_PointCoord;
in float gl_ClipDistance[];
in float gl_CullDistance[];
in int gl_PrimitiveID;
in int gl_SampleID;
in vec2 gl_SamplePosition;
in int gl_SampleMaskIn[1];
uniform int gl_NumSamples;
in int gl_Layer;
in int gl_ViewportIndex;
in bool gl_HelperInvocation;
out float gl_FragDepth;
out int gl_SampleMask[1];
out vec4 gl_FragColor;
out vec4 gl_FragData[];
in vec4 gl_Color;
in vec4 gl_SecondaryColor;
in vec4 gl_TexCoord[];
in float gl_FogFragCoord;
)";

const char* kComputePreamble = R"(
in uvec3 gl_NumWorkGroups;
const uvec3 gl_WorkGroupSize = uvec3(1, 1, 1);
in uvec3 gl_WorkGroupID;
in uvec3 gl_LocalInvocationID;
in uvec3 gl_GlobalInvocationID;
in uint gl_LocalInvocationIndex;
)";

}  // namespace

const std::vector<const Builtin*>* findBuiltins(const std::string& name) {
    auto& c = ctx();
    auto it = c.byName.find(name);
    return it == c.byName.end() ? nullptr : &it->second;
}

const std::string& builtinPreamble(Stage stage) {
    static std::string cache[(int)Stage::Count];
    static bool init = false;
    if (!init) {
        init = true;
        for (int s = 0; s < (int)Stage::Count; s++) {
            std::string p;
#define X(name, glenum, value) p += "const int " #name " = " + std::to_string(value) + ";\n";
#include "limits.def"
#undef X
            p += kCommonPreamble;
            switch ((Stage)s) {
                case Stage::Vertex: p += kVertexPreamble; break;
                case Stage::TessControl: p += kPerVertexInTess; p += kTessControlPreamble; break;
                case Stage::TessEval: p += kPerVertexInTess; p += kTessEvalPreamble; break;
                case Stage::Geometry: p += kPerVertexIn; p += kGeometryPreamble; break;
                case Stage::Fragment: p += kFragmentPreamble; break;
                case Stage::Compute: p += kComputePreamble; break;
                default: break;
            }
            cache[s] = p;
        }
    }
    return cache[(int)stage];
}

BuiltinVar builtinVarId(const std::string& n, Stage stage) {
    static const std::unordered_map<std::string, BuiltinVar> common = {
        {"gl_VertexID", BuiltinVar::VertexID}, {"gl_VertexIndex", BuiltinVar::VertexID},
        {"gl_InstanceID", BuiltinVar::InstanceID}, {"gl_InstanceIndex", BuiltinVar::InstanceID},
        {"gl_BaseVertex", BuiltinVar::BaseVertex}, {"gl_BaseVertexARB", BuiltinVar::BaseVertex},
        {"gl_BaseInstance", BuiltinVar::BaseInstance}, {"gl_BaseInstanceARB", BuiltinVar::BaseInstance},
        {"gl_DrawID", BuiltinVar::DrawID}, {"gl_DrawIDARB", BuiltinVar::DrawID},
        {"gl_Position", BuiltinVar::Position}, {"gl_PointSize", BuiltinVar::PointSize},
        {"gl_ClipDistance", BuiltinVar::ClipDistance}, {"gl_CullDistance", BuiltinVar::CullDistance},
        {"gl_ClipVertex", BuiltinVar::ClipVertex},
        {"gl_Vertex", BuiltinVar::Vertex}, {"gl_Normal", BuiltinVar::Normal},
        {"gl_MultiTexCoord0", BuiltinVar::MultiTexCoord}, {"gl_MultiTexCoord1", BuiltinVar::MultiTexCoord},
        {"gl_MultiTexCoord2", BuiltinVar::MultiTexCoord}, {"gl_MultiTexCoord3", BuiltinVar::MultiTexCoord},
        {"gl_MultiTexCoord4", BuiltinVar::MultiTexCoord}, {"gl_MultiTexCoord5", BuiltinVar::MultiTexCoord},
        {"gl_MultiTexCoord6", BuiltinVar::MultiTexCoord}, {"gl_MultiTexCoord7", BuiltinVar::MultiTexCoord},
        {"gl_FogCoord", BuiltinVar::FogCoord},
        {"gl_FrontColor", BuiltinVar::FrontColor}, {"gl_BackColor", BuiltinVar::BackColor},
        {"gl_FrontSecondaryColor", BuiltinVar::FrontSecondaryColor}, {"gl_BackSecondaryColor", BuiltinVar::BackSecondaryColor},
        {"gl_TexCoord", BuiltinVar::TexCoord}, {"gl_FogFragCoord", BuiltinVar::FogFragCoord},
        {"gl_in", BuiltinVar::In}, {"gl_out", BuiltinVar::Out},
        {"gl_PrimitiveIDIn", BuiltinVar::PrimitiveIDIn}, {"gl_PrimitiveID", BuiltinVar::PrimitiveID},
        {"gl_InvocationID", BuiltinVar::InvocationID}, {"gl_Layer", BuiltinVar::Layer},
        {"gl_ViewportIndex", BuiltinVar::ViewportIndex}, {"gl_PatchVerticesIn", BuiltinVar::PatchVerticesIn},
        {"gl_TessLevelOuter", BuiltinVar::TessLevelOuter}, {"gl_TessLevelInner", BuiltinVar::TessLevelInner},
        {"gl_TessCoord", BuiltinVar::TessCoord},
        {"gl_FragCoord", BuiltinVar::FragCoord}, {"gl_FrontFacing", BuiltinVar::FrontFacing},
        {"gl_PointCoord", BuiltinVar::PointCoord}, {"gl_FragColor", BuiltinVar::FragColor},
        {"gl_FragData", BuiltinVar::FragData}, {"gl_FragDepth", BuiltinVar::FragDepth},
        {"gl_SampleID", BuiltinVar::SampleID}, {"gl_SamplePosition", BuiltinVar::SamplePosition},
        {"gl_SampleMaskIn", BuiltinVar::SampleMaskIn}, {"gl_SampleMask", BuiltinVar::SampleMask},
        {"gl_HelperInvocation", BuiltinVar::HelperInvocation},
        {"gl_NumWorkGroups", BuiltinVar::NumWorkGroups}, {"gl_WorkGroupID", BuiltinVar::WorkGroupID},
        {"gl_LocalInvocationID", BuiltinVar::LocalInvocationID}, {"gl_GlobalInvocationID", BuiltinVar::GlobalInvocationID},
        {"gl_LocalInvocationIndex", BuiltinVar::LocalInvocationIndex}, {"gl_WorkGroupSize", BuiltinVar::WorkGroupSize},
        {"gl_DepthRange", BuiltinVar::DepthRange}, {"gl_NumSamples", BuiltinVar::NumSamples},
        {"gl_SubgroupSize", BuiltinVar::SubgroupSize}, {"gl_SubgroupInvocationID", BuiltinVar::SubgroupInvocationID},
        {"gl_SubgroupEqMask", BuiltinVar::SubgroupEqMask}, {"gl_SubgroupGeMask", BuiltinVar::SubgroupGeMask},
        {"gl_SubgroupGtMask", BuiltinVar::SubgroupGtMask}, {"gl_SubgroupLeMask", BuiltinVar::SubgroupLeMask},
        {"gl_SubgroupLtMask", BuiltinVar::SubgroupLtMask}, {"gl_NumSubgroups", BuiltinVar::NumSubgroups},
        {"gl_SubgroupID", BuiltinVar::SubgroupID},
    };
    if (n == "gl_Color") return stage == Stage::Fragment ? BuiltinVar::FragColorIn : BuiltinVar::Color;
    if (n == "gl_SecondaryColor") return stage == Stage::Fragment ? BuiltinVar::SecondaryColorIn : BuiltinVar::SecondaryColor;
    auto it = common.find(n);
    if (it != common.end()) return it->second;
    return BuiltinVar::CompatUniform;
}

}  // namespace glsl
