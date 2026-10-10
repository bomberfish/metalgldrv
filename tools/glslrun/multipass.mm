// glslrun multi-pass modes: emulate the runtime's dispatch for geometry shaders
// (mesh), tessellation (kernel + object/mesh) and transform feedback (kernel),
// plus a self-check of the tessellation primitive generator.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "glsl/glsl.h"

namespace glsl {
std::string buildMSLPrelude(const std::string& body);
}

using namespace glsl;

static std::string readAll(const std::string& p) {
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

struct PipelineArgs {
    std::vector<std::pair<Stage, std::string>> files;
    std::vector<float> verts;
    int vpp = 3;
    int instances = 1;
    std::vector<std::string> xfb;
    bool xfbSeparate = false;
    float defaultOuter[4] = {1, 1, 1, 1}, defaultInner[2] = {1, 1};
    bool dump = false;
    int expectPrims = -1;
    uint32_t xfbCapacity = 4096;
    bool tessCompute = false;  // --tess-compute: tessellation pass 1 as the mgl_tcs kernel (VariantKey::tessComputePass)
};

static id<MTLDevice> gDev;

static bool check(id obj, NSError* err, const char* what) {
    if (!obj) {
        fprintf(stderr, "%s: %s\n", what, err ? err.localizedDescription.UTF8String : "failed");
        return false;
    }
    return true;
}

int runPipeline(const PipelineArgs& pa) {
    LinkInput li;
    li.xfbVaryings = pa.xfb;
    li.xfbInterleaved = !pa.xfbSeparate;
    uint32_t mask = 0;
    for (auto& [st, path] : pa.files) {
        auto cr = compile(st, {readAll(path)});
        if (!cr.shader) { fprintf(stderr, "%s: %s", path.c_str(), cr.infoLog.c_str()); return 1; }
        li.shaders.push_back(cr.shader);
        mask |= 1u << (int)st;
    }
    auto lr = link(li);
    if (!lr.program) { fprintf(stderr, "%s", lr.infoLog.c_str()); return 3; }
    const Reflection& r = lr.reflection;
    bool tess = mask & (1u << (int)Stage::TessEval), gs = mask & (1u << (int)Stage::Geometry);
    bool xfbOnly = !pa.xfb.empty() && !tess;  // with a geometry shader: serial capture kernel (mgl_gs_xfb)
    VariantKey key;
    key.kind = tess ? PipelineKind::Tessellation : xfbOnly ? PipelineKind::XfbCapture : gs ? PipelineKind::MeshGeometry : PipelineKind::Classic;
    key.tessComputePass = pa.tessCompute;
    if (key.kind == PipelineKind::Classic && r.vsMeshPassthrough) {  // as the runtime: VS on a mesh stage
        key.kind = PipelineKind::MeshGeometry;
        key.primitive = pa.vpp == 1 ? 0 : pa.vpp == 2 ? 1 : 4;
    }
    for (auto& a : r.attributes) {
        VertexFetch& f = key.attribs[a.location];
        f.enabled = true;
        f.glType = 0x1406;
        f.size = 4;
        f.binding = (uint8_t)a.location;
    }
    auto mo = generateMSL(*lr.program, key);
    if (!mo.error.empty()) { fprintf(stderr, "%s\n", mo.error.c_str()); return 4; }
    if (pa.dump) printf("%s\n", mo.source.c_str());
    const int W = 64, H = 64;
    @autoreleasepool {
        gDev = MTLCreateSystemDefaultDevice();
        id<MTLDevice> dev = gDev;
        NSError* err = nil;
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.languageVersion = MTLLanguageVersion4_0;
        opts.mathMode = MTLMathModeSafe;
        id<MTLLibrary> lib = [dev newLibraryWithSource:@(mo.source.c_str()) options:opts error:&err];
        if (!check(lib, err, "compile")) return 5;
        auto entry = [&](EntryPoint::Kind k) -> const EntryPoint* {
            for (auto& e : mo.entryPoints) if (e.kind == k) return &e;
            return nullptr;
        };
        // vertex data: position at location 0 (float4), every attribute reads the same buffer
        size_t nverts = pa.verts.size() / 2;
        std::vector<float> v4(nverts * 4);
        for (size_t i = 0; i < nverts; i++) { v4[4 * i] = pa.verts[2 * i]; v4[4 * i + 1] = pa.verts[2 * i + 1]; v4[4 * i + 2] = 0; v4[4 * i + 3] = 1; }
        id<MTLBuffer> vb = [dev newBufferWithBytes:v4.data() length:std::max<size_t>(16, v4.size() * 4) options:MTLResourceStorageModeShared];
        std::vector<uint32_t> ids(nverts);
        for (size_t i = 0; i < nverts; i++) ids[i] = (uint32_t)i;
        id<MTLBuffer> ib = [dev newBufferWithBytes:ids.data() length:std::max<size_t>(4, ids.size() * 4) options:MTLResourceStorageModeShared];
        uint32_t prims = (uint32_t)(nverts / pa.vpp);
        id<MTLBuffer> vout = [dev newBufferWithLength:std::max<size_t>(16, (size_t)prims * pa.instances * 32 * std::max(16u, r.multipass.tcsVertexOutBytes)) options:MTLResourceStorageModeShared];
        id<MTLBuffer> pout = [dev newBufferWithLength:std::max<size_t>(16, (size_t)prims * pa.instances * std::max(32u, r.multipass.tcsPatchOutBytes)) options:MTLResourceStorageModeShared];
        id<MTLBuffer> counters = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
        memset(counters.contents, 0, 64);
        id<MTLBuffer> xfbBuf[4];
        for (int b = 0; b < 4; b++) {
            xfbBuf[b] = [dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];
            memset(xfbBuf[b].contents, 0xff, 4096);
        }
        MGLDrawInfo info{};
        for (auto& a : r.attributes) {
            info.vertexBuffer[a.location] = vb.gpuAddress;
            info.vertexBufferSize[a.location] = (uint32_t)vb.length;
            info.vertexStride[a.location] = 16;
        }
        info.indexBuffer = ib.gpuAddress;
        info.indexType = 4;
        info.count = (uint32_t)nverts;
        info.instanceCount = pa.instances;
        info.primitiveCount = prims;
        info.verticesPerPrimitive = pa.vpp;
        info.patchVertices = pa.vpp;
        info.tcsVertexOut = vout.gpuAddress;
        info.tcsPatchOut = pout.gpuAddress;
        info.counters = counters.gpuAddress;
        // one sub-draw: the decomposed list in indexBuffer
        MGLSubDraw sd{};
        id<MTLBuffer> sdBuf = [dev newBufferWithBytes:&sd length:sizeof sd options:MTLResourceStorageModeShared];
        info.subDraws = sdBuf.gpuAddress;
        info.subDrawCount = 1;
        info.subDrawIndexType = 0;
        for (int b = 0; b < 4; b++) {
            info.xfbBuffer[b] = xfbBuf[b].gpuAddress;
            info.xfbStride[b] = r.multipass.xfbStride[b];
            info.xfbCapacity[b] = pa.xfbCapacity;
        }
        // varying spill buffer (mesh producers past Metal's mesh output budget)
        id<MTLBuffer> spillBuf = [dev newBufferWithLength:16 << 20 options:MTLResourceStorageModeShared];
        id<MTLBuffer> spillCnt = [dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
        memset(spillCnt.contents, 0, 16);
        for (auto& ep : mo.entryPoints)
            if (ep.kind == EntryPoint::Mesh && ep.spillRecordBytes) {
                info.spillBuffer = spillBuf.gpuAddress;
                info.spillCounter = spillCnt.gpuAddress;
                info.spillCapacity = (uint32_t)((16u << 20) / ep.spillRecordBytes);
            }
        id<MTLBuffer> infoBuf = [dev newBufferWithBytes:&info length:sizeof info options:MTLResourceStorageModeShared];
        id<MTLBuffer> args = [dev newBufferWithLength:std::max<size_t>(8, r.slots.size() * 8) options:MTLResourceStorageModeShared];
        id<MTLBuffer> defaults = [dev newBufferWithLength:std::max<uint32_t>(16, r.defaultUniformSize) options:MTLResourceStorageModeShared];
        memset(defaults.contents, 0, defaults.length);
        if (!r.defaultUniformInit.empty()) memcpy(defaults.contents, r.defaultUniformInit.data(), r.defaultUniformInit.size());
        uint64_t* slots = (uint64_t*)args.contents;
        for (size_t i = 0; i < r.slots.size(); i++) slots[i] = r.slots[i].kind == SlotKind::DefaultUniforms ? defaults.gpuAddress : 0;
        MGLSys sys{};
        sys.depthRangeFar = 1;
        sys.pointSize = 1;
        sys.alphaFunc = 0x0207;
        sys.renderTargetHeight = H;
        for (int i = 0; i < 4; i++) sys.tessDefaultOuter[i] = pa.defaultOuter[i];
        for (int i = 0; i < 2; i++) sys.tessDefaultInner[i] = pa.defaultInner[i];
        id<MTLBuffer> sysBuf = [dev newBufferWithBytes:&sys length:sizeof sys options:MTLResourceStorageModeShared];
        NSArray* resources = @[ vb, ib, vout, pout, counters, xfbBuf[0], xfbBuf[1], xfbBuf[2], xfbBuf[3], defaults, spillBuf, spillCnt, sdBuf ];
        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLCommandBuffer> cb = [q commandBuffer];
        auto computePass = [&](const EntryPoint* k, MTLSize grid, bool threads) -> bool {
            id<MTLComputePipelineState> cps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@(k->name.c_str())] error:&err];
            if (!check(cps, err, "compute pipeline")) return false;
            id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
            [ce setComputePipelineState:cps];
            [ce setBuffer:args offset:0 atIndex:0];
            [ce setBuffer:sysBuf offset:0 atIndex:1];
            [ce setBuffer:infoBuf offset:0 atIndex:22];
            for (id<MTLBuffer> b in resources) [ce useResource:b usage:MTLResourceUsageRead | MTLResourceUsageWrite];
            MTLSize tg = MTLSizeMake(k->threadsPerThreadgroup[0], k->threadsPerThreadgroup[1], k->threadsPerThreadgroup[2]);
            if (threads) [ce dispatchThreads:grid threadsPerThreadgroup:tg];
            else [ce dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [ce endEncoding];
            return true;
        };
        if (xfbOnly) {
            // Same protocol as the runtime: (GS) count pass, CPU prefix sums, write pass.
            auto byName = [&](const char* n) -> const EntryPoint* {
                for (auto& e : mo.entryPoints) if (e.name == n) return &e;
                return nullptr;
            };
            uint32_t outVpp = gs ? (r.gsOutputPrimitive == 0 ? 1 : r.gsOutputPrimitive == 3 ? 2 : 3) : pa.vpp;
            uint32_t limit = UINT32_MAX;
            for (int b = 0; b < 4; b++)
                if ((r.multipass.xfbBufferMask & (1u << b)) && r.multipass.xfbStride[b])
                    limit = std::min<uint32_t>(limit, pa.xfbCapacity / r.multipass.xfbStride[b] / outVpp);
            MGLDrawInfo* ip = (MGLDrawInfo*)infoBuf.contents;
            for (int k = 0; k < 4; k++) ip->xfbPrimLimit[k] = limit;
            uint64_t generated = 0;
            if (!gs) {
                generated = (uint64_t)prims * pa.instances;
                if (!computePass(byName("mgl_xfb"), MTLSizeMake(nverts * pa.instances, 1, 1), true)) return 5;
            } else {
                uint32_t items = prims * pa.instances * std::max(1, r.gsInvocations);
                id<MTLBuffer> cnt = [dev newBufferWithLength:std::max<size_t>(16, items * 16) options:MTLResourceStorageModeShared];
                id<MTLBuffer> off = [dev newBufferWithLength:std::max<size_t>(16, items * 16) options:MTLResourceStorageModeShared];
                resources = [resources arrayByAddingObjectsFromArray:@[ cnt, off ]];
                ip->itemCount = items;
                ip->itemCounts = cnt.gpuAddress;
                ip->itemOffsets = off.gpuAddress;
                if (!computePass(byName("mgl_gs_count"), MTLSizeMake(items, 1, 1), true)) return 5;
                [cb commit];
                [cb waitUntilCompleted];
                cb = [q commandBuffer];
                const uint32_t* c = (const uint32_t*)cnt.contents;
                uint32_t* o = (uint32_t*)off.contents;
                for (uint32_t i = 0; i < items; i++) {
                    o[i * 4] = (uint32_t)generated;
                    generated += c[i * 4];
                }
                if (!computePass(byName("mgl_gs_write"), MTLSizeMake(items, 1, 1), true)) return 5;
            }
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.error) { fprintf(stderr, "GPU error: %s\n", cb.error.localizedDescription.UTF8String); return 6; }
            uint32_t written = (uint32_t)std::min<uint64_t>(generated, limit);
            printf("xfb: written=%u generated=%u\n", written, (uint32_t)generated);
            for (int b = 0; b < 4; b++) {
                if (!(r.multipass.xfbBufferMask & (1u << b))) continue;
                const float* f = (const float*)xfbBuf[b].contents;
                uint32_t n = r.multipass.xfbStride[b] / 4 * written * outVpp;
                printf("buffer %d (stride %u):", b, r.multipass.xfbStride[b]);
                for (uint32_t i = 0; i < n && i < 64; i++) printf(" %g", f[i]);
                printf("\n");
            }
            if (pa.expectPrims >= 0 && (int)written != pa.expectPrims) return 7;
            return 0;
        }
        // pass 1 as a kernel, unless the variant runs VS + TCS in the object stage
        if (tess && entry(EntryPoint::Kernel) && !computePass(entry(EntryPoint::Kernel), MTLSizeMake(prims, pa.instances, 1), false)) return 5;
        MTLMeshRenderPipelineDescriptor* md = [MTLMeshRenderPipelineDescriptor new];
        const EntryPoint* ms = entry(EntryPoint::Mesh);
        const EntryPoint* os = entry(EntryPoint::Object);
        md.meshFunction = [lib newFunctionWithName:@(ms->name.c_str())];
        if (os) md.objectFunction = [lib newFunctionWithName:@(os->name.c_str())];
        md.fragmentFunction = [lib newFunctionWithName:@"mgl_fs"];
        md.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
        if (os) md.payloadMemoryLength = os->payloadBytes;
        if (os) md.maxTotalThreadsPerObjectThreadgroup = os->threadsPerThreadgroup[0];
        id<MTLRenderPipelineState> pso = [dev newRenderPipelineStateWithMeshDescriptor:md options:0 reflection:nil error:&err];
        if (!check(pso, err, "mesh pipeline")) return 5;
        MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:W height:H mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget;
        td.storageMode = MTLStorageModeShared;
        id<MTLTexture> rt = [dev newTextureWithDescriptor:td];
        MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = rt;
        rp.colorAttachments[0].loadAction = MTLLoadActionClear;
        rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
        [e setRenderPipelineState:pso];
        for (int st = 0; st < 3; st++) {
            auto setB = [&](id<MTLBuffer> b, int idx) {
                if (st == 0) [e setObjectBuffer:b offset:0 atIndex:idx];
                else if (st == 1) [e setMeshBuffer:b offset:0 atIndex:idx];
                else [e setFragmentBuffer:b offset:0 atIndex:idx];
            };
            setB(args, 0);
            setB(sysBuf, 1);
            setB(infoBuf, 22);
        }
        for (id<MTLBuffer> b in resources)
            [e useResource:b usage:MTLResourceUsageRead | MTLResourceUsageWrite stages:MTLRenderStageObject | MTLRenderStageMesh | MTLRenderStageFragment];
        MTLSize meshTG = MTLSizeMake(ms->threadsPerThreadgroup[0], ms->threadsPerThreadgroup[1], ms->threadsPerThreadgroup[2]);
        if (os)
            [e drawMeshThreadgroups:MTLSizeMake(prims, pa.instances, 1) threadsPerObjectThreadgroup:MTLSizeMake(os->threadsPerThreadgroup[0], 1, 1)
                threadsPerMeshThreadgroup:meshTG];
        else
            [e drawMeshThreadgroups:MTLSizeMake(prims, pa.instances, std::max(1, r.gsInvocations)) threadsPerObjectThreadgroup:MTLSizeMake(1, 1, 1)
                threadsPerMeshThreadgroup:meshTG];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) { fprintf(stderr, "GPU error: %s\n", cb.error.localizedDescription.UTF8String); return 6; }
        std::vector<float> px(W * H * 4);
        [rt getBytes:px.data() bytesPerRow:W * 16 fromRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0];
        int pass = 0, fail = 0, empty = 0, minRow = H, maxRow = -1;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                const float* p = &px[(y * W + x) * 4];
                if (p[0] == 0 && p[1] == 0 && p[2] == 0) { empty++; continue; }
                minRow = std::min(minRow, y);
                maxRow = std::max(maxRow, y);
                if (p[0] == 0 && p[1] == 1) pass++;
                else {
                    if (fail < 4) printf("  pixel (%d,%d) = %g %g %g %g\n", x, y, p[0], p[1], p[2], p[3]);
                    fail++;
                }
            }
        printf("draw: pass=%d fail=%d uncovered=%d rows=%d..%d\n", pass, fail, empty, minRow, maxRow);
        return fail ? 7 : 0;
    }
}

// ---------------------------------------------------------------------------
// Tessellation primitive generator self-check
// ---------------------------------------------------------------------------
struct TessCase {
    uint32_t domain, spacing, points, ccw;
    float outer[4], inner[2];
};

static const char* kTessKernel = R"(
struct mgl_TCase { uint domain, spacing, points, ccw; float outer[4]; float inner[2]; float pad[2]; };
kernel void tesscheck(device const mgl_TCase* cases [[buffer(0)]], device float* out [[buffer(1)]], device uint* counts [[buffer(2)]],
                      device uint* mismatches [[buffer(3)]], uint c [[threadgroup_position_in_grid]]) {
    threadgroup float3 co[130];
    threadgroup float tp[130];
    threadgroup ushort idx[400];
    mgl_TCase k = cases[c];
    mgl_tess T = mgl_tess_setup(k.domain, k.spacing, k.points, k.ccw, float4(k.outer[0], k.outer[1], k.outer[2], k.outer[3]), float2(k.inner[0], k.inner[1]));
    uint ipp = k.points != 0u ? 1u : k.domain == 2u ? 2u : 3u;
    uint base = c * 262144u, n = 0u, bad = 0u;
    uint sides = k.domain == 0u ? 3u : 4u;
    for (uint u = 0; u < T.units; u++) {
        uint nv, np;
        mgl_tess_unit(T, u, co, tp, idx, nv, np);
        // the per-element functions must reproduce it exactly
        uint2 sz = mgl_tess_unit_size(T, u);
        if (sz.x != nv || sz.y != np) bad++;
        bool strip = k.domain != 2u && T.special == 0u && k.points == 0u && u < sides * T.rings;
        for (uint i = 0; i < nv; i++) {
            float t;
            float3 v = mgl_tess_unit_vertex(T, u, i, t);
            if (any(as_type<uint3>(v) != as_type<uint3>(co[i])) || (strip && as_type<uint>(t) != as_type<uint>(tp[i]))) bad++;
        }
        for (uint p = 0; p < np; p++) {
            uint3 t = mgl_tess_unit_prim(T, u, p, co, tp);
            for (uint i = 0; i < ipp; i++)
                if (t[i] != uint(idx[p * ipp + i])) bad++;
        }
        for (uint p = 0; p < np; p++)
            for (uint i = 0; i < ipp; i++) {
                float3 v = co[idx[p * ipp + i]];
                if (n < 87000u) { out[base + 3u * n] = v.x; out[base + 3u * n + 1u] = v.y; out[base + 3u * n + 2u] = v.z; }
                n++;
            }
    }
    counts[c] = n;
    mismatches[c] = bad;
}
)";

int runTessCheck(bool verbose) {
    std::vector<TessCase> cases;
    // triangles / quads with every spacing and a range of levels
    const float lv[] = {1.0f, 1.5f, 2.0f, 2.3f, 3.0f, 4.0f, 5.0f, 6.7f, 7.0f, 8.0f, 15.2f, 16.0f, 33.0f, 64.0f};
    for (uint32_t d = 0; d < 2; d++)
        for (uint32_t sp = 0; sp < 3; sp++)
            for (float a : lv)
                for (float b : {1.0f, 3.0f, 4.5f, 64.0f}) {
                    TessCase t{d, sp, 0, 1, {a, b, a, b}, {b, a}};
                    cases.push_back(t);
                }
    for (uint32_t sp = 0; sp < 3; sp++)
        for (float a : {1.0f, 3.0f, 7.2f})
            for (float b : {1.0f, 2.0f, 5.5f, 64.0f}) cases.push_back({2, sp, 0, 1, {a, b, 0, 0}, {0, 0}});
    cases.push_back({0, 0, 0, 1, {0.0f, 4, 4, 4}, {4, 4}});          // culled
    cases.push_back({1, 0, 0, 1, {NAN, 4, 4, 4}, {4, 4}});           // culled
    cases.push_back({0, 0, 0, 0, {3, 3, 3, 3}, {3, 3}});             // cw
    for (uint32_t d = 0; d < 2; d++)
        for (float a : {1.0f, 2.0f, 3.0f, 6.0f}) cases.push_back({d, 0, 1, 1, {a, a, a, a}, {a, a}});  // point mode
    // different level on every edge (uneven strips), every spacing and winding
    uint32_t rng = 7;
    auto rlevel = [&]() {
        rng = rng * 1664525u + 1013904223u;
        return 1.0f + (float)(rng >> 8) / (float)(1u << 24) * ((rng & 3u) == 0u ? 40.0f : 9.0f);
    };
    for (int i = 0; i < 400; i++) {
        uint32_t d = (uint32_t)i % 2u, sp = (uint32_t)(i / 2) % 3u, pts = i % 23 == 0 ? 1u : 0u;
        cases.push_back({d, sp, pts, (uint32_t)(i / 6) % 2u, {rlevel(), rlevel(), rlevel(), rlevel()}, {rlevel(), rlevel()}});
    }
    std::string body = kTessKernel;
    std::string src = buildMSLPrelude(body + "\n// mgl_tess_setup mgl_tess_unit mgl_tess_unit_size mgl_tess_unit_vertex mgl_tess_unit_prim\n") + body;
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError* err = nil;
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.languageVersion = MTLLanguageVersion4_0;
        opts.mathMode = getenv("METALGLDRV_FAST") ? MTLMathModeFast : MTLMathModeSafe;
        id<MTLLibrary> lib = [dev newLibraryWithSource:@(src.c_str()) options:opts error:&err];
        if (!check(lib, err, "compile")) return 5;
        id<MTLComputePipelineState> cps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"tesscheck"] error:&err];
        if (!check(cps, err, "pipeline")) return 5;
        struct GPUCase { uint32_t domain, spacing, points, ccw; float outer[4]; float inner[2]; float pad[2]; };
        std::vector<GPUCase> gc;
        for (auto& c : cases) {
            GPUCase g{c.domain, c.spacing, c.points, c.ccw, {c.outer[0], c.outer[1], c.outer[2], c.outer[3]}, {c.inner[0], c.inner[1]}, {0, 0}};
            gc.push_back(g);
        }
        id<MTLBuffer> cb0 = [dev newBufferWithBytes:gc.data() length:gc.size() * sizeof(GPUCase) options:MTLResourceStorageModeShared];
        id<MTLBuffer> outB = [dev newBufferWithLength:cases.size() * 262144 * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> cnt = [dev newBufferWithLength:cases.size() * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> mis = [dev newBufferWithLength:cases.size() * 4 options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLCommandBuffer> cmd = [q commandBuffer];
        id<MTLComputeCommandEncoder> ce = [cmd computeCommandEncoder];
        [ce setComputePipelineState:cps];
        [ce setBuffer:cb0 offset:0 atIndex:0];
        [ce setBuffer:outB offset:0 atIndex:1];
        [ce setBuffer:cnt offset:0 atIndex:2];
        [ce setBuffer:mis offset:0 atIndex:3];
        [ce dispatchThreadgroups:MTLSizeMake(cases.size(), 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [ce endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        const float* out = (const float*)outB.contents;
        const uint32_t* counts = (const uint32_t*)cnt.contents;
        int failures = 0;
        for (size_t ci = 0; ci < cases.size(); ci++) {
            const TessCase& c = cases[ci];
            uint32_t n = counts[ci];
            const float* v = out + ci * 262144;
            std::string label = std::string(c.domain == 0 ? "tri" : c.domain == 1 ? "quad" : "iso") + " sp" + std::to_string(c.spacing) +
                                (c.points ? " points" : "") + " outer=" + std::to_string(c.outer[0]) + "," + std::to_string(c.outer[1]) +
                                " inner=" + std::to_string(c.inner[0]);
            if (uint32_t m = ((const uint32_t*)mis.contents)[ci]) {
                printf("FAIL %s: per-element generation differs in %u values\n", label.c_str(), m);
                failures++;
            }
            bool culled = !(c.outer[0] > 0) || !(c.outer[1] > 0);
            if (culled) {
                if (n != 0) { printf("FAIL %s: culled patch produced %u vertices\n", label.c_str(), n); failures++; }
                continue;
            }
            if (n >= 87000) { printf("FAIL %s: output overflow\n", label.c_str()); failures++; continue; }
            if (c.points) {
                // every vertex distinct
                std::set<std::tuple<float, float, float>> s;
                for (uint32_t i = 0; i < n; i++) s.insert({v[3 * i], v[3 * i + 1], v[3 * i + 2]});
                if (s.size() != n) { printf("FAIL %s: %u points but %zu distinct\n", label.c_str(), n, s.size()); failures++; }
                continue;
            }
            if (c.domain == 2) {
                // isolines: lines * segments
                int lines = (int)std::ceil(std::min(64.0f, std::max(1.0f, c.outer[0])));
                if (n % 2 || (int)(n / 2) % lines) { printf("FAIL %s: %u line vertices\n", label.c_str(), n); failures++; }
                continue;
            }
            // triangles: area, orientation, watertightness
            double area = 0;
            int badWinding = 0;
            std::map<std::pair<std::tuple<float, float>, std::tuple<float, float>>, int> edges;
            for (uint32_t t = 0; t + 2 < n; t += 3) {
                float ax = v[3 * t], ay = v[3 * t + 1], bx = v[3 * t + 3], by = v[3 * t + 4], cx = v[3 * t + 6], cy = v[3 * t + 7];
                double a2 = (double)(bx - ax) * (cy - ay) - (double)(cx - ax) * (by - ay);
                if ((c.ccw && a2 < -1e-12) || (!c.ccw && a2 > 1e-12)) badWinding++;
                area += std::fabs(a2) / 2;
                // zero-area triangles add no surface: fractional_odd treats an inner level of 1 as 1 + epsilon,
                // whose inner ring lies on the outer edges (see mgl_tess_setup)
                if (a2 == 0) continue;
                std::tuple<float, float> P[3] = {{ax, ay}, {bx, by}, {cx, cy}};
                for (int k = 0; k < 3; k++) {
                    auto p0 = P[k], p1 = P[(k + 1) % 3];
                    if (p0 == p1) continue;
                    if (p1 < p0) std::swap(p0, p1);
                    edges[{p0, p1}]++;
                }
            }
            double expectArea = c.domain == 0 ? 0.5 : 1.0;
            int open = 0;
            for (auto& [e, cnt2] : edges) {
                if (cnt2 == 2) continue;
                // boundary edge: both endpoints on the domain boundary on the same side
                auto [x0, y0] = e.first;
                auto [x1, y1] = e.second;
                bool boundary;
                if (c.domain == 0) {
                    float w0 = 1 - x0 - y0, w1 = 1 - x1 - y1;
                    boundary = (x0 == 0 && x1 == 0) || (y0 == 0 && y1 == 0) || (std::fabs(w0) < 1e-6 && std::fabs(w1) < 1e-6);
                } else {
                    boundary = (x0 == 0 && x1 == 0) || (y0 == 0 && y1 == 0) || (x0 == 1 && x1 == 1) || (y0 == 1 && y1 == 1);
                }
                if (!(boundary && cnt2 == 1)) {
                    open++;
                    if (verbose && open < 12) printf("    edge (%.9g,%.9g)-(%.9g,%.9g) x%d\n", x0, y0, x1, y1, cnt2);
                }
            }
            bool ok = std::fabs(area - expectArea) < 1e-4 && badWinding == 0 && open == 0;
            if (!ok || verbose)
                printf("%s %s: %u tris, area %.6f, bad winding %d, non-manifold edges %d\n", ok ? "ok  " : "FAIL", label.c_str(), n / 3, area,
                       badWinding, open);
            if (!ok) failures++;
        }
        printf("tesscheck: %zu cases, %d failed\n", cases.size(), failures);
        return failures ? 7 : 0;
    }
}
