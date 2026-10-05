// glslrun: executes self-checking GLSL compute tests on the host GPU.
//
// Convention for test shaders:
//   layout(std430, binding = 0) buffer Results { uint res[]; };   // res[0] = number of checks, res[1+i] = 1 on pass
//   layout(std430, binding = 1) buffer Input { uint seq[]; };     // seq[i] == i (defeats constant folding)
//   bindings 2..7: zero-initialised 64 KiB scratch buffers; all blocks with the same binding alias one buffer
//   uniform block binding 0: 4 KiB with float values u[i] = i * 0.25
// One workgroup of the declared local size is dispatched (override with --groups N).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <fstream>
#include <sstream>

#include "glsl/glsl.h"

using namespace glsl;

static std::string readFile(const char* p) {
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// --draw mode: renders a triangle with a vertex+fragment program into a 64x64
// RGBA32F target cleared to black, using the runtime conventions (y flip via
// MGLSys, attribute location L at buffer index 2+L, float4 per vertex).
// attribute 0 = position (x, y, 0, 1) of the triangle given by --tri.
// The fragment shader reports per pixel: green (0,1,0) pass, red (1,0,0) fail.
// Prints pass/fail/uncovered counts and the range of memory rows covered.
static int runDraw(const char* vsPath, const char* fsPath, const float tri[6], uint32_t flags, bool dump) {
    LinkInput li;
    for (auto [st, path] : {std::pair{Stage::Vertex, vsPath}, std::pair{Stage::Fragment, fsPath}}) {
        auto cr = compile(st, {readFile(path)});
        if (!cr.shader) { fprintf(stderr, "%s: %s", path, cr.infoLog.c_str()); return 1; }
        li.shaders.push_back(cr.shader);
    }
    auto lr = link(li);
    if (!lr.program) { fprintf(stderr, "%s", lr.infoLog.c_str()); return 3; }
    VariantKey key;
    auto mo = generateMSL(*lr.program, key);
    if (!mo.error.empty()) { fprintf(stderr, "%s\n", mo.error.c_str()); return 4; }
    if (dump) printf("%s\n", mo.source.c_str());
    const int W = 64, H = 64;
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.languageVersion = MTLLanguageVersion4_0;
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:@(mo.source.c_str()) options:opts error:&err];
        if (!lib) { fprintf(stderr, "%s\n", err.localizedDescription.UTF8String); return 5; }
        MTLRenderPipelineDescriptor* pd = [MTLRenderPipelineDescriptor new];
        pd.vertexFunction = [lib newFunctionWithName:@"mgl_vs"];
        pd.fragmentFunction = [lib newFunctionWithName:@"mgl_fs"];
        pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
        MTLVertexDescriptor* vd = [MTLVertexDescriptor vertexDescriptor];
        const Reflection& r = lr.reflection;
        float verts[3][4];
        for (int i = 0; i < 3; i++) { verts[i][0] = tri[2 * i]; verts[i][1] = tri[2 * i + 1]; verts[i][2] = 0; verts[i][3] = 1; }
        id<MTLBuffer> vb = [dev newBufferWithBytes:verts length:sizeof verts options:MTLResourceStorageModeShared];
        for (auto& a : r.attributes) {
            vd.attributes[a.location].format = MTLVertexFormatFloat4;
            vd.attributes[a.location].bufferIndex = 2 + a.location;
            vd.attributes[a.location].offset = 0;
            vd.layouts[2 + a.location].stride = 16;
        }
        pd.vertexDescriptor = vd;
        id<MTLRenderPipelineState> pso = [dev newRenderPipelineStateWithDescriptor:pd error:&err];
        if (!pso) { fprintf(stderr, "%s\n", err.localizedDescription.UTF8String); return 5; }
        MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:W height:H mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget;
        td.storageMode = MTLStorageModeShared;
        id<MTLTexture> rt = [dev newTextureWithDescriptor:td];
        id<MTLBuffer> args = [dev newBufferWithLength:std::max<size_t>(8, r.slots.size() * 8) options:MTLResourceStorageModeShared];
        id<MTLBuffer> defaults = [dev newBufferWithLength:std::max<uint32_t>(16, r.defaultUniformSize) options:MTLResourceStorageModeShared];
        memset(defaults.contents, 0, defaults.length);
        uint64_t* slots = (uint64_t*)args.contents;
        for (size_t i = 0; i < r.slots.size(); i++) slots[i] = r.slots[i].kind == SlotKind::DefaultUniforms ? defaults.gpuAddress : 0;
        MGLSys sys{};
        sys.depthRangeNear = 0;
        sys.depthRangeFar = 1;
        sys.pointSize = 1;
        sys.flags = flags;
        sys.alphaFunc = 0x0207;
        sys.renderTargetHeight = H;
        sys.viewport[2] = W;
        sys.viewport[3] = H;
        MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = rt;
        rp.colorAttachments[0].loadAction = MTLLoadActionClear;
        rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
        [enc setRenderPipelineState:pso];
        // Metal default front face is clockwise; GL CCW triangles become CW after the y flip.
        [enc setVertexBuffer:args offset:0 atIndex:0];
        [enc setFragmentBuffer:args offset:0 atIndex:0];
        [enc setVertexBytes:&sys length:sizeof sys atIndex:1];
        [enc setFragmentBytes:&sys length:sizeof sys atIndex:1];
        for (auto& a : r.attributes) [enc setVertexBuffer:vb offset:0 atIndex:2 + a.location];
        [enc useResource:defaults usage:MTLResourceUsageRead stages:MTLRenderStageVertex | MTLRenderStageFragment];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
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
};
int runPipeline(const PipelineArgs& pa);
int runTessCheck(bool verbose);

static std::vector<float> parseFloats(const std::string& s) {
    std::vector<float> v;
    size_t p = 0;
    while (p < s.size()) {
        size_t q = s.find(',', p);
        if (q == std::string::npos) q = s.size();
        v.push_back(std::stof(s.substr(p, q - p)));
        p = q + 1;
    }
    return v;
}

int main(int argc, char** argv) {
    if (argc >= 2 && !strcmp(argv[1], "--tesscheck")) return runTessCheck(argc > 2 && !strcmp(argv[2], "-v"));
    if (argc >= 2 && !strcmp(argv[1], "--pipeline")) {
        PipelineArgs pa;
        pa.verts = {-1, -1, 3, -1, -1, 3};
        for (int i = 2; i < argc; i++) {
            std::string a = argv[i];
            auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
            if (a == "--verts") pa.verts = parseFloats(next());
            else if (a == "--vpp") pa.vpp = std::stoi(next());
            else if (a == "--inst") pa.instances = std::stoi(next());
            else if (a == "--xfb") {
                std::string v = next();
                size_t p = 0;
                while (p <= v.size()) {
                    size_t q = v.find(',', p);
                    if (q == std::string::npos) q = v.size();
                    if (q > p) pa.xfb.push_back(v.substr(p, q - p));
                    p = q + 1;
                }
            } else if (a == "--xfb-separate") pa.xfbSeparate = true;
            else if (a == "--outer") { auto f = parseFloats(next()); for (size_t k = 0; k < 4 && k < f.size(); k++) pa.defaultOuter[k] = f[k]; }
            else if (a == "--inner") { auto f = parseFloats(next()); for (size_t k = 0; k < 2 && k < f.size(); k++) pa.defaultInner[k] = f[k]; }
            else if (a == "--expect-prims") pa.expectPrims = std::stoi(next());
            else if (a == "--dump") pa.dump = true;
            else {
                std::string ext = a.substr(a.rfind('.') + 1);
                Stage st = ext == "vert" ? Stage::Vertex : ext == "tesc" ? Stage::TessControl : ext == "tese" ? Stage::TessEval
                         : ext == "geom" ? Stage::Geometry : Stage::Fragment;
                pa.files.push_back({st, a});
            }
        }
        return runPipeline(pa);
    }
    if (argc < 2) {
        fprintf(stderr, "usage: glslrun test.comp [--groups N] [--dump]\n"
                        "       glslrun --draw a.vert a.frag [--tri x0,y0,x1,y1,x2,y2] [--flags N] [--dump]\n");
        return 2;
    }
    if (!strcmp(argv[1], "--draw") && argc >= 4) {
        float tri[6] = {-1, -1, 3, -1, -1, 3};
        uint32_t flags = 0;
        bool dump = false;
        for (int i = 4; i < argc; i++) {
            if (!strcmp(argv[i], "--tri") && i + 1 < argc)
                sscanf(argv[++i], "%f,%f,%f,%f,%f,%f", &tri[0], &tri[1], &tri[2], &tri[3], &tri[4], &tri[5]);
            else if (!strcmp(argv[i], "--flags") && i + 1 < argc) flags = (uint32_t)atoi(argv[++i]);
            else if (!strcmp(argv[i], "--dump")) dump = true;
        }
        return runDraw(argv[2], argv[3], tri, flags, dump);
    }
    int groups = 1;
    bool dump = false;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--groups") && i + 1 < argc) groups = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump")) dump = true;
    }
    std::ifstream f(argv[1]);
    std::stringstream ss;
    ss << f.rdbuf();
    auto cr = compile(Stage::Compute, {ss.str()});
    if (!cr.shader) { fprintf(stderr, "%s", cr.infoLog.c_str()); return 1; }
    LinkInput li;
    li.shaders.push_back(cr.shader);
    auto lr = link(li);
    if (!lr.program) { fprintf(stderr, "%s", lr.infoLog.c_str()); return 3; }
    VariantKey key;
    key.kind = PipelineKind::Compute;
    auto mo = generateMSL(*lr.program, key);
    if (!mo.error.empty()) { fprintf(stderr, "%s\n", mo.error.c_str()); return 4; }
    if (dump) printf("%s\n", mo.source.c_str());
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.languageVersion = MTLLanguageVersion4_0;
        opts.mathMode = MTLMathModeSafe;
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:@(mo.source.c_str()) options:opts error:&err];
        if (!lib) { fprintf(stderr, "%s\n", err.localizedDescription.UTF8String); return 5; }
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mgl_cs"] error:&err];
        if (!pso) { fprintf(stderr, "%s\n", err.localizedDescription.UTF8String); return 5; }
        const Reflection& r = lr.reflection;
        const size_t kSize = 65536;
        id<MTLBuffer> ssbo[8];
        for (int i = 0; i < 8; i++) {
            ssbo[i] = [dev newBufferWithLength:kSize options:MTLResourceStorageModeShared];
            memset(ssbo[i].contents, 0, kSize);
        }
        uint32_t* seq = (uint32_t*)ssbo[1].contents;
        for (size_t i = 0; i < kSize / 4; i++) seq[i] = (uint32_t)i;
        id<MTLBuffer> ubo = [dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];
        float* uf = (float*)ubo.contents;
        for (int i = 0; i < 1024; i++) uf[i] = i * 0.25f;
        id<MTLBuffer> defaults = [dev newBufferWithLength:std::max<uint32_t>(16, r.defaultUniformSize) options:MTLResourceStorageModeShared];
        memset(defaults.contents, 0, defaults.length);
        id<MTLBuffer> acb = [dev newBufferWithLength:256 options:MTLResourceStorageModeShared];
        memset(acb.contents, 0, 256);
        id<MTLBuffer> args = [dev newBufferWithLength:std::max<size_t>(8, r.slots.size() * 8) options:MTLResourceStorageModeShared];
        uint64_t* slots = (uint64_t*)args.contents;
        for (size_t i = 0; i < r.slots.size(); i++) {
            const Slot& s = r.slots[i];
            switch (s.kind) {
                case SlotKind::DefaultUniforms: slots[i] = defaults.gpuAddress; break;
                case SlotKind::Ubo: slots[i] = ubo.gpuAddress; break;
                case SlotKind::Ssbo: slots[i] = ssbo[std::min(7, r.storageBlocks[s.index].binding)].gpuAddress; break;
                case SlotKind::BufferSize: slots[i] = kSize; break;
                case SlotKind::AtomicCounter: slots[i] = acb.gpuAddress; break;
                default: slots[i] = 0; break;
            }
        }
        id<MTLBuffer> sys = [dev newBufferWithLength:sizeof(MGLSys) options:MTLResourceStorageModeShared];
        memset(sys.contents, 0, sizeof(MGLSys));
        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:args offset:0 atIndex:0];
        [enc setBuffer:sys offset:0 atIndex:1];
        for (int i = 0; i < 8; i++) [enc useResource:ssbo[i] usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        [enc useResource:ubo usage:MTLResourceUsageRead];
        [enc useResource:defaults usage:MTLResourceUsageRead];
        [enc useResource:acb usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(r.localSize[0], r.localSize[1], r.localSize[2])];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) { fprintf(stderr, "GPU error: %s\n", cb.error.localizedDescription.UTF8String); return 6; }
        const uint32_t* res = (const uint32_t*)ssbo[0].contents;
        uint32_t n = res[0];
        int failed = 0;
        for (uint32_t i = 0; i < n; i++)
            if (res[1 + i] != 1) {
                printf("  check %u FAILED (value %u)\n", i, res[1 + i]);
                failed++;
            }
        printf("%s: %u checks, %d failed\n", argv[1], n, failed);
        return failed ? 7 : (n ? 0 : 8);
    }
}
