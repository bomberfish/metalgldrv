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
    uint32_t xfbCapacity = 4096;
    bool tessCompute = false;  // --tess-compute: tessellation pass 1 as the mgl_tcs kernel (VariantKey::tessComputePass)
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

// Test textures with known contents for Texture/Sampler/Image slots (compute mode).
//   float 2D:     RGBA32Float 4x4 with 3 mips, texel(level L, x, y) = (x, y, L, 100L + x + 4y)
//   int/uint 2D:  RGBA32Sint/Uint 4x4 (1 mip), texel = (x, y, 0, x + 4y)
//   2D array:     RGBA32Float 4x4x2, texel = (x, y, layer, x + 4y + 16 layer)
//   3D:           RGBA32Float 4x4x2, texel = (x, y, z, x + 4y + 16z)
//   2D shadow:    Depth32Float 4x4, depth = (x + 4y) / 16, compare LEQUAL
//   buffer:       RGBA32Float 16 texels, texel i = (i, 2i, 3i, 4i)
//   image2D:      RGBA32Float 4x4 = (x, y, 0, 1); uimage2D/iimage2D: R32Uint/R32Sint 4x4 = x + 4y
// Samplers: nearest, clamp to edge, nearest mip.
static uint64_t testTexture(id<MTLDevice> dev, uint32_t glType, bool sampler, std::vector<id>& keep) {
    auto fill = [&](id<MTLTexture> t, int w, int h, int d, int level, int comps, auto fn) {
        for (int z = 0; z < d; z++) {
            std::vector<uint32_t> px(w * h * comps);
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    for (int c = 0; c < comps; c++) px[(y * w + x) * comps + c] = fn(x, y, z, c);
            if (t.textureType == MTLTextureType2DArray)
                [t replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:level slice:z withBytes:px.data() bytesPerRow:w * comps * 4 bytesPerImage:0];
            else
                [t replaceRegion:MTLRegionMake3D(0, 0, z, w, h, 1) mipmapLevel:level slice:0 withBytes:px.data() bytesPerRow:w * comps * 4 bytesPerImage:w * h * comps * 4];
        }
    };
    auto F = [](float f) { uint32_t u; memcpy(&u, &f, 4); return u; };
    MTLTextureDescriptor* td = [MTLTextureDescriptor new];
    td.storageMode = MTLStorageModeShared;
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    td.width = 4;
    td.height = 4;
    id<MTLTexture> t = nil;
    switch (glType) {
        case 0x8B5E: case 0x8B63: {  // sampler2D, sampler2DRect
            td.pixelFormat = MTLPixelFormatRGBA32Float;
            td.mipmapLevelCount = glType == 0x8B5E ? 3 : 1;
            t = [dev newTextureWithDescriptor:td];
            for (int L = 0; L < (int)td.mipmapLevelCount; L++)
                fill(t, 4 >> L, 4 >> L, 1, L, 4, [&](int x, int y, int, int c) { float v[4] = {(float)x, (float)y, (float)L, (float)(100 * L + x + 4 * y)}; return F(v[c]); });
            break;
        }
        case 0x8DCA: case 0x8DD2: {  // isampler2D, usampler2D
            td.pixelFormat = glType == 0x8DCA ? MTLPixelFormatRGBA32Sint : MTLPixelFormatRGBA32Uint;
            t = [dev newTextureWithDescriptor:td];
            fill(t, 4, 4, 1, 0, 4, [&](int x, int y, int, int c) { uint32_t v[4] = {(uint32_t)x, (uint32_t)y, 0, (uint32_t)(x + 4 * y)}; return v[c]; });
            break;
        }
        case 0x8DC1: {  // sampler2DArray
            td.textureType = MTLTextureType2DArray;
            td.arrayLength = 2;
            td.pixelFormat = MTLPixelFormatRGBA32Float;
            t = [dev newTextureWithDescriptor:td];
            fill(t, 4, 4, 2, 0, 4, [&](int x, int y, int z, int c) { float v[4] = {(float)x, (float)y, (float)z, (float)(x + 4 * y + 16 * z)}; return F(v[c]); });
            break;
        }
        case 0x8B5F: {  // sampler3D
            td.textureType = MTLTextureType3D;
            td.depth = 2;
            td.pixelFormat = MTLPixelFormatRGBA32Float;
            t = [dev newTextureWithDescriptor:td];
            fill(t, 4, 4, 2, 0, 4, [&](int x, int y, int z, int c) { float v[4] = {(float)x, (float)y, (float)z, (float)(x + 4 * y + 16 * z)}; return F(v[c]); });
            break;
        }
        case 0x8B62: {  // sampler2DShadow
            td.pixelFormat = MTLPixelFormatDepth32Float;
            td.usage = MTLTextureUsageShaderRead;
            t = [dev newTextureWithDescriptor:td];
            fill(t, 4, 4, 1, 0, 1, [&](int x, int y, int, int) { return F((x + 4 * y) / 16.0f); });
            break;
        }
        case 0x8DC2: {  // samplerBuffer
            td.textureType = MTLTextureTypeTextureBuffer;
            td.pixelFormat = MTLPixelFormatRGBA32Float;
            td.width = 16;
            td.height = 1;
            id<MTLBuffer> b = [dev newBufferWithLength:16 * 16 options:MTLResourceStorageModeShared];
            float* f = (float*)b.contents;
            for (int i = 0; i < 16; i++) { f[4 * i] = i; f[4 * i + 1] = 2 * i; f[4 * i + 2] = 3 * i; f[4 * i + 3] = 4 * i; }
            t = [b newTextureWithDescriptor:td offset:0 bytesPerRow:256];
            keep.push_back(b);
            break;
        }
        case 0x904D: {  // image2D
            td.pixelFormat = MTLPixelFormatRGBA32Float;
            t = [dev newTextureWithDescriptor:td];
            fill(t, 4, 4, 1, 0, 4, [&](int x, int y, int, int c) { float v[4] = {(float)x, (float)y, 0, 1}; return F(v[c]); });
            break;
        }
        case 0x9063: case 0x9058: {  // uimage2D, iimage2D
            td.pixelFormat = glType == 0x9063 ? MTLPixelFormatR32Uint : MTLPixelFormatR32Sint;
            td.usage |= MTLTextureUsageShaderAtomic;
            t = [dev newTextureWithDescriptor:td];
            fill(t, 4, 4, 1, 0, 1, [&](int x, int y, int, int) { return (uint32_t)(x + 4 * y); });
            break;
        }
        default:
            fprintf(stderr, "glslrun: no test texture for GL type 0x%x\n", glType);
            return 0;
    }
    keep.push_back(t);
    if (!sampler) return t.gpuResourceID._impl;
    MTLSamplerDescriptor* sd = [MTLSamplerDescriptor new];
    sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterNearest;
    sd.mipFilter = MTLSamplerMipFilterNearest;
    sd.sAddressMode = sd.tAddressMode = sd.rAddressMode = MTLSamplerAddressModeClampToEdge;
    sd.supportArgumentBuffers = YES;
    if (glType == 0x8B62) sd.compareFunction = MTLCompareFunctionLessEqual;
    id<MTLSamplerState> ss = [dev newSamplerStateWithDescriptor:sd];
    keep.push_back(ss);
    return ss.gpuResourceID._impl;
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
            else if (a == "--xfb-capacity") pa.xfbCapacity = (uint32_t)std::stoi(next());
            else if (a == "--dump") pa.dump = true;
            else if (a == "--tess-compute") pa.tessCompute = true;
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
        std::vector<id<MTLBuffer>> keep;
        std::vector<id> keepTex;
        for (size_t i = 0; i < r.slots.size(); i++) {
            const Slot& s = r.slots[i];
            switch (s.kind) {
                case SlotKind::Texture: slots[i] = testTexture(dev, r.uniforms[s.index].type, false, keepTex); break;
                case SlotKind::Sampler: slots[i] = testTexture(dev, r.uniforms[s.index].type, true, keepTex); break;
                case SlotKind::Image: slots[i] = testTexture(dev, r.uniforms[s.index].type, false, keepTex); break;
                case SlotKind::DefaultUniforms: slots[i] = defaults.gpuAddress; break;
                case SlotKind::Ubo: slots[i] = ubo.gpuAddress; break;
                case SlotKind::Ssbo: slots[i] = ssbo[std::min(7, r.storageBlocks[s.index].binding)].gpuAddress; break;
                case SlotKind::BufferSize: slots[i] = kSize; break;
                case SlotKind::LodBias: slots[i] = 0; break;
                case SlotKind::AtomicCounter: slots[i] = acb.gpuAddress; break;
                case SlotKind::SubroutineIndices: {
                    const StageSubroutines& ss = r.subroutines[s.index];
                    id<MTLBuffer> sb = [dev newBufferWithLength:std::max(4, ss.numLocations * 4) options:MTLResourceStorageModeShared];
                    uint32_t* sel = (uint32_t*)sb.contents;
                    for (auto& u : ss.uniforms)
                        for (int k = 0; k < u.arraySize; k++) sel[u.location + k] = u.compatible.empty() ? 0 : (uint32_t)u.compatible[0];
                    keep.push_back(sb);
                    slots[i] = sb.gpuAddress;
                    break;
                }
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
        for (auto b : keep) [enc useResource:b usage:MTLResourceUsageRead];
        for (id x : keepTex)
            if ([x conformsToProtocol:@protocol(MTLResource)]) [enc useResource:(id<MTLResource>)x usage:MTLResourceUsageRead | MTLResourceUsageWrite];
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
