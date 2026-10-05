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

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: glslrun test.comp [--groups N] [--dump]\n");
        return 2;
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
