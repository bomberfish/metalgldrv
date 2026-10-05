// Validates src/glsl/runtime/fp64.metal on the GPU against host IEEE doubles.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const char* kKernels = R"MSL(
kernel void t_binop(device const ulong* a [[buffer(0)]], device const ulong* b [[buffer(1)]], device ulong* out [[buffer(2)]],
                    constant uint& op [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    ulong x = a[i], y = b[i], r = 0;
    switch (op) {
    case 0: r = mgl_f64_add(x, y); break;
    case 1: r = mgl_f64_sub(x, y); break;
    case 2: r = mgl_f64_mul(x, y); break;
    case 3: r = mgl_f64_div(x, y); break;
    case 4: r = mgl_f64_sqrt(x); break;
    case 5: r = mgl_f64_lt(x, y) | (mgl_f64_le(x, y) << 1) | (mgl_f64_eq(x, y) << 2); break;
    case 6: r = as_type<uint>(mgl_f64_to_float(x)); break;
    case 7: r = mgl_f64_from_float(as_type<float>(uint(x))); break;
    case 8: r = uint(mgl_f64_to_int(x)); break;
    case 9: r = mgl_f64_from_int(int(uint(x))); break;
    case 10: r = mgl_f64_floor(x); break;
    case 11: r = mgl_f64_ceil(x); break;
    case 12: r = mgl_f64_trunc(x); break;
    case 13: r = mgl_f64_roundEven(x); break;
    case 14: r = mgl_f64_fract(x); break;
    case 15: r = mgl_f64_fma(x, y, a[(i * 7 + 3) % 1048576]); break;
    }
    out[i] = r;
}
)MSL";

static uint64_t bits(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }
static double dbl(uint64_t u) { double d; memcpy(&d, &u, 8); return d; }

int main(int argc, char** argv) {
    @autoreleasepool {
        std::string path = argc > 1 ? argv[1] : "src/glsl/runtime/fp64.metal";
        NSString* lib = [NSString stringWithContentsOfFile:@(path.c_str()) encoding:NSUTF8StringEncoding error:nil];
        if (!lib) { fprintf(stderr, "cannot read %s\n", path.c_str()); return 2; }
        NSString* src = [NSString stringWithFormat:@"#include <metal_stdlib>\nusing namespace metal;\n%@\n%s", lib, kKernels];
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError* err = nil;
        MTLCompileOptions* co = [MTLCompileOptions new];
        co.mathMode = MTLMathModeSafe;
        id<MTLLibrary> L = [dev newLibraryWithSource:src options:co error:&err];
        if (!L) { fprintf(stderr, "compile failed: %s\n", err.localizedDescription.UTF8String); return 2; }
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[L newFunctionWithName:@"t_binop"] error:&err];
        id<MTLCommandQueue> q = [dev newCommandQueue];
        const int N = 1 << 20;
        std::mt19937_64 rng(1234);
        std::vector<uint64_t> A(N), B(N);
        const uint64_t specials[] = {0, 0x8000000000000000ull, bits(1.0), bits(-1.0), bits(INFINITY), bits(-INFINITY), bits(NAN),
                                     1, 0x000FFFFFFFFFFFFFull, 0x0010000000000000ull, 0x7FEFFFFFFFFFFFFFull, bits(0.5), bits(2.5),
                                     bits(-2.5), bits(3.5), bits(1e300), bits(1e-300), bits(0.1), bits(4294967296.0), bits(-2147483648.0)};
        const int NS = sizeof(specials) / sizeof(specials[0]);
        for (int i = 0; i < N; i++) {
            auto gen = [&]() -> uint64_t {
                int k = rng() % 8;
                if (k == 0) return specials[rng() % NS];
                if (k == 1) return bits((double)(int64_t)(rng() % 2000001) / 1000.0 - 1000.0);  // moderate decimals
                if (k == 2) {  // nearby exponents to stress cancellation
                    uint64_t m = rng() & 0x800FFFFFFFFFFFFFull;
                    return m | (uint64_t(1020 + rng() % 8) << 52);
                }
                return rng();  // arbitrary bit patterns (incl. subnormals/NaNs)
            };
            A[i] = gen();
            B[i] = (i & 3) == 0 ? (A[i] ^ (rng() & 0xFF)) : gen();  // near-equal pairs
        }
        id<MTLBuffer> ba = [dev newBufferWithBytes:A.data() length:N * 8 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bb = [dev newBufferWithBytes:B.data() length:N * 8 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bo = [dev newBufferWithLength:N * 8 options:MTLResourceStorageModeShared];
        const char* names[] = {"add", "sub", "mul", "div", "sqrt", "cmp", "to_float", "from_float", "to_int", "from_int",
                               "floor", "ceil", "trunc", "roundEven", "fract", "fma"};
        int totalFail = 0;
        for (uint32_t op = 0; op < 16; op++) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:pso];
            [e setBuffer:ba offset:0 atIndex:0];
            [e setBuffer:bb offset:0 atIndex:1];
            [e setBuffer:bo offset:0 atIndex:2];
            [e setBytes:&op length:4 atIndex:3];
            [e dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            const uint64_t* out = (const uint64_t*)bo.contents;
            int fails = 0;
            for (int i = 0; i < N; i++) {
                double x = dbl(A[i]), y = dbl(B[i]);
                uint64_t expect;
                bool nanOk = false;
                switch (op) {
                case 0: expect = bits(x + y); nanOk = true; break;
                case 1: expect = bits(x - y); nanOk = true; break;
                case 2: expect = bits(x * y); nanOk = true; break;
                case 3: expect = bits(x / y); nanOk = true; break;
                case 4: expect = bits(sqrt(x)); nanOk = true; break;
                case 5: expect = (uint64_t)(x < y) | ((uint64_t)(x <= y) << 1) | ((uint64_t)(x == y) << 2); break;
                case 6: {
                    float f = (float)x;
                    // flush double subnormals -> 0 behavior: our lib flushes |x| < 2^-1022
                    if (fabs(x) < 2.2250738585072014e-308 && x != 0) f = std::signbit(x) ? -0.0f : 0.0f;
                    uint32_t u; memcpy(&u, &f, 4); expect = u; nanOk = std::isnan(x);
                    break;
                }
                case 7: { float f; uint32_t u = (uint32_t)A[i]; memcpy(&f, &u, 4); expect = bits((double)f); nanOk = std::isnan(f); break; }
                case 8: {
                    if (std::isnan(x) || x >= 2147483648.0 || x < -2147483648.0) { expect = out[i]; break; }
                    expect = (uint32_t)(int32_t)x;
                    break;
                }
                case 9: expect = bits((double)(int32_t)(uint32_t)A[i]); break;
                case 10: expect = bits(floor(x)); nanOk = true; break;
                case 11: expect = bits(ceil(x)); nanOk = true; break;
                case 12: expect = bits(trunc(x)); nanOk = true; break;
                case 13: expect = bits(nearbyint(x)); nanOk = true; break;
                case 14: expect = bits(x - floor(x)); nanOk = true; break;
                case 15: expect = bits(fma(x, y, dbl(A[(i * 7 + 3) % 1048576]))); nanOk = true; break;
                default: expect = 0;
                }
                bool ok = out[i] == expect;
                if (!ok && nanOk) {
                    bool en = op == 6 ? false : std::isnan(dbl(expect));
                    bool gn = op == 6 ? false : std::isnan(dbl(out[i]));
                    if (op == 6) {
                        float fe, fg; uint32_t ue = (uint32_t)expect, ug = (uint32_t)out[i];
                        memcpy(&fe, &ue, 4); memcpy(&fg, &ug, 4);
                        ok = std::isnan(fe) && std::isnan(fg);
                    } else if (op == 7) {
                        ok = std::isnan(dbl(expect)) && std::isnan(dbl(out[i]));
                    } else {
                        ok = en && gn;
                    }
                }
                if (!ok) {
                    if (fails < 5)
                        fprintf(stderr, "  %s(%016llx [%.17g], %016llx [%.17g]) = %016llx expected %016llx\n", names[op], (unsigned long long)A[i], x,
                                (unsigned long long)B[i], y, (unsigned long long)out[i], (unsigned long long)expect);
                    fails++;
                }
            }
            printf("%-10s %s (%d/%d mismatches)\n", names[op], fails ? "FAIL" : "ok", fails, N);
            totalFail += fails;
        }
        return totalFail ? 1 : 0;
    }
}
