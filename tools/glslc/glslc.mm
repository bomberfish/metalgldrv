// glslc: host tool for the metalgldrv GLSL compiler.
//   glslc -S vert a.vert -S frag a.frag [--msl out.metal] [--no-validate] [-q]
// Compiles, links, prints reflection + MSL, and validates the MSL with the
// host Metal compiler (MTLLanguageVersion4_0).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "glsl/glsl.h"

using namespace glsl;

static bool stageFromName(const std::string& s, Stage& st) {
    if (s == "vert" || s == "vs" || s == "vertex") st = Stage::Vertex;
    else if (s == "frag" || s == "fs" || s == "fragment") st = Stage::Fragment;
    else if (s == "comp" || s == "cs" || s == "compute") st = Stage::Compute;
    else if (s == "geom" || s == "gs") st = Stage::Geometry;
    else if (s == "tesc") st = Stage::TessControl;
    else if (s == "tese") st = Stage::TessEval;
    else return false;
    return true;
}

static const char* slotKindName(SlotKind k) {
    switch (k) {
        case SlotKind::DefaultUniforms: return "DefaultUniforms";
        case SlotKind::SystemValues: return "SystemValues";
        case SlotKind::Ubo: return "Ubo";
        case SlotKind::Ssbo: return "Ssbo";
        case SlotKind::AtomicCounter: return "AtomicCounter";
        case SlotKind::Texture: return "Texture";
        case SlotKind::Sampler: return "Sampler";
        case SlotKind::Image: return "Image";
        case SlotKind::BufferSize: return "BufferSize";
        case SlotKind::XfbBuffer: return "XfbBuffer";
        case SlotKind::SubroutineIndices: return "SubroutineIndices";
    }
    return "?";
}

static void printReflection(const Reflection& r) {
    printf("== reflection ==\n");
    for (auto& a : r.attributes) printf("attribute %s type=0x%x size=%d location=%d (%d locs)\n", a.name.c_str(), a.type, a.arraySize, a.location, a.locationsUsed);
    for (auto& o : r.fragOutputs) printf("output %s type=0x%x size=%d location=%d index=%d\n", o.name.c_str(), o.type, o.arraySize, o.location, o.index);
    for (size_t i = 0; i < r.uniforms.size(); i++) {
        auto& u = r.uniforms[i];
        printf("uniform[%zu] %s type=0x%x size=%d block=%d offset=%d astride=%d mstride=%d rowmajor=%d loc=%d slot=%d binding=%d\n", i,
               u.name.c_str(), u.type, u.arraySize, u.blockIndex, u.offset, u.arrayStride, u.matrixStride, u.rowMajor, u.location, u.slot,
               u.binding);
    }
    for (size_t i = 0; i < r.uniformBlocks.size(); i++) {
        auto& b = r.uniformBlocks[i];
        printf("ubo[%zu] %s binding=%d size=%u slot=%d members=%zu\n", i, b.name.c_str(), b.binding, b.dataSize, b.slot, b.activeUniforms.size());
    }
    for (size_t i = 0; i < r.storageBlocks.size(); i++) {
        auto& b = r.storageBlocks[i];
        printf("ssbo[%zu] %s binding=%d size=%u slot=%d sizeSlot=%d\n", i, b.name.c_str(), b.binding, b.dataSize, b.slot, b.sizeSlot);
    }
    for (auto& bv : r.bufferVariables)
        printf("buffervar %s type=0x%x size=%d block=%d offset=%d astride=%d mstride=%d top=%d/%d\n", bv.name.c_str(), bv.type,
               bv.arraySize, bv.blockIndex, bv.offset, bv.arrayStride, bv.matrixStride, bv.topLevelArraySize, bv.topLevelArrayStride);
    for (auto& a : r.atomicCounterBuffers) printf("acb binding=%d size=%u slot=%d\n", a.binding, a.dataSize, a.slot);
    for (size_t i = 0; i < r.slots.size(); i++)
        printf("slot %zu: %s index=%u element=%u target=0x%x\n", i, slotKindName(r.slots[i].kind), r.slots[i].index, r.slots[i].element,
               r.slots[i].textureTarget);
    printf("multipass: tcsVertexOutBytes=%u tcsPatchOutBytes=%u xfbStride=%u,%u,%u,%u mask=%x flat=%d layer=%d vpi=%d\n",
           r.multipass.tcsVertexOutBytes, r.multipass.tcsPatchOutBytes, r.multipass.xfbStride[0], r.multipass.xfbStride[1],
           r.multipass.xfbStride[2], r.multipass.xfbStride[3], r.multipass.xfbBufferMask, r.usesFlatShading, r.writesLayer,
           r.writesViewportIndex);
    for (auto& x : r.xfbVaryings) printf("xfb %s type=0x%x size=%d buffer=%d offset=%d\n", x.name.c_str(), x.type, x.arraySize, x.buffer, x.offset);
    for (int st = 0; st < (int)Stage::Count; st++) {
        auto& ss = r.subroutines[st];
        for (size_t i = 0; i < ss.functions.size(); i++) printf("subroutine[%d] %zu: %s\n", st, i, ss.functions[i].c_str());
        for (auto& u : ss.uniforms) {
            printf("subroutine uniform[%d] %s location=%d size=%d compatible=", st, u.name.c_str(), u.location, u.arraySize);
            for (int c : u.compatible) printf("%d ", c);
            printf("\n");
        }
    }
    printf("defaultUniformSize=%u localSize=%d,%d,%d fp64=%d discard=%d\n", r.defaultUniformSize, r.localSize[0], r.localSize[1],
           r.localSize[2], r.usesFp64, r.usesDiscard);
}

bool validateMSL(const std::string& src, std::string& err) {
    @autoreleasepool {
        static id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.languageVersion = MTLLanguageVersion4_0;
        opts.mathMode = MTLMathModeSafe;
        opts.preserveInvariance = YES;
        NSError* e = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()] options:opts error:&e];
        if (!lib) {
            err = e ? e.localizedDescription.UTF8String : "unknown error";
            return false;
        }
        return true;
    }
}

int main(int argc, char** argv) {
    std::vector<std::pair<Stage, std::string>> inputs;
    std::vector<std::string> xfbNames;
    bool xfbSeparate = false;
    std::string mslOut;
    bool validate = true, quiet = false, compileOnly = false, timing = false, xfb = false;
    auto now = [] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    double t0 = 0, tc = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-S" && i + 2 < argc) {
            Stage st;
            if (!stageFromName(argv[i + 1], st)) { fprintf(stderr, "unknown stage %s\n", argv[i + 1]); return 2; }
            inputs.push_back({st, argv[i + 2]});
            i += 2;
        } else if (a == "--msl" && i + 1 < argc) mslOut = argv[++i];
        else if (a == "--no-validate") validate = false;
        else if (a == "-q") quiet = true;
        else if (a == "--compile-only") compileOnly = true;
        else if (a == "--time") timing = true;
        else if (a == "--xfb" && i + 1 < argc) {
            // comma separated varying names for glTransformFeedbackVaryings (interleaved)
            xfb = true;
            std::string v = argv[++i];
            size_t p = 0;
            while (p <= v.size()) {
                size_t q = v.find(',', p);
                if (q == std::string::npos) q = v.size();
                if (q > p) xfbNames.push_back(v.substr(p, q - p));
                p = q + 1;
            }
        } else if (a == "--xfb-separate") xfbSeparate = true;
        else { fprintf(stderr, "usage: glslc -S <stage> <file> ... [--msl out] [--no-validate] [-q]\n"); return 2; }
    }
    LinkInput li;
    li.xfbVaryings = xfbNames;
    li.xfbInterleaved = !xfbSeparate;
    for (auto& [st, path] : inputs) {
        std::ifstream f(path);
        if (!f) { fprintf(stderr, "cannot read %s\n", path.c_str()); return 2; }
        std::stringstream ss;
        ss << f.rdbuf();
        t0 = now();
        CompileResult cr = compile(st, {ss.str()});
        tc += now() - t0;
        if (!cr.infoLog.empty()) fprintf(stderr, "%s: %s", path.c_str(), cr.infoLog.c_str());
        if (!cr.shader) { fprintf(stderr, "%s: compile FAILED\n", path.c_str()); return 1; }
        li.shaders.push_back(cr.shader);
    }
    if (compileOnly) return 0;
    t0 = now();
    LinkResult lr = link(li);
    double tl = now() - t0;
    if (!lr.infoLog.empty()) fprintf(stderr, "link: %s", lr.infoLog.c_str());
    if (!lr.program) { fprintf(stderr, "link FAILED\n"); return 3; }
    if (!quiet) printReflection(lr.reflection);
    VariantKey key;
    uint32_t sm = lr.reflection.stageMask;
    key.kind = (sm & (1u << (int)Stage::Compute)) ? PipelineKind::Compute
             : (sm & (1u << (int)Stage::TessEval)) ? PipelineKind::Tessellation
             : (sm & (1u << (int)Stage::Geometry)) ? PipelineKind::MeshGeometry
             : xfb ? PipelineKind::XfbCapture : PipelineKind::Classic;
    t0 = now();
    MSLOutput mo = generateMSL(*lr.program, key);
    double tm = now() - t0;
    if (!mo.error.empty()) { fprintf(stderr, "msl: %s\n", mo.error.c_str()); return 4; }
    if (!quiet) printf("== msl ==\n%s\n", mo.source.c_str());
    if (!mslOut.empty()) {
        std::ofstream f(mslOut);
        f << mo.source;
    }
    double tv = 0;
    if (validate) {
        std::string err;
        if (getenv("METALGLDRV_REPEAT")) for (int i = 0; i < atoi(getenv("METALGLDRV_REPEAT")); i++) {
            t0 = now();
            validateMSL(mo.source + "\n// " + std::to_string(i) + "\n", err);
            fprintf(stderr, "warm metal compile %.2f ms\n", now() - t0);
        }
        t0 = now();
        if (!validateMSL(mo.source, err)) {
            fprintf(stderr, "metal validation FAILED:\n%s\n", err.c_str());
            return 5;
        }
        tv = now() - t0;
        if (!quiet) printf("== metal validation OK ==\n");
    }
    if (timing) fprintf(stderr, "time: compile %.2f ms, link %.2f ms, msl %.2f ms (%zu bytes), metal %.2f ms\n", tc, tl, tm, mo.source.size(), tv);
    return 0;
}
