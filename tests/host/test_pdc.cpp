// ARB_post_depth_coverage (METALGLDRV_STRICT=2, super strict only): gl_SampleMaskIn holds only the samples that
// passed the depth/stencil tests (depth/stencil pre-passes, draw.mm executePdc). Run mgltest with
// METALGLDRV_STRICT=2 to cover it; at lower levels the tests check the extension is absent.
//
// The fragment shaders write gl_SampleMaskIn | gl_PrimitiveID << 8 | instance << 16 into an R32UI multisample
// attachment. Scenes never overwrite a sample, so in every pixel the samples holding one value are exactly the
// samples its primitive wrote, i.e. the samples that passed the tests: that set must equal the value's mask.
#include "harness.h"
#include <set>
#include <vector>

namespace {
bool hasPdc() {
    GLint n = 0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &n);
    for (GLint i = 0; i < n; i++)
        if (!strcmp((const char*)glGetStringi(GL_EXTENSIONS, i), "GL_ARB_post_depth_coverage")) return true;
    return false;
}

const int W = 8, H = 8;
const uint32_t kClear = 0xffffffffu, kMarker = 0xf000u;

const char* kVS = R"(#version 450
layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 offs;  // per instance (tests that draw instanced)
layout(location = 2) in float dz;
flat out int inst;
void main() {
    inst = gl_InstanceID;
    gl_Position = vec4(pos.xy + offs, pos.z + dz, 1.0);
})";

const char* kPdcFS = R"(#version 450
#extension GL_ARB_post_depth_coverage : require
layout(post_depth_coverage) in;
flat in int inst;
layout(location = 0) out uint o;
void main() { o = uint(gl_SampleMaskIn[0]) | (uint(gl_PrimitiveID) << 8) | (uint(inst) << 16); })";

const char* kMarkerFS = R"(#version 450
layout(location = 0) out uint o;
void main() { o = 0xf000u; })";

// Vertices: 0-2 near triangle over the lower-left half (its hypotenuse crosses pixels between their samples),
// 3-8 far full-screen quad as two triangles, 9-12 the quad in fan order, 13-16 in strip order (each split along
// a diagonal that the near triangle doesn't hide a whole triangle of).
const float kVerts[] = {
    -1, -1, -0.5f, 1.07f, -1, -0.5f, -1, 1.07f, -0.5f,                          // near
    -1, -1, 0.5f, 1, -1, 0.5f, 1, 1, 0.5f, -1, -1, 0.5f, 1, 1, 0.5f, -1, 1, 0.5f,  // far triangles
    -1, -1, 0.5f, 1, -1, 0.5f, 1, 1, 0.5f, -1, 1, 0.5f,                          // far fan
    1, -1, 0.5f, 1, 1, 0.5f, -1, -1, 0.5f, -1, 1, 0.5f,                          // far strip
};

struct Target {
    GLuint fb = 0, color = 0, ds = 0, vao = 0, vbo = 0, ibo = 0;
    int samples = 4;
    explicit Target(int s) : samples(s) {
        GLenum target = s > 1 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D;
        glGenTextures(1, &color);
        glBindTexture(target, color);
        if (s > 1) glTexStorage2DMultisample(target, s, GL_R32UI, W, H, GL_TRUE);
        else glTexStorage2D(target, 1, GL_R32UI, W, H);
        glGenRenderbuffers(1, &ds);
        glBindRenderbuffer(GL_RENDERBUFFER, ds);
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, s > 1 ? s : 0, GL_DEPTH24_STENCIL8, W, H);
        glGenFramebuffers(1, &fb);
        glBindFramebuffer(GL_FRAMEBUFFER, fb);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, target, color, 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, ds);
        CHECK_EQ(glCheckFramebufferStatus(GL_FRAMEBUFFER), (GLenum)GL_FRAMEBUFFER_COMPLETE);
        glViewport(0, 0, W, H);
        GLuint clear[4] = {kClear, 0, 0, 0};
        glClearBufferuiv(GL_COLOR, 0, clear);
        glClearDepth(1.0);
        glClearStencil(0);
        glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof kVerts, kVerts, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, nullptr);
        glEnableVertexAttribArray(0);
        glVertexAttrib2f(1, 0, 0);
        glVertexAttrib1f(2, 0);
    }
    ~Target() {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteFramebuffers(1, &fb);
        glDeleteTextures(1, &color);
        glDeleteRenderbuffers(1, &ds);
        glDeleteBuffers(1, &vbo);
        if (ibo) glDeleteBuffers(1, &ibo);
        glDeleteVertexArrays(1, &vao);
    }
    void indices(const std::vector<uint16_t>& idx) {
        glGenBuffers(1, &ibo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * 2, idx.data(), GL_STATIC_DRAW);
    }
    // Every sample of the color attachment, [(y * W + x) * samples + s].
    std::vector<uint32_t> read() {
        char src[1024];
        snprintf(src, sizeof src,
                 "#version 450\nlayout(local_size_x = 1) in;\nuniform %s u;\nlayout(std430, binding = 0) buffer O { uint o[]; };\n"
                 "void main() { ivec2 p = ivec2(gl_GlobalInvocationID.xy); int s = int(gl_GlobalInvocationID.z);\n"
                 "  o[(p.y * %d + p.x) * %d + s] = texelFetch(u, p, %s).x; }\n",
                 samples > 1 ? "usampler2DMS" : "usampler2D", W, samples, samples > 1 ? "s" : "0");
        GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
        const char* s = src;
        glShaderSource(cs, 1, &s, nullptr);
        glCompileShader(cs);
        GLuint p = glCreateProgram();
        glAttachShader(p, cs);
        glDeleteShader(cs);
        glLinkProgram(p);
        GLuint b;
        glGenBuffers(1, &b);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, b);
        glBufferData(GL_SHADER_STORAGE_BUFFER, W * H * samples * 4, nullptr, GL_DYNAMIC_READ);
        glUseProgram(p);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(samples > 1 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, color);
        glUniform1i(glGetUniformLocation(p, "u"), 0);
        glDispatchCompute(W, H, samples);
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        std::vector<uint32_t> out(W * H * samples);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, out.size() * 4, out.data());
        glDeleteBuffers(1, &b);
        glDeleteProgram(p);
        return out;
    }
};

GLuint pdcProgram() {
    std::string log;
    GLuint p = compileProgram(kVS, kPdcFS, &log);
    if (!p) fprintf(stderr, "%s\n", log.c_str());
    CHECK(p != 0);
    return p;
}

struct Stats {
    int partial = 0;  // pixels where a primitive wrote some but not all samples
    int mixed = 0;    // pixels written by more than one primitive / instance (or the marker)
    std::set<uint32_t> tags;  // primitive / instance tags written
};

// The invariant of the file comment; also: a value's mask contains only covered samples (no clear value below a
// written sample's own bit).
Stats checkMasks(const std::vector<uint32_t>& v, int samples) {
    Stats st;
    uint32_t all = (1u << samples) - 1;
    for (int px = 0; px < W * H; px++) {
        std::set<uint32_t> vals;
        for (int s = 0; s < samples; s++) vals.insert(v[px * samples + s]);
        int written = 0;
        for (uint32_t val : vals) {
            if (val == kClear) continue;
            written++;
            if (val == kMarker) continue;
            uint32_t have = 0;
            for (int s = 0; s < samples; s++)
                if (v[px * samples + s] == val) have |= 1u << s;
            if ((val & 0xff) != have) {
                testFail(__FILE__, __LINE__, "pixel " + std::to_string(px % W) + "," + std::to_string(px / W) + ": value 0x" +
                                                 std::to_string(val) + " written to samples " + std::to_string(have));
                return st;
            }
            if (have != all) st.partial++;
            st.tags.insert(val >> 8);
        }
        if (written > 1) st.mixed++;
    }
    return st;
}
}  // namespace

TEST(pdc_extension_super_strict_only) {
    CHECK_EQ(hasPdc(), strictLevelFromEnv() >= 2);
    if (hasPdc()) return;
    // without the extension the qualifier is an unknown (ignored) layout qualifier, as before
    GLuint s = glCreateShader(GL_FRAGMENT_SHADER);
    const char* src = "#version 450\n#extension GL_ARB_post_depth_coverage : require\nvoid main() {}\n";
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 1;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    CHECK(!ok);
    glDeleteShader(s);
}

// The CTS case: a near primitive, then a far one with post_depth_coverage: only the samples the near one doesn't
// cover remain (depth test, then stencil test).
static void twoDraws(bool stencil) {
    Target t(4);
    GLuint marker = compileProgram(kVS, kMarkerFS), p = pdcProgram();
    if (stencil) {
        glEnable(GL_STENCIL_TEST);
        glStencilFunc(GL_ALWAYS, 1, 0xff);
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
    } else {
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
    }
    glUseProgram(marker);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    if (stencil) {
        glStencilFunc(GL_NOTEQUAL, 1, 0xff);
        glStencilMask(0);
    }
    glUseProgram(p);
    glDrawArrays(GL_TRIANGLES, 3, 6);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST);
    glStencilMask(0xff);
    std::vector<uint32_t> v = t.read();
    Stats st = checkMasks(v, 4);
    CHECK(st.partial > 0);
    CHECK(st.mixed > 0);
    // every sample is either the near primitive's or written by the far one
    for (uint32_t x : v) CHECK(x != kClear);
    glDeleteProgram(marker);
    glDeleteProgram(p);
    CHECK_GL();
}

TEST(pdc_depth_occluded_samples) {
    if (!hasPdc()) return;
    twoDraws(false);
}

TEST(pdc_stencil_occluded_samples) {
    if (!hasPdc()) return;
    twoDraws(true);
}

// One draw whose first primitive occludes parts of the later ones (depth writes): each primitive must see the
// depth buffer as the previous ones left it.
enum class Topo { Arrays, Elements, StripRestart, FanRestart, FanCustomRestart };
static void intraDraw(Topo topo, bool stencil, int samples) {
    Target t(samples);
    GLuint p = pdcProgram();
    if (stencil) {
        // the first primitive passes and increments; the later ones fail where it was
        glEnable(GL_STENCIL_TEST);
        glStencilFunc(GL_EQUAL, 0, 0xff);
        glStencilOp(GL_KEEP, GL_KEEP, GL_INCR);
    } else {
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
    }
    glUseProgram(p);
    switch (topo) {
    case Topo::Arrays: glDrawArrays(GL_TRIANGLES, 0, 9); break;
    case Topo::Elements:
        t.indices({0, 1, 2, 3, 4, 5, 6, 7, 8});
        glDrawElements(GL_TRIANGLES, 9, GL_UNSIGNED_SHORT, nullptr);
        break;
    case Topo::StripRestart:
        t.indices({0, 1, 2, 0xffff, 13, 14, 15, 16});
        glEnable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
        glDrawElements(GL_TRIANGLE_STRIP, 8, GL_UNSIGNED_SHORT, nullptr);
        glDisable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
        break;
    case Topo::FanRestart:
        t.indices({0, 1, 2, 0xffff, 9, 10, 11, 12});
        glEnable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
        glDrawElements(GL_TRIANGLE_FAN, 8, GL_UNSIGNED_SHORT, nullptr);
        glDisable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
        break;
    case Topo::FanCustomRestart:
        t.indices({0, 1, 2, 77, 9, 10, 11, 12});
        glEnable(GL_PRIMITIVE_RESTART);
        glPrimitiveRestartIndex(77);
        glDrawElements(GL_TRIANGLE_FAN, 8, GL_UNSIGNED_SHORT, nullptr);
        glDisable(GL_PRIMITIVE_RESTART);
        break;
    }
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST);
    std::vector<uint32_t> v = t.read();
    Stats st = checkMasks(v, samples);
    if (samples > 1) {
        CHECK(st.partial > 0);
        CHECK(st.mixed > 0);
    }
    // primitives 0 (near) and 1, 2 (far quad; restart doesn't reset gl_PrimitiveID)
    CHECK(st.tags == std::set<uint32_t>({0u, 1u, 2u}));
    for (uint32_t x : v) CHECK(x != kClear);
    glDeleteProgram(p);
    CHECK_GL();
}

TEST(pdc_intra_draw_depth) {
    if (!hasPdc()) return;
    intraDraw(Topo::Arrays, false, 4);
    intraDraw(Topo::Elements, false, 4);
}

TEST(pdc_intra_draw_restart_strips_fans) {
    if (!hasPdc()) return;
    intraDraw(Topo::StripRestart, false, 4);
    intraDraw(Topo::FanRestart, false, 4);
    intraDraw(Topo::FanCustomRestart, false, 4);
}

TEST(pdc_intra_draw_stencil) {
    if (!hasPdc()) return;
    intraDraw(Topo::Arrays, true, 4);
    intraDraw(Topo::StripRestart, true, 4);
}

TEST(pdc_intra_draw_2x) {
    if (!hasPdc()) return;
    intraDraw(Topo::Arrays, false, 2);
    intraDraw(Topo::FanRestart, true, 2);
}

// Single-sampled: gl_SampleMaskIn is one bit, set whenever the fragment is shaded.
TEST(pdc_single_sample) {
    if (!hasPdc()) return;
    intraDraw(Topo::Arrays, false, 1);
    intraDraw(Topo::FanRestart, true, 1);
}

// Instances are drawn in order: instance 0 occludes parts of instance 1 (depth writes). Per-instance attributes
// with divisors 1 and 2, gl_InstanceID and gl_PrimitiveID (which restarts per instance).
TEST(pdc_instanced) {
    if (!hasPdc()) return;
    Target t(4);
    GLuint p = pdcProgram();
    const float offs[] = {0, 0, 0.3f, 0.1f, 0.6f, 0.2f};
    const float dz[] = {-0.4f, 0.6f};  // instances 0 and 1 near, instance 2 far
    GLuint b[2];
    glGenBuffers(2, b);
    glBindBuffer(GL_ARRAY_BUFFER, b[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof offs, offs, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8, nullptr);
    glVertexAttribDivisor(1, 1);
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, b[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof dz, dz, GL_STATIC_DRAW);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 4, nullptr);
    glVertexAttribDivisor(2, 2);
    glEnableVertexAttribArray(2);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glUseProgram(p);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 3, 3);
    glDisable(GL_DEPTH_TEST);
    std::vector<uint32_t> v = t.read();
    Stats st = checkMasks(v, 4);
    CHECK(st.partial > 0);
    CHECK(st.mixed > 0);
    // instance i writes tag i << 8 (primitive 0 of each instance)
    CHECK(st.tags == std::set<uint32_t>({0u, 0x100u, 0x200u}));
    glDeleteBuffers(2, b);
    glDeleteProgram(p);
    CHECK_GL();
}

// Without a depth or stencil test gl_SampleMaskIn is the coverage.
TEST(pdc_no_tests) {
    if (!hasPdc()) return;
    Target t(4);
    GLuint p = pdcProgram();
    glUseProgram(p);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    Stats st = checkMasks(t.read(), 4);
    CHECK(st.partial > 0);
    glDeleteProgram(p);
    CHECK_GL();
}

// Pre-passes re-run the vertex shader on copies of its storage: its side effects happen once per vertex.
TEST(pdc_prepass_no_side_effects) {
    if (!hasPdc()) return;
    Target t(4);
    std::string log;
    GLuint p = compileProgram(R"(#version 450
layout(location = 0) in vec3 pos;
layout(std430, binding = 0) buffer C { uint n; uint last; };
flat out int inst;
void main() {
    uint k = atomicAdd(n, 1u);
    last = k;
    inst = 0;
    gl_Position = vec4(pos, 1.0);
})", kPdcFS, &log);
    if (!p) fprintf(stderr, "%s\n", log.c_str());
    CHECK(p != 0);
    GLuint b;
    glGenBuffers(1, &b);
    uint32_t zero[2] = {0, 0};
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, b);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 8, zero, GL_DYNAMIC_READ);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glUseProgram(p);
    glDrawArrays(GL_TRIANGLES, 0, 9);  // split per primitive: three pre-passes
    glDisable(GL_DEPTH_TEST);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    uint32_t got[2] = {0, 0};
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, b);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 8, got);
    CHECK_EQ(got[0], 9u);
    Stats st = checkMasks(t.read(), 4);
    CHECK(st.tags == std::set<uint32_t>({0u, 1u, 2u}));
    glDeleteBuffers(1, &b);
    glDeleteProgram(p);
    CHECK_GL();
}

// Layered framebuffer: the coverage attachment has the layers too (the vertex shader picks layer 1).
TEST(pdc_layered) {
    if (!hasPdc()) return;
    const int S = 4, L = 2;
    GLuint color, depth, fb;
    glGenTextures(1, &color);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, color);
    glTexStorage3DMultisample(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, S, GL_R32UI, W, H, L, GL_TRUE);
    glGenTextures(1, &depth);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, depth);
    glTexStorage3DMultisample(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, S, GL_DEPTH_COMPONENT32F, W, H, L, GL_TRUE);
    glGenFramebuffers(1, &fb);
    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, color, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depth, 0);
    CHECK_EQ(glCheckFramebufferStatus(GL_FRAMEBUFFER), (GLenum)GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, W, H);
    GLuint clear[4] = {kClear, 0, 0, 0};
    glClearBufferuiv(GL_COLOR, 0, clear);
    glClearDepth(1.0);
    glClear(GL_DEPTH_BUFFER_BIT);
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof kVerts, kVerts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, nullptr);
    glEnableVertexAttribArray(0);
    std::string log;
    GLuint p = compileProgram(R"(#version 450
#extension GL_ARB_shader_viewport_layer_array : require
layout(location = 0) in vec3 pos;
flat out int inst;
void main() {
    inst = 0;
    gl_Layer = 1;
    gl_Position = vec4(pos, 1.0);
})", kPdcFS, &log);
    if (!p) fprintf(stderr, "%s\n", log.c_str());
    CHECK(p != 0);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glUseProgram(p);
    glDrawArrays(GL_TRIANGLES, 0, 9);
    glDisable(GL_DEPTH_TEST);
    // layer 1, every sample
    GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
    char src[512];
    snprintf(src, sizeof src,
             "#version 450\nlayout(local_size_x = 1) in;\nuniform usampler2DMSArray u;\nlayout(std430, binding = 0) buffer O { uint o[]; };\n"
             "void main() { ivec2 q = ivec2(gl_GlobalInvocationID.xy); int s = int(gl_GlobalInvocationID.z);\n"
             "  o[(q.y * %d + q.x) * %d + s] = texelFetch(u, ivec3(q, 1), s).x; }\n", W, S);
    const char* sp = src;
    glShaderSource(cs, 1, &sp, nullptr);
    glCompileShader(cs);
    GLuint cp = glCreateProgram();
    glAttachShader(cp, cs);
    glDeleteShader(cs);
    glLinkProgram(cp);
    GLuint b;
    glGenBuffers(1, &b);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, b);
    glBufferData(GL_SHADER_STORAGE_BUFFER, W * H * S * 4, nullptr, GL_DYNAMIC_READ);
    glUseProgram(cp);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, color);
    glDispatchCompute(W, H, S);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    std::vector<uint32_t> v(W * H * S);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, v.size() * 4, v.data());
    Stats st = checkMasks(v, S);
    CHECK(st.partial > 0);
    CHECK(st.tags == std::set<uint32_t>({0u, 1u, 2u}));
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteBuffers(1, &b);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteFramebuffers(1, &fb);
    glDeleteTextures(1, &color);
    glDeleteTextures(1, &depth);
    glDeleteProgram(p);
    glDeleteProgram(cp);
    CHECK_GL();
}

// Geometry shader (mesh pipeline): one pre-pass for the draw. The near primitive is drawn first by another draw,
// and the two far triangles don't overlap, so the result is exact.
TEST(pdc_geometry_shader) {
    if (!hasPdc()) return;
    Target t(4);
    GLuint marker = compileProgram(kVS, kMarkerFS);
    const char* srcs[3] = {kVS, R"(#version 450
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
flat in int inst[];
flat out int ginst;
void main() {
    for (int i = 0; i < 3; i++) {
        ginst = inst[i];
        gl_PrimitiveID = gl_PrimitiveIDIn;
        gl_Position = gl_in[i].gl_Position;
        EmitVertex();
    }
})", R"(#version 450
#extension GL_ARB_post_depth_coverage : require
layout(post_depth_coverage) in;
flat in int ginst;
layout(location = 0) out uint o;
void main() { o = uint(gl_SampleMaskIn[0]) | (uint(gl_PrimitiveID) << 8) | (uint(ginst) << 16); })"};
    GLenum types[3] = {GL_VERTEX_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER};
    GLuint p = glCreateProgram();
    for (int i = 0; i < 3; i++) {
        GLuint s = glCreateShader(types[i]);
        glShaderSource(s, 1, &srcs[i], nullptr);
        glCompileShader(s);
        glAttachShader(p, s);
        glDeleteShader(s);
    }
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    CHECK(ok);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glUseProgram(marker);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glUseProgram(p);
    glDrawArrays(GL_TRIANGLES, 3, 6);
    glDisable(GL_DEPTH_TEST);
    std::vector<uint32_t> v = t.read();
    Stats st = checkMasks(v, 4);
    CHECK(st.partial > 0);
    CHECK(st.mixed > 0);
    CHECK(st.tags == std::set<uint32_t>({0u, 1u}));
    for (uint32_t x : v) CHECK(x != kClear);
    glDeleteProgram(marker);
    glDeleteProgram(p);
    CHECK_GL();
}
