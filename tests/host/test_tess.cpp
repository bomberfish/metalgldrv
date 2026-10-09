// Tessellation fast path (fused vertex pipeline when the levels make every patch one primitive) and
// multi-draw batching, compared bit for bit against the emulation (METALGLDRV_TESS_FAST=0,
// METALGLDRV_TESS_BATCH=0, METALGLDRV_TESS_PASS1=compute).
#include "harness.h"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" void mglTestTessCounters(uint64_t* passes, uint64_t* fused);

namespace {

GLuint shader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, nullptr, log);
        fprintf(stderr, "  compile failed: %s\n", log);
    }
    return s;
}

GLuint program(const char* vs, const char* tcs, const char* tes, const char* fs) {
    GLuint p = glCreateProgram();
    GLuint s[4] = {shader(GL_VERTEX_SHADER, vs), tcs ? shader(GL_TESS_CONTROL_SHADER, tcs) : 0, shader(GL_TESS_EVALUATION_SHADER, tes),
                   shader(GL_FRAGMENT_SHADER, fs)};
    for (GLuint x : s)
        if (x) glAttachShader(p, x);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(p, sizeof log, nullptr, log);
        fprintf(stderr, "  link failed: %s\n", log);
    }
    for (GLuint x : s)
        if (x) glDeleteShader(x);
    return p;
}

struct Target {
    GLuint fbo = 0, tex = 0;
    Target() {
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, 64, 64);
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
        glViewport(0, 0, 64, 64);
    }
    ~Target() {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteFramebuffers(1, &fbo);
        glDeleteTextures(1, &tex);
    }
    std::vector<float> read() {
        std::vector<float> px(64 * 64 * 4);
        glReadPixels(0, 0, 64, 64, GL_RGBA, GL_FLOAT, px.data());
        return px;
    }
};

// Renders `draw` with the given switches: fast path, batching, pass 1 as a compute kernel.
std::vector<float> render(Target& t, bool fast, bool batch, bool kernel, const std::function<void()>& draw) {
    setenv("METALGLDRV_TESS_FAST", fast ? "1" : "0", 1);
    setenv("METALGLDRV_TESS_BATCH", batch ? "1" : "0", 1);
    if (kernel) setenv("METALGLDRV_TESS_PASS1", "compute", 1);
    else unsetenv("METALGLDRV_TESS_PASS1");
    glClearColor(-1, -1, -1, -1);
    glClear(GL_COLOR_BUFFER_BIT);
    draw();
    std::vector<float> px = t.read();
    unsetenv("METALGLDRV_TESS_FAST");
    unsetenv("METALGLDRV_TESS_BATCH");
    unsetenv("METALGLDRV_TESS_PASS1");
    return px;
}

int differing(const std::vector<float>& a, const std::vector<float>& b) {
    int n = 0;
    for (size_t i = 0; i < a.size(); i += 4)
        if (memcmp(&a[i], &b[i], 16)) n++;
    return n;
}

const float* at(const std::vector<float>& px, int x, int y) { return &px[(y * 64 + x) * 4]; }

uint64_t passes() {
    uint64_t p, f;
    mglTestTessCounters(&p, &f);
    return p;
}
uint64_t fusedDraws() {
    uint64_t p, f;
    mglTestTessCounters(&p, &f);
    return f;
}

// Shrimple-like: the TCS passes the control points through and takes its levels from a uniform, the
// TES (triangles, fractional_odd_spacing) interpolates; interior domain points are displaced, so a level
// above 1 changes the image. Every built-in the fused path provides reaches the color.
const char* kVS = R"(#version 460 core
layout(location = 0) in vec2 p;
layout(location = 1) in vec4 c;
out V { vec4 color; flat int id; } vo;
void main() {
    vo.color = c;
    vo.id = gl_DrawID * 1000 + gl_BaseVertex * 10 + (gl_VertexID - gl_BaseVertex);
    gl_Position = vec4(p, 0.0, 1.0);
})";
const char* kTCS = R"(#version 460 core
layout(vertices = 3) out;
uniform float lvl;
in V { vec4 color; flat int id; } vi[];
out V { vec4 color; flat int id; } vo[];
patch out vec4 pd;
void main() {
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    vo[gl_InvocationID].color = vi[gl_InvocationID].color;
    vo[gl_InvocationID].id = vi[gl_InvocationID].id;
    if (gl_InvocationID == 0) {
        pd = vec4(float(gl_PrimitiveID), float(gl_PatchVerticesIn), vi[2].color.x, 1.0);
        gl_TessLevelOuter[0] = lvl;
        gl_TessLevelOuter[1] = lvl;
        gl_TessLevelOuter[2] = lvl;
        gl_TessLevelInner[0] = lvl;
    }
})";
const char* kTES = R"(#version 460 core
layout(triangles, fractional_odd_spacing, ccw) in;
in V { vec4 color; flat int id; } vi[];
patch in vec4 pd;
out vec4 color;
flat out ivec3 ids;
out vec4 pdata;
void main() {
    vec3 t = gl_TessCoord;
    gl_Position = t.x * gl_in[0].gl_Position + t.y * gl_in[1].gl_Position + t.z * gl_in[2].gl_Position +
                  vec4(0.0, 1.5 * t.x * t.y * t.z, 0.0, 0.0);
    color = t.x * vi[0].color + t.y * vi[1].color + t.z * vi[2].color;
    ids = ivec3(vi[0].id, gl_PrimitiveID, int(t.x + 2.0 * t.y + 4.0 * t.z));
    pdata = pd + vec4(gl_TessLevelOuter[0], gl_TessLevelInner[0], float(gl_PatchVerticesIn), 0.0);
})";
const char* kFS = R"(#version 460 core
in vec4 color;
flat in ivec3 ids;
in vec4 pdata;
out vec4 o;
void main() {
    o = vec4(color.r + float(ids.z) * 10.0, color.g, float(ids.x) + float(ids.y) * 0.25, pdata.x + pdata.y * 10.0 + pdata.z * 100.0 + pdata.w * 1000.0);
})";

struct Vtx {
    float p[2], c[4];
};

GLuint quadsVao(GLuint* bufs) {
    // quad A (bottom left) = vertices 0..3, quad B (top right) = vertices 4..7
    const Vtx v[8] = {{{-0.9f, -0.9f}, {0.1f, 0.2f, 0, 1}}, {{-0.1f, -0.9f}, {0.3f, 0.4f, 0, 1}}, {{-0.1f, -0.1f}, {0.5f, 0.6f, 0, 1}},
                      {{-0.9f, -0.1f}, {0.7f, 0.8f, 0, 1}}, {{0.1f, 0.1f}, {0.15f, 0.25f, 0, 1}},   {{0.9f, 0.1f}, {0.35f, 0.45f, 0, 1}},
                      {{0.9f, 0.9f}, {0.55f, 0.65f, 0, 1}}, {{0.1f, 0.9f}, {0.75f, 0.85f, 0, 1}}};
    const uint16_t idx[6] = {0, 1, 2, 2, 3, 0};  // shared index buffer, as Sodium's
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(2, bufs);
    glBindBuffer(GL_ARRAY_BUFFER, bufs[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof v, v, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufs[1]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void*)0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void*)8);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    return vao;
}

}  // namespace

// Fused fast path vs the emulation (object-stage and compute pass 1, batched and not) for a Sodium-like
// glMultiDrawElementsBaseVertex(GL_TRIANGLES) with gl_DrawID / gl_BaseVertex / gl_PrimitiveID / patch
// outputs / flat varyings, and the switch between the paths as the level uniform changes.
TEST(tessellation_fast_path) {
    GLuint p = program(kVS, kTCS, kTES, kFS);
    GLuint bufs[2];
    GLuint vao = quadsVao(bufs);
    Target t;
    glUseProgram(p);
    GLint lvl = glGetUniformLocation(p, "lvl");
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    setenv("METALGLDRV_IMPLICIT_PATCHES", "1", 1);
    // sub-draws: quad A (draw 0), an empty one (draw 1), quad B's second triangle (draw 2: indices 2 3 0
    // + 4), quad B's first triangle (draw 3: indices 0 1 2 + 4)
    const GLsizei counts[4] = {6, 0, 3, 3};
    const void* offs[4] = {(void*)0, (void*)0, (void*)6, (void*)0};
    const GLint bv[4] = {0, 0, 4, 4};
    auto multi = [&]() { glMultiDrawElementsBaseVertex(GL_TRIANGLES, counts, GL_UNSIGNED_SHORT, offs, 4, bv); };

    glUniform1f(lvl, 1.0f);
    std::vector<float> ref = render(t, false, false, false, multi);
    // expected values: (id of control point 0, primitive id within the draw) and the provoking corner (0,0,1)
    const float* a0 = at(ref, 12, 8);   // quad A, triangle 0 1 2 (below the diagonal)
    const float* a1 = at(ref, 8, 16);   // quad A, triangle 2 3 0
    const float* b0 = at(ref, 52, 40);  // quad B, triangle 4 5 6 (draw 3)
    const float* b1 = at(ref, 40, 52);  // quad B, triangle 6 7 4 (draw 2)
    CHECK(a0[2] == 0.0f);
    CHECK(a1[2] == 2.25f);
    CHECK(b0[2] == 3040.0f);
    CHECK(b1[2] == 2042.0f);
    CHECK(a0[0] >= 40.0f && a0[0] < 41.0f);  // flat corner (0,0,1) = 4
    // patch data: (primitive 0 + outer 1) + (TCS gl_PatchVerticesIn 3 + inner 1) * 10 +
    // (control point 2's color.x 0.5 + TES gl_PatchVerticesIn 3) * 100 + 1000
    CHECK(a0[3] == 1391.0f);
    uint64_t f0 = fusedDraws(), p0 = passes();
    std::vector<float> fast = render(t, true, true, false, multi);
    CHECK_EQ(fusedDraws() - f0, 1u);  // fast path, one draw for all sub-draws
    CHECK_EQ(passes() - p0, 1u);
    CHECK_EQ(differing(ref, fast), 0);
    p0 = passes();
    CHECK_EQ(differing(ref, render(t, true, false, false, multi)), 0);
    CHECK_EQ(passes() - p0, 3u);  // unbatched: one per non-empty sub-draw
    p0 = passes();
    CHECK_EQ(differing(ref, render(t, false, true, false, multi)), 0);
    CHECK_EQ(passes() - p0, 1u);
    CHECK_EQ(differing(ref, render(t, false, true, true, multi)), 0);
    CHECK_EQ(differing(ref, render(t, false, false, true, multi)), 0);
    // winding: the quads are front-facing (ccw) and the TES keeps the orientation
    glEnable(GL_CULL_FACE);
    for (GLenum face : {GL_BACK, GL_FRONT}) {
        glCullFace(face);
        std::vector<float> cref = render(t, false, false, false, multi);
        CHECK_EQ(differing(cref, render(t, true, true, false, multi)), 0);
        CHECK((at(cref, 12, 8)[0] == -1.0f) == (face == GL_FRONT));
    }
    glCullFace(GL_BACK);
    glDisable(GL_CULL_FACE);

    // level 3: tessellated (emulation), the displaced interior points change the image
    glUniform1f(lvl, 3.0f);
    f0 = fusedDraws();
    std::vector<float> l3 = render(t, true, true, false, multi);
    CHECK_EQ(fusedDraws() - f0, 0u);
    CHECK(differing(ref, l3) > 20);
    CHECK_EQ(differing(l3, render(t, false, false, false, multi)), 0);
    CHECK_EQ(differing(l3, render(t, false, true, true, multi)), 0);
    // back to 1 (and 0.5, which also gives one segment per edge): the fast path again
    glUniform1f(lvl, 1.0f);
    f0 = fusedDraws();
    CHECK_EQ(differing(ref, render(t, true, true, false, multi)), 0);
    CHECK_EQ(fusedDraws() - f0, 1u);
    glUniform1f(lvl, 0.5f);
    std::vector<float> half = render(t, false, false, false, multi);
    f0 = fusedDraws();
    CHECK_EQ(differing(half, render(t, true, true, false, multi)), 0);
    CHECK_EQ(fusedDraws() - f0, 1u);
    // 0: every patch is culled; not the fast path
    glUniform1f(lvl, 0.0f);
    f0 = fusedDraws();
    std::vector<float> culled = render(t, true, true, false, multi);
    CHECK_EQ(fusedDraws() - f0, 0u);
    CHECK(at(culled, 12, 8)[0] == -1.0f && at(culled, 52, 40)[0] == -1.0f);

    // single draws (GL_PATCHES, uint16 indices from an offset, base vertex) and non-indexed multi-draws
    glUniform1f(lvl, 1.0f);
    auto singles = [&]() {
        glDrawElementsBaseVertex(GL_PATCHES, 3, GL_UNSIGNED_SHORT, (void*)6, 4);
        glDrawArrays(GL_PATCHES, 0, 3);
        const GLint first[2] = {4, 1};
        const GLsizei cnt[2] = {3, 3};
        glMultiDrawArrays(GL_PATCHES, first, cnt, 2);
    };
    std::vector<float> sref = render(t, false, false, false, singles);
    f0 = fusedDraws();
    CHECK_EQ(differing(sref, render(t, true, true, false, singles)), 0);
    CHECK_EQ(fusedDraws() - f0, 3u);
    CHECK_EQ(differing(sref, render(t, false, true, true, singles)), 0);
    unsetenv("METALGLDRV_IMPLICIT_PATCHES");

    glDeleteBuffers(2, bufs);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    CHECK_GL();
}

// Quads domain, cw, constant levels written in a chained assignment, 4-vertex patches: winding against
// face culling (one front face setting shows the quads) and the fast path vs the emulation.
TEST(tessellation_fast_path_quads) {
    const char* vs = R"(#version 410 core
layout(location = 0) in vec2 p;
layout(location = 1) in vec4 c;
out vec4 vc;
void main() { vc = c; gl_Position = vec4(p, 0.0, 1.0); })";
    const char* tcs = R"(#version 410 core
layout(vertices = 4) out;
in vec4 vc[];
out vec4 tc[];
void main() {
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    tc[gl_InvocationID] = vc[gl_InvocationID] * 2.0;
    gl_TessLevelOuter[0] = gl_TessLevelOuter[1] = gl_TessLevelOuter[2] = gl_TessLevelOuter[3] = 1.0;
    gl_TessLevelInner[0] = gl_TessLevelInner[1] = 1.0;
})";
    const char* tes = R"(#version 410 core
layout(quads, equal_spacing, cw) in;
in vec4 tc[];
out vec4 col;
flat out vec2 corner;
void main() {
    vec2 u = gl_TessCoord.xy;
    gl_Position = mix(mix(gl_in[0].gl_Position, gl_in[1].gl_Position, u.x), mix(gl_in[3].gl_Position, gl_in[2].gl_Position, u.x), u.y);
    col = mix(mix(tc[0], tc[1], u.x), mix(tc[3], tc[2], u.x), u.y);
    corner = u;
})";
    const char* fs = R"(#version 410 core
in vec4 col;
flat in vec2 corner;
out vec4 o;
void main() { o = vec4(col.rg, corner); })";
    GLuint p = program(vs, tcs, tes, fs);
    GLuint bufs[2];
    GLuint vao = quadsVao(bufs);
    Target t;
    glUseProgram(p);
    glPatchParameteri(GL_PATCH_VERTICES, 4);
    glEnable(GL_CULL_FACE);
    auto draw = [&]() {
        const GLint first[2] = {0, 4};
        const GLsizei cnt[2] = {4, 4};
        glMultiDrawArrays(GL_PATCHES, first, cnt, 2);
    };
    int shown = 0;
    for (GLenum front : {GL_CCW, GL_CW}) {
        glFrontFace(front);
        std::vector<float> ref = render(t, false, false, false, draw);
        uint64_t f0 = fusedDraws();
        std::vector<float> fast = render(t, true, true, false, draw);
        CHECK_EQ(fusedDraws() - f0, 1u);
        CHECK_EQ(differing(ref, fast), 0);
        CHECK_EQ(differing(ref, render(t, false, true, true, draw)), 0);
        if (at(fast, 16, 16)[0] != -1.0f && at(fast, 48, 48)[0] != -1.0f) shown++;
    }
    CHECK_EQ(shown, 1);
    glDisable(GL_CULL_FACE);
    glFrontFace(GL_CCW);
    glDeleteBuffers(2, bufs);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    CHECK_GL();
}

// Without a TCS the levels are GL_PATCH_DEFAULT_*_LEVEL: 1 takes the fast path, 2 the emulation.
TEST(tessellation_fast_path_no_tcs) {
    const char* vs = R"(#version 410 core
layout(location = 0) in vec2 p;
layout(location = 1) in vec4 c;
out vec4 vc;
void main() { vc = c; gl_Position = vec4(p, 0.0, 1.0); })";
    const char* tes = R"(#version 410 core
layout(triangles, equal_spacing, ccw) in;
in vec4 vc[];
out vec4 col;
void main() {
    vec3 t = gl_TessCoord;
    gl_Position = t.x * gl_in[0].gl_Position + t.y * gl_in[1].gl_Position + t.z * gl_in[2].gl_Position + vec4(0.0, t.x * t.y * t.z, 0.0, 0.0);
    col = t.x * vc[0] + t.y * vc[1] + t.z * vc[2];
})";
    const char* fs = R"(#version 410 core
in vec4 col;
out vec4 o;
void main() { o = col; })";
    GLuint p = program(vs, nullptr, tes, fs);
    GLuint bufs[2];
    GLuint vao = quadsVao(bufs);
    Target t;
    glUseProgram(p);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    auto draw = [&]() { glDrawElementsBaseVertex(GL_PATCHES, 6, GL_UNSIGNED_SHORT, nullptr, 4); };
    for (float level : {1.0f, 2.0f}) {
        const float outer[4] = {level, level, level, level}, inner[2] = {level, level};
        glPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL, outer);
        glPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL, inner);
        std::vector<float> ref = render(t, false, false, false, draw);
        uint64_t f0 = fusedDraws();
        CHECK_EQ(differing(ref, render(t, true, true, false, draw)), 0);
        CHECK_EQ(fusedDraws() - f0, level == 1.0f ? 1u : 0u);
        CHECK(at(ref, 52, 40)[0] != -1.0f);
    }
    const float one[4] = {1, 1, 1, 1};
    glPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL, one);
    glPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL, one);
    glDeleteBuffers(2, bufs);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    CHECK_GL();
}

// Programs the fast path must not take (the analysis rejects them): fractional_even spacing (level 1
// clamps to 2), a TCS barrier(), levels computed from inputs, point_mode.
TEST(tessellation_fast_path_rejects) {
    const char* vs = R"(#version 410 core
layout(location = 0) in vec2 p;
void main() { gl_Position = vec4(p, 0.0, 1.0); })";
    const char* tcsLevel = R"(#version 410 core
layout(vertices = 3) out;
void main() {
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    gl_TessLevelOuter[0] = gl_TessLevelOuter[1] = gl_TessLevelOuter[2] = gl_TessLevelInner[0] = 1.0;
})";
    const char* tcsBarrier = R"(#version 410 core
layout(vertices = 3) out;
void main() {
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    barrier();
    gl_TessLevelOuter[0] = gl_TessLevelOuter[1] = gl_TessLevelOuter[2] = gl_TessLevelInner[0] = 1.0;
})";
    const char* tcsInput = R"(#version 410 core
layout(vertices = 3) out;
void main() {
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    gl_TessLevelOuter[0] = gl_TessLevelOuter[1] = gl_TessLevelOuter[2] = gl_TessLevelInner[0] = gl_in[0].gl_Position.w;
})";
    auto tes = [](const char* layout) {
        return std::string("#version 410 core\nlayout(") + layout + R"() in;
void main() { gl_Position = gl_TessCoord.x * gl_in[0].gl_Position + gl_TessCoord.y * gl_in[1].gl_Position + gl_TessCoord.z * gl_in[2].gl_Position; })";
    };
    const char* fs = R"(#version 410 core
out vec4 o;
void main() { o = vec4(0.25, 0.5, 0.75, 1.0); })";
    // vertex shaders reading (fine) and writing (not: it runs several times per vertex) a storage buffer
    const char* vsRead = R"(#version 430 core
layout(location = 0) in vec2 p;
layout(std430, binding = 0) readonly buffer B { vec4 offset; } b;
void main() { gl_Position = vec4(p, 0.0, 1.0) + b.offset; })";
    const char* vsWrite = R"(#version 430 core
layout(location = 0) in vec2 p;
layout(std430, binding = 0) buffer B { vec4 offset; uint count; } b;
void main() { atomicAdd(b.count, 1u); gl_Position = vec4(p, 0.0, 1.0); })";
    struct Case {
        const char* vs;
        const char* tcs;
        std::string tes;
        bool fused;
    } cases[] = {{vs, tcsLevel, tes("triangles, equal_spacing, ccw"), true},
                 {vs, tcsLevel, tes("triangles, fractional_even_spacing, ccw"), false},
                 {vs, tcsBarrier, tes("triangles, equal_spacing, ccw"), false},
                 {vs, tcsInput, tes("triangles, equal_spacing, ccw"), false},
                 {vs, tcsLevel, tes("triangles, equal_spacing, ccw, point_mode"), false},
                 {vsRead, tcsLevel, tes("triangles, equal_spacing, ccw"), true},
                 {vsWrite, tcsLevel, tes("triangles, equal_spacing, ccw"), false}};
    GLuint bufs[2];
    GLuint vao = quadsVao(bufs);
    Target t;
    GLuint ssbo;
    const float zero[8] = {};
    glGenBuffers(1, &ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof zero, zero, GL_DYNAMIC_DRAW);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    for (auto& c : cases) {
        GLuint p = program(c.vs, c.tcs, c.tes.c_str(), fs);
        glUseProgram(p);
        auto draw = [&]() { glDrawElements(GL_PATCHES, 6, GL_UNSIGNED_SHORT, nullptr); };
        std::vector<float> ref = render(t, false, false, false, draw);
        uint64_t f0 = fusedDraws();
        CHECK_EQ(differing(ref, render(t, true, true, false, draw)), 0);
        CHECK_EQ(fusedDraws() - f0, c.fused ? 1u : 0u);
        glDeleteProgram(p);
    }
    glDeleteBuffers(1, &ssbo);
    glDeleteBuffers(2, bufs);
    glDeleteVertexArrays(1, &vao);
    CHECK_GL();
}

// The draw kinds Minecraft 26 issues besides Sodium's multi-draws, fast path vs emulation: instanced with base
// vertex / base instance and an index offset, indirect multi-draws, 8-bit indices, and a depth-only target.
TEST(tessellation_fast_path_draw_kinds) {
    const char* vs = R"(#version 460 core
layout(location = 0) in vec2 p;
layout(location = 1) in vec4 c;
out V { vec4 color; flat int id; } vo;
void main() {
    vo.color = c;
    vo.id = gl_InstanceID * 100000 + gl_BaseInstance * 10000 + gl_DrawID * 1000 + gl_BaseVertex * 10 + (gl_VertexID - gl_BaseVertex);
    gl_Position = vec4(p + vec2(float(gl_InstanceID), 0.0), 0.5 * c.r, 1.0);
})";
    GLuint p = program(vs, kTCS, kTES, kFS);
    GLuint bufs[2];
    GLuint vao = quadsVao(bufs);
    Target t;
    glUseProgram(p);
    glUniform1f(glGetUniformLocation(p, "lvl"), 1.0f);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    setenv("METALGLDRV_IMPLICIT_PATCHES", "1", 1);
    GLuint indirect, ib8;
    const uint32_t cmds[10] = {3, 1, 0, 0, 0, 3, 1, 3, 4, 0};  // count, instances, first index, base vertex, base instance
    glGenBuffers(1, &indirect);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, indirect);
    glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof cmds, cmds, GL_STATIC_DRAW);
    const uint8_t idx8[6] = {4, 5, 6, 6, 7, 4};
    glGenBuffers(1, &ib8);
    auto draws = [&]() {
        glDrawElementsInstancedBaseVertexBaseInstance(GL_TRIANGLES, 3, GL_UNSIGNED_SHORT, (void*)6, 2, 0, 3);  // quad A's 2 3 0, twice
        glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_SHORT, nullptr, 2, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib8);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx8, idx8, GL_STATIC_DRAW);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, nullptr);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufs[1]);
    };
    std::vector<float> ref = render(t, false, false, false, draws);
    uint64_t f0 = fusedDraws();
    CHECK_EQ(differing(ref, render(t, true, true, false, draws)), 0);
    CHECK_EQ(fusedDraws() - f0, 4u);  // the instanced draw, both indirect commands, the 8-bit one
    CHECK(at(ref, 8, 16)[2] == 3.0f * 10000 + 2.0f);       // instance 0, base instance 3, vertex 2
    CHECK(at(ref, 40, 16)[2] == 100000.0f + 30000 + 2.0f);  // instance 1 (moved right by 1)
    CHECK_EQ(differing(ref, render(t, false, true, true, draws)), 0);
    // depth-only target (a shadow pass)
    GLuint fbo, depth;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenTextures(1, &depth);
    glBindTexture(GL_TEXTURE_2D, depth);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH_COMPONENT32F, 64, 64);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    CHECK(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
    glEnable(GL_DEPTH_TEST);
    auto depthPass = [&](bool fast) {
        setenv("METALGLDRV_TESS_FAST", fast ? "1" : "0", 1);
        glClearDepth(1.0);
        glClear(GL_DEPTH_BUFFER_BIT);
        draws();
        std::vector<float> d(64 * 64);
        glReadPixels(0, 0, 64, 64, GL_DEPTH_COMPONENT, GL_FLOAT, d.data());
        unsetenv("METALGLDRV_TESS_FAST");
        return d;
    };
    std::vector<float> dref = depthPass(false), dfast = depthPass(true);
    CHECK(memcmp(dref.data(), dfast.data(), dref.size() * 4) == 0);
    CHECK(dref[16 * 64 + 8] < 1.0f);
    glDisable(GL_DEPTH_TEST);
    glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &depth);
    unsetenv("METALGLDRV_IMPLICIT_PATCHES");
    glDeleteBuffers(1, &indirect);
    glDeleteBuffers(1, &ib8);
    glDeleteBuffers(2, bufs);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    CHECK_GL();
}

// One batched multi-draw of more patches than Apple GPUs take in one object/mesh grid (from 2^19 object
// threadgroups a draw takes seconds and faults or trips the watchdog): the emulation draws the grid in chunks.
// Every patch is a tiny triangle at its own depth, so the image doesn't depend on the order; the emulation must
// match the fast path and finish quickly.
TEST(tessellation_large_batch) {
    const char* vs = R"(#version 460 core
layout(location = 0) in vec3 p;
out V { vec4 color; flat int id; } vo;
void main() { vo.color = vec4(p.z, fract(p.z * 256.0), 0.0, 1.0); vo.id = gl_VertexID; gl_Position = vec4(p.xy, p.z, 1.0); })";
    GLuint prog = program(vs, kTCS, kTES, kFS);
    glUseProgram(prog);
    glUniform1f(glGetUniformLocation(prog, "lvl"), 1.0f);
    const int patches = 1000000, half = patches / 2;  // ~300 MB of payload in one grid
    std::vector<float> v((size_t)patches * 9);
    uint32_t rng = 1;
    for (int i = 0; i < patches; i++) {
        rng = rng * 1664525u + 1013904223u;
        float x = (float)(rng >> 8 & 1023) / 512.0f - 1.0f, y = (float)(rng >> 18) / 8192.0f - 1.0f;
        float z = (float)i / patches;  // unique depth
        const float tri[9] = {x, y, z, x + 0.03f, y, z, x, y + 0.03f, z};
        memcpy(&v[(size_t)i * 9], tri, sizeof tri);
    }
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(v.size() * 4), v.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, nullptr);
    glEnableVertexAttribArray(0);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    Target t;
    GLuint depth;
    glGenRenderbuffers(1, &depth);
    glBindRenderbuffer(GL_RENDERBUFFER, depth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, 64, 64);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);
    glEnable(GL_DEPTH_TEST);
    auto draw = [&]() {
        glClear(GL_DEPTH_BUFFER_BIT);
        const GLint first[2] = {0, half * 3};
        const GLsizei count[2] = {half * 3, half * 3};
        glMultiDrawArrays(GL_PATCHES, first, count, 2);
    };
    std::vector<float> fast = render(t, true, true, false, draw);
    auto t0 = std::chrono::steady_clock::now();
    uint64_t p0 = passes();
    std::vector<float> emu = render(t, false, true, false, draw);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    CHECK_EQ(passes() - p0, 1u);
    CHECK(ms < 3000.0);
    CHECK_EQ(differing(fast, emu), 0);
    CHECK(at(emu, 32, 32)[0] != -1.0f);
    glDisable(GL_DEPTH_TEST);
    glDeleteRenderbuffers(1, &depth);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);
    CHECK_GL();
}

// Indices far past the vertex buffers (GL leaves the values undefined; nothing may fault): the GPU-side index
// reads of the fast path and the emulation fetch the attributes' default values instead, and later draws work.
TEST(tessellation_garbage_indices) {
    GLuint p = program(kVS, kTCS, kTES, kFS);
    GLuint bufs[2];
    GLuint vao = quadsVao(bufs);
    Target t;
    glUseProgram(p);
    glUniform1f(glGetUniformLocation(p, "lvl"), 1.0f);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    GLuint ib;
    const uint32_t bad[6] = {0x7fffffffu, 0xfffffff0u, 123456789u, 4000000000u, 1u << 30, 99999u};
    glGenBuffers(1, &ib);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof bad, bad, GL_STATIC_DRAW);
    for (bool fast : {true, false}) {
        render(t, fast, true, false, [&]() {
            const GLsizei counts[2] = {3, 3};
            const void* offs[2] = {nullptr, (void*)12};
            const GLint bv[2] = {0, 1 << 20};
            glMultiDrawElementsBaseVertex(GL_PATCHES, counts, GL_UNSIGNED_INT, offs, 2, bv);
        });
    }
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufs[1]);
    std::vector<float> after = render(t, true, true, false, [&]() { glDrawElements(GL_PATCHES, 6, GL_UNSIGNED_SHORT, nullptr); });
    CHECK(at(after, 12, 8)[2] == 0.0f);  // quad A drawn normally: control point 0's id
    glDeleteBuffers(1, &ib);
    glDeleteBuffers(2, bufs);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    CHECK_GL();
}
