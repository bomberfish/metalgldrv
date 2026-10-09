// tessbench: Sodium + Iris terrain with a tessellation shader pack (Shrimple), as a benchmark.
//
// Every frame issues `--calls` glMultiDrawElementsBaseVertex(GL_TRIANGLES) calls of `--subdraws`
// sub-draws of `--tris` triangles each, all from one shared quad index buffer (as Sodium does), with
// a VS + TCS + TES + FS program like Shrimple's gbuffers_terrain: the TCS passes the control points
// through and writes its levels from a uniform (`--level`, 1 by default, which makes tessellation
// a no-op), the TES (triangles, fractional_odd_spacing, ccw) interpolates the inputs at
// gl_TessCoord. GL_TRIANGLES draws run as patches through METALGLDRV_IMPLICIT_PATCHES (set here).
//
//   tessbench [--frames N] [--warmup N] [--calls N] [--subdraws N] [--tris N] [--level L]
//             [--notess] [--single] [--size WxH]
//
// --notess draws the same geometry with a VS + FS program (the cost without tessellation);
// --single issues one glDrawElementsBaseVertex per sub-draw instead of the multi-draws.
// Reports per-frame CPU time (issuing the frame's GL calls), GPU time (GL_TIME_ELAPSED around the
// frame) and wall time (CPU + glFinish), as median / mean / min over the measured frames.
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include "glapi.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

const char* kVS = R"(#version 410 core
layout(location = 0) in vec3 a_Pos;      // unsigned shorts, region-local, 1/32 block units
layout(location = 1) in vec4 a_Color;    // unsigned bytes, normalized
layout(location = 2) in vec2 a_TexCoord; // unsigned shorts, normalized
layout(location = 3) in ivec2 a_Light;   // unsigned bytes, integer
uniform mat4 u_ModelViewProjection;
uniform vec3 u_RegionOffset;
out VertexData {
    vec4 color;
    vec2 lmcoord;
    vec2 texcoord;
    vec3 localPos;
    vec2 localCoord;
    vec3 localNormal;
    vec4 localTangent;
    flat int blockId;
    flat mat2 atlasBounds;
} vOut;
void main() {
    vec3 pos = a_Pos * (1.0 / 32.0) + u_RegionOffset;
    vOut.color = a_Color;
    vOut.lmcoord = vec2(a_Light) / 255.0;
    vOut.texcoord = a_TexCoord;
    vOut.localPos = pos;
    vOut.localCoord = fract(a_TexCoord * 64.0);
    vOut.localNormal = normalize(vec3(a_Color.rg - 0.5, 1.0));
    vOut.localTangent = vec4(1.0, 0.0, 0.0, 1.0);
    vOut.blockId = gl_VertexID & 255;
    vOut.atlasBounds = mat2(a_TexCoord, a_TexCoord + 1.0 / 64.0);
    gl_Position = u_ModelViewProjection * vec4(pos, 1.0);
}
)";

const char* kTCS = R"(#version 410 core
layout(vertices = 3) out;
uniform float u_TessLevel;
in VertexData {
    vec4 color;
    vec2 lmcoord;
    vec2 texcoord;
    vec3 localPos;
    vec2 localCoord;
    vec3 localNormal;
    vec4 localTangent;
    flat int blockId;
    flat mat2 atlasBounds;
} vIn[];
out VertexData {
    vec4 color;
    vec2 lmcoord;
    vec2 texcoord;
    vec3 localPos;
    vec2 localCoord;
    vec3 localNormal;
    vec4 localTangent;
    flat int blockId;
    flat mat2 atlasBounds;
} vOut[];
void main() {
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    vOut[gl_InvocationID].color = vIn[gl_InvocationID].color;
    vOut[gl_InvocationID].lmcoord = vIn[gl_InvocationID].lmcoord;
    vOut[gl_InvocationID].texcoord = vIn[gl_InvocationID].texcoord;
    vOut[gl_InvocationID].localPos = vIn[gl_InvocationID].localPos;
    vOut[gl_InvocationID].localCoord = vIn[gl_InvocationID].localCoord;
    vOut[gl_InvocationID].localNormal = vIn[gl_InvocationID].localNormal;
    vOut[gl_InvocationID].localTangent = vIn[gl_InvocationID].localTangent;
    vOut[gl_InvocationID].blockId = vIn[gl_InvocationID].blockId;
    vOut[gl_InvocationID].atlasBounds = vIn[gl_InvocationID].atlasBounds;
    if (gl_InvocationID == 0) {
        gl_TessLevelOuter[0] = u_TessLevel;
        gl_TessLevelOuter[1] = u_TessLevel;
        gl_TessLevelOuter[2] = u_TessLevel;
        gl_TessLevelInner[0] = u_TessLevel;
    }
}
)";

const char* kTES = R"(#version 410 core
layout(triangles, fractional_odd_spacing, ccw) in;
in VertexData {
    vec4 color;
    vec2 lmcoord;
    vec2 texcoord;
    vec3 localPos;
    vec2 localCoord;
    vec3 localNormal;
    vec4 localTangent;
    flat int blockId;
    flat mat2 atlasBounds;
} vIn[];
out VertexData {
    vec4 color;
    vec2 lmcoord;
    vec2 texcoord;
    vec3 localPos;
    vec2 localCoord;
    vec3 localNormal;
    vec4 localTangent;
    flat int blockId;
    flat mat2 atlasBounds;
} vOut;
#define _interpolate(v0, v1, v2) (gl_TessCoord.x * (v0) + gl_TessCoord.y * (v1) + gl_TessCoord.z * (v2))
void main() {
    gl_Position = _interpolate(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_in[2].gl_Position);
    vOut.color = _interpolate(vIn[0].color, vIn[1].color, vIn[2].color);
    vOut.lmcoord = _interpolate(vIn[0].lmcoord, vIn[1].lmcoord, vIn[2].lmcoord);
    vOut.texcoord = _interpolate(vIn[0].texcoord, vIn[1].texcoord, vIn[2].texcoord);
    vOut.localPos = _interpolate(vIn[0].localPos, vIn[1].localPos, vIn[2].localPos);
    vOut.localCoord = _interpolate(vIn[0].localCoord, vIn[1].localCoord, vIn[2].localCoord);
    vOut.localNormal = _interpolate(vIn[0].localNormal, vIn[1].localNormal, vIn[2].localNormal);
    vOut.localTangent = _interpolate(vIn[0].localTangent, vIn[1].localTangent, vIn[2].localTangent);
    vOut.blockId = vIn[0].blockId;
    vOut.atlasBounds = vIn[0].atlasBounds;
}
)";

const char* kFS = R"(#version 410 core
in VertexData {
    vec4 color;
    vec2 lmcoord;
    vec2 texcoord;
    vec3 localPos;
    vec2 localCoord;
    vec3 localNormal;
    vec4 localTangent;
    flat int blockId;
    flat mat2 atlasBounds;
} vIn;
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outNormal;
void main() {
    vec2 uv = vIn.atlasBounds * vIn.localCoord;
    float shade = 0.5 + 0.5 * dot(vIn.localNormal, normalize(vec3(0.3, 0.8, 0.5)));
    outColor = vec4(vIn.color.rgb * shade * (0.5 + 0.5 * vIn.lmcoord.x), 1.0) + vec4(uv * 0.0001, float(vIn.blockId) * 1e-6, 0.0);
    outNormal = vec4(vIn.localNormal * 0.5 + 0.5, vIn.localTangent.w * vIn.texcoord.x);
}
)";

GLuint compile(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096] = {};
        glGetShaderInfoLog(s, sizeof log, nullptr, log);
        fprintf(stderr, "compile failed: %s\n", log);
        exit(1);
    }
    return s;
}

GLuint link(std::initializer_list<GLuint> shaders) {
    GLuint p = glCreateProgram();
    for (GLuint s : shaders) glAttachShader(p, s);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096] = {};
        glGetProgramInfoLog(p, sizeof log, nullptr, log);
        fprintf(stderr, "link failed: %s\n", log);
        exit(1);
    }
    for (GLuint s : shaders) glDeleteShader(s);
    return p;
}

struct Stats {
    double median, mean, min;
};
Stats stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (double x : v) sum += x;
    return {v[v.size() / 2], sum / v.size(), v.front()};
}

// Sodium's vertex: 20 bytes.
struct Vertex {
    uint16_t pos[3];
    uint16_t pad;
    uint8_t color[4];
    uint16_t uv[2];
    uint8_t light[2];
    uint16_t pad2;
};

}  // namespace

int main(int argc, char** argv) {
    int frames = 50, warmup = 10, calls = 300, subdraws = 7, tris = 500, width = 1920, height = 1080;
    float level = 1.0f;
    bool notess = false, single = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s needs a value\n", a.c_str());
                exit(2);
            }
            return argv[++i];
        };
        if (a == "--frames") frames = atoi(next());
        else if (a == "--warmup") warmup = atoi(next());
        else if (a == "--calls") calls = atoi(next());
        else if (a == "--subdraws") subdraws = atoi(next());
        else if (a == "--tris") tris = atoi(next());
        else if (a == "--level") level = (float)atof(next());
        else if (a == "--notess") notess = true;
        else if (a == "--single") single = true;
        else if (a == "--size") {
            if (sscanf(next(), "%dx%d", &width, &height) != 2) return 2;
        } else {
            fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    tris &= ~1;  // whole quads
    setenv("METALGLDRV_IMPLICIT_PATCHES", "1", 1);

    EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (!eglInitialize(dpy, nullptr, nullptr)) return 2;
    eglBindAPI(EGL_OPENGL_API);
    EGLint cfgAttr[] = {EGL_RED_SIZE, 8, EGL_DEPTH_SIZE, 24, EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_NONE};
    EGLConfig cfg;
    EGLint n;
    eglChooseConfig(dpy, cfgAttr, &cfg, 1, &n);
    EGLint pb[] = {EGL_WIDTH, 64, EGL_HEIGHT, 64, EGL_NONE};
    EGLSurface surf = eglCreatePbufferSurface(dpy, cfg, pb);
    EGLint ctxAttr[] = {EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 6, EGL_CONTEXT_OPENGL_PROFILE_MASK,
                        EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL_NONE};
    EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ctxAttr);
    eglMakeCurrent(dpy, surf, surf, ctx);

    GLuint prog = notess ? link({compile(GL_VERTEX_SHADER, kVS), compile(GL_FRAGMENT_SHADER, kFS)})
                         : link({compile(GL_VERTEX_SHADER, kVS), compile(GL_TESS_CONTROL_SHADER, kTCS),
                                 compile(GL_TESS_EVALUATION_SHADER, kTES), compile(GL_FRAGMENT_SHADER, kFS)});
    glUseProgram(prog);
    GLint uMvp = glGetUniformLocation(prog, "u_ModelViewProjection");
    GLint uRegion = glGetUniformLocation(prog, "u_RegionOffset");
    GLint uLevel = glGetUniformLocation(prog, "u_TessLevel");
    // orthographic: x, y in [0, 64) blocks, z in [0, 64)
    const float mvp[16] = {2.0f / 64, 0, 0, 0, 0, 2.0f / 64, 0, 0, 0, 0, -2.0f / 64, 0, -1, -1, 1, 1};
    glUniformMatrix4fv(uMvp, 1, GL_FALSE, mvp);
    if (uLevel >= 0) glUniform1f(uLevel, level);

    // G-buffer: two RGBA8 colour attachments + depth
    GLuint fbo, tex[2], depth;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenTextures(2, tex);
    for (int i = 0; i < 2; i++) {
        glBindTexture(GL_TEXTURE_2D, tex[i]);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, width, height);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, tex[i], 0);
    }
    glGenRenderbuffers(1, &depth);
    glBindRenderbuffer(GL_RENDERBUFFER, depth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);
    GLenum bufs[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    glDrawBuffers(2, bufs);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "framebuffer incomplete\n");
        return 1;
    }
    glViewport(0, 0, width, height);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    // Geometry: every sub-draw is a run of small quads (one "facing" of a chunk section); the
    // vertex data of all sub-draws lives in one buffer, addressed by base vertex.
    const int quads = tris / 2, vertsPerDraw = quads * 4, totalDraws = calls * subdraws;
    std::vector<Vertex> verts((size_t)totalDraws * vertsPerDraw);
    uint32_t rng = 12345;
    auto rnd = [&]() { return (rng = rng * 1664525u + 1013904223u) >> 8; };
    for (size_t q = 0; q < verts.size() / 4; q++) {
        uint16_t x = (uint16_t)(rnd() % (16 * 32)), y = (uint16_t)(rnd() % (16 * 32)), z = (uint16_t)(rnd() % (64 * 32));
        uint16_t s = (uint16_t)(8 + rnd() % 16);  // quads of 1/4 .. 3/4 block
        uint8_t r = (uint8_t)rnd(), g = (uint8_t)rnd(), b = (uint8_t)rnd();
        const uint16_t dx[4] = {0, s, s, 0}, dy[4] = {0, 0, s, s};
        for (int k = 0; k < 4; k++) {
            Vertex& v = verts[q * 4 + k];
            v = {};
            v.pos[0] = x + dx[k];
            v.pos[1] = y + dy[k];
            v.pos[2] = z;
            v.color[0] = r, v.color[1] = g, v.color[2] = b, v.color[3] = 255;
            v.uv[0] = (uint16_t)(dx[k] * 100), v.uv[1] = (uint16_t)(dy[k] * 100);
            v.light[0] = 240, v.light[1] = (uint8_t)(k * 60);
        }
    }
    // Sodium's shared index buffer: 0 1 2 2 3 0, 4 5 6 6 7 4, ...
    std::vector<uint32_t> idx((size_t)quads * 6);
    for (int q = 0; q < quads; q++) {
        const uint32_t o[6] = {0, 1, 2, 2, 3, 0};
        for (int k = 0; k < 6; k++) idx[(size_t)q * 6 + k] = (uint32_t)q * 4 + o[k];
    }
    GLuint vao, vbo, ibo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(verts.size() * sizeof(Vertex)), verts.data(), GL_STATIC_DRAW);
    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)(idx.size() * 4), idx.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_UNSIGNED_SHORT, GL_FALSE, sizeof(Vertex), (void*)0);
    glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(Vertex), (void*)8);
    glVertexAttribPointer(2, 2, GL_UNSIGNED_SHORT, GL_TRUE, sizeof(Vertex), (void*)12);
    glVertexAttribIPointer(3, 2, GL_UNSIGNED_BYTE, sizeof(Vertex), (void*)16);
    for (int l = 0; l < 4; l++) glEnableVertexAttribArray(l);
    if (!notess) glPatchParameteri(GL_PATCH_VERTICES, 3);

    std::vector<GLsizei> counts(subdraws, (GLsizei)idx.size());
    std::vector<const void*> offsets(subdraws, nullptr);
    std::vector<GLint> baseVertex(subdraws);
    GLuint query;
    glGenQueries(1, &query);
    std::vector<double> cpuMs, gpuMs, wallMs;
    for (int f = 0; f < warmup + frames; f++) {
        auto t0 = std::chrono::steady_clock::now();
        glBeginQuery(GL_TIME_ELAPSED, query);
        glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        for (int c = 0; c < calls; c++) {
            glUniform3f(uRegion, (float)(c % 4) * 16.0f, (float)(c / 4 % 4) * 16.0f, 0.0f);
            for (int s = 0; s < subdraws; s++) baseVertex[s] = (c * subdraws + s) * vertsPerDraw;
            if (single) {
                for (int s = 0; s < subdraws; s++) glDrawElementsBaseVertex(GL_TRIANGLES, counts[s], GL_UNSIGNED_INT, offsets[s], baseVertex[s]);
            } else {
                glMultiDrawElementsBaseVertex(GL_TRIANGLES, counts.data(), GL_UNSIGNED_INT, offsets.data(), subdraws, baseVertex.data());
            }
        }
        glEndQuery(GL_TIME_ELAPSED);
        auto t1 = std::chrono::steady_clock::now();
        glFinish();
        auto t2 = std::chrono::steady_clock::now();
        GLuint64 ns = 0;
        glGetQueryObjectui64v(query, GL_QUERY_RESULT, &ns);
        if (GLenum e = glGetError()) {
            fprintf(stderr, "GL error 0x%x\n", e);
            return 1;
        }
        if (f < warmup) continue;
        cpuMs.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        wallMs.push_back(std::chrono::duration<double, std::milli>(t2 - t0).count());
        gpuMs.push_back(ns / 1e6);
    }
    uint8_t px[4] = {};
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(width / 4, height / 4, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    Stats c = stats(cpuMs), g = stats(gpuMs), w = stats(wallMs);
    printf("tessbench: %s%s, %d calls x %d sub-draws x %d triangles (%.2f M triangles/frame), level %g, %dx%d, %d frames\n",
           notess ? "VS+FS" : "VS+TCS+TES+FS", single ? ", single draws" : ", multi-draws", calls, subdraws, tris,
           (double)calls * subdraws * tris / 1e6, level, width, height, frames);
    printf("  cpu  ms/frame: median %8.3f  mean %8.3f  min %8.3f\n", c.median, c.mean, c.min);
    printf("  gpu  ms/frame: median %8.3f  mean %8.3f  min %8.3f\n", g.median, g.mean, g.min);
    printf("  wall ms/frame: median %8.3f  mean %8.3f  min %8.3f  (%.1f fps)\n", w.median, w.mean, w.min, 1000.0 / w.median);
    printf("  probe pixel: %u %u %u %u\n", px[0], px[1], px[2], px[3]);
    glDeleteQueries(1, &query);
    glDeleteProgram(prog);
    eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroyContext(dpy, ctx);
    eglDestroySurface(dpy, surf);
    eglTerminate(dpy);
    return 0;
}
