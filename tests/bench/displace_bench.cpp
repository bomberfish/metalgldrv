// displacebench: displacement-mapped tessellation like Unigine Heaven's materials, as a benchmark that
// runs on both drivers: built as `displacebench` (linked against metalgldrv's CGL) and
// `displacebench-apple` (linked against OpenGL.framework).
//
// A perspective view of a terrain grid drawn as `--draws` glDrawElements(GL_PATCHES) calls; the TCS
// (Heaven's) culls patches outside the view and sets levels from per-vertex weights times an edge-length
// factor, the TES (Heaven's) displaces along the normal by a height texture fetched with textureLod,
// which makes it non-affine in gl_TessCoord. The fragment shader writes two G-buffer targets.
//
//   displacebench [--frames N] [--warmup N] [--grid N] [--draws N] [--weight W] [--notess] [--size WxH]
//
// --grid N: N x N quads (2 N^2 triangles = patches); --weight W: per-vertex weights in [W/2, W], so
// edge levels up to 2W (1 gives about one triangle per patch); --notess: the same geometry with VS + FS.
// Reports GPU time (GL_TIME_ELAPSED around the frame) and wall time (issue + glFinish) per frame.
#include <OpenGL/OpenGL.h>
#include <OpenGL/gl3.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

const char* kVS = R"(#version 410 core
layout(location = 0) in vec4 s_attribute_0;  // position, w = 1
layout(location = 1) in vec4 s_attribute_1;  // texcoord
layout(location = 2) in vec4 s_attribute_2;  // normal, w = tessellation weight
uniform mat4 s_projection;
out vec4 s_texcoord_0;
out vec4 s_texcoord_1;
out vec4 s_texcoord_9;
void main() {
    gl_Position = s_attribute_0;
    s_texcoord_0 = s_attribute_1;
    s_texcoord_1 = vec4(s_attribute_2.xyz * 0.5 + 0.5, 1.0);
    s_texcoord_9 = s_attribute_2;
}
)";

// Without tessellation the VS projects.
const char* kVSNoTess = R"(#version 410 core
layout(location = 0) in vec4 s_attribute_0;
layout(location = 1) in vec4 s_attribute_1;
layout(location = 2) in vec4 s_attribute_2;
uniform mat4 s_projection;
out vec4 s_texcoord_0;
out vec4 s_texcoord_1;
out vec4 s_texcoord_2;
void main() {
    gl_Position = s_projection * s_attribute_0;
    s_texcoord_0 = s_attribute_1;
    s_texcoord_1 = vec4(s_attribute_2.xyz, 0.0);
    s_texcoord_2 = vec4(s_attribute_2.xyz * 0.5 + 0.5, 1.0);
}
)";

const char* kTCS = R"(#version 410 core
layout(vertices = 3) out;
uniform mat4 s_projection;
uniform vec3 s_material_tessellation_distance;
in vec4 s_texcoord_0[];
in vec4 s_texcoord_1[];
in vec4 s_texcoord_9[];
out vec4 s_tess_texcoord_0[];
out vec4 s_tess_texcoord_1[];
out vec4 s_tess_texcoord_9[];
vec4 getPosition(vec4 vertex) { return s_projection * vertex; }
void main() {
    vec4 p0 = getPosition(gl_in[0].gl_Position);
    vec4 p1 = getPosition(gl_in[1].gl_Position);
    vec4 p2 = getPosition(gl_in[2].gl_Position);
    p0.xy /= p0.w;
    p1.xy /= p1.w;
    p2.xy /= p2.w;
    if (!all(lessThan(vec4(min(min(p0.xy, p1.xy), p2.xy), -max(max(p0.xy, p1.xy), p2.xy)), vec4(1.2)))) {
        gl_TessLevelOuter[0] = 0.0;
        gl_TessLevelOuter[1] = 0.0;
        gl_TessLevelOuter[2] = 0.0;
        gl_TessLevelInner[0] = 0.0;
        return;
    }
    float l0 = length(gl_in[1].gl_Position.xyz - gl_in[2].gl_Position.xyz);
    float l1 = length(gl_in[2].gl_Position.xyz - gl_in[0].gl_Position.xyz);
    float l2 = length(gl_in[0].gl_Position.xyz - gl_in[1].gl_Position.xyz);
    float f0 = (s_texcoord_9[1].w + s_texcoord_9[2].w) * clamp(l0 * s_material_tessellation_distance.z, 0.0, 1.0);
    float f1 = (s_texcoord_9[2].w + s_texcoord_9[0].w) * clamp(l1 * s_material_tessellation_distance.z, 0.0, 1.0);
    float f2 = (s_texcoord_9[0].w + s_texcoord_9[1].w) * clamp(l2 * s_material_tessellation_distance.z, 0.0, 1.0);
    gl_TessLevelOuter[0] = f0;
    gl_TessLevelOuter[1] = f1;
    gl_TessLevelOuter[2] = f2;
    gl_TessLevelInner[0] = (f0 + f1 + f2) * (1.0 / 3.0);
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    s_tess_texcoord_0[gl_InvocationID] = s_texcoord_0[gl_InvocationID];
    s_tess_texcoord_1[gl_InvocationID] = s_texcoord_1[gl_InvocationID];
    s_tess_texcoord_9[gl_InvocationID] = s_texcoord_9[gl_InvocationID];
}
)";

const char* kTES = R"(#version 410 core
layout(triangles, fractional_odd_spacing, ccw) in;
uniform mat4 s_projection;
uniform vec3 s_camera_position;
uniform vec3 s_material_tessellation_distance;
uniform vec4 s_material_tessellation_factor;
uniform sampler2D s_texture_20;
in vec4 s_tess_texcoord_0[];
in vec4 s_tess_texcoord_1[];
in vec4 s_tess_texcoord_9[];
out vec4 s_texcoord_0;
out vec4 s_texcoord_1;
out vec4 s_texcoord_2;
void main() {
    vec4 vertex = gl_in[0].gl_Position * gl_TessCoord.x + gl_in[1].gl_Position * gl_TessCoord.y + gl_in[2].gl_Position * gl_TessCoord.z;
    vec3 normal = s_tess_texcoord_9[0].xyz * gl_TessCoord.x + s_tess_texcoord_9[1].xyz * gl_TessCoord.y + s_tess_texcoord_9[2].xyz * gl_TessCoord.z;
    vec2 texcoord = s_tess_texcoord_0[0].xy * gl_TessCoord.x + s_tess_texcoord_0[1].xy * gl_TessCoord.y + s_tess_texcoord_0[2].xy * gl_TessCoord.z;
    float distance = length(vertex.xyz - s_camera_position);
    float lod = clamp((distance - s_material_tessellation_distance.x) * s_material_tessellation_distance.y, 0.0, 8.0);
    float offset = pow(clamp(1.0 - (distance - s_material_tessellation_distance.x) * s_material_tessellation_distance.y, 0.0, 1.0),
                       s_material_tessellation_factor.y);
    offset *= (textureLod(s_texture_20, texcoord, lod).x - 127.0 / 255.0) * s_material_tessellation_factor.x;
    vertex.xyz += normal * offset;
    gl_Position = s_projection * vertex;
    s_texcoord_0.xy = texcoord;
    s_texcoord_0.zw = s_tess_texcoord_0[0].zw * gl_TessCoord.x + s_tess_texcoord_0[1].zw * gl_TessCoord.y + s_tess_texcoord_0[2].zw * gl_TessCoord.z;
    s_texcoord_1 = vec4(normal, 0.0);
    s_texcoord_2 = s_tess_texcoord_1[0] * gl_TessCoord.x + s_tess_texcoord_1[1] * gl_TessCoord.y + s_tess_texcoord_1[2] * gl_TessCoord.z;
}
)";

const char* kFS = R"(#version 410 core
uniform sampler2D s_texture_20;
in vec4 s_texcoord_0;
in vec4 s_texcoord_1;
in vec4 s_texcoord_2;
layout(location = 0) out vec4 s_frag_data_0;
layout(location = 1) out vec4 s_frag_data_1;
void main() {
    float h = texture(s_texture_20, s_texcoord_0.xy * 4.0).x;
    s_frag_data_0 = vec4(s_texcoord_2.xyz * (0.5 + h), 1.0);
    s_frag_data_1 = vec4(normalize(s_texcoord_1.xyz) * 0.5 + 0.5, h);
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

// Column-major perspective * look-at.
void viewProjection(float* m, float aspect, const float eye[3], const float at[3]) {
    float f[3] = {at[0] - eye[0], at[1] - eye[1], at[2] - eye[2]};
    float fl = sqrtf(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
    for (float& x : f) x /= fl;
    const float up[3] = {0, 0, 1};
    float s[3] = {f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0]};
    float sl = sqrtf(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
    for (float& x : s) x /= sl;
    float u[3] = {s[1] * f[2] - s[2] * f[1], s[2] * f[0] - s[0] * f[2], s[0] * f[1] - s[1] * f[0]};
    float view[16] = {s[0], u[0], -f[0], 0, s[1], u[1], -f[1], 0, s[2], u[2], -f[2], 0, 0, 0, 0, 1};
    view[12] = -(s[0] * eye[0] + s[1] * eye[1] + s[2] * eye[2]);
    view[13] = -(u[0] * eye[0] + u[1] * eye[1] + u[2] * eye[2]);
    view[14] = f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2];
    float n = 0.5f, fa = 2000.0f, t = 1.0f / tanf(0.5f * 60.0f * 3.14159265f / 180.0f);
    float proj[16] = {t / aspect, 0, 0, 0, 0, t, 0, 0, 0, 0, (fa + n) / (n - fa), -1, 0, 0, 2 * fa * n / (n - fa), 0};
    for (int c = 0; c < 4; c++)
        for (int r = 0; r < 4; r++) {
            float acc = 0;
            for (int k = 0; k < 4; k++) acc += proj[k * 4 + r] * view[c * 4 + k];
            m[c * 4 + r] = acc;
        }
}

}  // namespace

int main(int argc, char** argv) {
    int frames = 60, warmup = 10, grid = 400, draws = 300, width = 1920, height = 1080;
    float weight = 4.0f;
    bool notess = false;
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
        else if (a == "--grid") grid = atoi(next());
        else if (a == "--draws") draws = atoi(next());
        else if (a == "--weight") weight = (float)atof(next());
        else if (a == "--notess") notess = true;
        else if (a == "--size") {
            if (sscanf(next(), "%dx%d", &width, &height) != 2) return 2;
        } else {
            fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }

    CGLPixelFormatAttribute attrs[] = {kCGLPFAOpenGLProfile, (CGLPixelFormatAttribute)kCGLOGLPVersion_3_2_Core, kCGLPFAAccelerated,
                                       (CGLPixelFormatAttribute)0};
    CGLPixelFormatObj pf = nullptr;
    GLint npf = 0;
    if (CGLChoosePixelFormat(attrs, &pf, &npf) != kCGLNoError || !pf) {
        fprintf(stderr, "no pixel format\n");
        return 1;
    }
    CGLContextObj cgl = nullptr;
    if (CGLCreateContext(pf, nullptr, &cgl) != kCGLNoError) return 1;
    CGLDestroyPixelFormat(pf);
    CGLSetCurrentContext(cgl);

    GLuint prog = notess ? link({compile(GL_VERTEX_SHADER, kVSNoTess), compile(GL_FRAGMENT_SHADER, kFS)})
                         : link({compile(GL_VERTEX_SHADER, kVS), compile(GL_TESS_CONTROL_SHADER, kTCS),
                                 compile(GL_TESS_EVALUATION_SHADER, kTES), compile(GL_FRAGMENT_SHADER, kFS)});
    glUseProgram(prog);
    float vp[16];
    const float eye[3] = {-20.0f, -20.0f, 25.0f}, at[3] = {60.0f, 60.0f, 0.0f};
    viewProjection(vp, (float)width / height, eye, at);
    glUniformMatrix4fv(glGetUniformLocation(prog, "s_projection"), 1, GL_FALSE, vp);
    glUniform3f(glGetUniformLocation(prog, "s_camera_position"), eye[0], eye[1], eye[2]);
    // start fading at 10 units over 100; full edge factor at edges of 1 unit
    glUniform3f(glGetUniformLocation(prog, "s_material_tessellation_distance"), 10.0f, 1.0f / 100.0f, 1.0f);
    glUniform4f(glGetUniformLocation(prog, "s_material_tessellation_factor"), 0.6f, 1.0f, 0.0f, 0.0f);
    glUniform1i(glGetUniformLocation(prog, "s_texture_20"), 0);

    // height texture: 1024^2 noise, mipmapped
    std::vector<uint8_t> heights(1024 * 1024);
    uint32_t rng = 12345;
    auto rnd = [&]() { return (rng = rng * 1664525u + 1013904223u) >> 8; };
    for (int y = 0; y < 1024; y++)
        for (int x = 0; x < 1024; x++)
            heights[y * 1024 + x] = (uint8_t)(127 + 60 * sinf(x * 0.05f) * cosf(y * 0.07f) + (rnd() % 40) - 20);
    GLuint htex;
    glGenTextures(1, &htex);
    glBindTexture(GL_TEXTURE_2D, htex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, 1024, 1024, 0, GL_RED, GL_UNSIGNED_BYTE, heights.data());
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    // G-buffer: two RGBA8 targets + depth
    GLuint fbo, tex[2], depth;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenTextures(2, tex);
    for (int i = 0; i < 2; i++) {
        glBindTexture(GL_TEXTURE_2D, tex[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, tex[i], 0);
    }
    glBindTexture(GL_TEXTURE_2D, htex);
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

    // terrain grid: (grid+1)^2 vertices over [0, 120]^2, 2 grid^2 triangles in `draws` index ranges
    struct Vertex {
        float pos[4], uv[4], normal[4];
    };
    std::vector<Vertex> verts((size_t)(grid + 1) * (grid + 1));
    const float span = 120.0f;
    for (int y = 0; y <= grid; y++)
        for (int x = 0; x <= grid; x++) {
            Vertex& v = verts[(size_t)y * (grid + 1) + x];
            float fx = span * x / grid, fy = span * y / grid;
            float w = weight * (0.5f + 0.5f * (rnd() % 1000) / 999.0f);
            v = {{fx, fy, 0.5f * sinf(fx * 0.2f) * cosf(fy * 0.15f), 1}, {fx / 8, fy / 8, (float)x / grid, (float)y / grid}, {0, 0, 1, w}};
        }
    std::vector<uint32_t> idx;
    idx.reserve((size_t)grid * grid * 6);
    for (int y = 0; y < grid; y++)
        for (int x = 0; x < grid; x++) {
            uint32_t a = y * (grid + 1) + x, b = a + 1, c = a + grid + 1, d = c + 1;
            const uint32_t q[6] = {a, b, d, a, d, c};
            idx.insert(idx.end(), q, q + 6);
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
    for (int l = 0; l < 3; l++) {
        glVertexAttribPointer(l, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(size_t)(l * 16));
        glEnableVertexAttribArray(l);
    }
    if (!notess) glPatchParameteri(GL_PATCH_VERTICES, 3);
    const size_t tris = idx.size() / 3, perDraw = (tris + draws - 1) / draws;

    GLuint query;
    glGenQueries(1, &query);
    std::vector<double> gpuMs, wallMs;
    for (int f = 0; f < warmup + frames; f++) {
        auto t0 = std::chrono::steady_clock::now();
        glBeginQuery(GL_TIME_ELAPSED, query);
        glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        for (size_t first = 0; first < tris; first += perDraw) {
            size_t n = std::min(perDraw, tris - first);
            glDrawElements(notess ? GL_TRIANGLES : GL_PATCHES, (GLsizei)(n * 3), GL_UNSIGNED_INT, (void*)(first * 12));
        }
        glEndQuery(GL_TIME_ELAPSED);
        glFinish();
        auto t1 = std::chrono::steady_clock::now();
        GLuint64 ns = 0;
        glGetQueryObjectui64v(query, GL_QUERY_RESULT, &ns);
        if (GLenum e = glGetError()) {
            fprintf(stderr, "GL error 0x%x\n", e);
            return 1;
        }
        if (f < warmup) continue;
        wallMs.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        gpuMs.push_back(ns / 1e6);
    }
    uint8_t px[4] = {};
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(width / 2, height / 3, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    Stats g = stats(gpuMs), w = stats(wallMs);
    printf("displacebench (%s): %s, %zu patches in %d draws, weight %g, %dx%d, %d frames\n", (const char*)glGetString(GL_RENDERER),
           notess ? "VS+FS" : "VS+TCS+TES+FS", tris, draws, weight, width, height, frames);
    printf("  gpu  ms/frame: median %8.3f  mean %8.3f  min %8.3f\n", g.median, g.mean, g.min);
    printf("  wall ms/frame: median %8.3f  mean %8.3f  min %8.3f\n", w.median, w.mean, w.min);
    printf("  probe pixel: %u %u %u %u\n", px[0], px[1], px[2], px[3]);
    CGLSetCurrentContext(nullptr);
    CGLDestroyContext(cgl);
    return 0;
}
