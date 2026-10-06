// End-to-end draw tests (GLSL -> MSL -> Metal 4).
#include "harness.h"
#include <cmath>
#include <vector>

static const char* kVS = R"(#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec4 aColor;
out vec4 vColor;
void main() { vColor = aColor; gl_Position = vec4(aPos, 0.0, 1.0); }
)";
static const char* kFS = R"(#version 330 core
in vec4 vColor;
out vec4 fragColor;
void main() { fragColor = vColor; }
)";

struct Vtx {
    float x, y;
    float r, g, b, a;
};

static GLuint makeVao(const Vtx* v, int n, GLuint* vboOut) {
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vtx) * n, v, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void*)0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void*)8);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    if (vboOut) *vboOut = vbo;
    return vao;
}

TEST(draw_triangle) {
    GLuint p = compileProgram(kVS, kFS);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    CHECK(ok);
    // bottom-left half of the viewport: triangle (-1,-1) (1,-1) (-1,1)
    Vtx v[3] = {{-1, -1, 1, 0, 0, 1}, {1, -1, 1, 0, 0, 1}, {-1, 1, 1, 0, 0, 1}};
    GLuint vbo, vao = makeVao(v, 3, &vbo);
    glClearColor(0, 0, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(p);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    CHECK_PIXEL(2, 2, 255, 0, 0, 255);      // bottom-left in GL coordinates
    CHECK_PIXEL(61, 61, 0, 0, 255, 255);    // top-right untouched
    CHECK_PIXEL(2, 60, 255, 0, 0, 255);     // near top-left corner (inside)
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    CHECK_GL();
}

TEST(draw_interpolation_and_viewport) {
    GLuint p = compileProgram(kVS, kFS);
    Vtx v[4] = {{-1, -1, 1, 0, 0, 1}, {1, -1, 0, 1, 0, 1}, {-1, 1, 0, 0, 1, 1}, {1, 1, 1, 1, 1, 1}};
    GLuint vbo, vao = makeVao(v, 4, &vbo);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(p);
    glViewport(32, 0, 32, 32);  // bottom-right quadrant
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glViewport(0, 0, 64, 64);
    uint8_t px[4];
    readPixel(33, 1, px);
    CHECK(px[0] > 200 && px[1] < 40 && px[2] < 40);  // near red corner
    readPixel(62, 1, px);
    CHECK(px[1] > 200 && px[0] < 40);                // near green corner
    readPixel(33, 30, px);
    CHECK(px[2] > 200 && px[0] < 40);                // near blue corner
    CHECK_PIXEL(10, 10, 0, 0, 0, 255);               // outside viewport
    CHECK_PIXEL(40, 40, 0, 0, 0, 255);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    CHECK_GL();
}

TEST(draw_uniforms_and_ubo) {
    const char* vs = R"(#version 330 core
layout(location = 0) in vec2 aPos;
uniform vec2 uOffset;
uniform mat2 uRot;
void main() { gl_Position = vec4(uRot * aPos + uOffset, 0.0, 1.0); }
)";
    const char* fs = R"(#version 330 core
layout(std140) uniform Colors { vec4 tint; float scale; };
uniform vec4 uColor;
out vec4 o;
void main() { o = uColor * tint * scale; }
)";
    GLuint p = compileProgram(vs, fs);
    glUseProgram(p);
    GLint lo = glGetUniformLocation(p, "uOffset"), lr = glGetUniformLocation(p, "uRot"), lc = glGetUniformLocation(p, "uColor");
    CHECK(lo >= 0 && lr >= 0 && lc >= 0);
    glUniform2f(lo, 0.5f, 0.5f);
    float rot[4] = {1, 0, 0, 1};
    glUniformMatrix2fv(lr, 1, GL_FALSE, rot);
    glUniform4f(lc, 1, 1, 1, 1);
    GLuint bi = glGetUniformBlockIndex(p, "Colors");
    CHECK(bi != GL_INVALID_INDEX);
    GLint bsize = 0;
    glGetActiveUniformBlockiv(p, bi, GL_UNIFORM_BLOCK_DATA_SIZE, &bsize);
    CHECK_EQ(bsize, 32);
    glUniformBlockBinding(p, bi, 3);
    float data[8] = {0, 1, 0, 1, 0.5f, 0, 0, 0};
    GLuint ubo;
    glGenBuffers(1, &ubo);
    glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    glBufferData(GL_UNIFORM_BUFFER, sizeof data, data, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 3, ubo);
    Vtx v[3] = {{-0.5f, -0.5f}, {0.5f, -0.5f}, {-0.5f, 0.5f}};
    GLuint vbo, vao = makeVao(v, 3, &vbo);
    glDisableVertexAttribArray(1);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    // triangle moved to the upper-right quadrant: (0,0)-(1,0)-(0,1) NDC
    CHECK_PIXEL(36, 36, 0, 128, 0, 128);
    CHECK_PIXEL(20, 20, 0, 0, 0, 0);
    // Update the UBO between draws: the earlier draw must keep the old contents.
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    float data2[8] = {1, 0, 0, 1, 1, 0, 0, 0};
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof data2, data2);
    glUniform2f(lo, -0.5f, -0.5f);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    CHECK_PIXEL(36, 36, 0, 128, 0, 128);
    CHECK_PIXEL(4, 4, 255, 0, 0, 255);
    glDeleteBuffers(1, &ubo);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    CHECK_GL();
}

TEST(draw_texture_sampling) {
    const char* vs = R"(#version 330 core
layout(location = 0) in vec2 aPos;
out vec2 uv;
void main() { uv = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }
)";
    const char* fs = R"(#version 330 core
in vec2 uv;
uniform sampler2D tex;
out vec4 o;
void main() { o = texture(tex, uv); }
)";
    GLuint p = compileProgram(vs, fs);
    // 2x2 texture: bottom row red/green, top row blue/white (GL row 0 = bottom)
    uint32_t px[4] = {0xff0000ffu, 0xff00ff00u, 0xffff0000u, 0xffffffffu};
    GLuint t;
    glGenTextures(1, &t);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glUseProgram(p);
    glUniform1i(glGetUniformLocation(p, "tex"), 2);
    Vtx v[4] = {{-1, -1}, {1, -1}, {-1, 1}, {1, 1}};
    GLuint vbo, vao = makeVao(v, 4, &vbo);
    glDisableVertexAttribArray(1);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    CHECK_PIXEL(8, 8, 255, 0, 0, 255);
    CHECK_PIXEL(56, 8, 0, 255, 0, 255);
    CHECK_PIXEL(8, 56, 0, 0, 255, 255);
    CHECK_PIXEL(56, 56, 255, 255, 255, 255);
    // incomplete texture (mipmapped min filter, no mips) samples (0,0,0,1)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    CHECK_PIXEL(8, 8, 0, 0, 0, 255);
    glActiveTexture(GL_TEXTURE0);
    glDeleteTextures(1, &t);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    CHECK_GL();
}

TEST(draw_depth_and_blend) {
    const char* vs = R"(#version 330 core
layout(location = 0) in vec2 aPos;
uniform float uZ;
void main() { gl_Position = vec4(aPos, uZ, 1.0); }
)";
    const char* fs = R"(#version 330 core
uniform vec4 uColor;
out vec4 o;
void main() { o = uColor; }
)";
    GLuint p = compileProgram(vs, fs);
    glUseProgram(p);
    GLint lz = glGetUniformLocation(p, "uZ"), lc = glGetUniformLocation(p, "uColor");
    Vtx v[4] = {{-1, -1}, {1, -1}, {-1, 1}, {1, 1}};
    GLuint vbo, vao = makeVao(v, 4, &vbo);
    glDisableVertexAttribArray(1);
    glClearColor(0, 0, 0, 1);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glUniform1f(lz, 0.5f);
    glUniform4f(lc, 1, 0, 0, 1);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glUniform1f(lz, 0.8f);  // behind -> rejected
    glUniform4f(lc, 0, 1, 0, 1);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    CHECK_PIXEL(10, 10, 255, 0, 0, 255);
    float d = 0;
    glReadPixels(10, 10, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &d);
    CHECK(fabsf(d - 0.75f) < 1e-4f);  // GL window z = (0.5 + 1) / 2
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUniform4f(lc, 0, 0, 1, 0.5f);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glDisable(GL_BLEND);
    CHECK_PIXEL(10, 10, 128, 0, 128, 191);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    CHECK_GL();
}

TEST(draw_elements_instanced_fan) {
    const char* vs = R"(#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec4 aColor;
out vec4 vColor;
void main() {
    vColor = aColor;
    gl_Position = vec4(aPos * 0.2 + vec2(-0.75 + 0.5 * float(gl_InstanceID), 0.0), 0.0, 1.0);
}
)";
    GLuint p = compileProgram(vs, kFS);
    glUseProgram(p);
    Vtx v[4] = {{-1, -1, 1, 1, 0, 1}, {1, -1, 1, 1, 0, 1}, {1, 1, 1, 1, 0, 1}, {-1, 1, 1, 1, 0, 1}};
    GLuint vbo, vao = makeVao(v, 4, &vbo);
    GLuint ebo;
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    uint16_t idx[6] = {0, 1, 2, 0, 2, 3};
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, nullptr, 4);
    // instances at x centers -0.75,-0.25,0.25,0.75 (pixels 8,24,40,56), y center 32
    CHECK_PIXEL(8, 32, 255, 255, 0, 255);
    CHECK_PIXEL(24, 32, 255, 255, 0, 255);
    CHECK_PIXEL(56, 32, 255, 255, 0, 255);
    CHECK_PIXEL(16, 32, 0, 0, 0, 255);
    CHECK_PIXEL(8, 50, 0, 0, 0, 255);
    // triangle fan (CPU-expanded) + ubyte indices
    glClear(GL_COLOR_BUFFER_BIT);
    uint8_t idx8[4] = {0, 1, 2, 3};
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx8, idx8, GL_STATIC_DRAW);
    glDrawElements(GL_TRIANGLE_FAN, 4, GL_UNSIGNED_BYTE, nullptr);
    CHECK_PIXEL(8, 32, 255, 255, 0, 255);
    CHECK_PIXEL(24, 32, 0, 0, 0, 255);
    glDeleteBuffers(1, &ebo);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    CHECK_GL();
}

TEST(draw_fbo_mrt_and_cull) {
    const char* fs = R"(#version 330 core
layout(location = 0) out vec4 o0;
layout(location = 1) out uvec4 o1;
void main() { o0 = vec4(0.0, 1.0, 0.0, 1.0); o1 = uvec4(gl_FragCoord.xy, 42u, 7u); }
)";
    GLuint p = compileProgram(kVS, fs);
    GLuint tex[2], fb;
    glGenTextures(2, tex);
    glBindTexture(GL_TEXTURE_2D, tex[0]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 16, 16);
    glBindTexture(GL_TEXTURE_2D, tex[1]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32UI, 16, 16);
    glGenFramebuffers(1, &fb);
    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex[0], 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, tex[1], 0);
    GLenum bufs[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    glDrawBuffers(2, bufs);
    glViewport(0, 0, 16, 16);
    float zero[4] = {0, 0, 0, 0};
    GLuint uzero[4] = {0, 0, 0, 0};
    glClearBufferfv(GL_COLOR, 0, zero);
    glClearBufferuiv(GL_COLOR, 1, uzero);
    glUseProgram(p);
    // CCW full-screen quad; then cull back faces with a CW triangle that must vanish
    Vtx v[4] = {{-1, -1}, {1, -1}, {-1, 1}, {1, 1}};
    GLuint vbo, vao = makeVao(v, 4, &vbo);
    glDisableVertexAttribArray(1);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glDrawArrays(GL_TRIANGLES, 0, 3);  // (-1,-1),(1,-1),(-1,1): CCW -> drawn
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    CHECK_PIXEL(1, 1, 0, 255, 0, 255);
    CHECK_PIXEL(14, 14, 0, 0, 0, 0);
    glDrawArrays(GL_TRIANGLES, 1, 3);  // (1,-1),(-1,1),(1,1): CW -> culled
    CHECK_PIXEL(14, 14, 0, 0, 0, 0);
    glDisable(GL_CULL_FACE);
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    GLuint u[4] = {};
    glReadPixels(3, 2, 1, 1, GL_RGBA_INTEGER, GL_UNSIGNED_INT, u);
    CHECK_EQ(u[0], 3u);  // gl_FragCoord.x = 3.5
    CHECK_EQ(u[1], 2u);  // gl_FragCoord.y = 2.5 (GL convention, bottom-up)
    CHECK_EQ(u[2], 42u);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, 64, 64);
    glDeleteFramebuffers(1, &fb);
    glDeleteTextures(2, tex);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    CHECK_GL();
}

TEST(compute_ssbo) {
    const char* cs = R"(#version 430 core
layout(local_size_x = 64) in;
layout(std430, binding = 1) buffer Data { uint values[]; };
uniform uint uMul;
void main() { uint i = gl_GlobalInvocationID.x; if (i < values.length()) values[i] = i * uMul; }
)";
    GLuint s = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(s, 1, &cs, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    CHECK(ok);
    GLuint p = glCreateProgram();
    glAttachShader(p, s);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    CHECK(ok);
    GLuint b;
    glGenBuffers(1, &b);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, b);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 1000 * 4, nullptr, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, b);
    glUseProgram(p);
    glUniform1ui(glGetUniformLocation(p, "uMul"), 3);
    glDispatchCompute(16, 1, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    std::vector<uint32_t> out(1000);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 4000, out.data());
    CHECK_EQ(out[0], 0u);
    CHECK_EQ(out[999], 2997u);
    CHECK_EQ(out[500], 1500u);
    glDeleteBuffers(1, &b);
    glDeleteShader(s);
    glDeleteProgram(p);
    CHECK_GL();
}

// gl_LocalInvocationIndex / gl_LocalInvocationID for multi-dimensional work groups
// (compute_shader.one-work-group).
static GLuint computeProgram(const std::string& src) {
    const char* c = src.c_str();
    GLuint s = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(s, 1, &c, nullptr);
    glCompileShader(s);
    GLuint p = glCreateProgram();
    glAttachShader(p, s);
    glDeleteShader(s);
    glLinkProgram(p);
    return p;
}
TEST(compute_local_ids_respecified_buffer) {
    GLuint b = 0, ib = 0;
    glGenBuffers(1, &b);
    glGenBuffers(1, &ib);
    struct It { int x, y, z, binding; bool indirect; } its[] = {{16, 1, 1, 0, true}, {8, 8, 1, 1, false}, {4, 4, 4, 2, true}, {1, 2, 3, 3, false}, {32, 1, 32, 7, true}};
    for (const It& it : its) {
        char src[1024];
        snprintf(src, sizeof src,
                 "#version 430 core\nlayout(local_size_x = %d, local_size_y = %d, local_size_z = %d) in;\n"
                 "layout(std430, binding = %d) buffer Output { uvec4 local_id[]; } g_out;\n"
                 "void main() {\n  if (gl_WorkGroupSize == uvec3(%d, %d, %d) && gl_WorkGroupID == uvec3(0) &&\n"
                 "      gl_GlobalInvocationID == gl_LocalInvocationID) {\n"
                 "    g_out.local_id[gl_LocalInvocationIndex] = uvec4(gl_LocalInvocationID, 0);\n"
                 "  } else {\n    g_out.local_id[gl_LocalInvocationIndex] = uvec4(0xffff);\n  }\n}\n",
                 it.x, it.y, it.z, it.binding, it.x, it.y, it.z);
        GLuint p = computeProgram(src);
        int n = it.x * it.y * it.z;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, it.binding, b);
        glBufferData(GL_SHADER_STORAGE_BUFFER, 16 * n, nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        glUseProgram(p);
        if (it.indirect) {
            const GLuint groups[3] = {1, 1, 1};
            glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, ib);
            glBufferData(GL_DISPATCH_INDIRECT_BUFFER, sizeof groups, groups, GL_STATIC_DRAW);
            glDispatchComputeIndirect(0);
        } else {
            glDispatchCompute(1, 1, 1);
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, b);
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        const GLuint* d = (const GLuint*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, 16 * n, GL_MAP_READ_BIT);
        CHECK(d != nullptr);
        if (d) {
            int bad = 0;
            for (int i = 0; i < n; i++) {
                int x = i % it.x, y = (i / it.x) % it.y, z = i / (it.x * it.y);
                if (d[4 * i] != (GLuint)x || d[4 * i + 1] != (GLuint)y || d[4 * i + 2] != (GLuint)z) bad++;
            }
            CHECK_EQ(bad, 0);
        }
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        glUseProgram(0);
        glDeleteProgram(p);
    }
    glDeleteBuffers(1, &b);
    glDeleteBuffers(1, &ib);
    CHECK_GL();
}

// 124 varying components into the fragment shader (Metal's limit; GL 4.6 asks for 128).
static int kNV = getenv("NV") ? atoi(getenv("NV")) : 31;  // TODO: GL needs 32 (128 comps); Metal stops at 124
TEST(varyings_124_components) {
    std::string vs = "#version 460 core\nlayout(location=0) in vec2 p;\n", fs = "#version 460 core\nout vec4 o;\n";
    std::string vb = "void main(){ gl_Position = vec4(p,0,1);\n", fb = "void main(){ vec4 s = vec4(0);\n";
    for (int i = 0; i < kNV; i++) {
        vs += "out vec4 v" + std::to_string(i) + ";\n";
        fs += "in vec4 v" + std::to_string(i) + ";\n";
        vb += " v" + std::to_string(i) + " = vec4(" + std::to_string(i) + ".0/1024.0);\n";
        fb += " s += v" + std::to_string(i) + ";\n";
    }
    vs += vb + "}\n";
    fs += fb + " o = s; }\n";
    std::string log;
    GLuint p = compileProgram(vs.c_str(), fs.c_str(), &log);
    if (!p) fprintf(stderr, "  %s\n", log.c_str());
    CHECK(p != 0);
    GLuint vao, vb2;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    float q[] = {-1, -1, 1, -1, -1, 1, 1, 1};
    glGenBuffers(1, &vb2);
    glBindBuffer(GL_ARRAY_BUFFER, vb2);
    glBufferData(GL_ARRAY_BUFFER, sizeof q, q, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);
    glUseProgram(p);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    // sum of i/1024 for i<31 = 465/1024 -> 116
    CHECK_PIXEL(8, 8, 116, 116, 116, 116);
    glDeleteProgram(p);
    glDeleteBuffers(1, &vb2);
    glDeleteVertexArrays(1, &vao);
    CHECK_GL();
}
