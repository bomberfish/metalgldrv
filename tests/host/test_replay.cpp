#include <unistd.h>
#include <dirent.h>
#include <algorithm>
#include <vector>
#include <chrono>
// Replays dumped application shaders (METALGLDRV_REPLAY_DIR) — diagnostic, skipped otherwise.
#include "harness.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <mach-o/dyld.h>

static std::string readFile(const std::string& p) {
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static GLuint compileLink(const std::string& vs, const std::string& fs) {
    GLuint p = glCreateProgram();
    const std::string* srcs[2] = {&vs, &fs};
    for (int i = 0; i < 2; i++) {
        GLuint s = glCreateShader(i ? GL_FRAGMENT_SHADER : GL_VERTEX_SHADER);
        const char* c = srcs[i]->c_str();
        glShaderSource(s, 1, &c, nullptr);
        glCompileShader(s);
        GLint ok = 0;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[4096];
            glGetShaderInfoLog(s, sizeof log, nullptr, log);
            fprintf(stderr, "  compile failed: %s\n", log);
        }
        glAttachShader(p, s);
        glDeleteShader(s);
    }
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(p, sizeof log, nullptr, log);
        fprintf(stderr, "  link failed: %s\n", log);
    }
    return p;
}

// Sodium compact vertex: 20-bit positions (scale 32/2^20, offset -8).
static void packPos(float x, float y, float z, uint32_t out[2]) {
    uint32_t v[3];
    float p[3] = {x, y, z};
    for (int i = 0; i < 3; i++) v[i] = (uint32_t)((p[i] + 8.0f) / 32.0f * 1048576.0f);
    out[0] = (v[0] >> 10) | (v[1] >> 10) << 10 | (v[2] >> 10) << 20;
    out[1] = (v[0] & 0x3ff) | (v[1] & 0x3ff) << 10 | (v[2] & 0x3ff) << 20;
}

static float drawTerrain(const std::string& vs, const std::string& fs) {
    GLuint p = compileLink(vs, fs);
    glUseProgram(p);
    float id[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    const char* mats[] = {"iris_ProjectionMatrix", "iris_ModelViewMatrix", "gbufferModelView", "gbufferModelViewInverse",
                          "gbufferProjection", "gbufferProjectionInverse", "iris_ProjectionMatrixInverse", "iris_ModelViewMatrixInverse"};
    for (auto m : mats) glUniformMatrix4fv(glGetUniformLocation(p, m), 1, GL_FALSE, id);
    float id3[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    glUniformMatrix3fv(glGetUniformLocation(p, "iris_NormalMatrix"), 1, GL_FALSE, id3);
    glUniform1f(glGetUniformLocation(p, "viewWidth"), 64);
    glUniform1f(glGetUniformLocation(p, "viewHeight"), 64);
    glUniform1f(glGetUniformLocation(p, "far"), 100);
    glUniform1f(glGetUniformLocation(p, "near"), 0.05f);
    glUniform1i(glGetUniformLocation(p, "tex"), 0);
    // every other sampler on its own (empty) unit, as in the game
    GLint nu = 0;
    glGetProgramiv(p, GL_ACTIVE_UNIFORMS, &nu);
    for (int i = 0, unit = 1; i < nu; i++) {
        char name[256];
        GLint size;
        GLenum type;
        glGetActiveUniform(p, i, sizeof name, nullptr, &size, &type, name);
        bool sampler = (type >= GL_SAMPLER_1D && type <= GL_SAMPLER_2D_RECT_SHADOW) || type == GL_SAMPLER_2D_ARRAY ||
                       type == GL_SAMPLER_2D_ARRAY_SHADOW || type == GL_SAMPLER_CUBE_SHADOW || type == GL_INT_SAMPLER_2D ||
                       type == GL_UNSIGNED_INT_SAMPLER_2D;
        if (sampler && strcmp(name, "tex") != 0) glUniform1i(glGetUniformLocation(p, name), unit++);
    }
    GLuint ubo;
    glGenBuffers(1, &ubo);
    std::vector<int> fades(64 * 4, -1);
    glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    glBufferData(GL_UNIFORM_BUFFER, fades.size() * 4, fades.data(), GL_STATIC_DRAW);
    GLuint bi = glGetUniformBlockIndex(p, "iris_ChunkData");
    if (bi != GL_INVALID_INDEX) {
        glUniformBlockBinding(p, bi, 0);
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo);
    }
    // opaque white texture
    GLuint t;
    glGenTextures(1, &t);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, t);
    // left half opaque, right half transparent: some fragments of the draw discard
    std::vector<uint32_t> px(16 * 16, 0xffffffffu);
    for (int y = 0; y < 16; y++)
        for (int x = 8; x < 16; x++) px[y * 16 + x] = 0x00ffffffu;
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 16, 16, 0, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    // vertices: 32-byte stride like Sodium (pos uvec2, color, texcoord, light/data, entity, normal, midtex)
    struct V { uint32_t pos[2]; uint8_t color[4]; uint16_t tex[2]; uint8_t light[4]; uint32_t entity; uint32_t normal; uint16_t mid[2]; };
    static_assert(sizeof(V) == 32, "");
    V v[4] = {};
    float q[4][2] = {{-1, -1}, {1, -1}, {-1, 1}, {1, 1}};
    for (int i = 0; i < 4; i++) {
        packPos(q[i][0], q[i][1], 0.0f, v[i].pos);
        memset(v[i].color, 255, 4);
        v[i].tex[0] = q[i][0] < 0 ? 0 : 32767;  // u spans the texture: left opaque, right transparent
        v[i].tex[1] = 16384;
        v[i].mid[0] = v[i].mid[1] = 16384;
    }
    GLuint vao, vb, ib;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vb);
    glBindBuffer(GL_ARRAY_BUFFER, vb);
    glBufferData(GL_ARRAY_BUFFER, sizeof v, v, GL_STATIC_DRAW);
    auto loc = [&](const char* n) { return glGetAttribLocation(p, n); };
    auto ia = [&](const char* n, int size, GLenum type, size_t off) {
        GLint l = loc(n);
        if (l < 0) return;
        glVertexAttribIPointer(l, size, type, 32, (void*)off);
        glEnableVertexAttribArray(l);
    };
    auto fa = [&](const char* n, int size, GLenum type, bool norm, size_t off) {
        GLint l = loc(n);
        if (l < 0) return;
        glVertexAttribPointer(l, size, type, norm, 32, (void*)off);
        glEnableVertexAttribArray(l);
    };
    ia("a_Position", 2, GL_UNSIGNED_INT, 0);
    fa("a_Color", 4, GL_UNSIGNED_BYTE, true, 8);
    ia("a_TexCoord", 2, GL_UNSIGNED_SHORT, 12);
    ia("a_LightAndData", 4, GL_UNSIGNED_BYTE, 16);
    ia("mc_Entity", 1, GL_UNSIGNED_INT, 20);
    ia("iris_Normal", 1, GL_UNSIGNED_INT, 24);
    fa("mc_midTexCoord", 2, GL_UNSIGNED_SHORT, false, 28);
    uint32_t idx[6] = {0, 1, 2, 2, 1, 3};
    glGenBuffers(1, &ib);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
    // target: color + depth texture
    GLuint fb, ct, dt;
    glGenFramebuffers(1, &fb);
    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    glGenTextures(1, &ct);
    glBindTexture(GL_TEXTURE_2D, ct);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 64, 64, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glGenTextures(1, &dt);
    glBindTexture(GL_TEXTURE_2D, dt);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, 64, 64, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ct, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, dt, 0);
    // Iris gbuffer layout seen on device: R11G11B10F, RGBA8, RGBA8_SNORM
    GLuint extra[2];
    glGenTextures(2, extra);
    GLenum fmts[2] = {GL_RGBA8, GL_RGBA8_SNORM};
    for (int i = 0; i < 2; i++) {
        glBindTexture(GL_TEXTURE_2D, extra[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, fmts[i], 64, 64, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1 + i, GL_TEXTURE_2D, extra[i], 0);
    }
    glBindTexture(GL_TEXTURE_2D, ct);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R11F_G11F_B10F, 64, 64, 0, GL_RGB, GL_FLOAT, nullptr);
    GLenum bufs[3] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
    glDrawBuffers(3, bufs);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, t);
    glViewport(0, 0, 64, 64);
    glClearDepth(1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glFinish();
    auto t0 = std::chrono::steady_clock::now();
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
    glFinish();
    fprintf(stderr, "  draw took %.1f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    float d = -1, dr = -1;
    glReadPixels(16, 32, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &d);   // opaque side
    glReadPixels(48, 32, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &dr);  // transparent side
    uint8_t c[4] = {};
    glReadPixels(16, 32, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, c);
    uint8_t c2[4] = {};
    glReadPixels(48, 32, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, c2);
    fprintf(stderr, "  opaque side depth %f color %d,%d,%d,%d; transparent side depth %f color %d,%d,%d,%d\n", d, c[0], c[1], c[2],
            c[3], dr, c2[0], c2[1], c2[2], c2[3]);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return d;
}

TEST(replay_iris_terrain) {
    const char* env = getenv("METALGLDRV_REPLAY_DIR");
    if (!env) return;
    std::string dir = env;
    if (dir == "@exe") {  // iOS runner: files shipped next to the executable
        char path[4096];
        uint32_t n = sizeof path;
        _NSGetExecutablePath(path, &n);
        dir = std::string(path);
        dir = dir.substr(0, dir.rfind('/')) + "/replay";
    }
    std::string prog = getenv("METALGLDRV_REPLAY_PROG") ? getenv("METALGLDRV_REPLAY_PROG") : "472";
    std::string vs = readFile(dir + "/p" + prog + "_8b31.glsl"), fs = readFile(dir + "/p" + prog + "_8b30.glsl");
    // Extra fragment-shader variants v_*.glsl in the same directory: report each one's result.
    if (DIR* dp = opendir(dir.c_str())) {
        std::vector<std::string> names;
        while (dirent* de = readdir(dp))
            if (!strncmp(de->d_name, "v_", 2) && strstr(de->d_name, ".glsl")) names.push_back(de->d_name);
        closedir(dp);
        std::sort(names.begin(), names.end());
        for (auto& n : names) {
            fprintf(stderr, "variant %s\n", n.c_str());
            // v_NAME.glsl + m_NAME.metal: compile the hand-edited MSL in place of the generated one
            std::string msl = dir + "/m_" + n.substr(2, n.size() - 7) + ".metal";
            if (access(msl.c_str(), R_OK) == 0) setenv("METALGLDRV_MSL_OVERRIDE", msl.c_str(), 1);
            std::string opts = dir + "/o_" + n.substr(2, n.size() - 7) + ".txt";  // compile options
            if (access(opts.c_str(), R_OK) == 0) setenv("METALGLDRV_MSL_OPTS", readFile(opts).c_str(), 1);
            float d = drawTerrain(vs, readFile(dir + "/" + n));
            unsetenv("METALGLDRV_MSL_OVERRIDE");
            unsetenv("METALGLDRV_MSL_OPTS");
            fprintf(stderr, "variant %s => %s\n", n.c_str(), d < 0.99f ? "GOOD" : "BAD");
        }
    }
    float d1 = drawTerrain(vs, fs);
    std::string fs2 = fs;
    size_t at = fs2.find("if (color.a <= 0.5f) discard;");
    if (at != std::string::npos) fs2.replace(at, strlen("if (color.a <= 0.5f) discard;"), "if (color.a <= 0.5f) return;");
    float d2 = drawTerrain(vs, fs2);
    fprintf(stderr, "  depth with discard %f, with return %f\n", d1, d2);
    CHECK(d1 < 0.99f);
    CHECK(d2 < 0.99f);
}
