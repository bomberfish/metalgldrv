// packreplay: draws shader pack programs (stage files assembled from a pack, e.g. with tests/glsl/corpus.py's
// include resolution) with the draw calls Minecraft 26 + Sodium + Iris issue, for reproducing driver problems with
// real shaders. Run it under MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1.
//
//   packreplay PREFIX... [--frames N] [--uniform NAME=V]
//
// For each PREFIX, PREFIX.vert, PREFIX.frag and the optional PREFIX.tesc / .tese / .geom are linked; every active
// attribute gets per-vertex data (float or integer as declared), the projection matrices (the compatibility matrix
// stack and mat4 uniforms named *Projection*) an orthographic projection of the quads (0..64), other matrices identity, vec2
// uniforms named *pixel* the reciprocal target size and other *Size / *Resolution the target size, sampler2D uniforms a texture; every SSBO binding 0..15 holds a
// zeroed 256 KB buffer; each program draws into its own G-buffer (one attachment per fragment output, integer
// outputs RGBA32UI / RGBA32I). Each frame, every program in turn draws with GL_TRIANGLES
// (METALGLDRV_IMPLICIT_PATCHES=1 for tessellated programs):
//   Sodium-like glMultiDrawElementsBaseVertex over a shared quad index buffer (uint32; index offsets, empty sub-draws),
//   Minecraft's glMultiDrawElementsIndirect / glDrawElementsIndirect, glDrawElementsInstancedBaseVertexBaseInstance
//   (uint16, index offset, 2 instances, base instance 3), glDrawElementsInstancedBaseVertex, glDrawElements with uint8
//   indices, glDrawArrays,
// then the multi-draws again into a depth-only FBO (a shadow pass).
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include "glapi.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

const int kWidth = 1920, kHeight = 1080, kQuads = 2000, kVerts = kQuads * 4;

std::string readFile(const std::string& p) {
    std::ifstream f(p);
    if (!f) return "";
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

GLuint compile(GLenum type, const std::string& src, const std::string& path) {
    GLuint s = glCreateShader(type);
    const char* p = src.c_str();
    glShaderSource(s, 1, &p, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        std::vector<char> log(1 << 16);
        glGetShaderInfoLog(s, (GLsizei)log.size(), nullptr, log.data());
        fprintf(stderr, "%s: compile failed: %s\n", path.c_str(), log.data());
        exit(1);
    }
    return s;
}

struct Program {
    std::string prefix;
    GLuint prog = 0, vao = 0, fbo = 0;
    bool tess = false;
    int probeAttachment = -1;
};

Program setup(const std::string& prefix, const std::vector<std::pair<std::string, float>>& floatUniforms) {
    Program P;
    P.prefix = prefix;
    P.prog = glCreateProgram();
    const struct { const char* ext; GLenum type; } stages[] = {{".vert", GL_VERTEX_SHADER}, {".tesc", GL_TESS_CONTROL_SHADER},
                                                               {".tese", GL_TESS_EVALUATION_SHADER}, {".geom", GL_GEOMETRY_SHADER},
                                                               {".frag", GL_FRAGMENT_SHADER}};
    for (auto& st : stages) {
        std::string src = readFile(prefix + st.ext);
        if (src.empty()) continue;
        if (st.type == GL_TESS_EVALUATION_SHADER) P.tess = true;
        glAttachShader(P.prog, compile(st.type, src, prefix + st.ext));
    }
    glLinkProgram(P.prog);
    GLint ok = 0;
    glGetProgramiv(P.prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        std::vector<char> log(1 << 16);
        glGetProgramInfoLog(P.prog, (GLsizei)log.size(), nullptr, log.data());
        fprintf(stderr, "%s: link failed: %s\n", prefix.c_str(), log.data());
        exit(1);
    }
    glUseProgram(P.prog);
    // uniforms
    const float ortho[16] = {2.0f / 64, 0, 0, 0, 0, 2.0f / 64, 0, 0, 0, 0, -2.0f / 64, 0, -1, -1, 0, 1};
    GLint nu = 0;
    glGetProgramiv(P.prog, GL_ACTIVE_UNIFORMS, &nu);
    int unit = 0;
    for (GLint i = 0; i < nu; i++) {
        char name[256];
        GLint size;
        GLenum type;
        glGetActiveUniform(P.prog, (GLuint)i, sizeof name, nullptr, &size, &type, name);
        GLint loc = glGetUniformLocation(P.prog, name);
        if (loc < 0) continue;
        std::string nm = name;
        if (type == GL_FLOAT_MAT4) {  // projections map the quads (0..64) onto the target, other matrices are identity
            const float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
            bool proj = nm.find("rojection") != std::string::npos && nm.find("Inverse") == std::string::npos;
            for (int k = 0; k < size; k++) glUniformMatrix4fv(loc + k, 1, GL_FALSE, proj ? ortho : identity);
        } else if (type == GL_FLOAT_MAT3) {
            const float identity3[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
            for (int k = 0; k < size; k++) glUniformMatrix3fv(loc + k, 1, GL_FALSE, identity3);
        } else if (type == GL_FLOAT_VEC2 && nm.find("pixel") != std::string::npos) {
            glUniform2f(loc, 1.0f / kWidth, 1.0f / kHeight);
        } else if (type == GL_FLOAT_VEC2 && (nm.find("Size") != std::string::npos || nm.find("Resolution") != std::string::npos)) {
            glUniform2f(loc, (float)kWidth, (float)kHeight);
        } else if (type == GL_SAMPLER_2D) {
            GLuint t;
            glGenTextures(1, &t);
            glActiveTexture(GL_TEXTURE0 + unit);
            glBindTexture(GL_TEXTURE_2D, t);
            std::vector<uint8_t> px(16 * 16 * 4, 255);  // white: alpha 1, depth textures at the far plane
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 16, 16, 0, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glUniform1i(loc, unit++);
        }
    }
    for (auto& [nm, v] : floatUniforms)
        if (GLint loc = glGetUniformLocation(P.prog, nm.c_str()); loc >= 0) glUniform1f(loc, v);
    glActiveTexture(GL_TEXTURE0);
    // vertex data for every active attribute: quads over the target
    glGenVertexArrays(1, &P.vao);
    glBindVertexArray(P.vao);
    GLint na = 0;
    glGetProgramiv(P.prog, GL_ACTIVE_ATTRIBUTES, &na);
    uint32_t rng = 7;
    auto rnd = [&]() { return ((rng = rng * 1664525u + 1013904223u) >> 8) / 16777216.0f; };
    for (GLint i = 0; i < na; i++) {
        char name[256];
        GLint size;
        GLenum type;
        glGetActiveAttrib(P.prog, (GLuint)i, sizeof name, nullptr, &size, &type, name);
        GLint loc = glGetAttribLocation(P.prog, name);
        if (getenv("PACKREPLAY_VERBOSE")) fprintf(stderr, "%s: attribute %s type 0x%x location %d\n", prefix.c_str(), name, type, loc);
        if (loc < 0) continue;
        bool isInt = type == GL_INT || type == GL_INT_VEC2 || type == GL_INT_VEC3 || type == GL_INT_VEC4 || type == GL_UNSIGNED_INT ||
                     type == GL_UNSIGNED_INT_VEC2 || type == GL_UNSIGNED_INT_VEC3 || type == GL_UNSIGNED_INT_VEC4;
        std::vector<float> f((size_t)kVerts * 4);
        std::vector<int32_t> iv((size_t)kVerts * 4);
        bool pos = strstr(name, "Position") || strstr(name, "Vertex");
        for (int q = 0; q < kQuads; q++) {
            float x = rnd() * 60, y = rnd() * 60, z = rnd() * 60, s = 0.5f + rnd();
            const float dx[4] = {0, s, s, 0}, dy[4] = {0, 0, s, s};
            for (int k = 0; k < 4; k++) {
                size_t v = (size_t)(q * 4 + k) * 4;
                if (pos) f[v] = x + dx[k], f[v + 1] = y + dy[k], f[v + 2] = z, f[v + 3] = 1;
                else f[v] = dx[k] / s, f[v + 1] = dy[k] / s, f[v + 2] = 0.5f, f[v + 3] = 1;
                iv[v] = q & 255, iv[v + 1] = k, iv[v + 2] = 15, iv[v + 3] = 0;
            }
        }
        GLuint b;
        glGenBuffers(1, &b);
        glBindBuffer(GL_ARRAY_BUFFER, b);
        if (isInt) {
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(iv.size() * 4), iv.data(), GL_STATIC_DRAW);
            glVertexAttribIPointer((GLuint)loc, 4, GL_INT, 16, nullptr);
        } else {
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(f.size() * 4), f.data(), GL_STATIC_DRAW);
            glVertexAttribPointer((GLuint)loc, 4, GL_FLOAT, GL_FALSE, 16, nullptr);
        }
        glEnableVertexAttribArray((GLuint)loc);
    }
    // G-buffer: one attachment per fragment output location
    glGenFramebuffers(1, &P.fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, P.fbo);
    GLenum bufs[8] = {GL_NONE, GL_NONE, GL_NONE, GL_NONE, GL_NONE, GL_NONE, GL_NONE, GL_NONE};
    int nbufs = 0;
    GLint nout = 0;
    glGetProgramInterfaceiv(P.prog, GL_PROGRAM_OUTPUT, GL_ACTIVE_RESOURCES, &nout);
    for (GLint i = 0; i < nout; i++) {
        const GLenum props[2] = {GL_TYPE, GL_LOCATION};
        GLint v[2] = {0, -1};
        glGetProgramResourceiv(P.prog, GL_PROGRAM_OUTPUT, (GLuint)i, 2, props, 2, nullptr, v);
        if (getenv("PACKREPLAY_VERBOSE")) fprintf(stderr, "%s: output type 0x%x location %d\n", prefix.c_str(), v[0], v[1]);
        if (v[1] < 0 || v[1] >= 8) continue;
        bool u = v[0] == GL_UNSIGNED_INT || v[0] == GL_UNSIGNED_INT_VEC2 || v[0] == GL_UNSIGNED_INT_VEC3 || v[0] == GL_UNSIGNED_INT_VEC4;
        bool s = v[0] == GL_INT || v[0] == GL_INT_VEC2 || v[0] == GL_INT_VEC3 || v[0] == GL_INT_VEC4;
        GLuint t;
        glGenTextures(1, &t);
        glBindTexture(GL_TEXTURE_2D, t);
        glTexStorage2D(GL_TEXTURE_2D, 1, u ? GL_RGBA32UI : s ? GL_RGBA32I : GL_RGBA8, kWidth, kHeight);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + v[1], GL_TEXTURE_2D, t, 0);
        bufs[v[1]] = GL_COLOR_ATTACHMENT0 + v[1];
        nbufs = std::max(nbufs, v[1] + 1);
        if (!u && !s && P.probeAttachment < 0) P.probeAttachment = v[1];
    }
    GLuint depth;
    glGenTextures(1, &depth);
    glBindTexture(GL_TEXTURE_2D, depth);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH_COMPONENT24, kWidth, kHeight);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth, 0);
    glDrawBuffers(std::max(nbufs, 1), bufs);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) fprintf(stderr, "%s: G-buffer FBO incomplete\n", prefix.c_str());
    return P;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> prefixes;
    int frames = 3;
    std::vector<std::pair<std::string, float>> floatUniforms;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--frames" && i + 1 < argc) frames = atoi(argv[++i]);
        else if (a == "--uniform" && i + 1 < argc) {
            std::string v = argv[++i];
            size_t eq = v.find('=');
            if (eq == std::string::npos) return 2;
            floatUniforms.push_back({v.substr(0, eq), (float)atof(v.c_str() + eq + 1)});
        } else if (a.size() > 2 && a.compare(0, 2, "--") == 0) {
            fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        } else {
            prefixes.push_back(a);
        }
    }
    if (prefixes.empty()) {
        fprintf(stderr, "usage: packreplay PREFIX... [--frames N] [--uniform NAME=V]\n");
        return 2;
    }
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
    // compatibility profile: packs use gl_Vertex / gl_ModelViewMatrix / gl_ProjectionMatrix (Iris rewrites them
    // into its own inputs; here the matrix stacks feed them)
    EGLint ctxAttr[] = {EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 6, EGL_CONTEXT_OPENGL_PROFILE_MASK,
                        EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT, EGL_NONE};
    EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ctxAttr);
    eglMakeCurrent(dpy, surf, surf, ctx);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 64, 0, 64, -64, 64);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // shared resources: SSBOs, index buffers, indirect commands, shadow FBO
    std::vector<GLuint> ssbos(16);
    glGenBuffers(16, ssbos.data());
    std::vector<uint8_t> zeros(256 << 10, 0);
    for (int b = 0; b < 16; b++) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbos[b]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)zeros.size(), zeros.data(), GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, b, ssbos[b]);
    }
    std::vector<uint32_t> idx32((size_t)kQuads * 6);
    std::vector<uint16_t> idx16((size_t)kQuads * 6);
    std::vector<uint8_t> idx8(36);
    for (int q = 0; q < kQuads; q++) {
        const uint32_t o[6] = {0, 1, 2, 2, 3, 0};
        for (int k = 0; k < 6; k++) {
            idx32[(size_t)q * 6 + k] = (uint32_t)q * 4 + o[k];
            idx16[(size_t)q * 6 + k] = (uint16_t)(q * 4 + o[k]);
            if (q < 6) idx8[(size_t)q * 6 + k] = (uint8_t)(q * 4 + o[k]);
        }
    }
    GLuint ib32, ib16, ib8, indirect;
    glGenBuffers(1, &ib32);
    glGenBuffers(1, &ib16);
    glGenBuffers(1, &ib8);
    glGenBuffers(1, &indirect);
    // DrawElementsIndirectCommand: count, instanceCount, firstIndex, baseVertex, baseInstance
    std::vector<uint32_t> cmds;
    for (int k = 0; k < 16; k++) {
        const uint32_t c[5] = {(uint32_t)(k % 5 == 4 ? 0 : 6 * (40 + k * 7)), 1, (uint32_t)(6 * (k % 3)), (uint32_t)(k * 4 * 37), 0};
        cmds.insert(cmds.end(), c, c + 5);
    }
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, indirect);
    glBufferData(GL_DRAW_INDIRECT_BUFFER, (GLsizeiptr)(cmds.size() * 4), cmds.data(), GL_STATIC_DRAW);
    GLuint depthFbo, shadowDepth;
    glGenFramebuffers(1, &depthFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, depthFbo);
    glGenTextures(1, &shadowDepth);
    glBindTexture(GL_TEXTURE_2D, shadowDepth);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH_COMPONENT32F, 1024, 1024);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadowDepth, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) fprintf(stderr, "depth-only FBO incomplete\n");

    std::vector<Program> progs;
    for (auto& p : prefixes) progs.push_back(setup(p, floatUniforms));
    // element buffers are VAO state: give every VAO the uint32 buffer, then fill the others
    for (auto& P : progs) {
        glBindVertexArray(P.vao);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib32);
    }
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)(idx32.size() * 4), idx32.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib16);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)(idx16.size() * 2), idx16.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib8);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)idx8.size(), idx8.data(), GL_STATIC_DRAW);
    glEnable(GL_DEPTH_TEST);
    glPatchParameteri(GL_PATCH_VERTICES, 3);

    // Sodium: 40 regions x 7 facings, sub-draws of 50..250 quads from the shared index buffer, some empty
    auto sodium = [&]() {
        for (int c = 0; c < 40; c++) {
            GLsizei counts[7];
            const void* offs[7];
            GLint bv[7];
            for (int s = 0; s < 7; s++) {
                int q = (c * 7 + s) % 5 == 4 ? 0 : 50 + (c * 37 + s * 11) % 200;
                counts[s] = q * 6;
                offs[s] = (const void*)(uintptr_t)(((c + s) % 3) * 6 * 4);  // index offset: whole quads
                bv[s] = ((c * 7 + s) * 4 * 3) % (kVerts - 4 * 260);
            }
            glMultiDrawElementsBaseVertex(GL_TRIANGLES, counts, GL_UNSIGNED_INT, offs, 7, bv);
        }
    };
    for (int f = 0; f < frames; f++) {
        for (auto& P : progs) {
            glUseProgram(P.prog);
            glBindVertexArray(P.vao);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib32);
            glBindFramebuffer(GL_FRAMEBUFFER, depthFbo);
            glViewport(0, 0, 1024, 1024);
            glClear(GL_DEPTH_BUFFER_BIT);
            sodium();
            glBindFramebuffer(GL_FRAMEBUFFER, P.fbo);
            glViewport(0, 0, kWidth, kHeight);
            glClearColor(0.0f, 1.0f, 0.0f, 0.0f);  // fragments that ran differ from it
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            sodium();
            glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, nullptr, 16, 0);
            glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, (const void*)(uintptr_t)(20 * 3));
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib16);
            glDrawElementsInstancedBaseVertexBaseInstance(GL_TRIANGLES, 600, GL_UNSIGNED_SHORT, (const void*)(uintptr_t)(12 * 6 * 2), 2, 40, 3);
            glDrawElementsInstancedBaseVertex(GL_TRIANGLES, 300, GL_UNSIGNED_SHORT, (const void*)(uintptr_t)(6 * 2), 3, 400);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib8);
            glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_BYTE, nullptr);
            glDrawArrays(GL_TRIANGLES, 300, 600);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib32);
        }
        glFinish();
        if (GLenum e = glGetError()) fprintf(stderr, "frame %d: GL error 0x%x\n", f, e);
    }
    for (auto& P : progs) {
        long shaded = -1;  // pixels of the first float attachment that differ from the clear color
        if (P.probeAttachment >= 0) {
            std::vector<uint8_t> px((size_t)kWidth * kHeight * 4);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, P.fbo);
            glReadBuffer(GL_COLOR_ATTACHMENT0 + P.probeAttachment);
            glReadPixels(0, 0, kWidth, kHeight, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
            shaded = 0;
            for (size_t i = 0; i < px.size(); i += 4) shaded += !(px[i] == 0 && px[i + 1] == 255 && px[i + 2] == 0 && px[i + 3] == 0);
        }
        printf("packreplay %s: %d frames done, %ld pixels shaded\n", P.prefix.c_str(), frames, shaded);
    }
    eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroyContext(dpy, ctx);
    eglDestroySurface(dpy, surf);
    eglTerminate(dpy);
    return 0;
}
