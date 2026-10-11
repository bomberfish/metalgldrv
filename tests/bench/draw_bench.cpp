// drawbench: the driver's CPU cost per draw call, as games issue them, on metalgldrv and on Apple's OpenGL (built
// as `drawbench`, linked against metalgldrv's CGL, and `drawbench-apple`, linked against OpenGL.framework).
//
// Every frame draws --draws small meshes (36-index cubes) into a small offscreen target (color + depth, so the GPU
// isn't the limit). Between draws it changes state the way a scene renderer does: the program (one of --programs,
// changed every --batch draws), the textures (--textures per program, from a pool of 16), the VAO (one of 4), the
// per-draw uniforms (a mat4 and a vec4) and, with --ubo, a per-draw uniform block range (glBindBufferRange into a
// buffer written once per frame) or, with --ubo-sub, a glBufferSubData per draw into one small UBO.
//
//   drawbench [--frames N] [--warmup N] [--draws N] [--programs N] [--batch N] [--textures N] [--ubo | --ubo-sub]
//             [--size N]
//
// Frames run back to back (glFlush); reports wall ms per frame and µs per draw over all frames.
#include <OpenGL/OpenGL.h>
#include <OpenGL/gl3.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

GLuint compile(GLenum type, const std::string& src) {
    GLuint s = glCreateShader(type);
    const char* p = src.c_str();
    glShaderSource(s, 1, &p, nullptr);
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

GLuint link(const std::string& vs, const std::string& fs) {
    GLuint p = glCreateProgram();
    GLuint a = compile(GL_VERTEX_SHADER, vs), b = compile(GL_FRAGMENT_SHADER, fs);
    glAttachShader(p, a);
    glAttachShader(p, b);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096] = {};
        glGetProgramInfoLog(p, sizeof log, nullptr, log);
        fprintf(stderr, "link failed: %s\n", log);
        exit(1);
    }
    glDeleteShader(a);
    glDeleteShader(b);
    return p;
}

}  // namespace

int main(int argc, char** argv) {
    int frames = 200, warmup = 10, draws = 3000, programs = 8, batch = 4, textures = 3, size = 128;
    bool ubo = false, uboSub = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> int {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s needs a value\n", a.c_str());
                exit(2);
            }
            return atoi(argv[++i]);
        };
        if (a == "--frames") frames = next();
        else if (a == "--warmup") warmup = next();
        else if (a == "--draws") draws = next();
        else if (a == "--programs") programs = std::max(1, next());
        else if (a == "--batch") batch = std::max(1, next());
        else if (a == "--textures") textures = std::min(4, std::max(0, next()));
        else if (a == "--size") size = next();
        else if (a == "--ubo") ubo = true;
        else if (a == "--ubo-sub") uboSub = true;
        else {
            fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    bool anyUbo = ubo || uboSub;

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

    // programs: the same structure with a different constant each (as different materials)
    std::vector<GLuint> progs;
    std::vector<GLint> locModel, locColor;
    for (int p = 0; p < programs; p++) {
        std::string vs = std::string("#version 410 core\n") + "layout(location = 0) in vec3 a_pos;\n" + "layout(location = 1) in vec3 a_normal;\n" +
                         "uniform mat4 u_viewProj;\nuniform mat4 u_model;\n" +
                         (anyUbo ? "layout(std140) uniform Object { mat4 o_model; vec4 o_tint; };\n" : "") +
                         "out vec3 v_normal;\nout vec2 v_uv;\nvoid main() {\n" + "    mat4 m = u_model" + (anyUbo ? " * o_model" : "") +
                         ";\n    gl_Position = u_viewProj * m * vec4(a_pos, 1.0);\n" + "    v_normal = mat3(m) * a_normal;\n" +
                         "    v_uv = a_pos.xy * 0.5 + 0.5 + " + std::to_string(p * 0.01) + ";\n}\n";
        std::string fs = "#version 410 core\nin vec3 v_normal;\nin vec2 v_uv;\nuniform vec4 u_color;\n";
        for (int t = 0; t < textures; t++) fs += "uniform sampler2D u_tex" + std::to_string(t) + ";\n";
        if (anyUbo) fs += "layout(std140) uniform Object { mat4 o_model; vec4 o_tint; };\n";
        fs += "out vec4 o;\nvoid main() {\n    vec4 c = u_color * max(dot(normalize(v_normal), vec3(0.3, 0.5, 0.8)), 0.2);\n";
        for (int t = 0; t < textures; t++) fs += "    c *= texture(u_tex" + std::to_string(t) + ", v_uv);\n";
        if (anyUbo) fs += "    c *= o_tint;\n";
        fs += "    o = c;\n}\n";
        GLuint pr = link(vs, fs);
        glUseProgram(pr);
        for (int t = 0; t < textures; t++) glUniform1i(glGetUniformLocation(pr, ("u_tex" + std::to_string(t)).c_str()), t);
        float vp[16] = {0.5f, 0, 0, 0, 0, 0.5f, 0, 0, 0, 0, 0.1f, 0, 0, 0, 0, 1};
        glUniformMatrix4fv(glGetUniformLocation(pr, "u_viewProj"), 1, GL_FALSE, vp);
        if (anyUbo) glUniformBlockBinding(pr, glGetUniformBlockIndex(pr, "Object"), 0);
        progs.push_back(pr);
        locModel.push_back(glGetUniformLocation(pr, "u_model"));
        locColor.push_back(glGetUniformLocation(pr, "u_color"));
    }

    // textures: 16 small RGBA8 textures with mipmaps
    GLuint tex[16];
    glGenTextures(16, tex);
    std::vector<uint32_t> px(64 * 64);
    for (int t = 0; t < 16; t++) {
        for (int i = 0; i < 64 * 64; i++) px[i] = 0xff000000u | (uint32_t)(t * 16 + (i & 15)) << ((t % 3) * 8) | 0x404040u;
        glBindTexture(GL_TEXTURE_2D, tex[t]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 64, 64, 0, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
        glGenerateMipmap(GL_TEXTURE_2D);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, t % 2 ? GL_REPEAT : GL_CLAMP_TO_EDGE);
    }

    // 4 VAOs: a cube (position + normal), slightly different each
    const float cube[8][3] = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1}, {-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1}};
    const uint16_t idx[36] = {0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0};
    GLuint vaos[4], vbos[4], ibo;
    glGenVertexArrays(4, vaos);
    glGenBuffers(4, vbos);
    glGenBuffers(1, &ibo);
    for (int v = 0; v < 4; v++) {
        float data[8][6];
        for (int i = 0; i < 8; i++)
            for (int k = 0; k < 3; k++) {
                data[i][k] = cube[i][k] * (0.1f + 0.02f * v);
                data[i][3 + k] = cube[i][k] * 0.577f;
            }
        glBindVertexArray(vaos[v]);
        glBindBuffer(GL_ARRAY_BUFFER, vbos[v]);
        glBufferData(GL_ARRAY_BUFFER, sizeof data, data, GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        if (v == 0) glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, (void*)0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, (void*)12);
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
    }

    // per-draw uniform block data: 80 bytes per object at 256-byte aligned offsets
    GLuint ubuf = 0;
    const int kUboStride = 256;
    std::vector<uint8_t> uboData((size_t)draws * kUboStride);
    for (int d = 0; d < draws; d++) {
        float* f = (float*)&uboData[(size_t)d * kUboStride];
        for (int k = 0; k < 16; k++) f[k] = k % 5 == 0 ? 1.0f : 0.0f;
        for (int k = 0; k < 4; k++) f[16 + k] = 0.5f + 0.5f * ((d >> k) & 1);
    }
    if (anyUbo) {
        glGenBuffers(1, &ubuf);
        glBindBuffer(GL_UNIFORM_BUFFER, ubuf);
        glBufferData(GL_UNIFORM_BUFFER, ubo ? (GLsizeiptr)uboData.size() : kUboStride, nullptr, GL_DYNAMIC_DRAW);
        if (uboSub) glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubuf);
    }

    GLuint fbo, color, depth;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenRenderbuffers(1, &color);
    glBindRenderbuffer(GL_RENDERBUFFER, color);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, size, size);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color);
    glGenRenderbuffers(1, &depth);
    glBindRenderbuffer(GL_RENDERBUFFER, depth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, size, size);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);
    glViewport(0, 0, size, size);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    auto frame = [&](int f) {
        glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if (ubo) {
            glBindBuffer(GL_UNIFORM_BUFFER, ubuf);
            glBufferSubData(GL_UNIFORM_BUFFER, 0, (GLsizeiptr)uboData.size(), uboData.data());
        }
        int prog = -1;
        for (int d = 0; d < draws; d++) {
            int p = (d / batch) % programs;
            if (p != prog) {
                prog = p;
                glUseProgram(progs[p]);
                for (int t = 0; t < textures; t++) {
                    glActiveTexture(GL_TEXTURE0 + t);
                    glBindTexture(GL_TEXTURE_2D, tex[(p * 3 + t * 5) % 16]);
                }
            }
            float x = (float)((d * 37) % 64) / 32.0f - 1.0f, y = (float)((d * 91 + f) % 64) / 32.0f - 1.0f;
            float model[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, (float)(d % 7) * 0.1f, 1};
            glUniformMatrix4fv(locModel[p], 1, GL_FALSE, model);
            glUniform4f(locColor[p], 0.5f + 0.5f * (d & 1), 0.7f, 0.2f + 0.1f * (d % 5), 1.0f);
            if (ubo) glBindBufferRange(GL_UNIFORM_BUFFER, 0, ubuf, (GLintptr)d * kUboStride, 80);
            if (uboSub) glBufferSubData(GL_UNIFORM_BUFFER, 0, 80, &uboData[(size_t)d * kUboStride]);
            glBindVertexArray(vaos[d & 3]);
            glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, nullptr);
        }
        glFlush();
    };
    for (int f = 0; f < warmup; f++) frame(f);
    glFinish();
    auto t0 = std::chrono::steady_clock::now();
    for (int f = 0; f < frames; f++) frame(f);
    glFinish();
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / frames;
    uint8_t pxl[4] = {};
    glReadPixels(size / 2, size / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pxl);
    printf("drawbench (%s): %d draws, %d programs (batch %d), %d textures%s, %dx%d, %d frames\n", (const char*)glGetString(GL_RENDERER), draws,
           programs, batch, textures, ubo ? ", UBO ranges" : uboSub ? ", UBO updates" : "", size, size, frames);
    printf("  wall ms/frame: %8.3f   us/draw: %6.3f   probe pixel: %u %u %u %u\n", ms, ms * 1000.0 / draws, pxl[0], pxl[1], pxl[2], pxl[3]);
    CGLSetCurrentContext(nullptr);
    CGLDestroyContext(cgl);
    return 0;
}
