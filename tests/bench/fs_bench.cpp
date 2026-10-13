// fsbench: GPU time of a full-screen fragment shader pass on metalgldrv and on Apple's OpenGL (built as `fsbench`,
// linked against metalgldrv's CGL, and `fsbench-apple`, linked against OpenGL.framework). For comparing how the two
// compile a given shader, e.g. one dumped from a game with METALGLDRV_DUMP_GLSL.
//
//   fsbench --fs FILE [--size WxH] [--target rgba8|rgba16f] [--passes N] [--frames N]
//           [--tex UNIT:FORMAT:WxH:FILL ...] [--uniform NAME=x[,y[,z[,w]]] ...]
//
// The vertex shader draws a full-screen triangle with outputs s_texcoord_0 .. s_texcoord_10 (vec4): xy = the
// position's texture coordinate, zw = 0 (s_texcoord_2 = a view ray (x, y, -1, 0)). Sampler uniforms s_texture_N use
// texture unit N. FORMAT is rgba8, rgba16f, rg32f or r8; FILL is rand, or gbuf (rg32f: x = a depth from 2 to 3, y =
// the bits of a packed random normal, as Unigine's deferred buffers). Unset uniforms keep 0. Each frame runs --passes
// passes into the target (blended additively, so that each is shaded), then glFlush; reports wall ms per pass over
// the frames after glFinish, and the sum of the target's pixels (to compare the drivers' results).
// FSB_LOGTEX=1: logs the Metal texture descriptors the driver creates. FSB_TEXSTORAGE=1: non-rg32f textures are immutable with one level (glTexStorage2D). FSB_CAPTURE=<file.gputrace> with
// MTL_CAPTURE_ENABLED=1: a Metal GPU capture of one frame (Apple's OpenGL runs on Metal too: its shaders are in it).
//
// Unigine Heaven's SSAO pass (2026-10-13): Apple's OpenGL 2.1 ms per pass, metalgldrv 7.2 ms, same image and the
// same compiled shader (METALGLDRV_SHADER_STATS); the difference was the textures' GPU-optimized (compressed) layout,
// which scattered fetches pay for: 32-bit float textures now go without it (texture.mm).
#include <OpenGL/OpenGL.h>
#include <OpenGL/gl3.h>
#include <dlfcn.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

bool mtlCaptureStart(const char* path);
void mtlCaptureStop();
void mtlLogTextures();

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

const char* kVS = R"(#version 150
out vec4 s_texcoord_0, s_texcoord_1, s_texcoord_2, s_texcoord_3, s_texcoord_4, s_texcoord_5, s_texcoord_6, s_texcoord_7,
         s_texcoord_8, s_texcoord_9, s_texcoord_10;
void main() {
    vec2 p = vec2((gl_VertexID & 1) * 4 - 1, (gl_VertexID & 2) * 2 - 1);
    vec2 uv = p * 0.5 + 0.5;
    s_texcoord_0 = vec4(uv, 0.0, 0.0);
    s_texcoord_1 = vec4(uv * 64.0, 0.0, 0.0);
    s_texcoord_2 = vec4(p, -1.0, 0.0);
    s_texcoord_3 = s_texcoord_4 = s_texcoord_5 = s_texcoord_6 = s_texcoord_7 = s_texcoord_8 = s_texcoord_9 = s_texcoord_10 = vec4(uv, 0.0, 1.0);
    gl_Position = vec4(p, 0.0, 1.0);
}
)";

struct TexSpec {
    int unit = 0, w = 0, h = 0;
    std::string format, fill;
};

GLuint makeTexture(const TexSpec& t, std::mt19937& rng) {
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glActiveTexture(GL_TEXTURE0 + t.unit);
    glBindTexture(GL_TEXTURE_2D, tex);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    size_t n = (size_t)t.w * t.h;
    if (t.format == "rg32f") {
        std::vector<float> d(n * 2);
        for (size_t i = 0; i < n; i++) {
            if (t.fill == "gbuf") {
                d[i * 2] = 2.0f + u(rng);
                uint32_t bits = (uint32_t)(u(rng) * 255) | (uint32_t)(u(rng) * 255) << 8 | (uint32_t)(u(rng) * 255) << 16;
                memcpy(&d[i * 2 + 1], &bits, 4);
            } else {
                d[i * 2] = u(rng);
                d[i * 2 + 1] = u(rng);
            }
        }
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, t.w, t.h, 0, GL_RG, GL_FLOAT, d.data());
    } else if (t.format == "rgba16f" || t.format == "rgba8" || t.format == "r8") {
        std::vector<float> d(n * 4);
        for (float& x : d) x = u(rng);
        GLenum internal = t.format == "rgba16f" ? GL_RGBA16F : t.format == "rgba8" ? GL_RGBA8 : GL_R8;
        if (getenv("FSB_TEXSTORAGE")) {
            auto texStorage2D = (void (*)(GLenum, GLsizei, GLenum, GLsizei, GLsizei))dlsym(RTLD_DEFAULT, "glTexStorage2D");
            if (!texStorage2D) {
                fprintf(stderr, "no glTexStorage2D\n");
                exit(1);
            }
            texStorage2D(GL_TEXTURE_2D, 1, internal, t.w, t.h);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, t.w, t.h, GL_RGBA, GL_FLOAT, d.data());
        } else {
            glTexImage2D(GL_TEXTURE_2D, 0, internal, t.w, t.h, 0, GL_RGBA, GL_FLOAT, d.data());
        }
    } else {
        fprintf(stderr, "unknown texture format %s\n", t.format.c_str());
        exit(1);
    }
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    return tex;
}

}  // namespace

int main(int argc, char** argv) {
    std::string fsPath, target = "rgba16f";
    int width = 960, height = 540, passes = 20, frames = 30;
    std::vector<TexSpec> texs;
    std::vector<std::pair<std::string, std::vector<float>>> uniforms;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s needs a value\n", a.c_str());
                exit(1);
            }
            return argv[++i];
        };
        if (a == "--fs") fsPath = next();
        else if (a == "--size") sscanf(next().c_str(), "%dx%d", &width, &height);
        else if (a == "--target") target = next();
        else if (a == "--passes") passes = atoi(next().c_str());
        else if (a == "--frames") frames = atoi(next().c_str());
        else if (a == "--tex") {
            TexSpec t;
            char fmt[32] = {}, fill[32] = {};
            if (sscanf(next().c_str(), "%d:%31[^:]:%dx%d:%31s", &t.unit, fmt, &t.w, &t.h, fill) != 5) {
                fprintf(stderr, "--tex UNIT:FORMAT:WxH:FILL\n");
                return 1;
            }
            t.format = fmt;
            t.fill = fill;
            texs.push_back(t);
        } else if (a == "--uniform") {
            std::string v = next();
            size_t eq = v.find('=');
            std::vector<float> vals;
            std::stringstream ss(v.substr(eq + 1));
            for (std::string x; std::getline(ss, x, ',');) vals.push_back(strtof(x.c_str(), nullptr));
            uniforms.push_back({v.substr(0, eq), vals});
        } else {
            fprintf(stderr, "usage: see the comment at the top of tests/bench/fs_bench.cpp\n");
            return 1;
        }
    }
    if (fsPath.empty()) {
        fprintf(stderr, "--fs FILE is required\n");
        return 1;
    }
    std::ifstream f(fsPath);
    std::stringstream fsSrc;
    fsSrc << f.rdbuf();

    if (getenv("FSB_LOGTEX")) mtlLogTextures();
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
    CGLSetCurrentContext(cgl);
    printf("%s (%s): %s, %dx%d %s, %d passes x %d frames\n", (const char*)glGetString(GL_VERSION), (const char*)glGetString(GL_RENDERER),
           fsPath.c_str(), width, height, target.c_str(), passes, frames);

    GLuint prog = glCreateProgram();
    glAttachShader(prog, compile(GL_VERTEX_SHADER, kVS));
    glAttachShader(prog, compile(GL_FRAGMENT_SHADER, fsSrc.str()));
    glBindFragDataLocation(prog, 0, "s_frag_color");
    glLinkProgram(prog);
    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096] = {};
        glGetProgramInfoLog(prog, sizeof log, nullptr, log);
        fprintf(stderr, "link failed: %s\n", log);
        return 1;
    }
    glUseProgram(prog);
    for (int u = 0; u < 16; u++) {
        GLint loc = glGetUniformLocation(prog, ("s_texture_" + std::to_string(u)).c_str());
        if (loc >= 0) glUniform1i(loc, u);
    }
    for (auto& [name, v] : uniforms) {
        GLint loc = glGetUniformLocation(prog, name.c_str());
        if (loc < 0) {
            fprintf(stderr, "note: uniform %s is not active\n", name.c_str());
            continue;
        }
        switch (v.size()) {
            case 1: glUniform1f(loc, v[0]); break;
            case 2: glUniform2f(loc, v[0], v[1]); break;
            case 3: glUniform3f(loc, v[0], v[1], v[2]); break;
            default: glUniform4f(loc, v[0], v[1], v[2], v[3]); break;
        }
    }
    std::mt19937 rng(1);
    for (auto& t : texs) makeTexture(t, rng);

    GLuint color = 0, fbo = 0, vao = 0;
    glGenTextures(1, &color);
    glActiveTexture(GL_TEXTURE15);
    glBindTexture(GL_TEXTURE_2D, color);
    glTexImage2D(GL_TEXTURE_2D, 0, target == "rgba8" ? GL_RGBA8 : GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "framebuffer incomplete\n");
        return 1;
    }
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glViewport(0, 0, width, height);
    // Additive blending: every pass is shaded (opaque full-screen draws would let the GPU's hidden-surface removal
    // shade only the last one).
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);

    auto frame = [&]() {
        for (int p = 0; p < passes; p++) glDrawArrays(GL_TRIANGLES, 0, 3);
        glFlush();
    };
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    for (int i = 0; i < 5; i++) frame();
    glFinish();
    // FSB_CAPTURE=<file.gputrace> (with MTL_CAPTURE_ENABLED=1): a Metal GPU capture of one frame, which shows the
    // driver's Metal shaders (also Apple's OpenGL's, which runs on Metal).
    if (const char* cap = getenv("FSB_CAPTURE")) {
        if (!mtlCaptureStart(cap)) fprintf(stderr, "Metal capture failed to start\n");
        frame();
        glFinish();
        mtlCaptureStop();
    }
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < frames; i++) frame();
    glFinish();
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    float px[4] = {};
    glReadPixels(width / 2, height / 2, 1, 1, GL_RGBA, GL_FLOAT, px);
    std::vector<float> all((size_t)width * height * 4);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT, all.data());
    double sum = 0;
    for (float v : all) sum += v;
    printf("  image sum %.6g\n", sum);
    printf("  %.3f ms per pass (%.1f ns per pixel)   probe %.4f %.4f %.4f %.4f   GL error 0x%x\n", ms / (frames * passes),
           ms * 1e6 / ((double)frames * passes * width * height), px[0], px[1], px[2], px[3], glGetError());
    CGLSetCurrentContext(nullptr);
    CGLDestroyContext(cgl);
    return 0;
}
