// METALGLDRV_STRICT features. The mode is read once at startup, so these tests check whichever mode
// the process runs in: run mgltest with METALGLDRV_STRICT=1 as well to cover the strict paths.
#include "harness.h"

namespace {
GLuint computeProgram(const char* src) {
    GLuint s = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(s, sizeof log, nullptr, log);
        fprintf(stderr, "%s\n", log);
    }
    GLuint p = glCreateProgram();
    glAttachShader(p, s);
    glDeleteShader(s);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    CHECK(ok);
    return p;
}

GLint maxImageSamples() {
    GLint n = -1;
    glGetIntegerv(GL_MAX_IMAGE_SAMPLES, &n);
    return n;
}

// Reads every sample of layer `layer` of a multisample texture (texelFetch) into `out`.
template <typename T>
void readSamples(GLenum target, GLuint tex, const char* samplerType, const char* vecType, int w, int h, int samples, int layer,
                 std::vector<T>& out) {
    bool arr = target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY;
    char src[1024];
    snprintf(src, sizeof src,
             "#version 430 core\nlayout(local_size_x = 1) in;\nuniform %s u_ms;\nuniform int u_layer;\n"
             "layout(std430, binding = 0) buffer O { %s o[]; };\n"
             "void main() { ivec2 p = ivec2(gl_GlobalInvocationID.xy); int s = int(gl_GlobalInvocationID.z);\n"
             "  o[(p.y * %d + p.x) * %d + s] = texelFetch(u_ms, %s, s); }\n",
             samplerType, vecType, w, samples, arr ? "ivec3(p, u_layer)" : "p");
    GLuint p = computeProgram(src);
    GLuint b;
    glGenBuffers(1, &b);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, b);
    glBufferData(GL_SHADER_STORAGE_BUFFER, w * h * samples * 16, nullptr, GL_DYNAMIC_READ);
    glUseProgram(p);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(target, tex);
    glUniform1i(glGetUniformLocation(p, "u_ms"), 0);
    glUniform1i(glGetUniformLocation(p, "u_layer"), layer);
    glDispatchCompute(w, h, samples);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    out.resize(w * h * samples * 4);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, w * h * samples * 16, out.data());
    glDeleteBuffers(1, &b);
    glDeleteProgram(p);
}
}  // namespace

// gl_MaxImageSamples agrees with GL_MAX_IMAGE_SAMPLES (0 by default, 4 under METALGLDRV_STRICT).
TEST(strict_max_image_samples_constant) {
    GLuint p = computeProgram(
        "#version 430 core\nlayout(local_size_x = 1) in;\nlayout(std430, binding = 0) buffer O { int o; };\n"
        "void main() { o = gl_MaxImageSamples; }\n");
    GLuint b;
    glGenBuffers(1, &b);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, b);
    int init = -1;
    glBufferData(GL_SHADER_STORAGE_BUFFER, 4, &init, GL_DYNAMIC_READ);
    glUseProgram(p);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    int v = -2;
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 4, &v);
    CHECK_EQ(v, maxImageSamples());
    CHECK(maxImageSamples() == 0 || maxImageSamples() == 4);
    glDeleteBuffers(1, &b);
    glDeleteProgram(p);
    CHECK_GL();
}

// Multisample image read-modify-write from a compute shader: the shadow starts from the texture's
// current (cleared) contents, and the writes reach every sample of the texture.
TEST(strict_ms_image_compute_rmw) {
    if (maxImageSamples() < 4) return;
    const int W = 7, H = 5, S = 4;
    GLuint tex, fbo;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, tex);
    glTexStorage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, S, GL_RGBA32F, W, H, GL_FALSE);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, tex, 0);
    const float clear[4] = {0.5f, 0.25f, 0.125f, 1.0f};
    glClearBufferfv(GL_COLOR, 0, clear);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    GLuint p = computeProgram(
        "#version 430 core\nlayout(local_size_x = 1) in;\nlayout(rgba32f, binding = 2) uniform image2DMS u_img;\n"
        "void main() { ivec2 p = ivec2(gl_GlobalInvocationID.xy); int s = int(gl_GlobalInvocationID.z);\n"
        "  if (imageSamples(u_img) != 4 || imageSize(u_img) != ivec2(7, 5)) return;\n"
        "  imageStore(u_img, p, s, imageLoad(u_img, p, s) + vec4(p, s, 0)); }\n");
    glBindImageTexture(2, tex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
    glUseProgram(p);
    glDispatchCompute(W, H, S);
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
    std::vector<float> got;
    readSamples(GL_TEXTURE_2D_MULTISAMPLE, tex, "sampler2DMS", "vec4", W, H, S, 0, got);
    int bad = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            for (int s = 0; s < S; s++) {
                const float* v = &got[((y * W + x) * S + s) * 4];
                if (v[0] != 0.5f + x || v[1] != 0.25f + y || v[2] != 0.125f + s || v[3] != 1.0f) bad++;
            }
    CHECK_EQ(bad, 0);
    glDeleteProgram(p);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    CHECK_GL();
}

// Non-layered binding of one layer of a multisample array, with image atomics from a fragment
// shader; the other layer keeps its contents.
TEST(strict_ms_image_array_layer_atomics) {
    if (maxImageSamples() < 4) return;
    const int W = 8, H = 8, S = 4;
    GLuint tex, fbo;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, tex);
    glTexStorage3DMultisample(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, S, GL_R32UI, W, H, 2, GL_FALSE);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    for (int l = 0; l < 2; l++) {
        glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex, 0, l);
        const GLuint c[4] = {10u + l, 0, 0, 0};
        glClearBufferuiv(GL_COLOR, 0, c);
    }
    // draw into an unrelated single-sample target; the fragment shader only does image atomics
    GLuint rt;
    glGenTextures(1, &rt);
    glBindTexture(GL_TEXTURE_2D, rt);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, W, H);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt, 0);
    glViewport(0, 0, W, H);
    GLuint prog = compileProgram(
        "#version 430 core\nvoid main() { gl_Position = vec4(vec2((gl_VertexID & 1) * 4 - 1, (gl_VertexID >> 1) * 4 - 1), 0, 1); }\n",
        "#version 430 core\nlayout(r32ui, binding = 1) uniform uimage2DMS u_img;\nout vec4 o;\n"
        "void main() { ivec2 p = ivec2(gl_FragCoord.xy);\n"
        "  for (int s = 0; s < imageSamples(u_img); s++) imageAtomicAdd(u_img, p, s, uint(s + 1));\n"
        "  o = vec4(1); }\n");
    CHECK(prog);
    glBindImageTexture(1, tex, 0, GL_FALSE, 1, GL_READ_WRITE, GL_R32UI);
    glUseProgram(prog);
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    for (int l = 0; l < 2; l++) {
        std::vector<uint32_t> got;
        readSamples(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, tex, "usampler2DMSArray", "uvec4", W, H, S, l, got);
        int bad = 0;
        for (int i = 0; i < W * H * S; i++) {
            uint32_t want = l == 0 ? 10u : 11u + 2u * (uint32_t)(i % S + 1);
            if (got[i * 4] != want) bad++;
        }
        CHECK_EQ(bad, 0);
    }
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);
    glDeleteFramebuffers(1, &fbo);
    GLuint texs[2] = {tex, rt};
    glDeleteTextures(2, texs);
    CHECK_GL();
}
