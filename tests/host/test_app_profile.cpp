// Application profiles (src/gl/app_profiles.cpp) and the video memory queries
// (GL_NVX_gpu_memory_info, GL_ATI_meminfo).
#include <cmath>
#include "harness.h"

extern "C" int mglTestResolveAppProfile(const char* exe, const char* const* env, char* out, int outSize);

namespace {
bool hasExtension(const char* name) {
    GLint n = 0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &n);
    for (GLint i = 0; i < n; i++)
        if (!strcmp((const char*)glGetStringi(GL_EXTENSIONS, i), name)) return true;
    return false;
}

std::string resolve(const char* exe, std::vector<const char*> env) {
    env.push_back(nullptr);
    char buf[2048];
    mglTestResolveAppProfile(exe, env.data(), buf, sizeof buf);
    return buf;
}

bool contains(const std::string& s, const char* sub) {
    if (s.find(sub) != std::string::npos) return true;
    fprintf(stderr, "  '%s' not in '%s'\n", sub, s.c_str());
    return false;
}
}  // namespace

TEST(nvx_ati_meminfo) {
    CHECK(hasExtension("GL_NVX_gpu_memory_info"));
    CHECK(hasExtension("GL_ATI_meminfo"));
    GLint dedicated = 0, total = 0, avail = -1, evictions = -1;
    glGetIntegerv(0x9047 /* GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX */, &dedicated);
    glGetIntegerv(0x9048 /* GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX */, &total);
    glGetIntegerv(0x9049 /* GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX */, &avail);
    glGetIntegerv(0x904A /* GPU_MEMORY_INFO_EVICTION_COUNT_NVX */, &evictions);
    CHECK_GL();
    CHECK(dedicated > 256 * 1024);  // KiB
    CHECK_EQ(total, dedicated);
    CHECK(avail >= 0 && avail <= total);
    CHECK_EQ(evictions, 0);
    for (GLenum pname : {GL_VBO_FREE_MEMORY_ATI, GL_TEXTURE_FREE_MEMORY_ATI, GL_RENDERBUFFER_FREE_MEMORY_ATI}) {
        GLint v[4] = {-1, -1, -1, -1};
        glGetIntegerv(pname, v);
        CHECK_GL();
        CHECK(v[0] > 0 && v[0] <= total);
        CHECK_EQ(v[1], v[0]);
        CHECK_EQ(v[2], 0);
        CHECK_EQ(v[3], 0);
    }
    GLint64 big = 0;
    glGetInteger64v(0x9048, &big);
    CHECK_EQ(big, (GLint64)total);
    // not indexed state
    GLint x;
    glGetIntegeri_v(0x9047, 0, &x);
    CHECK_EQ(glGetError(), (GLenum)GL_INVALID_ENUM);
}

// Without METALGLDRV_* overrides the strings are the driver's own (the test binary matches no profile).
TEST(app_profile_default_strings) {
    if (getenv("METALGLDRV_VENDOR") || getenv("METALGLDRV_RENDERER") || getenv("METALGLDRV_VERSION_STRING") ||
        getenv("METALGLDRV_APP_PROFILE") || getenv("METALGLDRV_HIDE_EXTENSIONS"))
        return;
    CHECK(!strcmp((const char*)glGetString(GL_VENDOR), "metalgldrv"));
    CHECK(!strncmp((const char*)glGetString(GL_VERSION), "4.6.0 ", 6));
    CHECK(!strncmp((const char*)glGetString(GL_SHADING_LANGUAGE_VERSION), "4.60", 4));
}

TEST(app_profile_resolution) {
    std::string s = resolve("glcts", {});
    CHECK(contains(s, "app profile 'default' (exe glcts)"));
    CHECK(s.find("vendor") == std::string::npos);
    // matched by executable name, case-insensitively
    s = resolve("doomx64.EXE", {});
    CHECK(contains(s, "app profile 'doom2016'"));
    CHECK(contains(s, "vendor \"ATI Technologies Inc.\""));
    CHECK(contains(s, "VRAM at most 8192 MB"));
    CHECK(contains(s, "GLSL quirks [vs_position_always_invariant]"));
    CHECK(contains(resolve("DOOMx64.exe", {"METALGLDRV_VENDOR=metalgldrv"}), "vendor \"metalgldrv\""));
    // forced by profile name or by executable name, or disabled
    CHECK(contains(resolve("foo", {"METALGLDRV_APP_PROFILE=doom2016"}), "'doom2016'"));
    CHECK(contains(resolve("foo", {"METALGLDRV_APP_PROFILE=DOOMx64.exe"}), "'doom2016'"));
    CHECK(contains(resolve("DOOMx64.exe", {"METALGLDRV_APP_PROFILE=none"}), "'default'"));
    // environment overrides
    s = resolve("foo", {"METALGLDRV_VENDOR=ACME", "METALGLDRV_RENDERER=Rocket", "METALGLDRV_VERSION_STRING=4.5.0 X",
                        "METALGLDRV_GLSL_VERSION_STRING=4.50", "METALGLDRV_HIDE_EXTENSIONS=GL_ARB_foo,GL_NV_*",
                        "METALGLDRV_EXTRA_EXTENSIONS=GL_ARB_bar GL_NV_baz", "METALGLDRV_VRAM_MB=4096",
                        "METALGLDRV_GLSL_QUIRKS=force_glsl_extensions_warn"});
    CHECK(contains(s, "vendor \"ACME\""));
    CHECK(contains(s, "renderer \"Rocket\""));
    CHECK(contains(s, "version \"4.5.0 X\""));
    CHECK(contains(s, "GLSL version \"4.50\""));
    CHECK(contains(s, "hide [GL_ARB_foo GL_NV_*]"));
    CHECK(contains(s, "extra [GL_ARB_bar]"));  // a hidden name is never added
    CHECK(contains(s, "VRAM 4096 MB"));
    CHECK(contains(s, "GLSL quirks [force_glsl_extensions_warn]"));
    // a quirk the profile sets can be turned off
    s = resolve("DOOMx64.exe", {"METALGLDRV_GLSL_QUIRKS=-vs_position_always_invariant"});
    CHECK(s.find("vs_position_always_invariant") == std::string::npos);
    CHECK(contains(resolve("x", {"METALGLDRV_LOG_UNSUPPORTED=1", "METALGLDRV_DUMP_DIR=/tmp/d"}), "dumps in /tmp/d"));
}

// glVertexAttribP{1,2,3}ui[v] (GL 3.3): the unpacked components, then the defaults (0, 0, 1).
TEST(vertex_attrib_packed_setters) {
    GLuint packed = (3u << 30) | (1023u << 20) | (0u << 10) | 511u;  // a=3 b=1023 g=0 r=511
    float v[4];
    glVertexAttribP1ui(3, GL_UNSIGNED_INT_2_10_10_10_REV, GL_FALSE, packed);
    glGetVertexAttribfv(3, GL_CURRENT_VERTEX_ATTRIB, v);
    CHECK(v[0] == 511 && v[1] == 0 && v[2] == 0 && v[3] == 1);
    glVertexAttribP2uiv(3, GL_UNSIGNED_INT_2_10_10_10_REV, GL_TRUE, &packed);
    glGetVertexAttribfv(3, GL_CURRENT_VERTEX_ATTRIB, v);
    CHECK(fabsf(v[0] - 511 / 1023.f) < 1e-6f && v[1] == 0 && v[2] == 0 && v[3] == 1);
    glVertexAttribP3ui(3, GL_UNSIGNED_INT_2_10_10_10_REV, GL_FALSE, packed);
    glGetVertexAttribfv(3, GL_CURRENT_VERTEX_ATTRIB, v);
    CHECK(v[0] == 511 && v[1] == 0 && v[2] == 1023 && v[3] == 1);
    glVertexAttribP4ui(3, GL_INT_2_10_10_10_REV, GL_FALSE, packed);
    glGetVertexAttribfv(3, GL_CURRENT_VERTEX_ATTRIB, v);
    CHECK(v[0] == 511 && v[1] == 0 && v[2] == -1 && v[3] == -1);
    // 10F_11F_11F: 1.0 = exponent 15, mantissa 0 in each field
    GLuint f = (15u << 6) | ((15u << 6) << 11) | ((15u << 5) << 22);
    glVertexAttribP3ui(3, GL_UNSIGNED_INT_10F_11F_11F_REV, GL_FALSE, f);
    glGetVertexAttribfv(3, GL_CURRENT_VERTEX_ATTRIB, v);
    CHECK(v[0] == 1 && v[1] == 1 && v[2] == 1 && v[3] == 1);
    CHECK_GL();
    glVertexAttribP2ui(3, GL_FLOAT, GL_FALSE, 0);
    CHECK_EQ(glGetError(), (GLenum)GL_INVALID_ENUM);
}

// EXT_framebuffer_object binds (DOOM 2016 binds every framebuffer through glBindFramebufferEXT): names
// that were never generated are created on first bind.
TEST(ext_framebuffer_object_binds) {
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 8, 8, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindFramebufferEXT(GL_FRAMEBUFFER, 77);
    CHECK_GL();
    GLint bound = 0;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &bound);
    CHECK_EQ(bound, 77);
    CHECK(glIsFramebuffer(77));
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    CHECK_EQ(glCheckFramebufferStatusEXT(GL_FRAMEBUFFER), (GLenum)GL_FRAMEBUFFER_COMPLETE);
    glClearColor(0, 1, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    CHECK_PIXEL(3, 3, 0, 255, 0, 255);
    glBindRenderbufferEXT(GL_RENDERBUFFER, 78);
    glGetIntegerv(GL_RENDERBUFFER_BINDING, &bound);
    CHECK_EQ(bound, 78);
    glBindRenderbufferEXT(GL_RENDERBUFFER, 0);
    glBindFramebufferEXT(GL_FRAMEBUFFER, 0);
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &bound);
    CHECK_EQ(bound, 0);
    GLuint f = 77, r = 78;
    glDeleteFramebuffers(1, &f);
    glDeleteRenderbuffers(1, &r);
    glDeleteTextures(1, &tex);
    CHECK_GL();
}

// DOOM 2016 compiles and links programs in one context of a share group and draws with them in another,
// with buffers and textures made in a third (it creates four shared 4.3 compatibility contexts).
TEST(shared_context_program_use) {
    EGLDisplay dpy = eglGetCurrentDisplay();
    EGLSurface surf = eglGetCurrentSurface(EGL_DRAW);
    EGLContext main = eglGetCurrentContext();
    EGLint cfgId = 0;
    eglQueryContext(dpy, main, EGL_CONFIG_ID, &cfgId);
    EGLint want[] = {EGL_CONFIG_ID, cfgId, EGL_NONE};
    EGLConfig cfg;
    EGLint n = 0;
    eglChooseConfig(dpy, want, &cfg, 1, &n);
    EGLint attr[] = {EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 3, EGL_CONTEXT_OPENGL_PROFILE_MASK,
                     EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT, EGL_NONE};
    EGLContext loader = eglCreateContext(dpy, cfg, main, attr);
    EGLContext compiler = eglCreateContext(dpy, cfg, loader, attr);
    CHECK(loader != EGL_NO_CONTEXT && compiler != EGL_NO_CONTEXT);

    // resources in the "loader" context
    eglMakeCurrent(dpy, surf, surf, loader);
    static const float pos[] = {-1, -1, 1, -1, -1, 1, 1, 1};
    GLuint vbo, tex;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof pos, pos, GL_STATIC_DRAW);
    const uint8_t texel[4] = {255, 128, 0, 255};
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, texel);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glFinish();

    // the program in the "compiler" context
    eglMakeCurrent(dpy, surf, surf, compiler);
    GLuint prog = compileProgram("#version 430 core\nin vec2 pos;\nvoid main() { gl_Position = vec4(pos, 0, 1); }",
                                 "#version 430 core\nuniform sampler2D t;\nout vec4 c;\nvoid main() { c = texture(t, vec2(0.5)); }");
    CHECK(prog != 0);

    // draw in the main context
    eglMakeCurrent(dpy, surf, surf, main);
    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog, "t"), 3);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, tex);
    glActiveTexture(GL_TEXTURE0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    GLint loc = glGetAttribLocation(prog, "pos");
    glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(loc);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    CHECK_PIXEL(10, 10, 255, 128, 0, 255);
    CHECK_GL();
    glDisableVertexAttribArray(loc);
    glUseProgram(0);
    glDeleteProgram(prog);
    glDeleteBuffers(1, &vbo);
    glDeleteTextures(1, &tex);
    eglDestroyContext(dpy, compiler);
    eglDestroyContext(dpy, loader);
}

namespace {
// A compute shader writes `words` into buffer `buf` (as DOOM 2016's page transcoder fills its PBOs).
void computeFill(GLuint buf, const std::vector<uint32_t>& words) {
    const char* src =
        "#version 430 core\nlayout(local_size_x = 1) in;\n"
        "layout(std430, binding = 0) buffer Out { uint o[]; };\nlayout(std430, binding = 1) buffer In { uint i[]; };\n"
        "void main() { o[gl_GlobalInvocationID.x] = i[gl_GlobalInvocationID.x]; }\n";
    GLuint s = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLuint p = glCreateProgram();
    glAttachShader(p, s);
    glLinkProgram(p);
    glDeleteShader(s);
    GLuint in;
    glGenBuffers(1, &in);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, in);
    glBufferData(GL_SHADER_STORAGE_BUFFER, words.size() * 4, words.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, in);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buf);
    glUseProgram(p);
    glDispatchCompute((GLuint)words.size(), 1, 1);
    glMemoryBarrier(GL_PIXEL_BUFFER_BARRIER_BIT);
    glUseProgram(0);
    glDeleteProgram(p);
    glDeleteBuffers(1, &in);
}
}  // namespace

// Texture uploads from a pixel unpack buffer written by the GPU (copied on the GPU when no conversion is needed).
TEST(pbo_upload_after_compute) {
    // RGBA8: a 4x4 image into texels (2..5, 1..4) of an 8x8 texture, with UNPACK_ROW_LENGTH 6 and skip 1 row
    GLuint pbo;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 64 * 4 + 64, nullptr, GL_DYNAMIC_COPY);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    std::vector<uint32_t> px(4 + 6 * 5);
    for (size_t i = 0; i < px.size(); i++) px[i] = 0xff000000u | (uint32_t)(i * 5) << 8 | (uint32_t)i;
    computeFill(pbo, px);
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 8, 8);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 6);
    glPixelStorei(GL_UNPACK_SKIP_ROWS, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 2, 1, 4, 4, GL_RGBA, GL_UNSIGNED_BYTE, (const void*)16);  // starts at word 4
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    CHECK_GL();
    std::vector<uint32_t> out(64);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, out.data());
    bool ok = true;
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++) ok &= out[(1 + y) * 8 + 2 + x] == px[4 + (1 + y) * 6 + x];
    CHECK(ok);

    // BC1: a compute shader writes two 4x4 blocks (opaque red, opaque blue) and they're uploaded side by side
    if (hasExtension("GL_EXT_texture_compression_s3tc")) {
        computeFill(pbo, {0, 0, 0x0000F800u, 0u, 0x0000001Fu, 0u});
        GLuint ct;
        glGenTextures(1, &ct);
        glBindTexture(GL_TEXTURE_2D, ct);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_COMPRESSED_RGBA_S3TC_DXT1_EXT, 8, 4);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glCompressedTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 8, 4, GL_COMPRESSED_RGBA_S3TC_DXT1_EXT, 16, (const void*)8);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        CHECK_GL();
        uint32_t blocks[4] = {};
        glGetCompressedTexImage(GL_TEXTURE_2D, 0, blocks);
        CHECK(blocks[0] == 0x0000F800u && blocks[1] == 0 && blocks[2] == 0x0000001Fu && blocks[3] == 0);
        CHECK_GL();
        glDeleteTextures(1, &ct);
    }
    glDeleteTextures(1, &tex);
    glDeleteBuffers(1, &pbo);
}

// METALGLDRV_EXTENSION_YEAR: the profile option, and the year table it uses.
extern "C" int mglTestExtensionYear(const char* name);
TEST(app_profile_extension_year) {
    CHECK(contains(resolve("quake3.exe", {"METALGLDRV_EXTENSION_YEAR=2003"}), "extension string up to 2003"));
    CHECK_EQ(mglTestExtensionYear("GL_ARB_multitexture"), 1998);
    CHECK_EQ(mglTestExtensionYear("GL_ARB_compute_shader"), 2012);
    CHECK_EQ(mglTestExtensionYear("GL_APPLE_fence"), 2002);
    CHECK_EQ(mglTestExtensionYear("GL_NOT_an_extension"), 0);
}
