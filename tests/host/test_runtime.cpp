// Runtime tests that don't require the shader compiler.
#include "harness.h"
#include <cmath>
#include <vector>

static GLuint makeTex2D(GLenum ifmt, int w, int h, GLenum fmt, GLenum type, const void* data) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, ifmt, w, h, 0, fmt, type, data);
    return t;
}

TEST(texformats_roundtrip) {
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    struct Case { GLenum ifmt, fmt, type; int bpp; };
    const Case cases[] = {
        {GL_R8, GL_RED, GL_UNSIGNED_BYTE, 1},       {GL_RG8, GL_RG, GL_UNSIGNED_BYTE, 2},
        {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, 4},   {GL_R16, GL_RED, GL_UNSIGNED_SHORT, 2},
        {GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT, 8},    {GL_R32F, GL_RED, GL_FLOAT, 4},
        {GL_RGBA32F, GL_RGBA, GL_FLOAT, 16},        {GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, 4},
        {GL_RGBA8I, GL_RGBA_INTEGER, GL_BYTE, 4},   {GL_RGB10_A2, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV, 4},
        {GL_R11F_G11F_B10F, GL_RGB, GL_UNSIGNED_INT_10F_11F_11F_REV, 4},
        {GL_RG16UI, GL_RG_INTEGER, GL_UNSIGNED_SHORT, 4},
        {GL_RGBA8, GL_BGRA, GL_UNSIGNED_BYTE, 4},   {GL_RGBA8, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, 4},
    };
    for (const Case& c : cases) {
        std::vector<uint8_t> in(8 * 4 * c.bpp), out(in.size(), 0xcd);
        for (size_t i = 0; i < in.size(); i++) in[i] = (uint8_t)(i * 37 + 11);
        if (c.type == GL_FLOAT)  // keep finite floats
            for (size_t i = 0; i < in.size() / 4; i++) ((float*)in.data())[i] = (float)i * 0.25f - 3.0f;
        if (c.type == GL_HALF_FLOAT)
            for (size_t i = 0; i < in.size() / 2; i++) ((uint16_t*)in.data())[i] = (uint16_t)(0x3c00 + i);
        if (c.type == GL_UNSIGNED_INT_10F_11F_11F_REV)
            for (size_t i = 0; i < in.size() / 4; i++) ((uint32_t*)in.data())[i] = 0x3c0 | (0x3c0u << 11) | (0x1e0u << 22);
        GLuint t = makeTex2D(c.ifmt, 8, 4, c.fmt, c.type, in.data());
        glGetTexImage(GL_TEXTURE_2D, 0, c.fmt, c.type, out.data());
        bool same = in == out;
        if (!same) {
            fprintf(stderr, "  format 0x%x (fmt 0x%x type 0x%x) mismatch: in %02x%02x%02x%02x out %02x%02x%02x%02x\n", c.ifmt, c.fmt, c.type, in[0],
                    in[1], in[2], in[3], out[0], out[1], out[2], out[3]);
        }
        CHECK(same);
        glDeleteTextures(1, &t);
    }
    CHECK_GL();
}

TEST(texture_pbo_upload) {
    GLuint pbo;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    uint32_t px[16 * 16];
    for (int i = 0; i < 256; i++) px[i] = 0xff000000u | (uint32_t)i * 0x010203u;
    glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeof px + 64, nullptr, GL_STREAM_DRAW);
    glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 64, sizeof px, px);
    GLuint t = makeTex2D(GL_RGBA8, 16, 16, GL_RGBA, GL_UNSIGNED_BYTE, (const void*)64);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    uint32_t out[256] = {};
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, out);
    CHECK(memcmp(px, out, sizeof px) == 0);
    // pack PBO readback
    GLuint pack;
    glGenBuffers(1, &pack);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pack);
    glBufferData(GL_PIXEL_PACK_BUFFER, sizeof px, nullptr, GL_STREAM_READ);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    const uint32_t* m = (const uint32_t*)glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, sizeof px, GL_MAP_READ_BIT);
    CHECK(m && memcmp(px, m, sizeof px) == 0);
    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glDeleteBuffers(1, &pbo);
    glDeleteBuffers(1, &pack);
    glDeleteTextures(1, &t);
    CHECK_GL();
}

TEST(texture_subimage_and_rowlength) {
    std::vector<uint32_t> base(32 * 32, 0xff0000ffu);
    GLuint t = makeTex2D(GL_RGBA8, 32, 32, GL_RGBA, GL_UNSIGNED_BYTE, base.data());
    // 4x4 patch taken from a 10-wide source image starting at (2,1)
    std::vector<uint32_t> src(10 * 6);
    for (int i = 0; i < 60; i++) src[i] = 0xff000000u | (uint32_t)i;
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 10);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, 2);
    glPixelStorei(GL_UNPACK_SKIP_ROWS, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 5, 7, 4, 4, GL_RGBA, GL_UNSIGNED_BYTE, src.data());
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
    glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
    std::vector<uint32_t> out(32 * 32);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, out.data());
    CHECK_EQ(out[7 * 32 + 5], 0xff000000u | 12u);   // src (2,1)
    CHECK_EQ(out[10 * 32 + 8], 0xff000000u | 45u);  // src (5,4)
    CHECK_EQ(out[6 * 32 + 5], 0xff0000ffu);
    glDeleteTextures(1, &t);
    CHECK_GL();
}

TEST(texture_mipmaps_generate) {
    std::vector<uint32_t> px(16 * 16, 0xff4080c0u);
    GLuint t = makeTex2D(GL_RGBA8, 16, 16, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
    glGenerateMipmap(GL_TEXTURE_2D);
    GLint w = 0;
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 4, GL_TEXTURE_WIDTH, &w);
    CHECK_EQ(w, 1);
    uint32_t out = 0;
    glGetTexImage(GL_TEXTURE_2D, 4, GL_RGBA, GL_UNSIGNED_BYTE, &out);
    CHECK_EQ(out, 0xff4080c0u);
    glDeleteTextures(1, &t);
    CHECK_GL();
}

TEST(texture_storage_and_levels) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D_ARRAY, t);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 3, GL_RGBA8, 8, 8, 4);
    GLint v = 0;
    glGetTexParameteriv(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_IMMUTABLE_LEVELS, &v);
    CHECK_EQ(v, 3);
    uint32_t layer[16];
    for (int i = 0; i < 16; i++) layer[i] = 0xff000000u | (uint32_t)(i + 1);
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 2, 0, 0, 3, 2, 2, 1, GL_RGBA, GL_UNSIGNED_BYTE, layer);
    std::vector<uint32_t> out(2 * 2 * 4);
    glGetTexImage(GL_TEXTURE_2D_ARRAY, 2, GL_RGBA, GL_UNSIGNED_BYTE, out.data());
    CHECK_EQ(out[3 * 4 + 0], layer[0]);
    CHECK_EQ(out[3 * 4 + 3], layer[3]);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, 8, 8, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    CHECK_EQ(glGetError(), (GLenum)GL_INVALID_OPERATION);  // immutable
    glDeleteTextures(1, &t);
    CHECK_GL();
}

TEST(texture_compressed_roundtrip) {
    // One BC1 block (8 bytes) for a 4x4 image: two 565 endpoints + indices.
    uint8_t block[8] = {0x00, 0xf8, 0x1f, 0x00, 0x00, 0x00, 0x00, 0x00};  // color0 = red, all indices 0
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glCompressedTexImage2D(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGBA_S3TC_DXT1_EXT, 4, 4, 0, 8, block);
    CHECK_GL();
    GLint compressed = 0, size = 0;
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_COMPRESSED, &compressed);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_COMPRESSED_IMAGE_SIZE, &size);
    CHECK_EQ(compressed, 1);
    CHECK_EQ(size, 8);
    uint8_t back[8] = {};
    glGetCompressedTexImage(GL_TEXTURE_2D, 0, back);
    CHECK(memcmp(block, back, 8) == 0);
    // Sampling path: copy into an FBO-readable texture is shader work; just check decode via blit.
    glDeleteTextures(1, &t);
    CHECK_GL();
}

TEST(copy_tex_and_copy_image) {
    glClearColor(0, 1, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_SCISSOR_TEST);
    glScissor(0, 0, 8, 8);
    glClearColor(1, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_SCISSOR_TEST);
    GLuint t = makeTex2D(GL_RGBA8, 16, 16, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 4, 4, 16, 16);  // window (4,4)-(20,20)
    uint32_t out[256];
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, out);
    CHECK_EQ(out[0], 0xff0000ffu);          // (4,4) red
    CHECK_EQ(out[4 * 16 + 4], 0xff00ff00u); // (8,8) green
    GLuint t2 = makeTex2D(GL_RGBA8, 16, 16, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    // §18.3.3: both textures must be complete under their own sampler state.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glCopyImageSubData(t, GL_TEXTURE_2D, 0, 0, 0, 0, t2, GL_TEXTURE_2D, 0, 8, 8, 0, 4, 4, 1);
    glBindTexture(GL_TEXTURE_2D, t2);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, out);
    CHECK_EQ(out[8 * 16 + 8], 0xff0000ffu);
    glDeleteTextures(1, &t);
    glDeleteTextures(1, &t2);
    CHECK_GL();
}

TEST(mrt_and_integer_clears) {
    GLuint tex[3], fb;
    glGenTextures(3, tex);
    GLenum fmts[3] = {GL_RGBA8, GL_RGBA32UI, GL_R32F};
    for (int i = 0; i < 3; i++) {
        glBindTexture(GL_TEXTURE_2D, tex[i]);
        glTexStorage2D(GL_TEXTURE_2D, 1, fmts[i], 8, 8);
    }
    glGenFramebuffers(1, &fb);
    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    for (int i = 0; i < 3; i++) glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, tex[i], 0);
    GLenum bufs[3] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
    glDrawBuffers(3, bufs);
    CHECK_EQ(glCheckFramebufferStatus(GL_FRAMEBUFFER), (GLenum)GL_FRAMEBUFFER_COMPLETE);
    float c0[4] = {0.5f, 0.25f, 1, 1};
    GLuint c1[4] = {7, 8, 9, 0xffffffffu};
    float c2[4] = {-3.5f, 0, 0, 0};
    glClearBufferfv(GL_COLOR, 0, c0);
    glClearBufferuiv(GL_COLOR, 1, c1);
    glClearBufferfv(GL_COLOR, 2, c2);
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    GLuint ui[4] = {};
    glReadPixels(1, 1, 1, 1, GL_RGBA_INTEGER, GL_UNSIGNED_INT, ui);
    CHECK_EQ(ui[0], 7u);
    CHECK_EQ(ui[3], 0xffffffffu);
    glReadBuffer(GL_COLOR_ATTACHMENT2);
    float f = 0;
    glReadPixels(2, 2, 1, 1, GL_RED, GL_FLOAT, &f);
    CHECK(f == -3.5f);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    CHECK_PIXEL(3, 3, 128, 64, 255, 255);
    // masked + scissored clear on attachment 0 (quad path)
    glEnable(GL_SCISSOR_TEST);
    glScissor(4, 4, 4, 4);
    glColorMaski(0, GL_TRUE, GL_FALSE, GL_FALSE, GL_FALSE);
    float c3[4] = {0, 1, 0, 0};
    glClearBufferfv(GL_COLOR, 0, c3);
    glColorMaski(0, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDisable(GL_SCISSOR_TEST);
    CHECK_PIXEL(5, 5, 0, 64, 255, 255);
    CHECK_PIXEL(3, 3, 128, 64, 255, 255);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fb);
    glDeleteTextures(3, tex);
    CHECK_GL();
}

TEST(msaa_resolve_blit) {
    GLuint rb, fb;
    glGenRenderbuffers(1, &rb);
    glBindRenderbuffer(GL_RENDERBUFFER, rb);
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_RGBA8, 16, 16);
    GLint samples = 0;
    glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_SAMPLES, &samples);
    CHECK_EQ(samples, 4);
    glGenFramebuffers(1, &fb);
    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
    CHECK_EQ(glCheckFramebufferStatus(GL_FRAMEBUFFER), (GLenum)GL_FRAMEBUFFER_COMPLETE);
    glClearColor(0.25f, 0.5f, 0.75f, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glBlitFramebuffer(0, 0, 16, 16, 0, 0, 16, 16, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    CHECK_PIXEL(8, 8, 64, 128, 191, 255);
    glDeleteFramebuffers(1, &fb);
    glDeleteRenderbuffers(1, &rb);
    CHECK_GL();
}

TEST(fbo_completeness_errors) {
    GLuint fb;
    glGenFramebuffers(1, &fb);
    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    CHECK_EQ(glCheckFramebufferStatus(GL_FRAMEBUFFER), (GLenum)GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT);
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t, 0);  // no image yet
    CHECK_EQ(glCheckFramebufferStatus(GL_FRAMEBUFFER), (GLenum)GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT);
    glClear(GL_COLOR_BUFFER_BIT);
    CHECK_EQ(glGetError(), (GLenum)GL_INVALID_FRAMEBUFFER_OPERATION);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT9, GL_TEXTURE_2D, t, 0);
    CHECK_EQ(glGetError(), (GLenum)GL_INVALID_OPERATION);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fb);
    glDeleteTextures(1, &t);
    glEnable(0x1234);
    CHECK_EQ(glGetError(), (GLenum)GL_INVALID_ENUM);
    glViewport(0, 0, -1, 1);
    CHECK_EQ(glGetError(), (GLenum)GL_INVALID_VALUE);
    CHECK_GL();
}

TEST(buffer_storage_persistent) {
    GLuint b;
    glCreateBuffers(1, &b);
    glNamedBufferStorage(b, 4096, nullptr, GL_MAP_WRITE_BIT | GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
    uint32_t* p = (uint32_t*)glMapNamedBufferRange(b, 0, 4096, GL_MAP_WRITE_BIT | GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
    CHECK(p != nullptr);
    if (p) {
        for (int i = 0; i < 1024; i++) p[i] = i;
        GLsync s = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        GLenum r = glClientWaitSync(s, GL_SYNC_FLUSH_COMMANDS_BIT, 1000000000ull);
        CHECK(r == GL_ALREADY_SIGNALED || r == GL_CONDITION_SATISFIED);
        glDeleteSync(s);
        uint32_t v = 0;
        glGetNamedBufferSubData(b, 4 * 100, 4, &v);
        CHECK_EQ(v, 100u);
    }
    glNamedBufferData(b, 16, nullptr, GL_STATIC_DRAW);
    CHECK_EQ(glGetError(), (GLenum)GL_INVALID_OPERATION);  // immutable
    glUnmapNamedBuffer(b);
    glDeleteBuffers(1, &b);
    CHECK_GL();
}

TEST(timer_query) {
    GLuint q;
    glGenQueries(1, &q);
    glBeginQuery(GL_TIME_ELAPSED, q);
    for (int i = 0; i < 20; i++) {
        glClearColor(i / 20.0f, 0, 0, 1);
        glEnable(GL_SCISSOR_TEST);
        glScissor(i, i, 10, 10);
        glClear(GL_COLOR_BUFFER_BIT);
    }
    glDisable(GL_SCISSOR_TEST);
    glEndQuery(GL_TIME_ELAPSED);
    GLuint64 ns = 0;
    glGetQueryObjectui64v(q, GL_QUERY_RESULT, &ns);
    CHECK(ns > 0 && ns < 1000000000ull);
    GLint avail = 0;
    glGetQueryObjectiv(q, GL_QUERY_RESULT_AVAILABLE, &avail);
    CHECK_EQ(avail, 1);
    glDeleteQueries(1, &q);
    GLint64 ts = 0;
    glGetInteger64v(GL_TIMESTAMP, &ts);
    CHECK(ts > 0);
    CHECK_GL();
}

TEST(debug_output) {
    struct Rec { int count = 0; GLenum lastType = 0; std::string last; } rec;
    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback([](GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* user) {
        Rec* r = (Rec*)user;
        r->count++;
        r->lastType = type;
        r->last = message;
    }, &rec);
    glDebugMessageInsert(GL_DEBUG_SOURCE_APPLICATION, GL_DEBUG_TYPE_MARKER, 1, GL_DEBUG_SEVERITY_NOTIFICATION, -1, "hello");
    CHECK_EQ(rec.count, 1);
    glBindBuffer(0xdead, 0);  // generates an error message
    CHECK_EQ(glGetError(), (GLenum)GL_INVALID_ENUM);
    CHECK_EQ(rec.count, 2);
    CHECK_EQ(rec.lastType, (GLenum)GL_DEBUG_TYPE_ERROR);
    // error messages name the error and the entry point the application called (without any logging enabled)
    CHECK(rec.last.find("GL_INVALID_ENUM in glBindBuffer") == 0);
    glGetStringi(GL_EXTENSIONS, 1u << 20);
    CHECK_EQ(glGetError(), (GLenum)GL_INVALID_VALUE);
    CHECK(rec.last.find("GL_INVALID_VALUE in glGetStringi") == 0);
    glDrawArraysInstanced(GL_TRIANGLES, 0, -1, 1);  // a wrapper: named after the entry point implementing it
    CHECK_EQ(glGetError(), (GLenum)GL_INVALID_VALUE);
    CHECK(rec.last == "GL_INVALID_VALUE in glDrawArraysInstancedBaseInstance");
    glDebugMessageCallback(nullptr, nullptr);
    GLuint b;
    glGenBuffers(1, &b);
    glBindBuffer(GL_ARRAY_BUFFER, b);
    glObjectLabel(GL_BUFFER, b, -1, "vbo");
    char label[16] = {};
    GLsizei len = 0;
    glGetObjectLabel(GL_BUFFER, b, sizeof label, &len, label);
    CHECK_EQ(len, 3);
    CHECK(strcmp(label, "vbo") == 0);
    glDeleteBuffers(1, &b);
    CHECK_GL();
}
