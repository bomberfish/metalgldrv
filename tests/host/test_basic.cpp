#include "harness.h"

TEST(strings) {
    const char* v = (const char*)glGetString(GL_VERSION);
    CHECK(v && strncmp(v, "4.6", 3) == 0);
    GLint major = 0, minor = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);
    CHECK_EQ(major, 4);
    CHECK_EQ(minor, 6);
    GLint n = 0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &n);
    CHECK(n > 50);
    CHECK_GL();
}

TEST(clear_color) {
    glClearColor(1, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    CHECK_PIXEL(0, 0, 255, 0, 0, 255);
    CHECK_PIXEL(63, 63, 255, 0, 0, 255);
    glClearColor(0, 0.5f, 1, 0.25f);
    glClear(GL_COLOR_BUFFER_BIT);
    CHECK_PIXEL(10, 20, 0, 128, 255, 64);
    CHECK_GL();
}

TEST(clear_scissor) {
    glClearColor(0, 0, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_SCISSOR_TEST);
    glScissor(0, 0, 16, 8);  // bottom-left corner in GL coordinates
    glClearColor(0, 1, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_SCISSOR_TEST);
    CHECK_PIXEL(0, 0, 0, 255, 0, 255);
    CHECK_PIXEL(15, 7, 0, 255, 0, 255);
    CHECK_PIXEL(16, 7, 0, 0, 255, 255);
    CHECK_PIXEL(0, 8, 0, 0, 255, 255);
    CHECK_PIXEL(63, 63, 0, 0, 255, 255);
    CHECK_GL();
}

TEST(clear_colormask) {
    glClearColor(1, 1, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glColorMask(GL_FALSE, GL_TRUE, GL_FALSE, GL_TRUE);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    CHECK_PIXEL(5, 5, 255, 0, 255, 0);
    CHECK_GL();
}

TEST(texture_roundtrip) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    uint8_t px[4 * 4 * 4];
    for (int i = 0; i < 16; i++) {
        px[i * 4 + 0] = i * 16;
        px[i * 4 + 1] = 255 - i * 16;
        px[i * 4 + 2] = i;
        px[i * 4 + 3] = 200;
    }
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    uint8_t out[64] = {};
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, out);
    CHECK(memcmp(px, out, 64) == 0);
    // RGB upload with alignment 1, read back as RGBA (alpha = 1)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    uint8_t rgb[3 * 3 * 3];
    for (int i = 0; i < 27; i++) rgb[i] = i * 9;
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, 3, 3, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb);
    uint8_t o2[36] = {};
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, o2);
    CHECK_EQ(o2[0], 0);
    CHECK_EQ(o2[4], 27);
    CHECK_EQ(o2[3], 255);
    CHECK_EQ(o2[35], 255);
    CHECK_EQ(o2[34], 26 * 9);
    glDeleteTextures(1, &t);
    CHECK_GL();
}

TEST(fbo_texture_clear) {
    GLuint t, fb;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 32, 32, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glGenFramebuffers(1, &fb);
    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t, 0);
    CHECK_EQ(glCheckFramebufferStatus(GL_FRAMEBUFFER), (GLenum)GL_FRAMEBUFFER_COMPLETE);
    glClearColor(0.2f, 0.4f, 0.6f, 0.8f);
    glClear(GL_COLOR_BUFFER_BIT);
    CHECK_PIXEL(31, 31, 51, 102, 153, 204);
    // blit into the default framebuffer, scaled
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glBlitFramebuffer(0, 0, 32, 32, 0, 0, 16, 16, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    CHECK_PIXEL(0, 0, 51, 102, 153, 204);
    CHECK_PIXEL(15, 15, 51, 102, 153, 204);
    CHECK_PIXEL(16, 16, 0, 0, 0, 255);
    glDeleteFramebuffers(1, &fb);
    glDeleteTextures(1, &t);
    CHECK_GL();
}

TEST(buffer_roundtrip) {
    GLuint b;
    glGenBuffers(1, &b);
    glBindBuffer(GL_ARRAY_BUFFER, b);
    uint32_t data[256];
    for (int i = 0; i < 256; i++) data[i] = i * 3;
    glBufferData(GL_ARRAY_BUFFER, sizeof data, data, GL_STATIC_DRAW);
    uint32_t sub[4] = {7, 8, 9, 10};
    glBufferSubData(GL_ARRAY_BUFFER, 16, 16, sub);
    uint32_t out[256];
    glGetBufferSubData(GL_ARRAY_BUFFER, 0, sizeof out, out);
    CHECK_EQ(out[0], 0u);
    CHECK_EQ(out[4], 7u);
    CHECK_EQ(out[7], 10u);
    CHECK_EQ(out[8], 24u);
    void* p = glMapBufferRange(GL_ARRAY_BUFFER, 0, 16, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_RANGE_BIT);
    CHECK(p != nullptr);
    if (p) memset(p, 0xff, 16);
    CHECK(glUnmapBuffer(GL_ARRAY_BUFFER));
    glGetBufferSubData(GL_ARRAY_BUFFER, 0, 4, out);
    CHECK_EQ(out[0], 0xffffffffu);
    GLuint b2;
    glGenBuffers(1, &b2);
    glBindBuffer(GL_COPY_WRITE_BUFFER, b2);
    glBufferData(GL_COPY_WRITE_BUFFER, 64, nullptr, GL_STREAM_COPY);
    glCopyBufferSubData(GL_ARRAY_BUFFER, GL_COPY_WRITE_BUFFER, 16, 0, 16);
    glGetBufferSubData(GL_COPY_WRITE_BUFFER, 0, 16, out);
    CHECK_EQ(out[0], 7u);
    glDeleteBuffers(1, &b);
    glDeleteBuffers(1, &b2);
    CHECK_GL();
}

TEST(depth_clear_readback) {
    glClearDepth(0.25);
    glClear(GL_DEPTH_BUFFER_BIT);
    float d = 0;
    glReadPixels(3, 3, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &d);
    CHECK(fabsf(d - 0.25f) < 1e-6f);
    glClearStencil(0x5a);
    glClear(GL_STENCIL_BUFFER_BIT);
    uint8_t s = 0;
    glReadPixels(3, 3, 1, 1, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, &s);
    CHECK_EQ(s, 0x5a);
    CHECK_GL();
}
