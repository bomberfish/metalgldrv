// Texture copy / pixel storage / format tests.
#include "harness.h"
#include <algorithm>
#include <cstdint>
#include <cstring>

// RGB9_E5 data must round-trip bit-exactly (non-canonical encodings included)
// through mutable multi-level 3D textures and glCopyImageSubData.
TEST(copy_image_rgb9e5_3d_bitexact) {
    const uint32_t v = 0xf8fc0000u;  // non-canonical encoding of 8064.0
    std::vector<uint32_t> l0(4 * 4 * 4, v), l1(2 * 2 * 2, v);
    GLuint t[2];
    glGenTextures(2, t);
    for (GLuint name : t) {
        glBindTexture(GL_TEXTURE_3D, name);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB9_E5, 4, 4, 4, 0, GL_RGB, GL_UNSIGNED_INT_5_9_9_9_REV, l0.data());
        glTexImage3D(GL_TEXTURE_3D, 1, GL_RGB9_E5, 2, 2, 2, 0, GL_RGB, GL_UNSIGNED_INT_5_9_9_9_REV, l1.data());
        glTexImage3D(GL_TEXTURE_3D, 2, GL_RGB9_E5, 1, 1, 1, 0, GL_RGB, GL_UNSIGNED_INT_5_9_9_9_REV, l1.data());
    }
    CHECK_GL();
    std::vector<uint32_t> out(64, 0);
    glBindTexture(GL_TEXTURE_3D, t[0]);
    glGetTexImage(GL_TEXTURE_3D, 0, GL_RGB, GL_UNSIGNED_INT_5_9_9_9_REV, out.data());
    CHECK_EQ(out[0], v);
    CHECK_EQ(out[63], v);
    glCopyImageSubData(t[0], GL_TEXTURE_3D, 0, 0, 0, 0, t[1], GL_TEXTURE_3D, 1, 0, 0, 0, 2, 2, 2);
    CHECK_GL();
    glBindTexture(GL_TEXTURE_3D, t[1]);
    glGetTexImage(GL_TEXTURE_3D, 1, GL_RGB, GL_UNSIGNED_INT_5_9_9_9_REV, out.data());
    CHECK_EQ(out[0], v);
    CHECK_EQ(out[7], v);
    glDeleteTextures(2, t);
}

static GLuint makeMip3D(GLenum target, int w0, int h0, int d0, const std::vector<uint32_t>& data) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(target, t);
    for (int l = 0; l < 3; l++) {
        int w = std::max(1, w0 >> l), h = std::max(1, h0 >> l), d = target == GL_TEXTURE_3D ? std::max(1, d0 >> l) : d0;
        glTexImage3D(target, l, GL_RGB9_E5, w, h, d, 0, GL_RGB, GL_UNSIGNED_INT_5_9_9_9_REV, data.data());
        glTexParameteri(target, GL_TEXTURE_BASE_LEVEL, 0);
        glTexParameteri(target, GL_TEXTURE_MAX_LEVEL, 2);
    }
    return t;
}

TEST(copy_image_3d_depth_region) {
    for (GLenum target : {GL_TEXTURE_3D, GL_TEXTURE_2D_ARRAY}) {
        std::vector<uint32_t> a(30 * 30 * 24, 0x11111111u), b(30 * 30 * 24, 0x22222222u);
        GLuint s = makeMip3D(target, 7, 7, 12, a);
        GLuint d = makeMip3D(target, 30, 30, target == GL_TEXTURE_3D ? 24 : 12, b);
        glCopyImageSubData(s, target, 0, 0, 0, 0, d, target, 1, 14, 14, 0, 1, 1, 12);
        CHECK_GL();
        std::vector<uint32_t> out(15 * 15 * 12, 0);
        glBindTexture(target, d);
        glGetTexImage(target, 1, GL_RGB, GL_UNSIGNED_INT_5_9_9_9_REV, out.data());
        CHECK_EQ(out[14 * 15 + 14], 0x11111111u);
        CHECK_EQ(out[11 * 225 + 14 * 15 + 14], 0x11111111u);
        CHECK_EQ(out[0], 0x22222222u);
        GLuint ts[2] = {s, d};
        glDeleteTextures(2, ts);
    }
}

static GLuint tex2D(GLenum ifmt, int w, int h, GLenum fmt, GLenum type, const void* data) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, ifmt, w, h, 0, fmt, type, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    return t;
}

// Raw bit reinterpretation between same-size formats of one view class.
TEST(copy_image_reinterpret) {
    uint32_t a[4] = {0x3ff003ffu, 0x12345678u, 0xc0000001u, 0xdeadbeefu}, out[4] = {};
    GLuint s = tex2D(GL_RGB10_A2, 2, 2, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV, a);
    GLuint d = tex2D(GL_R11F_G11F_B10F, 2, 2, GL_RGB, GL_UNSIGNED_INT_10F_11F_11F_REV, nullptr);
    glCopyImageSubData(s, GL_TEXTURE_2D, 0, 0, 0, 0, d, GL_TEXTURE_2D, 0, 0, 0, 0, 2, 2, 1);
    CHECK_GL();
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_INT_10F_11F_11F_REV, out);
    CHECK(memcmp(a, out, 16) == 0);
    GLuint r = tex2D(GL_RGBA8UI, 2, 2, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, nullptr);
    glCopyImageSubData(d, GL_TEXTURE_2D, 0, 0, 0, 0, r, GL_TEXTURE_2D, 0, 0, 0, 0, 2, 2, 1);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, out);
    CHECK(memcmp(a, out, 16) == 0);
    // different view classes / sizes
    GLuint x = tex2D(GL_RGBA16F, 2, 2, GL_RGBA, GL_HALF_FLOAT, nullptr);
    glCopyImageSubData(s, GL_TEXTURE_2D, 0, 0, 0, 0, x, GL_TEXTURE_2D, 0, 0, 0, 0, 2, 2, 1);
    CHECK_EQ(glGetError(), (GLenum)GL_INVALID_OPERATION);
    GLuint ts[4] = {s, d, r, x};
    glDeleteTextures(4, ts);
}

// Compressed <-> uncompressed: one 128-bit block <-> one RGBA32UI texel.
TEST(copy_image_compressed_uncompressed) {
    uint32_t blocks[2 * 2 * 4];
    for (int i = 0; i < 16; i++) blocks[i] = 0x01020304u * (i + 1);
    GLuint c;
    glGenTextures(1, &c);
    glBindTexture(GL_TEXTURE_2D, c);
    glCompressedTexImage2D(GL_TEXTURE_2D, 0, GL_COMPRESSED_RG_RGTC2, 8, 8, 0, sizeof blocks, blocks);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    GLuint u = tex2D(GL_RGBA32UI, 2, 2, GL_RGBA_INTEGER, GL_UNSIGNED_INT, nullptr);
    glCopyImageSubData(c, GL_TEXTURE_2D, 0, 0, 0, 0, u, GL_TEXTURE_2D, 0, 0, 0, 0, 8, 8, 1);
    CHECK_GL();
    uint32_t out[16] = {};
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, out);
    CHECK(memcmp(blocks, out, sizeof out) == 0);
    // and back, into the second block row
    glCopyImageSubData(u, GL_TEXTURE_2D, 0, 0, 0, 0, c, GL_TEXTURE_2D, 0, 0, 4, 0, 2, 1, 1);
    CHECK_GL();
    uint32_t back[16] = {};
    glBindTexture(GL_TEXTURE_2D, c);
    glGetCompressedTexImage(GL_TEXTURE_2D, 0, back);
    CHECK(memcmp(back + 8, blocks, 32) == 0);
    // compressed storage specified through glTexImage2D
    GLuint c2;
    glGenTextures(1, &c2);
    glBindTexture(GL_TEXTURE_2D, c2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_COMPRESSED_RG_RGTC2, 16, 16, 0, GL_RG, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    CHECK_GL();
    glCopyImageSubData(c2, GL_TEXTURE_2D, 0, 0, 0, 0, u, GL_TEXTURE_2D, 0, 0, 0, 0, 4, 4, 1);
    CHECK_GL();
    glDeleteTextures(1, &c2);
    // unaligned compressed region
    glCopyImageSubData(c, GL_TEXTURE_2D, 0, 2, 0, 0, u, GL_TEXTURE_2D, 0, 0, 0, 0, 4, 4, 1);
    CHECK_EQ(glGetError(), (GLenum)GL_INVALID_VALUE);
    GLuint ts[2] = {c, u};
    glDeleteTextures(2, ts);
}
