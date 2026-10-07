// ARB_sparse_texture(2) / ARB_sparse_texture_clamp / ARB_sparse_buffer.
#include "harness.h"

static bool hasExtension(const char* name) {
    GLint n = 0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &n);
    for (GLint i = 0; i < n; i++)
        if (!strcmp((const char*)glGetStringi(GL_EXTENSIONS, i), name)) return true;
    return false;
}

static GLuint sparseComputeProgram(const char* cs) {
    GLuint s = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(s, 1, &cs, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, nullptr, log);
        printf("%s\n", log);
    }
    GLuint p = glCreateProgram();
    glAttachShader(p, s);
    glDeleteShader(s);
    glLinkProgram(p);
    return p;
}

// Commits half of a 2D texture: committed texels keep uploads, uncommitted ones read as zero
// (API reads and sparseTexelFetchARB residency), and decommitting drops the data.
TEST(sparse_texture_commitment) {
    if (!hasExtension("GL_ARB_sparse_texture2")) {
        printf("  (no GL_ARB_sparse_texture2, skipped)\n");
        return;
    }
    GLint px = 0, py = 0, pz = 0, n = 0;
    glGetInternalformativ(GL_TEXTURE_2D, GL_RGBA8, GL_NUM_VIRTUAL_PAGE_SIZES_ARB, 1, &n);
    glGetInternalformativ(GL_TEXTURE_2D, GL_RGBA8, GL_VIRTUAL_PAGE_SIZE_X_ARB, 1, &px);
    glGetInternalformativ(GL_TEXTURE_2D, GL_RGBA8, GL_VIRTUAL_PAGE_SIZE_Y_ARB, 1, &py);
    glGetInternalformativ(GL_TEXTURE_2D, GL_RGBA8, GL_VIRTUAL_PAGE_SIZE_Z_ARB, 1, &pz);
    CHECK_EQ(n, 1);
    CHECK_EQ(px, 128);
    CHECK_EQ(py, 128);
    CHECK_EQ(pz, 1);
    const int w = 2 * px, h = py;
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SPARSE_ARB, GL_TRUE);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, w, h);
    CHECK_GL();
    GLint sparse = 0;
    glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SPARSE_ARB, &sparse);
    CHECK_EQ(sparse, 1);
    glTexPageCommitmentARB(GL_TEXTURE_2D, 0, 0, 0, 0, px, py, 1, GL_TRUE);
    CHECK_GL();
    glTexPageCommitmentARB(GL_TEXTURE_2D, 0, 1, 0, 0, px, py, 1, GL_TRUE);
    CHECK_EQ(glGetError(), (GLenum)GL_INVALID_VALUE);
    std::vector<uint32_t> data((size_t)w * h, 0x80402010u), out((size_t)w * h, 0xffffffffu);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, out.data());
    CHECK_GL();
    CHECK_EQ(out[5 * w + 7], 0x80402010u);
    CHECK_EQ(out[5 * w + px + 7], 0u);

    const char* cs = R"(#version 450
#extension GL_ARB_sparse_texture2 : require
#extension GL_ARB_sparse_texture_clamp : require
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer B { int r[]; };
uniform sampler2D s;
void main() {
    vec4 v;
    int c0 = sparseTexelFetchARB(s, ivec2(3, 3), 0, v);
    int c1 = sparseTexelFetchARB(s, ivec2(200, 3), 0, v);
    r[0] = sparseTexelsResidentARB(c0) ? 1 : 0;
    r[1] = sparseTexelsResidentARB(c1) ? 1 : 0;
    r[2] = int(textureClampARB(s, vec2(0.1), 0.0).x * 255.0 + 0.5);
}
)";
    GLuint p = sparseComputeProgram(cs);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    CHECK(ok);
    GLuint b;
    glGenBuffers(1, &b);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, b);
    int init[3] = {-1, -1, -1};
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof init, init, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, b);
    glUseProgram(p);
    glUniform1i(glGetUniformLocation(p, "s"), 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    int res[3] = {};
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof res, res);
    CHECK_GL();
    CHECK_EQ(res[0], 1);
    CHECK_EQ(res[1], 0);
    CHECK_EQ(res[2], 0x10);

    // Decommitted contents are undefined (the GPU may keep stale translations of an unmap for a
    // while); recommitting gives writable memory again.
    glTexPageCommitmentARB(GL_TEXTURE_2D, 0, 0, 0, 0, px, py, 1, GL_FALSE);
    glTexPageCommitmentARB(GL_TEXTURE_2D, 0, 0, 0, 0, px, py, 1, GL_TRUE);
    std::fill(data.begin(), data.end(), 0x01020304u);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, out.data());
    CHECK_GL();
    CHECK_EQ(out[5 * w + 7], 0x01020304u);
    glDeleteProgram(p);
    glDeleteBuffers(1, &b);
    glDeleteTextures(1, &t);
}

// Sparse buffer pages: committed pages keep data written by BufferSubData, uncommitted pages
// read as zero, and copies to and from the buffer go through the GPU.
TEST(sparse_buffer_commitment) {
    if (!hasExtension("GL_ARB_sparse_buffer")) {
        printf("  (no GL_ARB_sparse_buffer, skipped)\n");
        return;
    }
    GLint page = 0;
    glGetIntegerv(GL_SPARSE_BUFFER_PAGE_SIZE_ARB, &page);
    CHECK(page > 0 && page <= 65536);
    GLuint b[2];
    glGenBuffers(2, b);
    glBindBuffer(GL_COPY_WRITE_BUFFER, b[0]);
    glBufferStorage(GL_COPY_WRITE_BUFFER, page * 3, nullptr, GL_SPARSE_STORAGE_BIT_ARB | GL_DYNAMIC_STORAGE_BIT);
    CHECK_GL();
    CHECK(glMapBufferRange(GL_COPY_WRITE_BUFFER, 0, 16, GL_MAP_READ_BIT) == nullptr);
    CHECK_EQ(glGetError(), (GLenum)GL_INVALID_OPERATION);
    glBufferPageCommitmentARB(GL_COPY_WRITE_BUFFER, page, page, GL_TRUE);
    CHECK_GL();
    std::vector<uint8_t> src((size_t)page * 3);
    for (size_t i = 0; i < src.size(); i++) src[i] = (uint8_t)(i * 7 + 1);
    glBufferSubData(GL_COPY_WRITE_BUFFER, 0, (GLsizeiptr)src.size(), src.data());
    std::vector<uint8_t> out(src.size(), 0xcc);
    glGetBufferSubData(GL_COPY_WRITE_BUFFER, 0, (GLsizeiptr)out.size(), out.data());
    CHECK_GL();
    CHECK_EQ(out[10], 0);
    CHECK_EQ(out[page + 10], src[page + 10]);
    CHECK_EQ(out[2 * page + 10], 0);
    // copy the committed page out to a regular buffer
    glBindBuffer(GL_COPY_READ_BUFFER, b[1]);
    glBufferData(GL_COPY_READ_BUFFER, page, nullptr, GL_STATIC_READ);
    glCopyBufferSubData(GL_COPY_WRITE_BUFFER, GL_COPY_READ_BUFFER, page, 0, page);
    uint8_t v = 0;
    glGetBufferSubData(GL_COPY_READ_BUFFER, 33, 1, &v);
    CHECK_EQ(v, src[page + 33]);
    glBufferPageCommitmentARB(GL_COPY_WRITE_BUFFER, page, page, GL_FALSE);
    CHECK_GL();
    glDeleteBuffers(2, b);
}

// Several pages committed in one call, written by BufferSubData and read back by a GPU copy.
TEST(sparse_buffer_multi_page) {
    if (!hasExtension("GL_ARB_sparse_buffer")) return;
    GLint page = 0;
    glGetIntegerv(GL_SPARSE_BUFFER_PAGE_SIZE_ARB, &page);
    GLuint b[2];
    glGenBuffers(2, b);
    glBindBuffer(GL_COPY_WRITE_BUFFER, b[0]);
    glBufferStorage(GL_COPY_WRITE_BUFFER, page * 4, nullptr, GL_SPARSE_STORAGE_BIT_ARB | GL_DYNAMIC_STORAGE_BIT);
    glBufferPageCommitmentARB(GL_COPY_WRITE_BUFFER, 0, page * 4, GL_TRUE);
    std::vector<uint8_t> src((size_t)page * 4);
    for (size_t i = 0; i < src.size(); i++) src[i] = (uint8_t)(i / page + 1);
    glBufferSubData(GL_COPY_WRITE_BUFFER, 0, (GLsizeiptr)src.size(), src.data());
    glBindBuffer(GL_COPY_READ_BUFFER, b[1]);
    glBufferData(GL_COPY_READ_BUFFER, page * 4, nullptr, GL_STATIC_READ);
    glCopyBufferSubData(GL_COPY_WRITE_BUFFER, GL_COPY_READ_BUFFER, 0, 0, page * 4);
    std::vector<uint8_t> out(src.size());
    glGetBufferSubData(GL_COPY_READ_BUFFER, 0, (GLsizeiptr)out.size(), out.data());
    CHECK_GL();
    for (int p = 0; p < 4; p++) CHECK_EQ(out[(size_t)p * page + 5], p + 1);
    uint8_t one = 0x78;
    glClearBufferData(GL_COPY_WRITE_BUFFER, GL_R8, GL_RED, GL_UNSIGNED_BYTE, &one);
    glGetBufferSubData(GL_COPY_WRITE_BUFFER, 0, (GLsizeiptr)out.size(), out.data());
    for (int p = 0; p < 4; p++) CHECK_EQ(out[(size_t)p * page + 5], 0x78);
    glDeleteBuffers(2, b);
}

// Tiles of a deleted sparse buffer are reused by the next one.
TEST(sparse_buffer_reuse) {
    if (!hasExtension("GL_ARB_sparse_buffer")) return;
    GLint page = 0;
    glGetIntegerv(GL_SPARSE_BUFFER_PAGE_SIZE_ARB, &page);
    std::vector<uint8_t> src((size_t)page * 2, 0x5a), out(src.size());
    for (int round = 0; round < 4; round++) {
        GLuint b;
        glGenBuffers(1, &b);
        glBindBuffer(GL_COPY_WRITE_BUFFER, b);
        glBufferStorage(GL_COPY_WRITE_BUFFER, page * 2, nullptr, GL_SPARSE_STORAGE_BIT_ARB | GL_DYNAMIC_STORAGE_BIT);
        glBufferPageCommitmentARB(GL_COPY_WRITE_BUFFER, 0, page * 2, GL_TRUE);
        src[0] = (uint8_t)round;
        glBufferSubData(GL_COPY_WRITE_BUFFER, 0, (GLsizeiptr)src.size(), src.data());
        glGetBufferSubData(GL_COPY_WRITE_BUFFER, 0, (GLsizeiptr)out.size(), out.data());
        CHECK_EQ(out[0], round);
        CHECK_EQ(out[page + 3], 0x5a);
        glDeleteBuffers(1, &b);
        glFinish();
    }
}

// Shader writes (SSBO) and indirect dispatch arguments in a sparse buffer.
TEST(sparse_buffer_shader_access) {
    if (!hasExtension("GL_ARB_sparse_buffer")) return;
    GLint page = 0;
    glGetIntegerv(GL_SPARSE_BUFFER_PAGE_SIZE_ARB, &page);
    GLuint b;
    glGenBuffers(1, &b);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, b);
    glBufferStorage(GL_SHADER_STORAGE_BUFFER, page * 2, nullptr, GL_SPARSE_STORAGE_BIT_ARB | GL_DYNAMIC_STORAGE_BIT);
    glBufferPageCommitmentARB(GL_SHADER_STORAGE_BUFFER, 0, page * 2, GL_TRUE);
    GLuint args[8] = {2, 1, 1, 0, 0, 0, 0, 0};  // committed memory starts undefined
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof args, args);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, b);
    GLuint p = sparseComputeProgram(R"(#version 450
layout(local_size_x = 4) in;
layout(std430, binding = 0) buffer B { uint args[3]; uint pad; uint v[]; };
void main() { atomicAdd(v[0], 1u); v[1 + gl_GlobalInvocationID.x] = 7u; }
)");
    glUseProgram(p);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    GLuint r[3] = {};
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 16, sizeof r, r);
    CHECK_EQ(r[0], 4u);
    CHECK_EQ(r[1], 7u);
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, b);
    glDispatchComputeIndirect(0);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 16, sizeof r, r);
    CHECK_GL();
    CHECK_EQ(r[0], 12u);
    glDeleteProgram(p);
    glDeleteBuffers(1, &b);
}

// Indirect dispatch arguments straddling two pages, copied in from another buffer.
TEST(sparse_buffer_indirect_straddle) {
    if (!hasExtension("GL_ARB_sparse_buffer")) return;
    GLint page = 0;
    glGetIntegerv(GL_SPARSE_BUFFER_PAGE_SIZE_ARB, &page);
    GLuint helperBo;
    glGenBuffers(1, &helperBo);
    GLuint p = sparseComputeProgram(R"(#version 430 core
layout(local_size_x = 1023) in;
layout(binding = 0, offset = 0) uniform atomic_uint ac;
void main() { atomicCounterIncrement(ac); }
)");
    // As the CTS: a fresh 1 GB sparse buffer per round, committed, used, decommitted and deleted.
    for (int round = 0; round < 4; round++) {
        GLuint helper[4] = {8, 1, 1, 0};
        glBindBuffer(GL_ARRAY_BUFFER, helperBo);
        glBufferData(GL_ARRAY_BUFFER, sizeof helper, helper, GL_STATIC_DRAW);
        GLuint b;
        glGenBuffers(1, &b);
        glBindBuffer(GL_ARRAY_BUFFER, b);
        glBufferStorage(GL_ARRAY_BUFFER, 1024768 * 1024, nullptr, GL_SPARSE_STORAGE_BIT_ARB | (round & 1 ? GL_DYNAMIC_STORAGE_BIT : 0));
        glBufferPageCommitmentARB(GL_ARRAY_BUFFER, 0, page * 2, GL_TRUE);
        CHECK_GL();
        glUseProgram(p);
        glBindBufferRange(GL_ATOMIC_COUNTER_BUFFER, 0, helperBo, 12, 4);
        glBindBuffer(GL_COPY_READ_BUFFER, helperBo);
        glBindBuffer(GL_COPY_WRITE_BUFFER, b);
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, page - 4, 12);
        glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, b);
        glDispatchComputeIndirect(page - 4);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        GLuint r = 0;
        glGetBufferSubData(GL_COPY_READ_BUFFER, 12, 4, &r);
        CHECK_GL();
        CHECK_EQ(r, 8u * 1023u);
        glBindBuffer(GL_ARRAY_BUFFER, b);
        glBufferPageCommitmentARB(GL_ARRAY_BUFFER, 0, page * 2, GL_FALSE);
        glDeleteBuffers(1, &b);
    }
    glDeleteProgram(p);
    glDeleteBuffers(1, &helperBo);
}

// A buffer texture over a sparse buffer.
TEST(sparse_buffer_texture) {
    if (!hasExtension("GL_ARB_sparse_buffer")) return;
    GLint page = 0;
    glGetIntegerv(GL_SPARSE_BUFFER_PAGE_SIZE_ARB, &page);
    GLuint b[2], t;
    glGenBuffers(2, b);
    glBindBuffer(GL_TEXTURE_BUFFER, b[0]);
    glBufferStorage(GL_TEXTURE_BUFFER, 1024768 * 1024, nullptr, GL_SPARSE_STORAGE_BIT_ARB);  // as the CTS: mostly uncommitted
    glBufferPageCommitmentARB(GL_TEXTURE_BUFFER, 0, page * 2, GL_TRUE);
    std::vector<uint32_t> data((size_t)page / 2, 0x04030201u);
    glBindBuffer(GL_COPY_READ_BUFFER, b[1]);
    glBufferData(GL_COPY_READ_BUFFER, page * 2, data.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_COPY_WRITE_BUFFER, b[0]);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, page * 2);
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_BUFFER, t);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA8, b[0]);
    GLuint p = sparseComputeProgram(R"(#version 430 core
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer B { uint r[2]; };
uniform samplerBuffer s;
void main() { vec4 v = texelFetch(s, 0); r[0] = uint(v.x * 255.0 + 0.5); vec4 w = texelFetch(s, 5000); r[1] = uint(w.w * 255.0 + 0.5); }
)");
    GLuint o;
    glGenBuffers(1, &o);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, o);
    GLuint z[2] = {99, 99};
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof z, z, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, o);
    glUseProgram(p);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof z, z);
    CHECK_GL();
    CHECK_EQ(z[0], 1u);
    CHECK_EQ(z[1], 4u);
    glDeleteProgram(p);
    glDeleteTextures(1, &t);
    glDeleteBuffers(2, b);
    glDeleteBuffers(1, &o);
}

// Binding a texture as an image of another format reallocates it with view usage and copies its
// contents; the copy follows an image store to a multisample texture (dropped: Metal can't write those).
TEST(image_bind_realloc_after_ms_store) {
    {
        GLuint ms;
        glGenTextures(1, &ms);
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, ms);
        glTexStorage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 2, GL_R32I, 256, 256, GL_TRUE);
        GLuint p = sparseComputeProgram(R"(#version 430 core
layout(local_size_x = 1) in;
layout(location = 1) writeonly uniform highp iimage2DMS img;
void main() { imageStore(img, ivec2(gl_WorkGroupID.xy), 0, ivec4(1)); }
)");
        glUseProgram(p);
        glBindImageTexture(0, ms, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32I);
        glUniform1i(1, 0);
        glDispatchCompute(4, 4, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
    }
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R8, 128, 64);
    std::vector<uint8_t> z(128 * 64, 0);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 128, 64, GL_RED, GL_UNSIGNED_BYTE, z.data());
    glBindImageTexture(0, t, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R8UI);
    glFinish();
    CHECK_GL();
    glDeleteTextures(1, &t);
}

// Residency of committed / uncommitted pages for formats with square (RGBA8, R8) and non-square
// (R16, RGBA16F) Metal tiles: half of a 2x2-page texture is committed.
TEST(sparse_texture_residency_formats) {
    if (!hasExtension("GL_ARB_sparse_texture2")) return;
    const GLenum fmts[] = {GL_R16, GL_RGBA16F, GL_RGBA8, GL_R8};
    for (GLenum fmt : fmts) {
        GLint px = 0, py = 0;
        glGetInternalformativ(GL_TEXTURE_2D, fmt, GL_VIRTUAL_PAGE_SIZE_X_ARB, 1, &px);
        glGetInternalformativ(GL_TEXTURE_2D, fmt, GL_VIRTUAL_PAGE_SIZE_Y_ARB, 1, &py);
        GLuint t;
        glGenTextures(1, &t);
        glBindTexture(GL_TEXTURE_2D, t);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SPARSE_ARB, GL_TRUE);
        glTexStorage2D(GL_TEXTURE_2D, 1, fmt, 2 * px, 2 * py);
        glTexPageCommitmentARB(GL_TEXTURE_2D, 0, 0, 0, 0, px, 2 * py, 1, GL_TRUE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        GLuint p = sparseComputeProgram(R"(#version 450
#extension GL_ARB_sparse_texture2 : require
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer B { int r[64]; };
uniform sampler2D s;
uniform ivec2 page;
void main() {
    vec4 v;
    for (int j = 0; j < 8; j++)
        for (int i = 0; i < 8; i++)
            r[j * 8 + i] = sparseTexelsResidentARB(sparseTexelFetchARB(s, ivec2(i * page.x / 4 + 1, j * page.y / 4 + 1), 0, v)) ? 1 : 0;
}
)");
        GLuint o;
        glGenBuffers(1, &o);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, o);
        std::vector<int> z(64, -1);
        glBufferData(GL_SHADER_STORAGE_BUFFER, 256, z.data(), GL_DYNAMIC_COPY);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, o);
        glUseProgram(p);
        glUniform2i(glGetUniformLocation(p, "page"), px, py);
        glDispatchCompute(1, 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 256, z.data());
        CHECK_GL();
        bool ok = true;
        for (int j = 0; j < 8; j++)
            for (int i = 0; i < 8; i++) ok &= z[j * 8 + i] == (i < 4 ? 1 : 0);
        if (!ok) {
            printf("  format 0x%x page %dx%d residency (columns = quarter pages):\n", fmt, px, py);
            for (int j = 0; j < 8; j++) {
                printf("   ");
                for (int i = 0; i < 8; i++) printf("%d", z[j * 8 + i]);
                printf("\n");
            }
        }
        CHECK(ok);
        glDeleteProgram(p);
        glDeleteTextures(1, &t);
        glDeleteBuffers(1, &o);
    }
}

// textureClampARB / sparseTextureClampARB on a mipmapped texture: drawing each level at its own size
// gives an implicit LOD equal to the level, which the clamp keeps. Levels 0 and 2 are written by image
// stores from a program deleted right after its dispatch, level 1 by an upload.
static void clampLevels(bool sparse) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    if (sparse) glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SPARSE_ARB, GL_TRUE);
    glTexStorage2D(GL_TEXTURE_2D, 3, GL_R8, 1024, 1024);
    for (int l = 0; l < 3; l++) {
        int w = 1024 >> l;
        if (sparse) glTexPageCommitmentARB(GL_TEXTURE_2D, l, 0, 0, 0, w, w, 1, GL_TRUE);
        if (l != 1) {
            GLuint fill = sparseComputeProgram(R"(#version 430 core
layout(local_size_x = 1) in;
layout(location = 1) writeonly uniform highp image2D img;
layout(location = 2) uniform float v;
void main() { imageStore(img, ivec2(gl_WorkGroupID.xy), vec4(v, 0, 0, 1)); }
)");
            glUseProgram(fill);
            glBindImageTexture(0, t, l, GL_TRUE, 0, GL_WRITE_ONLY, GL_R8);
            glUniform1i(1, 0);
            glUniform1f(2, (64 + 40 * l) / 255.0f);
            glDispatchCompute(w, w, 1);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glDeleteProgram(fill);  // still current: its pipeline must outlive the dispatch
        } else {
            std::vector<uint8_t> d((size_t)w * w, (uint8_t)(64 + 40 * l));
            glTexSubImage2D(GL_TEXTURE_2D, l, 0, 0, w, w, GL_RED, GL_UNSIGNED_BYTE, d.data());
        }
    }
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    GLuint p = compileProgram(R"(#version 450
in vec2 pos;
out vec2 uv;
void main() { uv = pos * 0.5 + 0.5; gl_Position = vec4(pos, 0.0, 1.0); }
)", R"(#version 450
#extension GL_ARB_sparse_texture2 : enable
#extension GL_ARB_sparse_texture_clamp : enable
in vec2 uv;
out vec4 color;
uniform sampler2D s;
uniform float minLod;
void main() {
    vec4 v;
    int code = sparseTextureClampARB(s, uv, minLod, v);
    color = vec4(v.r, textureClampARB(s, uv, minLod).r, sparseTexelsResidentARB(code) ? 1.0 : 0.0, 1.0);
}
)");
    GLuint rt, fb;
    glGenTextures(1, &rt);
    glBindTexture(GL_TEXTURE_2D, rt);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 1024, 1024);
    glGenFramebuffers(1, &fb);
    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, t);
    glUseProgram(p);
    glUniform1i(glGetUniformLocation(p, "s"), 0);
    float quad[] = {-1, -1, 1, -1, -1, 1, 1, 1};
    GLuint vb, va;
    glGenVertexArrays(1, &va);
    glBindVertexArray(va);
    glGenBuffers(1, &vb);
    glBindBuffer(GL_ARRAY_BUFFER, vb);
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);
    for (int l = 0; l < 3; l++) {
        int w = 1024 >> l;
        glViewport(0, 0, w, w);
        glUniform1f(glGetUniformLocation(p, "minLod"), (float)l);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        std::vector<uint8_t> all((size_t)w * w * 4);
        glReadPixels(0, 0, w, w, GL_RGBA, GL_UNSIGNED_BYTE, all.data());
        CHECK_GL();
        int bad = 0, first = -1;
        for (int i = 0; i < w * w; i++) {
            const uint8_t* q = &all[(size_t)i * 4];
            if (abs(q[0] - (64 + 40 * l)) > 1 || abs(q[1] - (64 + 40 * l)) > 1 || q[2] != 255) {
                if (!bad++) first = i;
            }
        }
        if (bad)
            printf("  %s level %d: %d bad pixels, first (%d,%d) = %d %d %d, expected %d\n", sparse ? "sparse" : "regular", l, bad, first % w,
                   first / w, all[(size_t)first * 4], all[(size_t)first * 4 + 1], all[(size_t)first * 4 + 2], 64 + 40 * l);
        CHECK_EQ(bad, 0);
    }
    glDeleteFramebuffers(1, &fb);
    glDeleteTextures(1, &rt);
    glDeleteTextures(1, &t);
    glDeleteProgram(p);
    glDeleteBuffers(1, &vb);
    glDeleteVertexArrays(1, &va);
}

TEST(texture_clamp_levels) {
    if (!hasExtension("GL_ARB_sparse_texture_clamp")) return;
    clampLevels(false);
    clampLevels(true);
}
