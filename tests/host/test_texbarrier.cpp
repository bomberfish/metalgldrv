// Mirror of KHR-GL46.texture_barrier.{disjoint,overlapping}-texels: 8 R32UI textures that are both
// color attachments and sampled with texelFetch, a shuffled irregular grid of 8192 triangles drawn
// with several glDrawArrays per pass, glTexSubImage2D re-initialization and glGetTexImage checks.
#include <cmath>
#include "harness.h"

namespace {
const int N = 8, GRID = 64, TRIS = GRID * GRID * 2, VERTS = TRIS * 3;

void gridVertices(std::vector<float>& out) {
    struct V { float x, y; };
    struct T { V a, b, c; };
    static V grid[GRID + 1][GRID + 1];
    for (int x = 0; x <= GRID; x++)
        for (int y = 0; y <= GRID; y++) {
            float nx = (float)x / GRID, ny = (float)y / GRID;
            grid[x][y].x = nx * 2.f - 1.f + sinf(nx * (float)M_PI * 13.f) * 0.3f / GRID;
            grid[x][y].y = ny * 2.f - 1.f + sinf(ny * (float)M_PI * 13.f) * 0.3f / GRID;
        }
    std::vector<T> list(TRIS);
    for (int x = 0; x < GRID; x++)
        for (int y = 0; y < GRID; y++) {
            list[(x + y * GRID) * 2] = {grid[x][y], grid[x + 1][y], grid[x + 1][y + 1]};
            list[(x + y * GRID) * 2 + 1] = {grid[x + 1][y + 1], grid[x][y + 1], grid[x][y]};
        }
    for (int i = TRIS - 2; i > 0; --i) {
        int j = (int)((list[i].b.y + list[i].c.x + 13.f) * 1345.13f) % i;
        std::swap(list[i], list[j]);
    }
    out.clear();
    for (auto& t : list) out.insert(out.end(), {t.a.x, t.a.y, t.b.x, t.b.y, t.c.x, t.c.y});
}

// Runs `testPasses` x `renderPasses`; blockSize/modulo per pass from the callbacks. Returns bad texel count.
int runBarrierCase(int testPasses, int renderPasses, bool barrier, int (*blockSize)(int, int), int (*modulo)(int, int)) {
    const int W = 64, H = 64;
    GLuint tex[N], fbo, vao, vbo;
    glGenTextures(N, tex);
    for (int i = 0; i < N; i++) {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, tex[i]);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, W, H);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    GLenum bufs[N];
    for (int i = 0; i < N; i++) {
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, tex[i], 0);
        bufs[i] = GL_COLOR_ATTACHMENT0 + i;
    }
    glDrawBuffers(N, bufs);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    std::vector<float> v;
    gridVertices(v);
    glBufferData(GL_ARRAY_BUFFER, v.size() * sizeof(float), v.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    const char* vs = "#version 400 core\nin vec2 Position;\nvoid main() { gl_Position = vec4(Position, 0.0, 1.0); }";
    const char* fs = R"(#version 400 core
#define NUM_TEXTURES 8
uniform int blockSize;
uniform int modulo;
uniform usampler2D texInput[NUM_TEXTURES];
out uvec4 fragOutput[NUM_TEXTURES];
uvec4 func(uvec4 t0, uvec4 t1, uvec4 t2, uvec4 t3) { return t0 + t1 + t2 + t3; }
void main() {
    ivec2 texelCoord = ivec2(gl_FragCoord.xy);
    ivec2 blockCoord = texelCoord / blockSize;
    ivec2 xOffset = ivec2(blockSize, 0);
    ivec2 yOffset = ivec2(0, blockSize);
    if (((blockCoord.x + blockCoord.y) % 2) == modulo) {
        for (int i = 0; i < NUM_TEXTURES; ++i)
            fragOutput[i] = func(texelFetch(texInput[i], texelCoord + xOffset, 0), texelFetch(texInput[i], texelCoord - xOffset, 0),
                                 texelFetch(texInput[i], texelCoord + yOffset, 0), texelFetch(texInput[i], texelCoord - yOffset, 0));
    } else {
        discard;
    }
})";
    std::string log;
    GLuint p = compileProgram(vs, fs, &log);
    if (!p) fprintf(stderr, "%s\n", log.c_str());
    glUseProgram(p);
    for (int i = 0; i < N; i++) glUniform1i(glGetUniformLocation(p, ("texInput[" + std::to_string(i) + "]").c_str()), i);
    std::vector<std::vector<uint32_t>> ref(N, std::vector<uint32_t>(W * H));
    std::vector<uint32_t> got(W * H);
    auto at = [&](std::vector<uint32_t>& r, int x, int y) -> uint32_t { return x < 0 || y < 0 || x >= W || y >= H ? 0 : r[y * W + x]; };
    int bad = 0;
    for (int tp = 0; tp < testPasses && !bad; tp++) {
        for (int i = 0; i < N; i++) {
            for (int x = 0; x < W; x++)
                for (int y = 0; y < H; y++) ref[i][x + y * W] = ((uint32_t)i << 24) + ((uint32_t)y << 12) + (uint32_t)x;
            glActiveTexture(GL_TEXTURE0 + i);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RED_INTEGER, GL_UNSIGNED_INT, ref[i].data());
        }
        for (int rp = 0; rp < renderPasses; rp++) {
            glUniform1i(glGetUniformLocation(p, "blockSize"), blockSize(tp, rp));
            glUniform1i(glGetUniformLocation(p, "modulo"), modulo(tp, rp));
            int minTris = TRIS / 7, first = 0;
            while (first < VERTS) {
                int count = std::min(VERTS - first, (first % 23 + minTris) * 3);
                glDrawArrays(GL_TRIANGLES, first, count);
                first += count;
            }
            if (barrier) glTextureBarrier();
        }
        for (int rp = 0; rp < renderPasses; rp++) {
            int B = blockSize(tp, rp), M = modulo(tp, rp);
            for (int x = 0; x < W; x++)
                for (int y = 0; y < H; y++)
                    if ((int)(((x / B) + (y / B)) % 2) == M)
                        for (int i = 0; i < N; i++)
                            ref[i][x + y * W] = at(ref[i], x + B, y) + at(ref[i], x - B, y) + at(ref[i], x, y + B) + at(ref[i], x, y - B);
        }
        for (int i = 0; i < N; i++) {
            glActiveTexture(GL_TEXTURE0 + i);
            glGetTexImage(GL_TEXTURE_2D, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, got.data());
            int nb = 0, firstBad = -1;
            for (int k = 0; k < W * H; k++)
                if (got[k] != ref[i][k]) { if (firstBad < 0) firstBad = k; nb++; }
            if (nb)
                fprintf(stderr, "  test pass %d tex %d: %d bad texels, first (%d,%d): got %08x want %08x\n", tp, i, nb, firstBad % W, firstBad / W,
                        got[firstBad], ref[i][firstBad]);
            if (nb && i == 0 && getenv("TB_VERBOSE"))
                for (int k = 0, shown = 0; k < W * H && shown < 40; k++)
                    if (got[k] != ref[i][k]) { fprintf(stderr, " (%d,%d)", k % W, k / W); shown++; }
            if (nb && i == 0 && getenv("TB_VERBOSE")) fprintf(stderr, "\n");
            bad += nb;
        }
    }
    glActiveTexture(GL_TEXTURE0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(N, tex);
    glDeleteProgram(p);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    return bad;
}
}  // namespace

TEST(texture_barrier_cts_disjoint) {
    CHECK_EQ(runBarrierCase(16, 5, false, [](int tp, int) { return tp + 1; }, [](int, int) { return 0; }), 0);
    CHECK_GL();
}

TEST(texture_barrier_cts_overlapping) {
    CHECK_EQ(runBarrierCase(1, 42, true, [](int, int rp) { return (5 + rp * 3) % 7 + 1; }, [](int, int rp) { return (rp * 3) % 2; }), 0);
    CHECK_GL();
}
