// GL_ARB_vertex_program / GL_ARB_fragment_program: assembly programs (translated to GLSL, arb_program.cpp).
#include "harness.h"

namespace {
struct ArbFns {
    void (*gen)(GLsizei, GLuint*);
    void (*bind)(GLenum, GLuint);
    void (*string)(GLenum, GLenum, GLsizei, const void*);
    void (*env4f)(GLenum, GLuint, GLfloat, GLfloat, GLfloat, GLfloat);
    void (*local4f)(GLenum, GLuint, GLfloat, GLfloat, GLfloat, GLfloat);
    void (*getiv)(GLenum, GLenum, GLint*);
    void (*del)(GLsizei, const GLuint*);
};
ArbFns fns() {
    return {(void (*)(GLsizei, GLuint*))eglGetProcAddress("glGenProgramsARB"),
            (void (*)(GLenum, GLuint))eglGetProcAddress("glBindProgramARB"),
            (void (*)(GLenum, GLenum, GLsizei, const void*))eglGetProcAddress("glProgramStringARB"),
            (void (*)(GLenum, GLuint, GLfloat, GLfloat, GLfloat, GLfloat))eglGetProcAddress("glProgramEnvParameter4fARB"),
            (void (*)(GLenum, GLuint, GLfloat, GLfloat, GLfloat, GLfloat))eglGetProcAddress("glProgramLocalParameter4fARB"),
            (void (*)(GLenum, GLenum, GLint*))eglGetProcAddress("glGetProgramivARB"),
            (void (*)(GLsizei, const GLuint*))eglGetProcAddress("glDeleteProgramsARB")};
}
GLuint load(const ArbFns& f, GLenum target, const char* src) {
    GLuint p = 0;
    f.gen(1, &p);
    f.bind(target, p);
    f.string(target, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei)strlen(src), src);
    GLenum e = glGetError();
    if (e) {
        GLint pos = 0;
        glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &pos);
        fprintf(stderr, "  program error at %d: %s\n", pos, (const char*)glGetString(GL_PROGRAM_ERROR_STRING_ARB));
    }
    CHECK_EQ(e, (GLenum)GL_NO_ERROR);
    return p;
}
void identity() {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}
void quad() {
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(-1, -1);
    glTexCoord2f(1, 0); glVertex2f(1, -1);
    glTexCoord2f(1, 1); glVertex2f(1, 1);
    glTexCoord2f(0, 1); glVertex2f(-1, 1);
    glEnd();
}
}  // namespace

// A vertex program with fixed-function fragment processing.
TEST(arb_vertex_program) {
    ArbFns f = fns();
    identity();
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    GLuint vp = load(f, GL_VERTEX_PROGRAM_ARB,
                     "!!ARBvp1.0\n"
                     "# position through the matrix rows, color from the environment\n"
                     "PARAM mvp[4] = { state.matrix.mvp };\n"
                     "TEMP p;\n"
                     "DP4 p.x, mvp[0], vertex.position;\n"
                     "DP4 p.y, mvp[1], vertex.position;\n"
                     "DP4 p.z, mvp[2], vertex.position;\n"
                     "DP4 p.w, mvp[3], vertex.position;\n"
                     "MUL result.position, p, {0.5, 0.5, 1, 1};  # the quad shrunk to the middle\n"
                     "MOV result.color, program.env[3];\n"
                     "END\n");
    f.env4f(GL_VERTEX_PROGRAM_ARB, 3, 0, 1, 0, 1);
    glEnable(GL_VERTEX_PROGRAM_ARB);
    quad();
    glDisable(GL_VERTEX_PROGRAM_ARB);
    CHECK_PIXEL(32, 32, 0, 255, 0, 255);
    CHECK_PIXEL(4, 4, 0, 0, 0, 255);
    GLint n = 0;
    f.getiv(GL_VERTEX_PROGRAM_ARB, GL_PROGRAM_INSTRUCTIONS_ARB, &n);
    CHECK_EQ(n, 6);
    f.del(1, &vp);
    CHECK_GL();
}

// A fragment program with fixed-function vertex processing: texture, local parameter, _SAT.
TEST(arb_fragment_program) {
    ArbFns f = fns();
    identity();
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    uint8_t px[4] = {255, 128, 0, 255};
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    GLuint fp = load(f, GL_FRAGMENT_PROGRAM_ARB,
                     "!!ARBfp1.0\n"
                     "TEMP t;\n"
                     "TEX t, fragment.texcoord[0], texture[0], 2D;\n"
                     "MUL_SAT result.color, t, program.local[1];\n"
                     "END\n");
    f.local4f(GL_FRAGMENT_PROGRAM_ARB, 1, 0.5f, 4, 1, 1);  // red halved, green 0.5 * 4 saturated
    glEnable(GL_FRAGMENT_PROGRAM_ARB);  // GL_TEXTURE_2D stays disabled: fragment programs don't need it
    quad();
    glDisable(GL_FRAGMENT_PROGRAM_ARB);
    CHECK_PIXEL(32, 32, 128, 255, 0, 255);
    f.del(1, &fp);
    glDeleteTextures(1, &tex);
    CHECK_GL();
}

// Both stages: a varying through result.texcoord, KIL and CMP.
TEST(arb_vertex_and_fragment_program) {
    ArbFns f = fns();
    identity();
    glClearColor(0, 0, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    GLuint vp = load(f, GL_VERTEX_PROGRAM_ARB,
                     "!!ARBvp1.0\nOPTION ARB_position_invariant;\n"
                     "MOV result.texcoord[2], vertex.position;\n"
                     "END\n");
    GLuint fp = load(f, GL_FRAGMENT_PROGRAM_ARB,
                     "!!ARBfp1.0\n"
                     "ATTRIB pos = fragment.texcoord[2];\n"
                     "TEMP c;\n"
                     "KIL pos.y;                      # only the top half survives\n"
                     "CMP c, pos.x, {1, 0, 0, 1}, {0, 1, 0, 1};  # left red, right green\n"
                     "MOV result.color, c;\n"
                     "END\n");
    glEnable(GL_VERTEX_PROGRAM_ARB);
    glEnable(GL_FRAGMENT_PROGRAM_ARB);
    quad();
    glDisable(GL_VERTEX_PROGRAM_ARB);
    glDisable(GL_FRAGMENT_PROGRAM_ARB);
    CHECK_PIXEL(16, 48, 255, 0, 0, 255);
    CHECK_PIXEL(48, 48, 0, 255, 0, 255);
    CHECK_PIXEL(16, 16, 0, 0, 255, 255);  // killed
    f.del(1, &vp);
    f.del(1, &fp);
    CHECK_GL();
}

// Load errors: INVALID_OPERATION with a position and a message; the old program stays.
TEST(arb_program_errors) {
    ArbFns f = fns();
    GLuint p = 0;
    f.gen(1, &p);
    f.bind(GL_FRAGMENT_PROGRAM_ARB, p);
    const char* bad = "!!ARBfp1.0\nTEMP t;\nFOO t, t;\nEND\n";
    f.string(GL_FRAGMENT_PROGRAM_ARB, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei)strlen(bad), bad);
    CHECK_EQ(glGetError(), (GLenum)GL_INVALID_OPERATION);
    GLint pos = -1;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &pos);
    CHECK_EQ(pos, 19);  // "FOO"
    const char* msg = (const char*)glGetString(GL_PROGRAM_ERROR_STRING_ARB);
    CHECK(msg && strstr(msg, "FOO"));
    const char* good = "!!ARBfp1.0\nMOV result.color, fragment.color;\nEND\n";
    f.string(GL_FRAGMENT_PROGRAM_ARB, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei)strlen(good), good);
    CHECK_GL();
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &pos);
    CHECK_EQ(pos, -1);
    f.del(1, &p);
}

// Every instruction and the binding kinds load, link and draw.
TEST(arb_program_coverage) {
    ArbFns f = fns();
    identity();
    const char* vps =
        "!!ARBvp1.0\n"
        "ATTRIB pos = vertex.position; ATTRIB nrm = vertex.normal; ATTRIB tc = vertex.texcoord[1]; ATTRIB gen = vertex.attrib[6];\n"
        "PARAM mv[4] = { state.matrix.modelview };\n"
        "PARAM mvit[] = { state.matrix.modelview.invtrans.row[0..2] };\n"
        "PARAM tex[4] = { state.matrix.texture[1].transpose };\n"
        "PARAM arr[3] = { program.env[0..1], 0.25 };\n"
        "PARAM light = state.light[0].position; PARAM half = state.light[0].half; PARAM att = state.light[1].attenuation;\n"
        "PARAM spot = state.light[0].spot.direction; PARAM mat = state.material.front.diffuse; PARAM shine = state.material.shininess;\n"
        "PARAM lp = state.lightprod[0].back.specular; PARAM scene = state.lightmodel.scenecolor; PARAM amb = state.lightmodel.ambient;\n"
        "PARAM fog = state.fog.params; PARAM clip = state.clip[0].plane; PARAM ps = state.point.size; PARAM dr = state.depth.range;\n"
        "PARAM eye = state.texgen[0].eye.s; PARAM obj = state.texgen.object.q; PARAM loc = program.local[2];\n"
        "TEMP a, b, c; ADDRESS a0;\n"
        "OUTPUT o = result.position; ALIAS pp = a;\n"
        "ARL a0.x, loc.x;\n"
        "MOV b, arr[a0.x + 1];\n"
        "DP4 a.x, mv[0], pos; DP4 a.y, mv[1], pos; DP4 a.z, mv[2], pos; DP4 a.w, mv[3], pos;\n"
        "DP3 c.x, mvit[0], nrm; DPH c.y, pos, light; DST c, b, b; EX2 c.x, b.y; EXP c, b.x; LG2 c.y, b.x; LOG c, b.x;\n"
        "FLR c, b; FRC c, b; LIT c, b; MAD c, b, b.x, c; MAX c, b, -c; MIN c, b, c.wzyx; POW c.w, b.x, b.y;\n"
        "RCP c.z, b.w; RSQ c.z, b.x; SGE c, b, c; SLT c, b, c; SUB c, b, c; SWZ c, b, -x, 1, 0, w; XPD c, b, c;\n"
        "ABS c, -b; ADD c, c, half; MUL c, c, att; ADD c, c, spot; ADD c, c, mat; ADD c, c, shine; ADD c, c, lp; ADD c, c, scene;\n"
        "ADD c, c, amb; ADD c, c, fog; ADD c, c, clip; ADD c, c, ps; ADD c, c, dr; ADD c, c, eye; ADD c, c, obj; ADD c, c, gen;\n"
        "DP4 c.x, tex[0], tc;\n"
        "MOV o, pp;\n"
        "MOV result.color, c; MOV result.color.secondary, b; MOV result.color.back, c; MOV result.texcoord[3], tc;\n"
        "MOV result.fogcoord.x, c.x; MOV result.pointsize.x, c.y;\n"
        "END\n";
    const char* fps =
        "!!ARBfp1.0\nOPTION ARB_fog_exp2; OPTION ARB_precision_hint_fastest;\n"
        "ATTRIB col = fragment.color; ATTRIB sec = fragment.color.secondary; ATTRIB t = fragment.texcoord[3];\n"
        "ATTRIB fpos = fragment.position; ATTRIB fogc = fragment.fogcoord;\n"
        "PARAM env = program.env[5]; PARAM tenv = state.texenv[1].color;\n"
        "TEMP a, b;\n"
        "TEX a, t, texture[0], 2D; TXP b, t, texture[1], CUBE; TXB a, t, texture[2], 3D; TEX b, t, texture[3], RECT;\n"
        "TEX a, t.x, texture[4], 1D;\n"
        "CMP a, a, b, col; COS b.x, a.y; SIN b.y, a.x; SCS b.xy, a.z; LRP_SAT a, b, col, sec; KIL -a;\n"
        "ADD a, a, fpos; ADD a, a, fogc; ADD a, a, env; MAD a, a, tenv, b;\n"
        "MOV result.color, a; MOV result.depth.z, b.x;\n"
        "END\n";
    GLuint vp = load(f, GL_VERTEX_PROGRAM_ARB, vps);
    GLuint fp = load(f, GL_FRAGMENT_PROGRAM_ARB, fps);
    glEnable(GL_VERTEX_PROGRAM_ARB);
    glEnable(GL_FRAGMENT_PROGRAM_ARB);
    quad();
    CHECK_GL();
    glDisable(GL_VERTEX_PROGRAM_ARB);
    glDisable(GL_FRAGMENT_PROGRAM_ARB);
    f.del(1, &vp);
    f.del(1, &fp);
}
