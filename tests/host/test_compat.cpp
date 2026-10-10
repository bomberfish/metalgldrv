// Compatibility-profile fixed-function tests.
#include "harness.h"

TEST(compat_immediate_and_matrices) {
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 64, 0, 64, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(32, 0, 0);  // right half
    glColor3f(1, 0, 0);
    glBegin(GL_QUADS);
    glVertex2f(0, 0);
    glVertex2f(32, 0);
    glVertex2f(32, 32);
    glVertex2f(0, 32);
    glEnd();
    CHECK_PIXEL(48, 16, 255, 0, 0, 255);
    CHECK_PIXEL(16, 16, 0, 0, 0, 255);
    CHECK_PIXEL(48, 48, 0, 0, 0, 255);
    // per-vertex colors + push/pop
    glPushMatrix();
    glLoadIdentity();
    glBegin(GL_TRIANGLES);
    glColor3f(0, 1, 0);
    glVertex2f(0, 32);
    glVertex2f(32, 32);
    glVertex2f(0, 64);
    glEnd();
    glPopMatrix();
    CHECK_PIXEL(4, 36, 0, 255, 0, 255);
    GLfloat m[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, m);
    CHECK(m[12] == 32.0f);
    CHECK_GL();
}

TEST(compat_client_arrays_and_lists) {
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    float pos[] = {-1, -1, 0, -1, -1, 0, 0, 0};
    unsigned char col[] = {0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255};
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glVertexPointer(2, GL_FLOAT, 0, pos);
    glColorPointer(4, GL_UNSIGNED_BYTE, 0, col);
    glDrawArrays(GL_QUADS, 0, 4);  // bottom-left quadrant blue
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    CHECK_PIXEL(16, 16, 0, 0, 255, 255);
    CHECK_PIXEL(48, 48, 0, 0, 0, 255);
    GLuint list = glGenLists(1);
    glNewList(list, GL_COMPILE);
    glColor3f(1, 1, 0);
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(0, 0);
    glVertex2f(1, 0);
    glVertex2f(1, 1);
    glVertex2f(0, 1);
    glEnd();
    glEndList();
    CHECK_PIXEL(48, 48, 0, 0, 0, 255);  // compile only: nothing drawn yet
    glCallList(list);
    CHECK_PIXEL(48, 48, 255, 255, 0, 255);
    glDeleteLists(list, 1);
    CHECK_GL();
}

TEST(compat_texture_env_alpha_fog) {
    uint32_t px[4] = {0xff00ff00u, 0x00000000u, 0xff00ff00u, 0x00000000u};  // green / transparent columns
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glEnable(GL_TEXTURE_2D);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glEnable(GL_ALPHA_TEST);
    glAlphaFunc(GL_GREATER, 0.5f);
    glClearColor(1, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glColor4f(1, 1, 1, 1);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(-1, -1);
    glTexCoord2f(1, 0); glVertex2f(1, -1);
    glTexCoord2f(1, 1); glVertex2f(1, 1);
    glTexCoord2f(0, 1); glVertex2f(-1, 1);
    glEnd();
    CHECK_PIXEL(10, 10, 0, 255, 0, 255);  // left column textured green
    CHECK_PIXEL(50, 10, 255, 0, 0, 255);  // right column alpha-tested away
    glDisable(GL_ALPHA_TEST);
    glDisable(GL_TEXTURE_2D);
    // linear fog: everything at eye z=0 -> distance 0 -> f = (end-0)/(end-start) = 1 (no fog) ; use start/end to force full fog
    glEnable(GL_FOG);
    glFogi(GL_FOG_MODE, GL_LINEAR);
    glFogf(GL_FOG_START, 10);
    glFogf(GL_FOG_END, 20);  // distance 0 < start -> f > 1 -> clamped 1 -> unfogged
    float fogColor[4] = {0, 0, 1, 1};
    glFogfv(GL_FOG_COLOR, fogColor);
    glColor3f(1, 1, 1);
    glRectf(-1, -1, 0, 0);
    CHECK_PIXEL(10, 10, 255, 255, 255, 255);
    glFogf(GL_FOG_START, -20);
    glFogf(GL_FOG_END, -10);  // f = (end - 0)/(end-start) < 0 -> fully fogged
    glRectf(-1, -1, 0, 0);
    CHECK_PIXEL(10, 10, 0, 0, 255, 255);
    glDisable(GL_FOG);
    glDeleteTextures(1, &t);
    CHECK_GL();
}

TEST(compat_lighting) {
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    float dir[4] = {0, 0, 1, 0};  // directional, towards the viewer
    glLightfv(GL_LIGHT0, GL_POSITION, dir);
    float zero[4] = {0, 0, 0, 1};
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, zero);
    float diff[4] = {1, 0.5f, 0.25f, 1};
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diff);
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, zero);
    glNormal3f(0, 0, 1);
    glRectf(-1, -1, 1, 1);
    CHECK_PIXEL(32, 32, 255, 128, 64, 255);
    glNormal3f(0, 0, -1);  // facing away -> black
    glRectf(-1, -1, 1, 1);
    CHECK_PIXEL(32, 32, 0, 0, 0, 255);
    glDisable(GL_LIGHTING);
    CHECK_GL();
}

// GL_APPLE_element_array: client index array, drawn from index `first`.
TEST(compat_apple_element_array) {
    auto elementPointer = (void (*)(GLenum, const void*))eglGetProcAddress("glElementPointerAPPLE");
    auto drawElementArray = (void (*)(GLenum, GLint, GLsizei))eglGetProcAddress("glDrawElementArrayAPPLE");
    auto multiDraw = (void (*)(GLenum, const GLint*, const GLsizei*, GLsizei))eglGetProcAddress("glMultiDrawElementArrayAPPLE");
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    // left half (x -1..0) and right half (0..1) quads
    float pos[] = {-1, -1, 0, -1, -1, 1, 0, 1, 0, -1, 1, -1, 0, 1, 1, 1};
    uint16_t idx[] = {9, 9, 0, 1, 2, 2, 1, 3, 4, 5, 6, 6, 5, 7};  // two junk indices first
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_FLOAT, 0, pos);
    glEnableClientState(0x8A0C /* GL_ELEMENT_ARRAY_APPLE */);
    elementPointer(GL_UNSIGNED_SHORT, idx);
    CHECK(glIsEnabled(0x8A0C));
    GLint type = 0;
    glGetIntegerv(0x8A0D /* GL_ELEMENT_ARRAY_TYPE_APPLE */, &type);
    CHECK_EQ(type, GL_UNSIGNED_SHORT);
    void* p = nullptr;
    glGetPointerv(0x8A0E /* GL_ELEMENT_ARRAY_POINTER_APPLE */, &p);
    CHECK(p == idx);
    glColor3f(1, 0, 0);
    drawElementArray(GL_TRIANGLES, 2, 6);  // left half
    CHECK_PIXEL(16, 32, 255, 0, 0, 255);
    CHECK_PIXEL(48, 32, 0, 0, 0, 255);
    glColor3f(0, 1, 0);
    GLint firsts[] = {2, 8};
    GLsizei counts[] = {6, 6};
    multiDraw(GL_TRIANGLES, firsts, counts, 2);
    CHECK_PIXEL(16, 32, 0, 255, 0, 255);
    CHECK_PIXEL(48, 32, 0, 255, 0, 255);
    glDisableClientState(0x8A0C);
    glDisableClientState(GL_VERTEX_ARRAY);
    CHECK_GL();
}

// GL_APPLE_float_pixels / ARB_texture_float legacy formats: luminance-alpha and intensity floats keep values above 1.
TEST(compat_apple_float_pixels) {
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    float la[2] = {2.5f, 0.25f};
    glTexImage2D(GL_TEXTURE_2D, 0, 0x8819 /* GL_LUMINANCE_ALPHA_FLOAT32_APPLE */, 1, 1, 0, GL_LUMINANCE_ALPHA, GL_FLOAT, la);
    CHECK_GL();
    GLint fmt = 0, type = 0;
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &fmt);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_LUMINANCE_TYPE, &type);
    CHECK_EQ(fmt, 0x8819);
    CHECK_EQ(type, GL_FLOAT);
    float out[4] = {};
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, out);
    CHECK(out[0] == 2.5f && out[1] == 2.5f && out[2] == 2.5f && out[3] == 0.25f);
    uint16_t half = 0x4500;  // 5.0
    glTexImage2D(GL_TEXTURE_2D, 0, 0x881D /* GL_INTENSITY_FLOAT16_APPLE */, 1, 1, 0, GL_LUMINANCE, 0x140B /* GL_HALF_APPLE */, &half);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, out);
    CHECK(out[0] == 5.0f && out[3] == 5.0f);
    GLboolean isFloat = GL_TRUE;
    glGetBooleanv(0x8A0F /* GL_COLOR_FLOAT_APPLE */, &isFloat);
    CHECK(!isFloat);  // the RGBA8 window surface
    glDeleteTextures(1, &tex);
    CHECK_GL();
}

// GL_APPLE_vertex_point_size: per-vertex point sizes (array or current value) in fixed-function drawing.
TEST(compat_apple_vertex_point_size) {
    auto sizePointer = (void (*)(GLenum, GLsizei, const void*))eglGetProcAddress("glPointSizePointerAPPLE");
    auto vertexPointSize = (void (*)(GLfloat))eglGetProcAddress("glVertexPointSizefAPPLE");
    CHECK(sizePointer && vertexPointSize);
    if (!sizePointer || !vertexPointSize) return;
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glPointSize(1);
    glEnable(0x8A26 /* GL_VERTEX_POINT_SIZE_APPLE */);
    // two points: left one 1px, right one 9px (from the array)
    float pos[] = {-0.5f, 0, 0.5f, 0};
    float sizes[] = {1, 9};
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_FLOAT, 0, pos);
    glEnableClientState(0x8B9C /* GL_POINT_SIZE_ARRAY_APPLE */);
    sizePointer(GL_FLOAT, 0, sizes);
    glColor3f(1, 0, 0);
    glDrawArrays(GL_POINTS, 0, 2);
    glDisableClientState(0x8B9C);
    glDisableClientState(GL_VERTEX_ARRAY);
    CHECK_PIXEL(48 + 3, 32 + 3, 255, 0, 0, 255);  // inside the 9px point
    CHECK_PIXEL(16 + 3, 32 + 3, 0, 0, 0, 255);    // outside the 1px point
    // immediate mode: current point size
    glColor3f(0, 1, 0);
    glBegin(GL_POINTS);
    vertexPointSize(7);
    glVertex2f(-0.5f, 0.5f);
    glEnd();
    CHECK_PIXEL(16 + 2, 48 + 2, 0, 255, 0, 255);
    GLfloat cur = 0;
    glGetFloatv(0x8A27 /* GL_CURRENT_POINT_SIZE_APPLE */, &cur);
    CHECK(cur == 7.0f);
    glDisable(0x8A26);
    CHECK_GL();
}

// GL_APPLE_specular_vector: Phong specular (light reflected about the normal) instead of the half vector.
// Light 45 degrees off the normal, viewer straight on: the half vector is 22.5 degrees off the normal
// (Blinn: cos 22.5 = 0.92), the reflection 45 degrees off the view vector (Phong: cos 45 = 0.71).
TEST(compat_apple_specular_vector) {
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    float pos[] = {1, 0, 1, 0};  // directional, 45 degrees
    float zero[] = {0, 0, 0, 1}, one[] = {1, 1, 1, 1};
    glLightfv(GL_LIGHT0, GL_POSITION, pos);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, zero);
    glLightfv(GL_LIGHT0, GL_AMBIENT, zero);
    glLightfv(GL_LIGHT0, GL_SPECULAR, one);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, zero);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, one);
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, zero);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, zero);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 1);
    auto quad = [] {
        glBegin(GL_QUADS);
        glNormal3f(0, 0, 1);
        glVertex2f(-1, -1);
        glVertex2f(1, -1);
        glVertex2f(1, 1);
        glVertex2f(-1, 1);
        glEnd();
    };
    quad();
    uint8_t blinn[4], phong[4];
    readPixel(32, 32, blinn);
    glLightModeli(0x85B0 /* GL_LIGHT_MODEL_SPECULAR_VECTOR_APPLE */, GL_TRUE);
    GLint v = 0;
    glGetIntegerv(0x85B0, &v);
    CHECK_EQ(v, 1);
    quad();
    readPixel(32, 32, phong);
    glLightModeli(0x85B0, GL_FALSE);
    glDisable(GL_LIGHTING);
    CHECK(abs(blinn[0] - 235) <= 3);  // 0.924
    CHECK(abs(phong[0] - 180) <= 3);  // 0.707
    CHECK_GL();
}

// GL_APPLE_vertex_array_range: the range is recorded; arrays edited in it draw with the new data after a flush.
TEST(compat_apple_vertex_array_range) {
    auto range = (void (*)(GLsizei, void*))eglGetProcAddress("glVertexArrayRangeAPPLE");
    auto flush = (void (*)(GLsizei, void*))eglGetProcAddress("glFlushVertexArrayRangeAPPLE");
    auto param = (void (*)(GLenum, GLint))eglGetProcAddress("glVertexArrayParameteriAPPLE");
    const GLenum VAR = 0x851D, VAR_LENGTH = 0x851E, STORAGE_HINT = 0x851F, VAR_POINTER = 0x8521, STORAGE_CACHED = 0x85BE;
    static float pos[8] = {-1, -1, 0, -1, -1, 0, 0, 0};  // bottom-left quadrant
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    range(sizeof pos, pos);
    param(STORAGE_HINT, STORAGE_CACHED);
    glEnableClientState(VAR);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_FLOAT, 0, pos);
    glColor3f(1, 1, 0);
    glDrawArrays(GL_QUADS, 0, 4);
    for (int i = 0; i < 8; i++) pos[i] += 1;  // move to the top-right quadrant
    flush(sizeof pos, pos);
    glColor3f(0, 1, 1);
    glDrawArrays(GL_QUADS, 0, 4);
    CHECK_PIXEL(16, 16, 255, 255, 0, 255);
    CHECK_PIXEL(48, 48, 0, 255, 255, 255);
    GLint v = 0;
    glGetIntegerv(VAR_LENGTH, &v);
    CHECK_EQ(v, (GLint)sizeof pos);
    glGetIntegerv(STORAGE_HINT, &v);
    CHECK_EQ(v, (GLint)STORAGE_CACHED);
    void* p = nullptr;
    glGetPointerv(VAR_POINTER, &p);
    CHECK(p == pos);
    CHECK(glIsEnabled(VAR));
    glDisableClientState(VAR);
    glDisableClientState(GL_VERTEX_ARRAY);
    CHECK_GL();
}

// Evaluators: a 1D vertex map drawn as a line mesh, a 2D vertex + color map drawn as a filled mesh.
TEST(compat_evaluators) {
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    // 2D: the left half (x -1..0), red at v=0 (bottom) to blue at v=1 (top)
    float patch[2][2][3] = {{{-1, -1, 0}, {-1, 1, 0}}, {{0, -1, 0}, {0, 1, 0}}};  // [u][v]
    float colors[2][2][4] = {{{1, 0, 0, 1}, {0, 0, 1, 1}}, {{1, 0, 0, 1}, {0, 0, 1, 1}}};
    glMap2f(GL_MAP2_VERTEX_3, 0, 1, 6, 2, 0, 1, 3, 2, &patch[0][0][0]);
    glMap2f(GL_MAP2_COLOR_4, 0, 1, 8, 2, 0, 1, 4, 2, &colors[0][0][0]);
    glEnable(GL_MAP2_VERTEX_3);
    glEnable(GL_MAP2_COLOR_4);
    glMapGrid2f(4, 0, 1, 4, 0, 1);
    glEvalMesh2(GL_FILL, 0, 4, 0, 4);
    glDisable(GL_MAP2_VERTEX_3);
    glDisable(GL_MAP2_COLOR_4);
    uint8_t p[4];
    readPixel(16, 2, p);
    CHECK(p[0] > 230 && p[2] < 25);  // near v=0: red
    readPixel(16, 61, p);
    CHECK(p[2] > 230 && p[0] < 25);  // near v=1: blue
    readPixel(16, 32, p);
    CHECK(abs(p[0] - 128) < 12 && abs(p[2] - 128) < 12);
    // 1D: a horizontal line across the right half at y = 0.5 (pixel row 48)
    float line[2][3] = {{0, 0.515625f, 0}, {1, 0.515625f, 0}};  // through pixel centers of row 48
    glMap1f(GL_MAP1_VERTEX_3, 0, 1, 3, 2, &line[0][0]);
    glEnable(GL_MAP1_VERTEX_3);
    glColor3f(0, 1, 0);
    glMapGrid1f(8, 0, 1);
    glEvalMesh1(GL_LINE, 0, 8);
    CHECK_PIXEL(48, 48, 0, 255, 0, 255);
    CHECK_PIXEL(48, 40, 0, 0, 0, 255);
    // A quadratic map evaluated directly: the midpoint of (0,0) (1,2) (2,0) is (1,1).
    float quad[3][3] = {{0, 0, 0}, {1, 2, 0}, {2, 0, 0}};
    glMap1f(GL_MAP1_VERTEX_3, 0, 1, 3, 3, &quad[0][0]);
    GLint order = 0;
    glGetMapiv(GL_MAP1_VERTEX_3, GL_ORDER, &order);
    CHECK_EQ(order, 3);
    float dom[2] = {};
    glGetMapfv(GL_MAP1_VERTEX_3, GL_DOMAIN, dom);
    CHECK(dom[0] == 0 && dom[1] == 1);
    float coeff[9] = {};
    glGetMapfv(GL_MAP1_VERTEX_3, GL_COEFF, coeff);
    CHECK(coeff[4] == 2);
    glDisable(GL_MAP1_VERTEX_3);
    CHECK(!glIsEnabled(GL_MAP1_VERTEX_3));
    GLint segs = 0;
    glGetIntegerv(GL_MAP1_GRID_SEGMENTS, &segs);
    CHECK_EQ(segs, 8);
    CHECK_GL();
}

// glVertexAttrib between Begin and End: generic attributes per vertex, attribute 0 makes the vertex.
TEST(compat_immediate_generic_attribs) {
    const char* vs = "#version 120\nattribute vec4 pos;\nattribute vec4 col;\nvarying vec4 c;\n"
                     "void main() { gl_Position = pos; c = col; }";
    const char* fs = "#version 120\nvarying vec4 c;\nvoid main() { gl_FragColor = c; }";
    GLuint v = glCreateShader(GL_VERTEX_SHADER), f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(v, 1, &vs, nullptr);
    glShaderSource(f, 1, &fs, nullptr);
    glCompileShader(v);
    glCompileShader(f);
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glBindAttribLocation(p, 0, "pos");
    glBindAttribLocation(p, 3, "col");
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    CHECK(ok);
    glUseProgram(p);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_QUADS);  // left half red, right half green (flat per quad: all four vertices alike)
    glVertexAttrib4f(3, 1, 0, 0, 1);
    glVertexAttrib2f(0, -1, -1);
    glVertexAttrib2f(0, 0, -1);
    glVertexAttrib2f(0, 0, 1);
    glVertexAttrib2f(0, -1, 1);
    glVertexAttrib4f(3, 0, 1, 0, 1);
    glVertexAttrib2f(0, 0, -1);
    glVertexAttrib2f(0, 1, -1);
    glVertexAttrib2f(0, 1, 1);
    glVertexAttrib2f(0, 0, 1);
    glEnd();
    CHECK_PIXEL(16, 32, 255, 0, 0, 255);
    CHECK_PIXEL(48, 32, 0, 255, 0, 255);
    glUseProgram(0);
    glDeleteProgram(p);
    glDeleteShader(v);
    glDeleteShader(f);
    CHECK_GL();
}

// GL_APPLE_vertex_program_evaluators: maps for generic attributes; attribute 0 makes the vertices.
TEST(compat_apple_vertex_program_evaluators) {
    auto map2 = (void (*)(GLuint, GLuint, GLfloat, GLfloat, GLint, GLint, GLfloat, GLfloat, GLint, GLint, const GLfloat*))
        eglGetProcAddress("glMapVertexAttrib2fAPPLE");
    auto enable = (void (*)(GLuint, GLenum))eglGetProcAddress("glEnableVertexAttribAPPLE");
    auto disable = (void (*)(GLuint, GLenum))eglGetProcAddress("glDisableVertexAttribAPPLE");
    auto isEnabled = (GLboolean(*)(GLuint, GLenum))eglGetProcAddress("glIsVertexAttribEnabledAPPLE");
    const GLenum MAP2 = 0x8A01, MAP2_ORDER = 0x8A08, MAP2_SIZE = 0x8A06;
    const char* vs = "#version 120\nattribute vec4 pos;\nattribute vec4 col;\nvarying vec4 c;\n"
                     "void main() { gl_Position = pos; c = col; }";
    const char* fs = "#version 120\nvarying vec4 c;\nvoid main() { gl_FragColor = c; }";
    GLuint v = glCreateShader(GL_VERTEX_SHADER), f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(v, 1, &vs, nullptr);
    glShaderSource(f, 1, &fs, nullptr);
    glCompileShader(v);
    glCompileShader(f);
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glBindAttribLocation(p, 0, "pos");
    glBindAttribLocation(p, 3, "col");
    glLinkProgram(p);
    glUseProgram(p);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    float pos[2][2][2] = {{{-1, -1}, {-1, 1}}, {{1, -1}, {1, 1}}};                                // [u][v] xy
    float col[2][2][4] = {{{1, 0, 0, 1}, {1, 0, 0, 1}}, {{0, 1, 0, 1}, {0, 1, 0, 1}}};            // red at u=0, green at u=1
    map2(0, 2, 0, 1, 4, 2, 0, 1, 2, 2, &pos[0][0][0]);
    map2(3, 4, 0, 1, 8, 2, 0, 1, 4, 2, &col[0][0][0]);
    enable(0, MAP2);
    enable(3, MAP2);
    CHECK(isEnabled(3, MAP2));
    GLint order[2] = {}, size = 0;
    glGetVertexAttribiv(3, MAP2_ORDER, order);
    glGetVertexAttribiv(3, MAP2_SIZE, &size);
    CHECK(order[0] == 2 && order[1] == 2 && size == 4);
    glMapGrid2f(8, 0, 1, 8, 0, 1);
    glEvalMesh2(GL_FILL, 0, 8, 0, 8);
    uint8_t px[4];
    readPixel(2, 32, px);
    CHECK(px[0] > 230 && px[1] < 25);
    readPixel(61, 32, px);
    CHECK(px[1] > 230 && px[0] < 25);
    disable(0, MAP2);
    disable(3, MAP2);
    glUseProgram(0);
    glDeleteProgram(p);
    glDeleteShader(v);
    glDeleteShader(f);
    CHECK_GL();
}
