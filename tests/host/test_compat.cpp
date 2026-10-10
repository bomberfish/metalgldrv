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

// GL_ARB_shader_objects / GL_ARB_vertex_shader: the handle-based API compiles, links and draws.
TEST(compat_arb_shader_objects) {
    typedef void* H;  // GLhandleARB on Apple platforms
    auto createShader = (H(*)(GLenum))eglGetProcAddress("glCreateShaderObjectARB");
    auto shaderSource = (void (*)(H, GLsizei, const char**, const GLint*))eglGetProcAddress("glShaderSourceARB");
    auto compile = (void (*)(H))eglGetProcAddress("glCompileShaderARB");
    auto createProgram = (H(*)())eglGetProcAddress("glCreateProgramObjectARB");
    auto attach = (void (*)(H, H))eglGetProcAddress("glAttachObjectARB");
    auto link = (void (*)(H))eglGetProcAddress("glLinkProgramARB");
    auto use = (void (*)(H))eglGetProcAddress("glUseProgramObjectARB");
    auto getParam = (void (*)(H, GLenum, GLint*))eglGetProcAddress("glGetObjectParameterivARB");
    auto getLog = (void (*)(H, GLsizei, GLsizei*, char*))eglGetProcAddress("glGetInfoLogARB");
    auto getHandle = (H(*)(GLenum))eglGetProcAddress("glGetHandleARB");
    auto uniformLoc = (GLint(*)(H, const char*))eglGetProcAddress("glGetUniformLocationARB");
    auto bindAttrib = (void (*)(H, GLuint, const char*))eglGetProcAddress("glBindAttribLocationARB");
    auto getAttached = (void (*)(H, GLsizei, GLsizei*, H*))eglGetProcAddress("glGetAttachedObjectsARB");
    auto delObj = (void (*)(H))eglGetProcAddress("glDeleteObjectARB");
    const char* vs = "attribute vec2 pos;\nvoid main() { gl_Position = vec4(pos, 0.0, 1.0); }";
    const char* fs = "uniform vec4 color;\nvoid main() { gl_FragColor = color; }";
    H v = createShader(GL_VERTEX_SHADER), f = createShader(GL_FRAGMENT_SHADER);
    shaderSource(v, 1, &vs, nullptr);
    shaderSource(f, 1, &fs, nullptr);
    compile(v);
    compile(f);
    GLint st = 0;
    getParam(f, 0x8B81 /* GL_OBJECT_COMPILE_STATUS_ARB */, &st);
    CHECK_EQ(st, 1);
    getParam(f, 0x8B4E /* GL_OBJECT_TYPE_ARB */, &st);
    CHECK_EQ(st, 0x8B48 /* GL_SHADER_OBJECT_ARB */);
    getParam(f, 0x8B4F /* GL_OBJECT_SUBTYPE_ARB */, &st);
    CHECK_EQ(st, GL_FRAGMENT_SHADER);
    H p = createProgram();
    attach(p, v);
    attach(p, f);
    bindAttrib(p, 0, "pos");
    link(p);
    getParam(p, 0x8B82 /* GL_OBJECT_LINK_STATUS_ARB */, &st);
    CHECK_EQ(st, 1);
    if (!st) {
        char log[512] = {};
        getLog(p, sizeof log, nullptr, log);
        fprintf(stderr, "  link log: %s\n", log);
    }
    H objs[4] = {};
    GLsizei n = 0;
    getAttached(p, 4, &n, objs);
    CHECK_EQ(n, 2);
    use(p);
    CHECK(getHandle(0x8B40 /* GL_PROGRAM_OBJECT_ARB */) == p);
    glUniform4f(uniformLoc(p, "color"), 0, 0, 1, 1);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    float quad[] = {-1, -1, 1, -1, -1, 1, 1, 1};
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, quad);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glDisableVertexAttribArray(0);
    CHECK_PIXEL(32, 32, 0, 0, 255, 255);
    use(nullptr);
    delObj(p);
    delObj(v);
    delObj(f);
    CHECK_GL();
}

// GL_EXT_stencil_two_side: the active face's state for front- and back-facing primitives.
TEST(compat_ext_stencil_two_side) {
    auto activeFace = (void (*)(GLenum))eglGetProcAddress("glActiveStencilFaceEXT");
    const GLenum TWO_SIDE = 0x8910, ACTIVE_FACE = 0x8911;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glClearColor(0, 0, 0, 1);
    glClearStencil(0);
    glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    auto quads = [] {  // left: counter-clockwise (front), right: clockwise (back)
        glBegin(GL_QUADS);
        glVertex2f(-1, -1); glVertex2f(0, -1); glVertex2f(0, 1); glVertex2f(-1, 1);
        glVertex2f(0, -1); glVertex2f(0, 1); glVertex2f(1, 1); glVertex2f(1, -1);
        glEnd();
    };
    glEnable(GL_STENCIL_TEST);
    glEnable(TWO_SIDE);
    activeFace(GL_BACK);
    glStencilFunc(GL_ALWAYS, 2, 0xff);
    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
    activeFace(GL_FRONT);
    glStencilFunc(GL_ALWAYS, 1, 0xff);
    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
    GLint face = 0;
    glGetIntegerv(ACTIVE_FACE, &face);
    CHECK_EQ(face, GL_FRONT);
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    quads();
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDisable(TWO_SIDE);  // both faces now use the front state
    glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
    glStencilFunc(GL_EQUAL, 1, 0xff);
    glColor3f(0, 1, 0);
    quads();
    glStencilFunc(GL_EQUAL, 2, 0xff);
    glColor3f(0, 0, 1);
    quads();
    glDisable(GL_STENCIL_TEST);
    CHECK_PIXEL(16, 32, 0, 255, 0, 255);  // front quad wrote 1
    CHECK_PIXEL(48, 32, 0, 0, 255, 255);  // back quad wrote 2
    CHECK(!glIsEnabled(TWO_SIDE));
    CHECK_GL();
}

// ARB_vertex_type_2_10_10_10_rev immediate mode: packed positions (unnormalized) and colors (normalized).
TEST(compat_packed_immediate) {
    auto vertexP2 = (void (*)(GLenum, GLuint))eglGetProcAddress("glVertexP2ui");
    auto colorP4 = (void (*)(GLenum, GLuint))eglGetProcAddress("glColorP4ui");
    auto pack = [](int x, int y) { return (GLuint)((x & 1023) | (y & 1023) << 10); };  // signed 10-bit x, y
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_QUADS);
    colorP4(GL_UNSIGNED_INT_2_10_10_10_REV, 0u | 1023u << 10 | 0u << 20 | 3u << 30);  // green, alpha 1
    vertexP2(GL_INT_2_10_10_10_REV, pack(-1, -1));
    vertexP2(GL_INT_2_10_10_10_REV, pack(1, -1));
    vertexP2(GL_INT_2_10_10_10_REV, pack(1, 1));
    vertexP2(GL_INT_2_10_10_10_REV, pack(-1, 1));
    glEnd();
    CHECK_PIXEL(32, 32, 0, 255, 0, 255);
    CHECK_GL();
}

// glGetnMap (ARB_robustness): checked against the buffer size.
TEST(compat_getn_map) {
    auto getnMapfv = (void (*)(GLenum, GLenum, GLsizei, GLfloat*))eglGetProcAddress("glGetnMapfvARB");
    float line[2][3] = {{0, 0, 0}, {1, 1, 0}};
    glMap1f(GL_MAP1_VERTEX_3, 0, 1, 3, 2, &line[0][0]);
    float coeff[6] = {};
    getnMapfv(GL_MAP1_VERTEX_3, GL_COEFF, sizeof coeff, coeff);
    CHECK(coeff[3] == 1 && coeff[4] == 1);
    CHECK_GL();
    getnMapfv(GL_MAP1_VERTEX_3, GL_COEFF, 4 * sizeof(float), coeff);
    CHECK_EQ(glGetError(), (GLenum)GL_INVALID_OPERATION);
}

// GL_NV_primitive_restart: restart inside Begin/End, and the enable / index as core primitive restart.
TEST(compat_nv_primitive_restart) {
    auto restart = (void (*)())eglGetProcAddress("glPrimitiveRestartNV");
    auto restartIndex = (void (*)(GLuint))eglGetProcAddress("glPrimitiveRestartIndexNV");
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f(1, 1, 0);
    // two separate strips; without the restart the strip would also cover the middle band
    glBegin(GL_TRIANGLE_STRIP);
    glVertex2f(-1, -1); glVertex2f(-0.5f, -1); glVertex2f(-1, 1); glVertex2f(-0.5f, 1);
    restart();
    glVertex2f(0.5f, -1); glVertex2f(1, -1); glVertex2f(0.5f, 1); glVertex2f(1, 1);
    glEnd();
    CHECK_PIXEL(4, 32, 255, 255, 0, 255);
    CHECK_PIXEL(60, 32, 255, 255, 0, 255);
    CHECK_PIXEL(32, 32, 0, 0, 0, 255);
    glEnable(0x8558 /* GL_PRIMITIVE_RESTART_NV */);
    CHECK(glIsEnabled(GL_PRIMITIVE_RESTART));
    restartIndex(77);
    GLint idx = 0;
    glGetIntegerv(0x8559 /* GL_PRIMITIVE_RESTART_INDEX_NV */, &idx);
    CHECK_EQ(idx, 77);
    glDisable(0x8558);
    CHECK_GL();
}

// GL_ATI_texture_float: ATI's names for the ARB_texture_float formats.
TEST(compat_ati_texture_float) {
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    float a = 3.0f;
    glTexImage2D(GL_TEXTURE_2D, 0, 0x8816 /* GL_ALPHA_FLOAT32_ATI */, 1, 1, 0, GL_ALPHA, GL_FLOAT, &a);
    float out[4] = {};
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, out);
    CHECK(out[0] == 0 && out[3] == 3.0f);
    glDeleteTextures(1, &tex);
    CHECK_GL();
}

// GL_ATI_texture_mirror_once: MIRROR_CLAMP_TO_EDGE_ATI mirrors once around 0, then clamps.
TEST(compat_ati_texture_mirror_once) {
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_1D, tex);
    uint8_t px[8] = {255, 0, 0, 255, 0, 255, 0, 255};  // red, green
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA8, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, 0x8743 /* GL_MIRROR_CLAMP_TO_EDGE_ATI */);
    GLint w = 0;
    glGetTexParameteriv(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, &w);
    CHECK_EQ(w, 0x8743);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glEnable(GL_TEXTURE_1D);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glBegin(GL_QUADS);  // s from -1.5 (left) to 0 (right)
    glTexCoord1f(-1.5f); glVertex2f(-1, -1);
    glTexCoord1f(0); glVertex2f(1, -1);
    glTexCoord1f(0); glVertex2f(1, 1);
    glTexCoord1f(-1.5f); glVertex2f(-1, 1);
    glEnd();
    glDisable(GL_TEXTURE_1D);
    CHECK_PIXEL(2, 32, 0, 255, 0, 255);   // |s| > 1: clamped to the last texel
    CHECK_PIXEL(52, 32, 255, 0, 0, 255);  // s in (-0.5, 0): mirrored into the first texel
    glDeleteTextures(1, &tex);
    CHECK_GL();
}

// GL_NV_texgen_reflection: REFLECTION_MAP_NV / NORMAL_MAP_NV texgen modes. A normal facing the viewer
// reflects straight back: NORMAL_MAP gives (0, 0, 1), REFLECTION_MAP (0, 0, 1) too (the eye looks down -Z).
TEST(compat_nv_texgen_reflection) {
    glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, 0x8512 /* GL_REFLECTION_MAP_NV */);
    glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, 0x8511 /* GL_NORMAL_MAP_NV */);
    GLint m = 0;
    glGetTexGeniv(GL_S, GL_TEXTURE_GEN_MODE, &m);
    CHECK_EQ(m, 0x8512);
    glGetTexGeniv(GL_T, GL_TEXTURE_GEN_MODE, &m);
    CHECK_EQ(m, 0x8511);
    CHECK_GL();
}

// GL_EXT_texture_env_add: the ADD texture environment (ARB_texture_env_add's).
TEST(compat_ext_texture_env_add) {
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    uint8_t px[4] = {0, 128, 0, 255};
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glEnable(GL_TEXTURE_2D);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_ADD);
    glColor3f(0.5f, 0.25f, 0);
    glRectf(-1, -1, 1, 1);
    glDisable(GL_TEXTURE_2D);
    uint8_t p[4];
    readPixel(32, 32, p);
    CHECK(abs(p[0] - 128) <= 2 && abs(p[1] - 192) <= 2 && p[2] <= 2);
    glDeleteTextures(1, &tex);
    CHECK_GL();
}

// Immediate-mode type variants: integer colors normalize, integer coordinates don't.
TEST(compat_immediate_variants) {
    GLshort c[3] = {32767, 0, 16384};
    glColor3sv(c);
    GLfloat cur[4] = {};
    glGetFloatv(GL_CURRENT_COLOR, cur);
    CHECK(cur[0] == 1.0f && cur[1] == 0 && fabsf(cur[2] - 0.5f) < 0.001f && cur[3] == 1.0f);
    glTexCoord3i(2, 3, 4);
    glGetFloatv(GL_CURRENT_TEXTURE_COORDS, cur);
    CHECK(cur[0] == 2 && cur[1] == 3 && cur[2] == 4 && cur[3] == 1);
    GLdouble mt[4] = {0.5, 0.25, 0, 2};
    glMultiTexCoord4dv(GL_TEXTURE0, mt);
    glGetFloatv(GL_CURRENT_TEXTURE_COORDS, cur);
    CHECK(cur[0] == 0.5f && cur[3] == 2);
    glSecondaryColor3ub(255, 0, 0);
    glGetFloatv(GL_CURRENT_SECONDARY_COLOR, cur);
    CHECK(cur[0] == 1.0f);
    // vertices from integer variants
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 64, 0, 64, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glColor3ub(0, 255, 0);
    GLint v0[4] = {0, 0, 0, 1}, v1[4] = {32, 0, 0, 1}, v2[4] = {32, 32, 0, 1}, v3[4] = {0, 32, 0, 1};
    glBegin(GL_QUADS);
    glVertex4iv(v0);
    glVertex4iv(v1);
    glVertex4iv(v2);
    glVertex4iv(v3);
    glEnd();
    CHECK_PIXEL(16, 16, 0, 255, 0, 255);
    CHECK_PIXEL(48, 48, 0, 0, 0, 255);
    CHECK_GL();
}

// GL_EXT_secondary_color: the EXT entry points, and the secondary color added with COLOR_SUM.
TEST(compat_ext_secondary_color) {
    auto sec3f = (void (*)(GLfloat, GLfloat, GLfloat))eglGetProcAddress("glSecondaryColor3fEXT");
    auto sec3ub = (void (*)(GLubyte, GLubyte, GLubyte))eglGetProcAddress("glSecondaryColor3ubEXT");
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glEnable(GL_COLOR_SUM);
    glColor3f(1, 0, 0);
    sec3f(0, 0, 1);
    glRectf(-1, -1, 0, 1);
    sec3ub(0, 255, 0);
    glRectf(0, -1, 1, 1);
    glDisable(GL_COLOR_SUM);
    CHECK_PIXEL(16, 32, 255, 0, 255, 255);
    CHECK_PIXEL(48, 32, 255, 255, 0, 255);
    CHECK_GL();
}

// GL_EXT_fog_coord: fog from per-vertex fog coordinates (FOG_COORDINATE_SOURCE_EXT = FOG_COORDINATE_EXT).
TEST(compat_ext_fog_coord) {
    auto fogCoordf = (void (*)(GLfloat))eglGetProcAddress("glFogCoordfEXT");
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glEnable(GL_FOG);
    glFogi(GL_FOG_MODE, GL_LINEAR);
    glFogf(GL_FOG_START, 0);
    glFogf(GL_FOG_END, 1);
    float fogColor[4] = {0, 0, 1, 1};
    glFogfv(GL_FOG_COLOR, fogColor);
    glFogi(0x8450 /* GL_FOG_COORDINATE_SOURCE_EXT */, 0x8451 /* GL_FOG_COORDINATE_EXT */);
    glColor3f(1, 0, 0);
    glBegin(GL_QUADS);
    fogCoordf(1);  // fully fogged
    glVertex2f(-1, -1); glVertex2f(1, -1); glVertex2f(1, 1); glVertex2f(-1, 1);
    glEnd();
    glDisable(GL_FOG);
    glFogi(0x8450, GL_FRAGMENT_DEPTH);
    CHECK_PIXEL(32, 32, 0, 0, 255, 255);
    CHECK_GL();
}

// Point parameters: distance attenuation and size bounds in fixed-function drawing.
TEST(compat_point_parameters) {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1, 1, -1, 1, 0, 10);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glPointSize(16);
    float att[3] = {0, 0, 1};  // size / distance
    glPointParameterfv(GL_POINT_DISTANCE_ATTENUATION, att);
    float got[3] = {};
    glGetFloatv(GL_POINT_DISTANCE_ATTENUATION, got);
    CHECK(got[2] == 1);
    glColor3f(1, 1, 1);
    glBegin(GL_POINTS);
    glVertex3f(0, 0, -4);  // eye distance 4: 16 / 4 = 4 pixels
    glEnd();
    CHECK_PIXEL(32 + 1, 32 + 1, 255, 255, 255, 255);
    CHECK_PIXEL(32 + 4, 32 + 4, 0, 0, 0, 255);
    glPointParameterf(GL_POINT_SIZE_MIN, 10);  // now clamped up to 10
    glBegin(GL_POINTS);
    glVertex3f(0, 0, -4);
    glEnd();
    CHECK_PIXEL(32 + 4, 32 + 4, 255, 255, 255, 255);
    float d[3] = {1, 0, 0};
    glPointParameterfv(GL_POINT_DISTANCE_ATTENUATION, d);
    glPointParameterf(GL_POINT_SIZE_MIN, 0);
    glPointSize(1);
    CHECK_GL();
}

// GL_EXT_point_parameters: the EXT entry points and enums (the GL 1.4 values).
TEST(compat_ext_point_parameters) {
    auto pf = (void (*)(GLenum, GLfloat))eglGetProcAddress("glPointParameterfEXT");
    auto pfv = (void (*)(GLenum, const GLfloat*))eglGetProcAddress("glPointParameterfvEXT");
    pf(0x8127 /* GL_POINT_SIZE_MAX_EXT */, 20);
    float att[3] = {1, 0.5f, 0};
    pfv(0x8129 /* GL_DISTANCE_ATTENUATION_EXT */, att);
    float v = 0, a[3] = {};
    glGetFloatv(GL_POINT_SIZE_MAX, &v);
    glGetFloatv(GL_POINT_DISTANCE_ATTENUATION, a);
    CHECK(v == 20 && a[1] == 0.5f);
    pf(0x8127, 511);
    float d[3] = {1, 0, 0};
    pfv(0x8129, d);
    CHECK_GL();
}

// GL_EXT_compiled_vertex_array: lock / unlock around draws of the locked range.
TEST(compat_ext_compiled_vertex_array) {
    auto lock = (void (*)(GLint, GLsizei))eglGetProcAddress("glLockArraysEXT");
    auto unlock = (void (*)())eglGetProcAddress("glUnlockArraysEXT");
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    float pos[] = {-1, -1, 1, -1, 1, 1, -1, 1};
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_FLOAT, 0, pos);
    lock(0, 4);
    GLint v = 0;
    glGetIntegerv(0x81A9 /* GL_ARRAY_ELEMENT_LOCK_COUNT_EXT */, &v);
    CHECK_EQ(v, 4);
    glColor3f(1, 0, 1);
    glDrawArrays(GL_QUADS, 0, 4);
    unlock();
    glDisableClientState(GL_VERTEX_ARRAY);
    CHECK_PIXEL(32, 32, 255, 0, 255, 255);
    unlock();
    CHECK_EQ(glGetError(), (GLenum)GL_INVALID_OPERATION);
}

// glArrayElement: vertices from the enabled client arrays inside Begin/End, also compiled into a list.
TEST(compat_array_element) {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    float pos[] = {-1, -1, 0, -1, 0, 1, -1, 1};
    uint8_t col[] = {0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255};
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glVertexPointer(2, GL_FLOAT, 0, pos);
    glColorPointer(4, GL_UNSIGNED_BYTE, 0, col);
    glBegin(GL_QUADS);
    for (int i = 0; i < 4; i++) glArrayElement(i);
    glEnd();
    CHECK_PIXEL(16, 32, 0, 255, 0, 255);
    GLuint list = glGenLists(1);
    glNewList(list, GL_COMPILE);
    glBegin(GL_QUADS);
    for (int i = 0; i < 4; i++) glArrayElement(i);
    glEnd();
    glEndList();
    for (auto& f : pos) f += (&f - pos) % 2 == 0 ? 1 : 0;  // move the arrays right: the list keeps the old ones
    glClear(GL_COLOR_BUFFER_BIT);
    glCallList(list);
    CHECK_PIXEL(16, 32, 0, 255, 0, 255);
    CHECK_PIXEL(48, 32, 0, 0, 0, 255);
    glDeleteLists(list, 1);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    CHECK_GL();
}

// GL_EXT_vertex_array: the EXT pointer calls (with their count) and glDrawArraysEXT / glArrayElementEXT.
TEST(compat_ext_vertex_array) {
    auto vertexPointer = (void (*)(GLint, GLenum, GLsizei, GLsizei, const void*))eglGetProcAddress("glVertexPointerEXT");
    auto colorPointer = (void (*)(GLint, GLenum, GLsizei, GLsizei, const void*))eglGetProcAddress("glColorPointerEXT");
    auto drawArrays = (void (*)(GLenum, GLint, GLsizei))eglGetProcAddress("glDrawArraysEXT");
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    float pos[] = {-1, -1, 1, -1, 1, 1, -1, 1};
    float col[] = {1, 0.5f, 0, 1, 1, 0.5f, 0, 1, 1, 0.5f, 0, 1, 1, 0.5f, 0, 1};
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    vertexPointer(2, GL_FLOAT, 0, 4, pos);
    colorPointer(4, GL_FLOAT, 0, 4, col);
    drawArrays(GL_QUADS, 0, 4);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    CHECK_PIXEL(32, 32, 255, 128, 0, 255);
    CHECK_GL();
}
