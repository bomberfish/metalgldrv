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
