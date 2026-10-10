// Apple's vendor extensions (GL_APPLE_*), as Apple's OpenGL exposes them, so code written against it runs
// unchanged. Most map onto the core features they were the precursors of.
#import "gl/context.h"
#include "util/log.h"

using namespace mgl;
#define API extern "C" __attribute__((visibility("default")))

// --- GL_APPLE_vertex_array_object -----------------------------------------------------------------------
// Gen/Delete/IsVertexArraysAPPLE are the core functions (gl.xml aliases). Binding differs: any name works,
// and binding one that GenVertexArraysAPPLE didn't return creates it (unlike ARB_vertex_array_object).
API void APIENTRY glBindVertexArrayAPPLE(GLuint array) {
    MGL_CTX();
    if (array && !ctx->vaos.isName(array)) ctx->vaos.reserve(array);
    glBindVertexArray(array);
}
