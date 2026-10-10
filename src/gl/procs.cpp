#include <cstring>
#include "glapi.h"
#include "gl/procs.h"
#include "gl/app_profile.h"
#include "util/log.h"

// Entry points that the Khronos registry doesn't list (Apple's own), sorted by name.
extern "C" void APIENTRY glFinishRenderAPPLE(void);
extern "C" void APIENTRY glFlushRenderAPPLE(void);
extern "C" void APIENTRY glSwapAPPLE(void);
extern "C" void APIENTRY glPointSizePointerAPPLE(GLenum type, GLsizei stride, const void* pointer);
extern "C" void APIENTRY glVertexPointSizefAPPLE(GLfloat size);
static const mgl_proc_entry kExtraProcs[] = {
    {"glFinishRenderAPPLE", (void (*)(void))&glFinishRenderAPPLE},
    {"glFlushRenderAPPLE", (void (*)(void))&glFlushRenderAPPLE},
    {"glPointSizePointerAPPLE", (void (*)(void))&glPointSizePointerAPPLE},
    {"glSwapAPPLE", (void (*)(void))&glSwapAPPLE},
    {"glVertexPointSizefAPPLE", (void (*)(void))&glVertexPointSizefAPPLE},
};

extern "C" void (*mgl_lookup_proc(const char* name))(void) {
    unsigned lo = 0, hi = mgl_gl_procs_count;
    while (lo < hi) {
        unsigned mid = (lo + hi) / 2;
        int c = strcmp(mgl_gl_procs[mid].name, name);
        if (c == 0) return mgl_gl_procs[mid].fn;
        if (c < 0) lo = mid + 1;
        else hi = mid;
    }
    if (name)
        for (const auto& e : kExtraProcs)
            if (!strcmp(e.name, name)) return e.fn;
    if (mgl::g_logUnsupported && name) mgl::logUnsupportedOnce(std::string("unsupported entry point ") + name + " (GetProcAddress)");
    return nullptr;
}
