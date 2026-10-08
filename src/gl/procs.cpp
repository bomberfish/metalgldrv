#include <cstring>
#include "glapi.h"
#include "gl/procs.h"
#include "gl/app_profile.h"
#include "util/log.h"

extern "C" void (*mgl_lookup_proc(const char* name))(void) {
    unsigned lo = 0, hi = mgl_gl_procs_count;
    while (lo < hi) {
        unsigned mid = (lo + hi) / 2;
        int c = strcmp(mgl_gl_procs[mid].name, name);
        if (c == 0) return mgl_gl_procs[mid].fn;
        if (c < 0) lo = mid + 1;
        else hi = mid;
    }
    if (mgl::g_logUnsupported && name) mgl::logUnsupportedOnce(std::string("unsupported entry point ") + name + " (GetProcAddress)");
    return nullptr;
}
