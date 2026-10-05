#include "util/log.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <os/log.h>
#include <mutex>
#include <unordered_set>
#include <string>
#include "gl/stub_report.h"

namespace mgl {
int g_logLevel = envInt("MGL_LOG", LOG_INFO);

static os_log_t mglLog() {
    static os_log_t l = os_log_create("ca.bomberfish.metalgldrv", "driver");
    return l;
}

void logf(int level, const char* fmt, ...) {
    if (level > g_logLevel) return;
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    static const char* tags[] = {"E", "W", "I", "D", "T"};
    fprintf(stderr, "[metalgldrv %s] %s\n", tags[level], buf);
    // MGL_LOG_FILE=<path> (or "documents" for $HOME/Documents/metalgldrv.log)
    static FILE* lf = [] () -> FILE* {
        const char* p = getenv("MGL_LOG_FILE");
        if (!p || !*p) return nullptr;
        std::string path = p;
        if (path == "documents") path = std::string(getenv("HOME") ? getenv("HOME") : ".") + "/Documents/metalgldrv.log";
        return fopen(path.c_str(), "w");
    }();
    if (lf) {
        fprintf(lf, "[%s] %s\n", tags[level], buf);
        fflush(lf);
    }
    os_log_type_t t = level == LOG_ERROR ? OS_LOG_TYPE_ERROR : level == LOG_WARN ? OS_LOG_TYPE_DEFAULT : OS_LOG_TYPE_INFO;
    os_log_with_type(mglLog(), t, "[%{public}s] %{public}s", tags[level], buf);
}

bool envFlag(const char* name, bool def) {
    const char* v = getenv(name);
    if (!v || !*v) return def;
    return !(strcmp(v, "0") == 0 || strcasecmp(v, "false") == 0 || strcasecmp(v, "no") == 0);
}

int envInt(const char* name, int def) {
    const char* v = getenv(name);
    return (v && *v) ? atoi(v) : def;
}
}  // namespace mgl

extern "C" void mgl_report_unimplemented(const char* name) {
    static std::mutex m;
    static std::unordered_set<std::string> seen;
    std::lock_guard<std::mutex> g(m);
    if (seen.insert(name).second) MGL_WARN("unimplemented: %s", name);
}
