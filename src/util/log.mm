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
int g_logLevel = envInt("METALGLDRV_LOG", LOG_INFO);
bool g_logUnsupported = envFlag("METALGLDRV_LOG_UNSUPPORTED");

static os_log_t mglLog() {
    static os_log_t l = os_log_create("ca.bomberfish.metalgldrv", "driver");
    return l;
}

void logf(int level, const char* fmt, ...) {
    if (level > g_logLevel) return;
    char small[2048];
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int n = vsnprintf(small, sizeof small, fmt, ap);
    va_end(ap);
    std::string big;
    const char* buf = small;
    if (n >= (int)sizeof small) {  // long messages (shader dumps) are not truncated
        big.resize((size_t)n + 1);
        vsnprintf(big.data(), big.size(), fmt, ap2);
        buf = big.c_str();
    }
    va_end(ap2);
    static const char* tags[] = {"E", "W", "I", "D", "T"};
    fprintf(stderr, "[metalgldrv %s] %s\n", tags[level], buf);
    // METALGLDRV_LOG_FILE=<path> (or "documents" for $HOME/Documents/metalgldrv.log)
    // Opened on the first message after the variable appears: the Fabric mod / JNI_OnLoad set it
    // from Java properties after the driver has already logged.
    static std::mutex lfMutex;
    std::lock_guard<std::mutex> lfLock(lfMutex);
    static FILE* lf = nullptr;
    static std::string lfPath;
    if (const char* p = getenv("METALGLDRV_LOG_FILE"); p && *p && lfPath != p) {
        lfPath = p;
        std::string path = p;
        if (path == "documents") path = std::string(getenv("HOME") ? getenv("HOME") : ".") + "/Documents/metalgldrv.log";
        if (lf) fclose(lf);
        lf = fopen(path.c_str(), "w");
    }
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
    if (seen.insert(name).second) MGL_WARN("unimplemented: %s%s", name, mgl::g_logUnsupported ? " (unsupported entry point called)" : "");
}
