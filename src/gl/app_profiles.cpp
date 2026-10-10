// Per-application profiles (app_profile.h). Each entry is matched against the executable name
// (case-insensitive); the environment can force a profile and override every option. The options
// a profile sets are documented in docs/ARCHITECTURE.md, "Application profiles"; why each game
// needs its entry is in the doc the entry cites.
#include "gl/app_profile.h"

#include <crt_externs.h>
#include <mutex>
#include <set>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <strings.h>
#include "util/log.h"

namespace mgl {

namespace {

struct ProfileEntry {
    const char* name;
    const char* exes;  // space-separated executable names
    const char* vendor = nullptr;
    const char* renderer = nullptr;
    const char* version = nullptr;
    const char* glslVersion = nullptr;
    const char* hideExtensions = nullptr;   // space-separated, '*' suffix = prefix
    const char* extraExtensions = nullptr;
    const char* glslQuirks = nullptr;       // METALGLDRV_GLSL_QUIRKS syntax
    uint64_t vramCapMB = 0;                 // cap of the reported video memory
    int extensionYear = 0;                  // AppProfile::extensionYear
    bool shareFramebuffers = false;         // AppProfile::shareFramebuffers
};

// The table. Keep entries minimal: every option here is a deviation from conformant behaviour for
// that application only.
const ProfileEntry kProfiles[] = {
    // DOOM (2016), idTech 6 OpenGL renderer (docs/doom2016.md). It reads video memory through GL_ATI_meminfo
    // only when GL_VENDOR is AMD's (GL_NVX_gpu_memory_info for NVIDIA's; others get "unlimited"), and the
    // AMD path is the one whose GL calls and shaders were checked against this driver (an apitrace from an
    // AMD driver). The cap keeps its streaming budgets in the range of a discrete GPU. Its depth pre-pass is
    // followed by GL_EQUAL tests from other programs without "invariant" (radv forces invariant geometry).
    // It creates and attaches its framebuffers in one context and renders with them in another.
    {.name = "doom2016",
     .exes = "DOOMx64.exe",
     .vendor = "ATI Technologies Inc.",
     .glslQuirks = "vs_position_always_invariant",
     .vramCapMB = 8192,
     .shareFramebuffers = true},
};

std::vector<std::string> splitList(const char* s) {
    std::vector<std::string> r;
    if (!s) return r;
    std::string cur;
    for (const char* p = s;; p++) {
        if (!*p || *p == ' ' || *p == ',' || *p == ';' || *p == '\t' || *p == '\n') {
            if (!cur.empty()) r.push_back(cur);
            cur.clear();
            if (!*p) break;
        } else {
            cur += *p;
        }
    }
    return r;
}

std::string baseName(const std::string& path) {
    size_t p = path.find_last_of("/\\");
    return p == std::string::npos ? path : path.substr(p + 1);
}

bool endsWithExe(const std::string& s) { return s.size() > 4 && strcasecmp(s.c_str() + s.size() - 4, ".exe") == 0; }

// Applies "name" / "-name" / "name=value" items to the quirks; returns false for unknown names.
bool applyQuirk(glsl::Quirks& q, const std::string& item) {
    bool on = item[0] != '-';
    std::string name = on ? item : item.substr(1);
    std::string value;
    size_t eq = name.find('=');
    if (eq != std::string::npos) {
        value = name.substr(eq + 1);
        name = name.substr(0, eq);
        on = !(value == "0" || value == "false" || value == "no");
    }
    if (name == "force_glsl_extensions_warn") q.forceExtensionsWarn = on;
    else if (name == "vs_position_always_invariant") q.vsPositionAlwaysInvariant = on;
    else return false;
    return true;
}

std::string quirksSummary(const glsl::Quirks& q) {
    std::string s;
    auto add = [&](const std::string& x) { s += (s.empty() ? "" : ",") + x; };
    if (q.forceExtensionsWarn) add("force_glsl_extensions_warn");
    if (q.vsPositionAlwaysInvariant) add("vs_position_always_invariant");
    return s;
}

std::string summary(const AppProfile& p) {
    std::string s = "app profile '" + p.name + "' (exe " + (p.exe.empty() ? "?" : p.exe) + ")";
    if (!p.vendor.empty()) s += ", vendor \"" + p.vendor + "\"";
    if (!p.renderer.empty()) s += ", renderer \"" + p.renderer + "\"";
    if (!p.version.empty()) s += ", version \"" + p.version + "\"";
    if (!p.glslVersion.empty()) s += ", GLSL version \"" + p.glslVersion + "\"";
    auto list = [](const std::vector<std::string>& v) {
        std::string r;
        for (auto& x : v) r += (r.empty() ? "" : " ") + x;
        return r;
    };
    if (!p.hideExtensions.empty()) s += ", hide [" + list(p.hideExtensions) + "]";
    if (!p.extraExtensions.empty()) s += ", extra [" + list(p.extraExtensions) + "]";
    if (p.extensionYear) s += ", extension string up to " + std::to_string(p.extensionYear);
    if (p.vramMB) s += ", VRAM " + std::to_string(p.vramMB) + " MB";
    else if (p.vramCapMB) s += ", VRAM at most " + std::to_string(p.vramCapMB) + " MB";
    std::string q = quirksSummary(p.glsl);
    if (!q.empty()) s += ", GLSL quirks [" + q + "]";
    if (p.shareFramebuffers) s += ", framebuffers shared between contexts";
    if (p.logUnsupported) s += ", logging unsupported features (dumps in " + p.dumpDir + ")";
    return s;
}

}  // namespace

bool AppProfile::hidden(const char* ext) const {
    for (const auto& h : hideExtensions) {
        if (!h.empty() && h.back() == '*') {
            if (strncmp(ext, h.c_str(), h.size() - 1) == 0) return true;
        } else if (h == ext) {
            return true;
        }
    }
    return false;
}

std::string detectExecutableName() {
    int argc = *_NSGetArgc();
    char** argv = *_NSGetArgv();
    // Under Wine the loader is argv[0] and the Windows image path (C:\...\GAME.exe, or a unix path)
    // follows it; Wine may also rewrite argv[0] to the image path itself.
    for (int i = 0; i < argc && argv && argv[i]; i++) {
        std::string b = baseName(argv[i]);
        if (endsWithExe(b)) return b;
    }
    const char* n = getprogname();
    if (n && *n) return n;
    return argc > 0 && argv && argv[0] ? baseName(argv[0]) : std::string();
}

AppProfile resolveAppProfile(const std::string& exe, const char* (*env)(const char*)) {
    AppProfile p;
    p.exe = exe;
    // METALGLDRV_APP_PROFILE=<profile name | executable name | none>
    const char* forced = env("METALGLDRV_APP_PROFILE");
    std::string match = forced && *forced ? forced : exe;
    const ProfileEntry* entry = nullptr;
    if (!(forced && (!strcasecmp(forced, "none") || !strcasecmp(forced, "default")))) {
        for (const auto& e : kProfiles) {
            if (!strcasecmp(match.c_str(), e.name)) entry = &e;
            for (const auto& x : splitList(e.exes))
                if (!strcasecmp(match.c_str(), x.c_str())) entry = &e;
            if (entry) break;
        }
        if (forced && *forced && !entry) MGL_WARN("METALGLDRV_APP_PROFILE=%s: no such profile", forced);
    }
    if (entry) {
        p.name = entry->name;
        if (entry->vendor) p.vendor = entry->vendor;
        if (entry->renderer) p.renderer = entry->renderer;
        if (entry->version) p.version = entry->version;
        if (entry->glslVersion) p.glslVersion = entry->glslVersion;
        p.hideExtensions = splitList(entry->hideExtensions);
        p.extraExtensions = splitList(entry->extraExtensions);
        for (const auto& q : splitList(entry->glslQuirks)) applyQuirk(p.glsl, q);
        p.vramCapMB = entry->vramCapMB;
        p.shareFramebuffers = entry->shareFramebuffers;
        p.extensionYear = entry->extensionYear;
    }
    auto str = [&](const char* var, std::string& out) {
        const char* v = env(var);
        if (v && *v) out = v;
    };
    str("METALGLDRV_VENDOR", p.vendor);
    str("METALGLDRV_RENDERER", p.renderer);
    str("METALGLDRV_VERSION_STRING", p.version);
    str("METALGLDRV_GLSL_VERSION_STRING", p.glslVersion);
    if (const char* v = env("METALGLDRV_HIDE_EXTENSIONS")) {
        for (auto& x : splitList(v)) p.hideExtensions.push_back(x);
    }
    if (const char* v = env("METALGLDRV_EXTRA_EXTENSIONS")) {
        for (auto& x : splitList(v)) p.extraExtensions.push_back(x);
    }
    if (const char* v = env("METALGLDRV_VRAM_MB"); v && *v) p.vramMB = strtoull(v, nullptr, 10);
    if (const char* v = env("METALGLDRV_EXTENSION_YEAR"); v && *v) p.extensionYear = atoi(v);
    if (const char* v = env("METALGLDRV_SHARE_FRAMEBUFFERS"); v && *v) p.shareFramebuffers = strcmp(v, "0") != 0;
    if (const char* v = env("METALGLDRV_GLSL_QUIRKS")) {
        for (auto& q : splitList(v))
            if (!applyQuirk(p.glsl, q)) MGL_WARN("METALGLDRV_GLSL_QUIRKS: unknown quirk '%s'", q.c_str());
    }
    const char* lu = env("METALGLDRV_LOG_UNSUPPORTED");  // also read early into g_logUnsupported (util/log.mm)
    p.logUnsupported = lu && *lu && strcmp(lu, "0") && strcasecmp(lu, "false") && strcasecmp(lu, "no");
    if (const char* d = env("METALGLDRV_DUMP_DIR"); d && *d) {
        p.dumpDir = d;
    } else {
        const char* t = env("TMPDIR");
        p.dumpDir = std::string(t && *t ? t : "/tmp/");
        if (p.dumpDir.back() != '/') p.dumpDir += '/';
        p.dumpDir += "metalgldrv-dumps";
    }
    // the extra list never re-adds a hidden name
    std::vector<std::string> extra;
    for (auto& x : p.extraExtensions)
        if (!p.hidden(x.c_str())) extra.push_back(x);
    p.extraExtensions = extra;
    return p;
}

const AppProfile& appProfile() {
    static const AppProfile p = [] {
        AppProfile r = resolveAppProfile(detectExecutableName(), [](const char* n) -> const char* { return getenv(n); });
        MGL_INFO("%s", summary(r).c_str());
        return r;
    }();
    return p;
}

void logUnsupportedOnce(const std::string& what) {
    static std::mutex m;
    static std::set<std::string> seen;
    std::lock_guard<std::mutex> g(m);
    if (seen.insert(what).second) MGL_WARN("%s", what.c_str());
}

std::string dumpFailedShader(const char* kind, unsigned name, const std::vector<std::string>& sources, const std::string& log) {
    static std::mutex m;
    static int counter = 0;
    std::lock_guard<std::mutex> g(m);
    const std::string& dir = appProfile().dumpDir;
    if (mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) return "";
    char path[1024];
    snprintf(path, sizeof path, "%s/%d-%04d-%s-%u.glsl", dir.c_str(), (int)getpid(), counter++, kind, name);
    FILE* f = fopen(path, "w");
    if (!f) return "";
    fprintf(f, "// metalgldrv: %s %u failed\n", kind, name);
    std::string l = log;
    size_t at = 0;
    while (at < l.size()) {  // the info log as a comment block
        size_t e = l.find('\n', at);
        if (e == std::string::npos) e = l.size();
        fprintf(f, "// %s\n", l.substr(at, e - at).c_str());
        at = e + 1;
    }
    for (size_t i = 0; i < sources.size(); i++) {
        if (sources.size() > 1) fprintf(f, "\n// ---- source string %zu ----\n", i);
        fputs(sources[i].c_str(), f);
    }
    fclose(f);
    return path;
}

}  // namespace mgl

// Test hook (tests/host/test_app_profile.cpp): resolves a profile for `exe` with the environment given as
// "NAME=value" strings (null-terminated array) and writes its summary line to `out`.
extern "C" __attribute__((visibility("default"))) int mglTestResolveAppProfile(const char* exe, const char* const* env, char* out, int outSize) {
    thread_local const char* const* t_env;
    t_env = env;
    mgl::AppProfile p = mgl::resolveAppProfile(exe ? exe : "", [](const char* n) -> const char* {
        size_t len = strlen(n);
        for (const char* const* e = t_env; e && *e; e++)
            if (!strncmp(*e, n, len) && (*e)[len] == '=') return *e + len + 1;
        return nullptr;
    });
    std::string s = mgl::summary(p);
    snprintf(out, (size_t)outSize, "%s", s.c_str());
    return (int)s.size();
}

// Test hook: the year table (ext_years.cpp).
#include "gl/ext_years.h"
extern "C" __attribute__((visibility("default"))) int mglTestExtensionYear(const char* name) { return mgl::extensionYear(name); }
