#include "util/stats.h"
#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include "util/log.h"

namespace mgl {
bool g_statsEnabled = envFlag("MGL_STATS", false);
static std::mutex g_m;
static std::map<std::string, uint64_t> g_counts;
static auto g_last = std::chrono::steady_clock::now();

void statCount(const char* event) {
    std::lock_guard<std::mutex> g(g_m);
    g_counts[event]++;
}

void statCaller(const char* event, void* ra) {
    Dl_info info{};
    std::string who = "?";
    if (dladdr(ra, &info) && info.dli_sname) who = info.dli_sname;
    std::lock_guard<std::mutex> g(g_m);
    g_counts[std::string(event) + " <- " + who]++;
    g_counts[event]++;
}

void statMaybeReport() {
    if (!g_statsEnabled) return;
    std::vector<std::pair<std::string, uint64_t>> v;
    {
        std::lock_guard<std::mutex> g(g_m);
        auto now = std::chrono::steady_clock::now();
        if (now - g_last < std::chrono::seconds(5)) return;
        g_last = now;
        v.assign(g_counts.begin(), g_counts.end());
        g_counts.clear();
    }
    std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second > b.second; });
    std::string out = "stats (5s):";
    for (size_t i = 0; i < v.size() && i < 25; i++) out += "\n  " + std::to_string(v[i].second) + "  " + v[i].first;
    MGL_INFO("%s", out.c_str());
}
}  // namespace mgl
