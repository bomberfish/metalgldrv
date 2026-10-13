#include "util/stats.h"
#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <map>
#include <mutex>
#include <unordered_map>
#include <thread>
#include <pthread.h>
#include <string>
#include <vector>
#include "util/log.h"
#include "mtl/device.h"
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <TargetConditionals.h>
#if TARGET_OS_IOS
#include <os/proc.h>
#endif

namespace mgl {
bool g_statsEnabled = envFlag("METALGLDRV_STATS", false);
static std::mutex g_m;
static std::map<std::string, uint64_t> g_counts;
static auto g_last = std::chrono::steady_clock::now();

void statCount(const char* event) {
    std::lock_guard<std::mutex> g(g_m);
    g_counts[event]++;
}

// Symbol name of a return address (dladdr is slow: cached per address).
static const std::string& callerName(void* ra) {
    static std::unordered_map<void*, std::string> names;
    auto it = names.find(ra);
    if (it != names.end()) return it->second;
    Dl_info info{};
    return names[ra] = dladdr(ra, &info) && info.dli_sname ? info.dli_sname : "?";
}

void statCaller(const char* event, void* ra) {
    std::lock_guard<std::mutex> g(g_m);
    const std::string& who = callerName(ra);
    g_counts[std::string(event) + " <- " + who]++;
    g_counts[event]++;
}

void statAddCaller(const char* event, void* ra, uint64_t amount) {
    std::lock_guard<std::mutex> g(g_m);
    const std::string& who = callerName(ra);
    g_counts[std::string(event) + " <- " + who] += amount;
    g_counts[event] += amount;
}

// ---- METALGLDRV_PROFILE_PASSES ----
bool g_profilePasses = false;
namespace {
std::mutex g_profMutex;
std::unordered_map<uint64_t, std::string> g_profTags;          // serial -> label (pending)
std::map<std::string, std::pair<double, uint64_t>> g_profTimes;  // label -> GPU seconds, count
}  // namespace

void profileTag(uint64_t serial, const char* label) {
    std::lock_guard<std::mutex> g(g_profMutex);
    std::string& t = g_profTags[serial];
    if (t.empty()) t = label;
    else if (t.size() < 200) t += std::string(" + ") + label;
}

void profileComplete(uint64_t serial, double gpuSeconds) {
    std::lock_guard<std::mutex> g(g_profMutex);
    auto it = g_profTags.find(serial);
    std::string label = it == g_profTags.end() ? "(untagged)" : it->second;
    if (it != g_profTags.end()) g_profTags.erase(it);
    auto& e = g_profTimes[label];
    e.first += gpuSeconds;
    e.second++;
}

void profileAdd(const char* label, double gpuSeconds) {
    std::lock_guard<std::mutex> g(g_profMutex);
    auto& e = g_profTimes[label];
    e.first += gpuSeconds;
    e.second++;
}

static std::string profileReport() {
    std::vector<std::pair<std::string, std::pair<double, uint64_t>>> v;
    {
        std::lock_guard<std::mutex> g(g_profMutex);
        v.assign(g_profTimes.begin(), g_profTimes.end());
        g_profTimes.clear();
    }
    std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second.first > b.second.first; });
    std::string out = "\n  GPU time by pass (5s):";
    char buf[320];
    for (size_t i = 0; i < v.size() && i < 25; i++) {
        snprintf(buf, sizeof buf, "\n  %8.1f ms %6llu x  %s", v[i].second.first * 1000, (unsigned long long)v[i].second.second, v[i].first.c_str());
        out += buf;
    }
    return out;
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
    task_vm_info_data_t vm{};
    mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
    task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vm, &n);
    uint64_t avail = 0;
#if TARGET_OS_IOS
    avail = os_proc_available_memory();
#endif
    Device* d = Device::get();
    char mem[256];
    snprintf(mem, sizeof mem, "mem: footprint %llu MB, virtual %llu MB, available %llu MB, metal allocated %llu MB",
             (unsigned long long)(vm.phys_footprint >> 20), (unsigned long long)(vm.virtual_size >> 20),
             (unsigned long long)(avail >> 20), (unsigned long long)(d ? [d->mtl currentAllocatedSize] >> 20 : 0));
    std::string out = std::string("stats (5s): ") + mem;
    if (d) {
        double busy = d->takeGpuBusySeconds();
        char gpu[96];
        double tess[2];
        d->takeTessGpuSeconds(tess);
        snprintf(gpu, sizeof gpu, ", GPU busy %.0f ms (%.0f%%) + tessellation %.0f ms (TCS %.0f)", busy * 1000, busy / 5 * 100,
                 (tess[0] + tess[1]) * 1000, tess[0] * 1000);
        out += gpu;
        // command buffers: everything up to the watermark is complete (and its resources reusable)
        uint64_t c[4];
        d->completionState(c);
        snprintf(gpu, sizeof gpu, ", serials: complete <= %llu (+%llu above), oldest pending %llu, next %llu", (unsigned long long)c[0],
                 (unsigned long long)c[1], (unsigned long long)c[2], (unsigned long long)c[3]);
        out += gpu;
    }
    if (g_profilePasses) out += profileReport();
    for (size_t i = 0; i < v.size() && i < 25; i++) out += "\n  " + std::to_string(v[i].second) + "  " + v[i].first;
    MGL_INFO("%s", out.c_str());
}

// ---- METALGLDRV_TRACE_CALLS ----
// 1: the 30 most called entry points per report; N > 1: the N most called.
bool g_traceCalls = envFlag("METALGLDRV_TRACE_CALLS", false);
namespace {
size_t g_traceTop = 30;
int g_traceSeq = 0;  // METALGLDRV_TRACE_SEQ=N: also log the first N calls of each thread, in order
std::mutex g_callMutex;
std::unordered_map<const char*, uint64_t> g_calls;  // keyed by __func__ (one pointer per entry point)
void callReporter() {
    for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        std::vector<std::pair<const char*, uint64_t>> v;
        {
            std::lock_guard<std::mutex> g(g_callMutex);
            v.assign(g_calls.begin(), g_calls.end());
            g_calls.clear();
        }
        std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second > b.second; });
        uint64_t total = 0;
        for (auto& e : v) total += e.second;
        std::string out = "GL calls (5s): " + std::to_string(total) + " total";
        for (size_t i = 0; i < v.size() && i < g_traceTop; i++) out += "\n  " + std::to_string(v[i].second) + "  " + v[i].first;
        MGL_INFO("%s", out.c_str());
    }
}
}  // namespace

// ---- METALGLDRV_HITCH_MS ----
int g_hitchMs = 0;
namespace {
bool g_countCalls = false;  // METALGLDRV_TRACE_CALLS / TRACE_SEQ (g_traceCalls is also set for hitch timing)
struct CallTime {
    uint64_t ticks = 0, count = 0, maxTicks = 0;
};
std::mutex g_hitchMutex;
std::map<std::pair<uint64_t, const char*>, CallTime> g_hitchCalls;  // (thread, entry point)
std::map<std::string, std::pair<double, int>> g_hitchCompiles;      // what -> ms, count
double ticksToMs(uint64_t t) {
    static mach_timebase_info_data_t tb = [] {
        mach_timebase_info_data_t i;
        mach_timebase_info(&i);
        return i;
    }();
    return (double)t * tb.numer / tb.denom / 1e6;
}
}  // namespace

void traceCallEnd(const char* name, uint64_t start) {
    uint64_t dt = mach_absolute_time() - start;
    uint64_t tid = pthread_mach_thread_np(pthread_self());
    std::lock_guard<std::mutex> g(g_hitchMutex);
    CallTime& c = g_hitchCalls[{tid, name}];
    c.ticks += dt;
    c.count++;
    c.maxTicks = std::max(c.maxTicks, dt);
}

void hitchCompile(const char* what, double ms) {
    if (g_hitchMs <= 0) return;
    std::lock_guard<std::mutex> g(g_hitchMutex);
    auto& e = g_hitchCompiles[what];
    e.first += ms;
    e.second++;
}

void hitchFrameEnd() {
    if (g_hitchMs <= 0) return;
    static uint64_t last = 0;
    uint64_t now = mach_absolute_time();
    std::map<std::pair<uint64_t, const char*>, CallTime> calls;
    std::map<std::string, std::pair<double, int>> compiles;
    {
        std::lock_guard<std::mutex> g(g_hitchMutex);
        calls.swap(g_hitchCalls);
        compiles.swap(g_hitchCompiles);
    }
    double frame = last ? ticksToMs(now - last) : 0;
    last = now;
    if (frame <= g_hitchMs) return;
    std::map<uint64_t, double> perThread;
    std::vector<std::pair<double, std::string>> top;
    for (auto& [k, c] : calls) {
        perThread[k.first] += ticksToMs(c.ticks);
        char b[160];
        snprintf(b, sizeof b, "thread %llu %s: %.1f ms in %llu call%s (longest %.1f ms)", (unsigned long long)k.first, k.second,
                 ticksToMs(c.ticks), (unsigned long long)c.count, c.count == 1 ? "" : "s", ticksToMs(c.maxTicks));
        top.push_back({ticksToMs(c.ticks), b});
    }
    std::sort(top.begin(), top.end(), [](auto& a, auto& b) { return a.first > b.first; });
    char head[96];
    snprintf(head, sizeof head, "hitch: %.0f ms frame; time in GL calls by thread:", frame);
    std::string out = head;
    for (auto& [t, ms] : perThread) out += " " + std::to_string(t) + "=" + std::to_string((int)ms) + "ms";
    for (auto& [what, e] : compiles) {
        char b[128];
        snprintf(b, sizeof b, "\n  driver work: %s %.1f ms (%d)", what.c_str(), e.first, e.second);
        out += b;
    }
    for (size_t i = 0; i < top.size() && i < 12; i++) out += "\n  " + top[i].second;
    MGL_INFO("%s", out.c_str());
}

void statsRefreshFromEnv() {
    envChanged();
    g_statsEnabled = envFlag("METALGLDRV_STATS", false);
    g_traceCalls = envFlag("METALGLDRV_TRACE_CALLS", false);
    g_countCalls = g_traceCalls;
    g_hitchMs = envInt("METALGLDRV_HITCH_MS", 0);
    int top = envInt("METALGLDRV_TRACE_CALLS", 0);
    g_traceTop = top > 1 ? (size_t)top : 30;
    g_traceSeq = envInt("METALGLDRV_TRACE_SEQ", 0);
    g_profilePasses = envFlag("METALGLDRV_PROFILE_PASSES", false);
    if (g_traceSeq > 0) g_traceCalls = g_countCalls = true;
    if (g_hitchMs > 0) g_traceCalls = true;
}

uint64_t traceCall(const char* name) {
    uint64_t start = g_hitchMs > 0 ? mach_absolute_time() : 0;
    if (!g_countCalls) return start;
    static std::once_flag once;
    std::call_once(once, [] { std::thread(callReporter).detach(); });
    if (g_traceSeq > 0) {
        static thread_local int seq = 0;
        if (seq < g_traceSeq) MGL_INFO("call %d (thread %llu): %s", seq, (unsigned long long)pthread_mach_thread_np(pthread_self()), name);
        seq++;
    }
    std::lock_guard<std::mutex> g(g_callMutex);
    g_calls[name]++;
    return start;
}
}  // namespace mgl
