#include "util/stats.h"
#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include "util/log.h"
#include "mtl/device.h"
#include <mach/mach.h>
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

void statCaller(const char* event, void* ra) {
    Dl_info info{};
    std::string who = "?";
    if (dladdr(ra, &info) && info.dli_sname) who = info.dli_sname;
    std::lock_guard<std::mutex> g(g_m);
    g_counts[std::string(event) + " <- " + who]++;
    g_counts[event]++;
}

void statAddCaller(const char* event, void* ra, uint64_t amount) {
    Dl_info info{};
    std::string who = "?";
    if (dladdr(ra, &info) && info.dli_sname) who = info.dli_sname;
    std::lock_guard<std::mutex> g(g_m);
    g_counts[std::string(event) + " <- " + who] += amount;
    g_counts[event] += amount;
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
    for (size_t i = 0; i < v.size() && i < 25; i++) out += "\n  " + std::to_string(v[i].second) + "  " + v[i].first;
    MGL_INFO("%s", out.c_str());
}
}  // namespace mgl
