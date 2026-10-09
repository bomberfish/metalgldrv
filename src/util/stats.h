#pragma once
#include <cstdint>
// Lightweight counters for diagnosing driver behavior in the field (METALGLDRV_STATS=1).
namespace mgl {
extern bool g_statsEnabled;
// Counts an event attributed to the calling function (resolved with dladdr).
void statCaller(const char* event, void* returnAddress);
void statCount(const char* event);
// Adds `amount` to a counter attributed to the caller.
void statAddCaller(const char* event, void* returnAddress, uint64_t amount);
// Logs and resets the counters at most every 5 seconds.
void statMaybeReport();
// Re-reads METALGLDRV_STATS and METALGLDRV_TRACE_CALLS (after the environment changed, e.g. from the conf file).
void statsRefreshFromEnv();
// METALGLDRV_PROFILE_PASSES=1 (debugging; changes performance): every render pass is committed in its own
// command buffer, tagged with a label, and the stats report lists GPU time per label.
extern bool g_profilePasses;
void profileTag(uint64_t serial, const char* label);
void profileComplete(uint64_t serial, double gpuSeconds);
}  // namespace mgl
#define MGL_STAT_CALLER(ev) do { if (::mgl::g_statsEnabled) ::mgl::statCaller(ev, __builtin_return_address(0)); } while (0)
#define MGL_STAT(ev) do { if (::mgl::g_statsEnabled) ::mgl::statCount(ev); } while (0)
