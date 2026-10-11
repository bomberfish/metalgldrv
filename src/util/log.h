#pragma once
#include <cstdarg>
#include <cstdint>

namespace mgl {
enum LogLevel { LOG_ERROR = 0, LOG_WARN = 1, LOG_INFO = 2, LOG_DEBUG = 3, LOG_TRACE = 4 };
extern int g_logLevel;
// METALGLDRV_LOG_UNSUPPORTED (also AppProfile::logUnsupported): log unsupported entry points, enums,
// #extension requests and GL errors once each.
extern bool g_logUnsupported;
void logf(int level, const char* fmt, ...) __attribute__((format(printf, 2, 3)));
// Environment helpers (MGL_* variables)
bool envFlag(const char* name, bool def = false);
int envInt(const char* name, int def);
// envFlag for switches read on every draw: cached per thread until envChanged() (getenv scans the environment)
bool envFlagCached(const char* name, bool def);
void envChanged();
}  // namespace mgl

#define MGL_ERR(...) ::mgl::logf(::mgl::LOG_ERROR, __VA_ARGS__)
#define MGL_WARN(...) ::mgl::logf(::mgl::LOG_WARN, __VA_ARGS__)
#define MGL_INFO(...) ::mgl::logf(::mgl::LOG_INFO, __VA_ARGS__)
#define MGL_DBG(...) do { if (::mgl::g_logLevel >= ::mgl::LOG_DEBUG) ::mgl::logf(::mgl::LOG_DEBUG, __VA_ARGS__); } while (0)
#define MGL_TRACE(...) do { if (::mgl::g_logLevel >= ::mgl::LOG_TRACE) ::mgl::logf(::mgl::LOG_TRACE, __VA_ARGS__); } while (0)
