// Per-application profiles: quirks picked by executable name (also under Wine), plus env overrides.
// The table and the matching rules are in app_profiles.cpp; docs/ARCHITECTURE.md ("Application
// profiles") lists the options. Without a matching profile or override nothing changes.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "glsl/glsl.h"

namespace mgl {

struct AppProfile {
    std::string name = "default";  // matched profile ("default" when none)
    std::string exe;               // detected executable name (the Windows .exe under Wine)
    // String overrides; empty = the driver's own strings.
    std::string vendor, renderer, version, glslVersion;
    // Extensions removed from GL_EXTENSIONS / glGetStringi and from GLSL (#extension, predefined
    // macros). Entries ending in '*' are prefixes.
    std::vector<std::string> hideExtensions;
    // Names advertised in addition to the supported ones (testing aid: nothing implements them).
    std::vector<std::string> extraExtensions;
    // GL_NVX_gpu_memory_info / GL_ATI_meminfo: the total is Metal's recommendedMaxWorkingSetSize, at most
    // vramCapMB (profiles; 0 = no cap), or exactly vramMB (METALGLDRV_VRAM_MB; 0 = unset).
    uint64_t vramMB = 0, vramCapMB = 0;
    glsl::Quirks glsl;  // compiler leniency
    // Framebuffer objects belong to the share group instead of one context (as on AMD's and NVIDIA's
    // Windows drivers): DOOM (2016) attaches its render targets in one context and draws in another.
    bool shareFramebuffers = false;
    // METALGLDRV_LOG_UNSUPPORTED: log unsupported entry points, enums, #extension requests and
    // shader failures (sources dumped to dumpDir) once each.
    bool logUnsupported = false;
    std::string dumpDir;

    bool hidden(const char* ext) const;
};

// Resolved once per process (logs a one-line summary the first time).
const AppProfile& appProfile();

// The executable name used for matching: the last path component of the first argv entry that ends in
// ".exe" (Wine's loader passes the Windows image path), else the process name.
std::string detectExecutableName();

// Builds the profile for an executable name and an environment lookup (exposed for tests).
AppProfile resolveAppProfile(const std::string& exe, const char* (*env)(const char*));

// METALGLDRV_LOG_UNSUPPORTED: logs `what` (a warning) once per unique string.
void logUnsupportedOnce(const std::string& what);

// Writes a failed shader's sources and log to appProfile().dumpDir; returns the file path ("" on failure).
std::string dumpFailedShader(const char* kind, unsigned name, const std::vector<std::string>& sources, const std::string& log);

}  // namespace mgl
