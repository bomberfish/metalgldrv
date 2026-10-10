// The public C API (include/metalgldrv/metalgldrv.h).
#import <Metal/Metal.h>
#include <TargetConditionals.h>
#include "mgl_version.h"
#include "metalgldrv/metalgldrv.h"
#include "util/log.h"
#if TARGET_OS_OSX
#include "macos/hooks.h"
#endif

#define API extern "C" __attribute__((visibility("default")))

API int metalgldrv_is_supported(void) {
    static const int supported = [] {
        @autoreleasepool {
            id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
            return dev && [dev supportsFamily:MTLGPUFamilyMetal4] ? 1 : 0;
        }
    }();
    return supported;
}

API int metalgldrv_activate(void) {
#if TARGET_OS_OSX
    if (mgl::macos::replacementActive()) return 1;
    if (!metalgldrv_is_supported()) {
        MGL_WARN("metalgldrv_activate: no Metal 4 GPU; the app keeps the system's OpenGL");
        return 0;
    }
    mgl::macos::activateReplacement("metalgldrv_activate");
    return 1;
#else
    return 0;  // iOS: no system OpenGL to replace; use EGL
#endif
}

API int metalgldrv_is_active(void) {
#if TARGET_OS_OSX
    return mgl::macos::replacementActive() ? 1 : 0;
#else
    return 0;
#endif
}

API const char* metalgldrv_version(void) { return MGL_VERSION_STRING; }
