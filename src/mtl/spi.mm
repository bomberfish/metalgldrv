#import "mtl/spi.h"
#import <objc/message.h>
#include <cstdlib>
#include <cstring>
#include <strings.h>
#include "util/log.h"

#ifndef METALGLDRV_PRIVATE_API
#define METALGLDRV_PRIVATE_API 1
#endif

namespace mgl::spi {

bool enabled() {
#if METALGLDRV_PRIVATE_API
    static const bool on = [] {
        const char* v = getenv("METALGLDRV_MORALLY_WRONG");
        bool off = v && (!strcmp(v, "0") || !strcasecmp(v, "false") || !strcasecmp(v, "no"));
        MGL_INFO("private Metal API %s", off ? "disabled (METALGLDRV_MORALLY_WRONG=0)" : "enabled");
        return !off;
    }();
    return on;
#else
    return false;
#endif
}

#if METALGLDRV_PRIVATE_API
// Sends `sel` with one argument of type T if the receiver implements it.
template <class T>
static bool send(id obj, const char* sel, T arg) {
    SEL s = sel_registerName(sel);
    if (!obj || ![obj respondsToSelector:s]) return false;
    ((void (*)(id, SEL, T))objc_msgSend)(obj, s, arg);
    return true;
}
#endif

static int g_lineWidth = 0;  // 1 supported, -1 not

bool lineWidthSupported() { return g_lineWidth > 0; }

void probeLineWidth(id encoder) {
#if METALGLDRV_PRIVATE_API
    if (enabled()) g_lineWidth = [encoder respondsToSelector:sel_registerName("setLineWidth:")] ? 1 : -1;
#endif
}

void setLineWidth(id encoder, float width) {
#if METALGLDRV_PRIVATE_API
    if (g_lineWidth > 0) ((void (*)(id, SEL, float))objc_msgSend)(encoder, sel_registerName("setLineWidth:"), width);
#endif
}

float maxLineWidth() { return lineWidthSupported() ? 16.0f : 1.0f; }

bool cubeSeamsSupported() {
#if METALGLDRV_PRIVATE_API
    static const bool ok = [] {
        if (!enabled() || ![MTLSamplerDescriptor instancesRespondToSelector:sel_registerName("setForceSeamsOnCubemapFiltering:")]) return false;
        id<MTLDevice> d = MTLCreateSystemDefaultDevice();
        SEL cap = sel_registerName("supportsForceSeamsOnCubemaps");
        return [d respondsToSelector:cap] && ((BOOL (*)(id, SEL))objc_msgSend)(d, cap);
    }();
    return ok;
#else
    return false;
#endif
}

void forceCubeSeams(MTLSamplerDescriptor* d) {
#if METALGLDRV_PRIVATE_API
    if (cubeSeamsSupported()) send<BOOL>(d, "setForceSeamsOnCubemapFiltering:", YES);
#endif
}

void markOpenGLQueue(MTLCommandQueueDescriptor* d) {
#if METALGLDRV_PRIVATE_API
    if (enabled()) send<BOOL>(d, "setIsOpenGLQueue:", YES);
#endif
}

}  // namespace mgl::spi
