// Activation of the macOS system-GL replacement (see hooks.h).
//
//  - Wine mode: winemac.so (after scripts/crossover-install.sh) links libmetalgldrv
//    in place of OpenGL.framework, so its CGL imports already bind to cgl.mm. It
//    still dlopen()s /System/Library/Frameworks/OpenGL.framework/OpenGL (RTLD_NOLOAD)
//    to dlsym the gl* entry points; we rebind winemac.so's dlopen import so that
//    path returns our own handle, and install the NSOpenGL hooks (its
//    WineOpenGLContext subclasses NSOpenGLContext).
//  - Replacement mode (Agent_OnLoad for `java -agentpath:libmetalgldrv.dylib`, or
//    METALGLDRV_HOOK_NSOPENGL=1): NSOpenGL hooks, plus, in every non-system image
//    (already loaded or loaded later): imports bound to OpenGL.framework resolve to
//    our exports, dlopen of OpenGL.framework returns us, and CFBundle function
//    lookups in com.apple.opengl (stock GLFW's glfwGetProcAddress) return ours. The
//    agent also points LWJGL 3 at us (org.lwjgl.opengl.libname) unless already set.
#include "macos/hooks.h"
#import <CoreFoundation/CoreFoundation.h>
#import <Foundation/Foundation.h>
#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <mach-o/loader.h>
#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include "gl/procs.h"
#include "macos/cgl_internal.h"
#include "macos/macho_rebind.h"
#include "util/log.h"

using namespace mgl;

namespace {

std::atomic<bool> g_replacement{false};
// Serializes rebinding. Heap-allocated on first use: the constructor below can run
// before this file's static initializers. Held only around rebindImage (which makes
// no dyld calls; the resolvers use our own tables), never around dyld APIs: the
// image callback runs under dyld's lock.
std::recursive_mutex& rebindMutex() {
    static auto* m = new std::recursive_mutex();
    return *m;
}

bool hasSuffix(const char* s, const char* suffix) {
    size_t n = strlen(s), m = strlen(suffix);
    return n >= m && strcmp(s + n - m, suffix) == 0;
}

// dlopen replacement for rebound images: OpenGL.framework paths open us instead.
// Other paths tail-call the real dlopen so it still sees the caller's image (for
// @loader_path / @rpath resolution).
void* hooked_dlopen(const char* path, int mode) {
    if (macho::isOpenGLFrameworkPath(path)) {
        if (void* h = dlopen(macos::ownPath(), RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD)) {
            static std::once_flag once;
            std::call_once(once, [path] { MGL_INFO("dlopen(\"%s\") -> %s", path, macos::ownPath()); });
            return h;
        }
    }
    [[clang::musttail]] return dlopen(path, mode);
}

bool isOpenGLBundleID(CFStringRef ident) {
    return ident && CFStringCompare(ident, CFSTR("com.apple.opengl"), kCFCompareCaseInsensitive) == kCFCompareEqualTo;
}

void* hooked_CFBundleGetFunctionPointerForName(CFBundleRef bundle, CFStringRef name) {
    if (bundle && name && isOpenGLBundleID(CFBundleGetIdentifier(bundle))) {
        char buf[256];
        if (!CFStringGetCString(name, buf, sizeof buf, kCFStringEncodingUTF8)) return nullptr;
        void* p = dlsym(macos::ownHandle(), buf);
        if (!p) MGL_DBG("CFBundleGetFunctionPointerForName(com.apple.opengl, %s): not provided", buf);
        return p;  // never Apple's: it would run on Apple's (absent) current context
    }
    return CFBundleGetFunctionPointerForName(bundle, name);
}

CFBundleRef hooked_CFBundleGetBundleWithIdentifier(CFStringRef ident) {
    CFBundleRef b = CFBundleGetBundleWithIdentifier(ident);
    if (!b && isOpenGLBundleID(ident)) {
        // OpenGL.framework isn't loaded (nothing links it any more): hand out a bundle
        // object for it without loading its code; function lookups come to us above.
        static CFBundleRef framework = [] {
            CFURLRef url = CFURLCreateWithFileSystemPath(nullptr, CFSTR("/System/Library/Frameworks/OpenGL.framework"),
                                                         kCFURLPOSIXPathStyle, true);
            CFBundleRef fb = url ? CFBundleCreate(nullptr, url) : nullptr;
            if (url) CFRelease(url);
            return fb;
        }();
        b = framework;
    }
    return b;
}

void* wineResolver(const char* symbol, const char* library, void* current, void* user) {
    if (strcmp(symbol, "dlopen") == 0) return (void*)&hooked_dlopen;
    return nullptr;
}

void* replacementResolver(const char* symbol, const char* library, void* current, void* user) {
    if (strcmp(symbol, "dlopen") == 0) return (void*)&hooked_dlopen;
    if (strcmp(symbol, "CFBundleGetFunctionPointerForName") == 0) return (void*)&hooked_CFBundleGetFunctionPointerForName;
    if (strcmp(symbol, "CFBundleGetBundleWithIdentifier") == 0) return (void*)&hooked_CFBundleGetBundleWithIdentifier;
    if (library && strstr(library, "/OpenGL.framework/")) {
        void* p = symbol[0] == 'g' && symbol[1] == 'l' ? (void*)mgl_lookup_proc(symbol) : nullptr;
        if (!p) p = cgl::exportedFunction(symbol);
        if (!p) MGL_DBG("%s: import %s from OpenGL.framework is not provided; left to Apple's", (const char*)user, symbol);
        return p;
    }
    return nullptr;
}

void onWinemac(const mach_header* mh, intptr_t slide, const char* path) {
    static std::once_flag once;
    std::call_once(once, [&] {
        int n;
        {
            std::lock_guard<std::recursive_mutex> g(rebindMutex());
            n = macho::rebindImage(mh, slide, wineResolver, nullptr);
        }
        MGL_INFO("CrossOver/Wine: %s: dlopen rebound (%d slot%s); OpenGL.framework -> %s", path, n, n == 1 ? "" : "s",
                 macos::ownPath());
        macos::installNSOpenGLHooks();
    });
}

void rebindForReplacement(const mach_header* mh, intptr_t slide, const char* path) {
    if (macho::isSystemImagePath(path)) return;
    if ((mh->flags & MH_DYLIB_IN_CACHE) || mh->magic != MH_MAGIC_64) return;
    if (strcmp(path, macos::ownPath()) == 0) return;
    int n;
    {
        std::lock_guard<std::recursive_mutex> g(rebindMutex());
        n = macho::rebindImage(mh, slide, replacementResolver, (void*)path);
    }
    if (n) MGL_INFO("rebound %d GL/loader import%s in %s", n, n == 1 ? "" : "s", path);
}

void onImageAdded(const mach_header* mh, intptr_t slide) {
    Dl_info info;
    if (!dladdr(mh, &info) || !info.dli_fname) return;
    const char* path = info.dli_fname;
    if (hasSuffix(path, "/winemac.so")) onWinemac(mh, slide, path);
    else if (g_replacement.load()) rebindForReplacement(mh, slide, path);
}

__attribute__((constructor)) void macosInit() {
    // May run before log.mm's initializer; same value it computes.
    g_logLevel = envInt("METALGLDRV_LOG", LOG_INFO);
    macos::ownPath();  // resolve (dladdr) now, outside any dyld callback
    if (envFlag("METALGLDRV_HOOK_NSOPENGL")) macos::activateReplacement("METALGLDRV_HOOK_NSOPENGL");
    // Called for every image already loaded (winemac.so when it is what loaded us)
    // and for every image loaded later.
    _dyld_register_func_for_add_image(onImageAdded);
}

}  // namespace

namespace mgl::macos {

const char* ownPath() {
    static std::string path = [] {
        Dl_info info;
        return dladdr((const void*)&ownPath, &info) && info.dli_fname ? std::string(info.dli_fname) : std::string();
    }();
    return path.c_str();
}

void* ownHandle() {
    static void* h = dlopen(ownPath(), RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD);
    return h;
}

void activateReplacement(const char* why) {
    if (g_replacement.exchange(true)) return;
    MGL_INFO("system GL replacement on (%s): %s", why, ownPath());
    installNSOpenGLHooks();
    // Images loaded from now on go through onImageAdded; do the ones already here.
    for (uint32_t i = 0, n = _dyld_image_count(); i < n; i++) {
        const mach_header* mh = _dyld_get_image_header(i);
        const char* path = _dyld_get_image_name(i);
        if (mh && path && !hasSuffix(path, "/winemac.so")) rebindForReplacement(mh, _dyld_get_image_vmaddr_slide(i), path);
    }
}

}  // namespace mgl::macos

// ---------------------------------------------------------------------------
// JVMTI agent: java -agentpath:/path/to/libmetalgldrv.dylib[=nolibname]
//
// Minimal JNI / JVMTI declarations (function tables are indexed per jni.h / jvmti.h).

namespace {

typedef int32_t jint;
typedef void* jobject;
typedef void* jmethodID;
typedef void JavaVM;   // const struct JNIInvokeInterface_**
typedef void JNIEnv;   // const struct JNINativeInterface_**
typedef void jvmtiEnv; // const struct jvmtiInterface_1_**

constexpr jint JNI_OK = 0;
constexpr jint JVMTI_VERSION_1_2 = 0x30010200;
constexpr int JVMTI_ENABLE = 1;
constexpr int JVMTI_EVENT_VM_INIT = 50;
// Function table slots.
constexpr int kInvokeGetEnv = 6;
constexpr int kJvmtiSetEventNotificationMode = 1, kJvmtiSetEventCallbacks = 121;
constexpr int kJniFindClass = 6, kJniExceptionClear = 17, kJniDeleteLocalRef = 23, kJniGetStaticMethodID = 113,
              kJniCallStaticObjectMethod = 114, kJniNewStringUTF = 167, kJniGetStringUTFChars = 169,
              kJniReleaseStringUTFChars = 170, kJniExceptionCheck = 228;
constexpr int kJvmtiEventCallbackSlots = 39;  // sizeof(jvmtiEventCallbacks) / sizeof(void*)

template <typename F>
F fn(void* env, int slot) {
    return (F)(*(void***)env)[slot];
}

bool g_setLibname = true;

void setLwjglLibname(JNIEnv* env) {
    auto FindClass = fn<jobject (*)(JNIEnv*, const char*)>(env, kJniFindClass);
    auto GetStaticMethodID = fn<jmethodID (*)(JNIEnv*, jobject, const char*, const char*)>(env, kJniGetStaticMethodID);
    auto CallStaticObjectMethod = fn<jobject (*)(JNIEnv*, jobject, jmethodID, ...)>(env, kJniCallStaticObjectMethod);
    auto NewStringUTF = fn<jobject (*)(JNIEnv*, const char*)>(env, kJniNewStringUTF);
    auto GetStringUTFChars = fn<const char* (*)(JNIEnv*, jobject, unsigned char*)>(env, kJniGetStringUTFChars);
    auto ReleaseStringUTFChars = fn<void (*)(JNIEnv*, jobject, const char*)>(env, kJniReleaseStringUTFChars);
    auto DeleteLocalRef = fn<void (*)(JNIEnv*, jobject)>(env, kJniDeleteLocalRef);
    auto ExceptionCheck = fn<unsigned char (*)(JNIEnv*)>(env, kJniExceptionCheck);
    auto ExceptionClear = fn<void (*)(JNIEnv*)>(env, kJniExceptionClear);

    static const char kKey[] = "org.lwjgl.opengl.libname";
    jobject sys = FindClass(env, "java/lang/System");
    jmethodID getProperty = sys ? GetStaticMethodID(env, sys, "getProperty", "(Ljava/lang/String;)Ljava/lang/String;") : nullptr;
    jmethodID setProperty =
        sys ? GetStaticMethodID(env, sys, "setProperty", "(Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;") : nullptr;
    if (!getProperty || !setProperty) {
        if (ExceptionCheck(env)) ExceptionClear(env);
        MGL_ERR("JVMTI agent: java.lang.System.getProperty/setProperty not found");
        return;
    }
    jobject key = NewStringUTF(env, kKey);
    jobject cur = CallStaticObjectMethod(env, sys, getProperty, key);
    if (cur) {
        const char* s = GetStringUTFChars(env, cur, nullptr);
        MGL_INFO("JVMTI agent: %s already set (%s); leaving it", kKey, s ? s : "?");
        if (s) ReleaseStringUTFChars(env, cur, s);
        DeleteLocalRef(env, cur);
    } else {
        jobject value = NewStringUTF(env, macos::ownPath());
        jobject prev = CallStaticObjectMethod(env, sys, setProperty, key, value);
        if (prev) DeleteLocalRef(env, prev);
        DeleteLocalRef(env, value);
        MGL_INFO("JVMTI agent: %s=%s", kKey, macos::ownPath());
    }
    if (ExceptionCheck(env)) ExceptionClear(env);
    DeleteLocalRef(env, key);
    DeleteLocalRef(env, sys);
}

void onVMInit(jvmtiEnv* jvmti, JNIEnv* env, void* thread) {
    if (g_setLibname) setLwjglLibname(env);
}

}  // namespace

extern "C" __attribute__((visibility("default"))) jint Agent_OnLoad(JavaVM* vm, char* options, void* reserved) {
    if (options && strstr(options, "nolibname")) g_setLibname = false;
    macos::activateReplacement("JVMTI agent");
    jvmtiEnv* jvmti = nullptr;
    auto GetEnv = fn<jint (*)(JavaVM*, void**, jint)>(vm, kInvokeGetEnv);
    if (GetEnv(vm, (void**)&jvmti, JVMTI_VERSION_1_2) != JNI_OK || !jvmti) {
        MGL_ERR("JVMTI agent: no JVMTI environment; set -Dorg.lwjgl.opengl.libname=%s yourself", macos::ownPath());
        return JNI_OK;
    }
    void* callbacks[kJvmtiEventCallbackSlots] = {};
    callbacks[0] = (void*)&onVMInit;  // jvmtiEventCallbacks.VMInit
    auto SetEventCallbacks = fn<jint (*)(jvmtiEnv*, const void*, jint)>(jvmti, kJvmtiSetEventCallbacks);
    auto SetEventNotificationMode = fn<jint (*)(jvmtiEnv*, int, int, void*, ...)>(jvmti, kJvmtiSetEventNotificationMode);
    jint e1 = SetEventCallbacks(jvmti, callbacks, (jint)sizeof callbacks);
    jint e2 = SetEventNotificationMode(jvmti, JVMTI_ENABLE, JVMTI_EVENT_VM_INIT, nullptr);
    if (e1 || e2) MGL_ERR("JVMTI agent: VMInit hook failed (%d/%d); set -Dorg.lwjgl.opengl.libname yourself", e1, e2);
    return JNI_OK;
}
