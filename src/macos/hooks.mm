// Activation of the macOS system-GL replacement (see hooks.h).
//
//  - Wine mode: winemac.so (after scripts/crossover-install.sh) links libmetalgldrv
//    in place of OpenGL.framework, so its CGL imports already bind to cgl.mm. It
//    still dlopen()s /System/Library/Frameworks/OpenGL.framework/OpenGL (RTLD_NOLOAD)
//    to dlsym the gl* entry points; we rebind winemac.so's dlopen import so that
//    path returns our own handle, and install the NSOpenGL hooks (its
//    WineOpenGLContext subclasses NSOpenGLContext).
//  - Replacement mode (Agent_OnLoad for `java -agentpath:libmetalgldrv.dylib`,
//    JNI_OnLoad for System.load of the dylib, or METALGLDRV_HOOK_NSOPENGL=1): NSOpenGL
//    hooks, plus, in every non-system image (already loaded or loaded later): imports
//    bound to OpenGL.framework resolve to our exports, dlopen of OpenGL.framework
//    returns us, CFBundle function lookups in com.apple.opengl (stock GLFW's
//    glfwGetProcAddress) return ours, and SDL's CVDisplayLink-from-CGL-context call
//    gets the context's display. The Java entry points also point LWJGL 3 at us
//    (org.lwjgl.opengl.libname) unless already set.
#include "macos/hooks.h"
#import <CoreFoundation/CoreFoundation.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>
#include <unistd.h>
#include <mach-o/dyld.h>
#include <mach-o/loader.h>
#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include "gl/procs.h"
#include "macos/cgl_internal.h"
#include "macos/macho_rebind.h"
#include "mgl_version.h"
#include "util/log.h"
#include "util/stats.h"

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

// "<pid>:<path>" of the driver image that turned replacement mode on in this process.
constexpr const char* kActiveImageEnv = "METALGLDRV_ACTIVE_IMAGE";
std::string activeImageMarker() { return std::to_string(getpid()) + ":" + mgl::macos::ownPath(); }

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

// SDL 3 (Minecraft 26.3+ windows through it) drives vsync from a CVDisplayLink it
// points at the context's display with this call, which would hand our CGL objects
// to Apple's CGL. Ours: use the display of the context's window (the main display
// until it has one).
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
CVReturn hooked_CVDisplayLinkSetCurrentCGDisplayFromOpenGLContext(CVDisplayLinkRef link, CGLContextObj ctx,
                                                                   CGLPixelFormatObj pix) {
    if (!cgl::isOurContext(ctx)) return CVDisplayLinkSetCurrentCGDisplayFromOpenGLContext(link, ctx, pix);
    CGDirectDisplayID display = macos::displayOfContext(ctx);
    if (!display) display = CGMainDisplayID();
    MGL_DBG("CVDisplayLinkSetCurrentCGDisplayFromOpenGLContext(%p) -> display %u", (void*)ctx, display);
    return CVDisplayLinkSetCurrentCGDisplay(link, display);
}
#pragma clang diagnostic pop

void* wineResolver(const char* symbol, const char* library, void* current, void* user) {
    if (strcmp(symbol, "dlopen") == 0) return (void*)&hooked_dlopen;
    return nullptr;
}

void* replacementResolver(const char* symbol, const char* library, void* current, void* user) {
    if (strcmp(symbol, "dlopen") == 0) return (void*)&hooked_dlopen;
    if (strcmp(symbol, "CFBundleGetFunctionPointerForName") == 0) return (void*)&hooked_CFBundleGetFunctionPointerForName;
    if (strcmp(symbol, "CFBundleGetBundleWithIdentifier") == 0) return (void*)&hooked_CFBundleGetBundleWithIdentifier;
    if (strcmp(symbol, "CVDisplayLinkSetCurrentCGDisplayFromOpenGLContext") == 0)
        return (void*)&hooked_CVDisplayLinkSetCurrentCGDisplayFromOpenGLContext;
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
    MGL_INFO("system GL replacement on (%s): metalgldrv %s, %s", why, MGL_VERSION, ownPath());
    // Lets a second copy of the driver in this process (another path) stand down.
    setenv(kActiveImageEnv, activeImageMarker().c_str(), 1);
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
// Java entry points.
//
//  - JVMTI agent: java -agentpath:/path/to/libmetalgldrv.dylib[=nolibname]
//  - JNI_OnLoad: System.load("/path/to/libmetalgldrv.dylib") from Java code that runs
//    before LWJGL creates its window (the Fabric mod in mods/fabric does it from a
//    preLaunch entrypoint). -Dmetalgldrv.enabled=false turns it into a no-op.
//
// Both copy Java system properties metalgldrv.<NAME> into METALGLDRV_<NAME>
// environment variables (unless already set) before the driver initializes, point
// LWJGL 3 at us (org.lwjgl.opengl.libname) unless that is already set, and report
// back through system properties:
//   metalgldrv.version     driver version (git describe)
//   metalgldrv.activation  "agent" / "jni", or "skipped: <reason>"
//   metalgldrv.library     path of the dylib that activated
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
constexpr jint JNI_VERSION_1_8 = 0x00010008;
constexpr jint JVMTI_VERSION_1_2 = 0x30010200;
constexpr int JVMTI_ENABLE = 1;
constexpr int JVMTI_EVENT_VM_INIT = 50;
// Function table slots.
constexpr int kInvokeGetEnv = 6;
constexpr int kJvmtiSetEventNotificationMode = 1, kJvmtiSetEventCallbacks = 121;
constexpr int kJniFindClass = 6, kJniExceptionClear = 17, kJniPushLocalFrame = 19, kJniPopLocalFrame = 20,
              kJniDeleteLocalRef = 23, kJniGetObjectClass = 31, kJniGetMethodID = 33, kJniCallObjectMethod = 34,
              kJniGetStaticMethodID = 113, kJniCallStaticObjectMethod = 114, kJniNewStringUTF = 167,
              kJniGetStringUTFChars = 169, kJniReleaseStringUTFChars = 170, kJniGetArrayLength = 171,
              kJniGetObjectArrayElement = 173, kJniExceptionCheck = 228;
constexpr int kJvmtiEventCallbackSlots = 39;  // sizeof(jvmtiEventCallbacks) / sizeof(void*)

template <typename F>
F fn(void* env, int slot) {
    return (F)(*(void***)env)[slot];
}

// java.lang.System property access over raw JNI. Each call runs in its own local
// frame and clears any pending exception.
struct JavaSystem {
    JNIEnv* env;
    jobject sys = nullptr;
    jmethodID getProperty = nullptr, setProperty = nullptr, getProperties = nullptr;

    template <typename F>
    F f(int slot) const {
        return fn<F>(env, slot);
    }
    bool check() const {  // false (and cleared) if an exception is pending
        if (!f<unsigned char (*)(JNIEnv*)>(kJniExceptionCheck)(env)) return true;
        f<void (*)(JNIEnv*)>(kJniExceptionClear)(env);
        return false;
    }
    jobject str(const char* s) const { return f<jobject (*)(JNIEnv*, const char*)>(kJniNewStringUTF)(env, s); }
    std::string utf(jobject s) const {
        const char* c = s ? f<const char* (*)(JNIEnv*, jobject, unsigned char*)>(kJniGetStringUTFChars)(env, s, nullptr) : nullptr;
        std::string r = c ? c : "";
        if (c) f<void (*)(JNIEnv*, jobject, const char*)>(kJniReleaseStringUTFChars)(env, s, c);
        return r;
    }
    template <typename... A>
    jobject callStatic(jmethodID m, A... args) const {
        return f<jobject (*)(JNIEnv*, jobject, jmethodID, ...)>(kJniCallStaticObjectMethod)(env, sys, m, args...);
    }
    bool push() const { return f<jint (*)(JNIEnv*, jint)>(kJniPushLocalFrame)(env, 16) == JNI_OK; }
    void pop() const {
        check();
        f<jobject (*)(JNIEnv*, jobject)>(kJniPopLocalFrame)(env, nullptr);
    }

    explicit JavaSystem(JNIEnv* e) : env(e) {
        auto GetStaticMethodID = f<jmethodID (*)(JNIEnv*, jobject, const char*, const char*)>(kJniGetStaticMethodID);
        sys = f<jobject (*)(JNIEnv*, const char*)>(kJniFindClass)(env, "java/lang/System");
        if (sys) {
            getProperty = GetStaticMethodID(env, sys, "getProperty", "(Ljava/lang/String;)Ljava/lang/String;");
            setProperty = GetStaticMethodID(env, sys, "setProperty", "(Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;");
            getProperties = GetStaticMethodID(env, sys, "getProperties", "()Ljava/util/Properties;");
        }
        if (!check() || !getProperty || !setProperty) {
            MGL_ERR("Java: java.lang.System.getProperty/setProperty not found");
            getProperty = nullptr;
        }
    }
    ~JavaSystem() {
        if (sys) f<void (*)(JNIEnv*, jobject)>(kJniDeleteLocalRef)(env, sys);
    }
    bool ok() const { return getProperty != nullptr; }

    // False when the property is unset (or on error).
    bool get(const char* key, std::string* value) const {
        if (!ok() || !push()) return false;
        jobject v = callStatic(getProperty, str(key));
        bool found = check() && v;
        if (found) *value = utf(v);
        pop();
        return found;
    }
    void set(const char* key, const char* value) const {
        if (!ok() || !push()) return;
        callStatic(setProperty, str(key), str(value));
        pop();
    }

    // Names of all system properties that start with `prefix`.
    std::vector<std::string> namesWithPrefix(const char* prefix) const {
        std::vector<std::string> out;
        if (!ok() || !getProperties || !push()) return out;
        auto GetObjectClass = f<jobject (*)(JNIEnv*, jobject)>(kJniGetObjectClass);
        auto GetMethodID = f<jmethodID (*)(JNIEnv*, jobject, const char*, const char*)>(kJniGetMethodID);
        auto CallObjectMethod = f<jobject (*)(JNIEnv*, jobject, jmethodID, ...)>(kJniCallObjectMethod);
        auto GetArrayLength = f<jint (*)(JNIEnv*, jobject)>(kJniGetArrayLength);
        auto GetObjectArrayElement = f<jobject (*)(JNIEnv*, jobject, jint)>(kJniGetObjectArrayElement);
        auto DeleteLocalRef = f<void (*)(JNIEnv*, jobject)>(kJniDeleteLocalRef);
        jobject props = callStatic(getProperties);
        jmethodID names = props && check() ? GetMethodID(env, GetObjectClass(env, props), "stringPropertyNames", "()Ljava/util/Set;") : nullptr;
        jobject set = names && check() ? CallObjectMethod(env, props, names) : nullptr;
        jmethodID toArray = set && check() ? GetMethodID(env, GetObjectClass(env, set), "toArray", "()[Ljava/lang/Object;") : nullptr;
        jobject arr = toArray && check() ? CallObjectMethod(env, set, toArray) : nullptr;
        if (arr && check()) {
            size_t plen = strlen(prefix);
            for (jint i = 0, n = GetArrayLength(env, arr); i < n; i++) {
                jobject e = GetObjectArrayElement(env, arr, i);
                if (!check() || !e) break;
                std::string name = utf(e);
                DeleteLocalRef(env, e);
                if (name.compare(0, plen, prefix) == 0) out.push_back(std::move(name));
            }
        }
        pop();
        return out;
    }
};

bool isFalse(const std::string& v) {
    return v == "0" || strcasecmp(v.c_str(), "false") == 0 || strcasecmp(v.c_str(), "no") == 0 || strcasecmp(v.c_str(), "off") == 0;
}

// Settings of the Java side (and our reports), not environment variables.
bool isReservedProperty(const std::string& name) {
    for (const char* r : {"enabled", "force", "version", "activation", "library"})
        if (strcasecmp(name.c_str(), r) == 0) return true;
    return false;
}

// metalgldrv.<NAME> system properties -> METALGLDRV_<NAME> environment variables
// (NAME upper-cased, '.' and '-' -> '_') unless the variable is already set; then
// re-reads the settings latched when the dylib loaded.
void copyPropertiesToEnv(const JavaSystem& js, const char* who) {
    static const char kPrefix[] = "metalgldrv.";
    for (const std::string& prop : js.namesWithPrefix(kPrefix)) {
        std::string name = prop.substr(sizeof kPrefix - 1), var = "METALGLDRV_", value;
        if (name.empty() || isReservedProperty(name)) continue;
        bool valid = true;
        for (char c : name) {
            if (c == '.' || c == '-') c = '_';
            if (!isalnum((unsigned char)c) && c != '_') valid = false;
            var += (char)toupper((unsigned char)c);
        }
        if (!valid || !js.get(prop.c_str(), &value)) continue;
        if (getenv(var.c_str())) {
            MGL_INFO("%s: %s is already in the environment; Java property %s ignored", who, var.c_str(), prop.c_str());
            continue;
        }
        setenv(var.c_str(), value.c_str(), 0);
        MGL_INFO("%s: %s=%s (from Java property %s)", who, var.c_str(), value.c_str(), prop.c_str());
    }
    g_logLevel = envInt("METALGLDRV_LOG", LOG_INFO);
    g_logUnsupported = envFlag("METALGLDRV_LOG_UNSUPPORTED");
    g_statsEnabled = envFlag("METALGLDRV_STATS", false);
}

bool g_setLibname = true;

void setLwjglLibname(const JavaSystem& js, const char* who) {
    static const char kKey[] = "org.lwjgl.opengl.libname";
    std::string cur;
    if (!js.get(kKey, &cur)) {
        js.set(kKey, macos::ownPath());
        MGL_INFO("%s: %s=%s", who, kKey, macos::ownPath());
    } else if (cur == macos::ownPath()) {
        MGL_INFO("%s: %s=%s", who, kKey, cur.c_str());
    } else {
        MGL_WARN("%s: %s is already set to %s; leaving it (LWJGL won't use metalgldrv's GL entry points)", who, kKey, cur.c_str());
    }
}

void reportActivation(const JavaSystem& js, const std::string& activation) {
    js.set("metalgldrv.version", MGL_VERSION);
    js.set("metalgldrv.activation", activation.c_str());
    if (activation.compare(0, 7, "skipped") != 0) js.set("metalgldrv.library", macos::ownPath());
}

// The driver needs a Metal 4 GPU (Apple silicon, macOS 26+). Without one we stay out
// of the way and the app keeps Apple's OpenGL. METALGLDRV_DEBUG_NO_METAL4=1 pretends
// there is none (tests).
bool haveMetal4(std::string* why) {
    if (envFlag("METALGLDRV_DEBUG_NO_METAL4")) {
        *why = "METALGLDRV_DEBUG_NO_METAL4 is set";
        return false;
    }
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            *why = "no Metal device";
            return false;
        }
        if (![dev supportsFamily:MTLGPUFamilyMetal4]) {
            *why = std::string(dev.name.UTF8String ?: "the GPU") + " does not support Metal 4";
            return false;
        }
    }
    return true;
}

// Another copy of the driver (another path) already turned replacement mode on in
// this process.
bool otherCopyActive(std::string* other) {
    const char* v = getenv(kActiveImageEnv);
    std::string pid = std::to_string(getpid()) + ":";
    if (!v || strncmp(v, pid.c_str(), pid.size()) != 0) return false;  // none, or a parent process's
    if (strcmp(v + pid.size(), macos::ownPath()) == 0) return false;
    *other = v + pid.size();
    return true;
}

// LWJGL's libglfw.dylib or libSDL3.dylib is loaded: a window, with Apple's GL
// context, may exist. Returns its file name.
const char* windowingLibraryLoaded() {
    for (uint32_t i = 0, n = _dyld_image_count(); i < n; i++) {
        const char* path = _dyld_get_image_name(i);
        const char* base = path ? strrchr(path, '/') : nullptr;
        if (base && (strncmp(base + 1, "libglfw", 7) == 0 || strncmp(base + 1, "libSDL3", 7) == 0)) return base + 1;
    }
    return nullptr;
}

void onVMInit(jvmtiEnv* jvmti, JNIEnv* env, void* thread) {
    JavaSystem js(env);
    copyPropertiesToEnv(js, "JVMTI agent");
    reportActivation(js, "agent");
    if (g_setLibname) setLwjglLibname(js, "JVMTI agent");
}

std::atomic<bool> g_jniLoaded{false};

}  // namespace

extern "C" __attribute__((visibility("default"))) jint Agent_OnLoad(JavaVM* vm, char* options, void* reserved) {
    if (options && strstr(options, "nolibname")) g_setLibname = false;
    std::string why;
    if (!haveMetal4(&why)) {
        MGL_WARN("JVMTI agent: %s; not activating (the app keeps Apple's OpenGL)", why.c_str());
        return JNI_OK;
    }
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

extern "C" __attribute__((visibility("default"))) jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    // Once per image: a System.load of this path from another class loader must not
    // activate again. (LWJGL loading us as its GL library uses dlopen: no JNI_OnLoad.)
    if (g_jniLoaded.exchange(true)) return JNI_VERSION_1_8;
    JNIEnv* env = nullptr;
    auto GetEnv = fn<jint (*)(JavaVM*, void**, jint)>(vm, kInvokeGetEnv);
    if (GetEnv(vm, (void**)&env, JNI_VERSION_1_8) != JNI_OK || !env) return JNI_VERSION_1_8;
    static const char kWho[] = "JNI_OnLoad";
    JavaSystem js(env);
    if (!js.ok()) return JNI_VERSION_1_8;
    if (g_replacement.load()) {  // -agentpath with this same dylib (or METALGLDRV_HOOK_NSOPENGL)
        MGL_INFO("%s: system GL replacement is already on", kWho);
        if (g_setLibname) setLwjglLibname(js, kWho);
        return JNI_VERSION_1_8;
    }
    std::string value;
    if (js.get("metalgldrv.enabled", &value) && isFalse(value)) {
        reportActivation(js, "skipped: disabled by metalgldrv.enabled");
        return JNI_VERSION_1_8;
    }
    copyPropertiesToEnv(js, kWho);
    std::string why;
    if (otherCopyActive(&why)) {
        MGL_WARN("%s: metalgldrv at %s is already active in this process; not activating %s", kWho, why.c_str(), macos::ownPath());
        reportActivation(js, "skipped: another copy is active: " + why);
        return JNI_VERSION_1_8;
    }
    if (!haveMetal4(&why)) {
        MGL_WARN("%s: %s; not activating (the app keeps Apple's OpenGL)", kWho, why.c_str());
        reportActivation(js, "skipped: " + why);
        return JNI_VERSION_1_8;
    }
    if (const char* lib = windowingLibraryLoaded())
        MGL_WARN("%s: %s is already loaded; a window with Apple's OpenGL context may already exist", kWho, lib);
    macos::activateReplacement(kWho);
    reportActivation(js, "jni");
    setLwjglLibname(js, kWho);
    return JNI_VERSION_1_8;
}
