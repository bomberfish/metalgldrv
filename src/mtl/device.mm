#import "mtl/device.h"
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#import <UIKit/UIKit.h>
#endif
#import <QuartzCore/CAMetalLayer.h>
#include "util/log.h"
#include "util/stats.h"

namespace mgl {

Device* Device::get() {
    static Device* dev = [] {
        Device* d = new Device();
        if (!d->init()) {
            delete d;
            return (Device*)nullptr;
        }
        return d;
    }();
    return dev;
}

bool Device::init() {
    @autoreleasepool {
        mtl = MTLCreateSystemDefaultDevice();
        if (!mtl) {
            MGL_ERR("no Metal device");
            return false;
        }
        if (![mtl supportsFamily:MTLGPUFamilyMetal4]) {
            MGL_ERR("device %s does not support Metal 4", mtl.name.UTF8String);
            return false;
        }
        MTL4CommandQueueDescriptor* qd = [MTL4CommandQueueDescriptor new];
        qd.label = @"metalgldrv";
        NSError* err = nil;
        queue = [mtl newMTL4CommandQueueWithDescriptor:qd error:&err];
        if (!queue) {
            MGL_ERR("newMTL4CommandQueue failed: %s", err.localizedDescription.UTF8String);
            return false;
        }
        MTL4CompilerDescriptor* cd = [MTL4CompilerDescriptor new];
        cd.label = @"metalgldrv";
        compiler = [mtl newCompilerWithDescriptor:cd error:&err];
        if (!compiler) {
            MGL_ERR("newCompiler failed: %s", err.localizedDescription.UTF8String);
            return false;
        }
        MTLResidencySetDescriptor* rd = [MTLResidencySetDescriptor new];
        rd.label = @"metalgldrv";
        rd.initialCapacity = 4096;
        residency = [mtl newResidencySetWithDescriptor:rd error:&err];
        if (!residency) {
            MGL_ERR("newResidencySet failed: %s", err.localizedDescription.UTF8String);
            return false;
        }
        [queue addResidencySet:residency];

#if TARGET_OS_IPHONE
        // Park GL threads while backgrounded (submissions would fail and lose work).
        NSNotificationCenter* nc = NSNotificationCenter.defaultCenter;
        [nc addObserverForName:UIApplicationDidEnterBackgroundNotification object:nil queue:nil
                    usingBlock:^(NSNotification*) { this->setBackground(true); }];
        [nc addObserverForName:UIApplicationWillEnterForegroundNotification object:nil queue:nil
                    usingBlock:^(NSNotification*) { this->setBackground(false); }];
#endif
        caps.hasFamilyApple9 = [mtl supportsFamily:MTLGPUFamilyApple9];
        caps.bcCompression = mtl.supportsBCTextureCompression;
        // Sampler min/max reduction: an Apple10 GPU (M5) applies reductionMode; an Apple9 one (M4 Pro)
        // accept the descriptor but always average.
        caps.samplerReduction = [mtl supportsFamily:MTLGPUFamilyApple10];
        caps.maxThreadsPerThreadgroup = (uint32_t)mtl.maxThreadsPerThreadgroup.width;
        caps.maxThreadgroupMemory = (uint32_t)mtl.maxThreadgroupMemoryLength;
        MGL_INFO("Metal 4 device: %s (apple9=%d, bc=%d)", mtl.name.UTF8String, caps.hasFamilyApple9, caps.bcCompression);
    }
    return true;
}

uint64_t Device::reserveSerial() {
    std::lock_guard<std::mutex> g(mutex_);
    return nextSerial_++;
}

bool Device::gpuTimes(uint64_t serial, double* start, double* end) {
    std::lock_guard<std::mutex> g(mutex_);
    auto it = gpuTimes_.find(serial);
    if (it == gpuTimes_.end()) return false;
    *start = it->second.first;
    *end = it->second.second;
    return true;
}

bool Device::isComplete(uint64_t serial) {
    std::lock_guard<std::mutex> g(mutex_);
    return isCompleteLocked(serial);
}

void Device::markComplete(uint64_t serial, NSError* error, double gpuStart, double gpuEnd) {
    if (error) MGL_ERR("GPU command buffer %llu failed: %s", (unsigned long long)serial, error.description.UTF8String);
    {
        std::lock_guard<std::mutex> g(mutex_);
        if (gpuEnd > 0) {
            gpuTimes_[serial] = {gpuStart, gpuEnd};
            while (gpuTimes_.size() > 4096) gpuTimes_.erase(gpuTimes_.begin());
        }
        submitted_.erase(serial);
        completedAbove_.insert(serial);
        while (!completedAbove_.empty() && *completedAbove_.begin() == watermark_ + 1) {
            watermark_++;
            completedAbove_.erase(completedAbove_.begin());
        }
    }
    cv_.notify_all();
}

void Device::waitFor(uint64_t serial) {
    if (serial == 0) return;
    {
        std::unique_lock<std::mutex> lk(mutex_);
        if (!isCompleteLocked(serial) && !submitted_.count(serial)) {
            // Recorded by a context that has not flushed yet; GL leaves this undefined.
            MGL_WARN("waiting on unsubmitted serial %llu", (unsigned long long)serial);
            return;
        }
        if (!cv_.wait_for(lk, std::chrono::seconds(10), [&] { return isCompleteLocked(serial); }))
            MGL_ERR("GPU wait for serial %llu timed out", (unsigned long long)serial);
    }
    collect();
}

void Device::waitIdle() {
    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait_for(lk, std::chrono::seconds(10), [&] { return submitted_.empty(); });
}

id<MTL4CommandAllocator> Device::acquireAllocator() {
    std::lock_guard<std::mutex> g(mutex_);
    while (!inflightAllocators_.empty() && isCompleteLocked(inflightAllocators_.front().serial)) {
        id<MTL4CommandAllocator> a = inflightAllocators_.front().allocator;
        [a reset];
        freeAllocators_.push_back(a);
        inflightAllocators_.pop_front();
    }
    if (!freeAllocators_.empty()) {
        id<MTL4CommandAllocator> a = freeAllocators_.back();
        freeAllocators_.pop_back();
        return a;
    }
    return [mtl newCommandAllocator];
}

id<MTL4CommandBuffer> Device::newCommandBuffer() { return [mtl newCommandBuffer]; }

void Device::setBackground(bool bg) {
    {
        std::lock_guard<std::mutex> g(bgMutex_);
        background_ = bg;
    }
    MGL_INFO("app %s", bg ? "entered background: parking GPU submissions" : "returned to foreground");
    bgCv_.notify_all();
}

void Device::waitForeground() {
    if (!background_.load()) return;
    std::unique_lock<std::mutex> lk(bgMutex_);
    bgCv_.wait(lk, [&] { return !background_.load(); });
}

void Device::submit(id<MTL4CommandBuffer> cb, id<MTL4CommandAllocator> alloc, uint64_t serial,
                    id<CAMetalDrawable> drawable) {
    waitForeground();
    MTL4CommitOptions* opts = [MTL4CommitOptions new];
    [opts addFeedbackHandler:^(id<MTL4CommitFeedback> fb) {
        this->markComplete(serial, fb.error, fb.GPUStartTime, fb.GPUEndTime);
    }];
    std::lock_guard<std::mutex> g(mutex_);
    if (residencyDirty_) {
        [residency commit];
        residencyDirty_ = false;
    }
    submitted_.insert(serial);
    if (drawable) [queue waitForDrawable:drawable];
    id<MTL4CommandBuffer> cbs[1] = {cb};
    [queue commit:cbs count:1 options:opts];
    if (drawable) {
        [queue signalDrawable:drawable];
        // ProMotion keeps a CAMetalLayer at 60 Hz unless presents ask for a shorter
        // frame duration; request the screen's maximum rate (METALGLDRV_MAX_FPS caps it).
        static const double minDuration = [] {
            double fps = 0;
#if TARGET_OS_IPHONE
            fps = (double)UIScreen.mainScreen.maximumFramesPerSecond;
#endif
            if (const char* e = getenv("METALGLDRV_MAX_FPS")) fps = atof(e);
            if (fps > 0) MGL_INFO("presenting at up to %.0f fps", fps);
            return fps > 0 ? 1.0 / fps : 0.0;
        }();
        if (minDuration > 0)
            [drawable presentAfterMinimumDuration:minDuration];
        else
            [drawable present];
    }
    if (alloc) inflightAllocators_.push_back({serial, alloc});
}

void Device::makeResident(id<MTLAllocation> a) {
    std::lock_guard<std::mutex> g(mutex_);
    [residency addAllocation:a];
    residencyDirty_ = true;
}

void Device::releaseAfter(id<MTLAllocation> a, uint64_t serial) {
    if (!a) return;
    std::lock_guard<std::mutex> g(mutex_);
    if (isCompleteLocked(serial)) {
        [residency removeAllocation:a];
        residencyDirty_ = true;
        return;
    }
    deferred_.push_back({serial, a, true});
}

void Device::retainUntil(id obj, uint64_t serial) {
    if (!obj) return;
    std::lock_guard<std::mutex> g(mutex_);
    if (isCompleteLocked(serial)) return;
    deferred_.push_back({serial, obj, false});
}

void Device::collect() {
    std::lock_guard<std::mutex> g(mutex_);
    // deferred_ is not strictly sorted (different contexts), so scan.
    for (auto it = deferred_.begin(); it != deferred_.end();) {
        if (isCompleteLocked(it->serial)) {
            if (it->removeResidency) {
                [residency removeAllocation:(id<MTLAllocation>)it->obj];
                residencyDirty_ = true;
            }
            it = deferred_.erase(it);
        } else {
            ++it;
        }
    }
}

id<MTLBuffer> Device::newBuffer(size_t size, MTLResourceOptions opts) {
    if (size == 0) size = 16;
    if (g_statsEnabled) statAddCaller("newBuffer KB", __builtin_return_address(0), size >> 10);
    id<MTLBuffer> b = [mtl newBufferWithLength:size options:opts | MTLResourceHazardTrackingModeUntracked];
    if (b) makeResident(b);
    return b;
}

id<MTLLibrary> Device::compileLibrary(NSString* source, NSString** errorOut, bool fastMath) {
    MTL4LibraryDescriptor* ld = [MTL4LibraryDescriptor new];
    ld.source = source;
    MTLCompileOptions* co = [MTLCompileOptions new];
    co.languageVersion = MTLLanguageVersion4_0;
    co.mathMode = fastMath ? MTLMathModeFast : MTLMathModeSafe;
    co.preserveInvariance = YES;
    // Debugging aid, read per compile: METALGLDRV_MSL_OPTS containing "noinv" and/or "size".
    if (const char* o = getenv("METALGLDRV_MSL_OPTS")) {
        if (strstr(o, "noinv")) co.preserveInvariance = NO;
        if (strstr(o, "size")) co.optimizationLevel = MTLLibraryOptimizationLevelSize;
    }
    ld.options = co;
    NSError* err = nil;
    id<MTLLibrary> lib = [compiler newLibraryWithDescriptor:ld error:&err];
    if (!lib && errorOut) *errorOut = err.localizedDescription;
    return lib;
}

}  // namespace mgl
