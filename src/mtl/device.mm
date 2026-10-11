#import "mtl/device.h"
#include <TargetConditionals.h>
#include <strings.h>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#if TARGET_OS_IPHONE
#import <UIKit/UIKit.h>
#endif
#import <QuartzCore/CAMetalLayer.h>
#include "mtl/spi.h"
#include "util/log.h"
#include "util/stats.h"
#include <chrono>

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
        // Framebuffer fetch ([[color(n)]] inputs) is an Apple-family feature; every GPU that runs
        // Metal 4 is Apple7 or later.
        caps.programmableBlending = [mtl supportsFamily:MTLGPUFamilyApple7];
        caps.maxThreadsPerThreadgroup = (uint32_t)mtl.maxThreadsPerThreadgroup.width;
        caps.maxThreadgroupMemory = (uint32_t)mtl.maxThreadgroupMemoryLength;
        if (@available(macOS 26.4, iOS 26.4, *)) {
            if (mtl.supportsPlacementSparse) sparseTileBytes = [mtl sparseTileSizeInBytesForSparsePageSize:kSparsePageSize];
        }
        caps.sparse = sparseTileBytes > 0;
        caps.strictLevel = parseAccuracyLevel();
        if (spi::enabled()) {
            // private API on Metal 4 encoders: ask a render encoder (recorded, never committed)
            MTLTextureDescriptor* td = texture2DDescriptor(MTLPixelFormatR8Unorm, 1, 1);
            td.usage = MTLTextureUsageRenderTarget;
            td.storageMode = MTLStorageModePrivate;
            id<MTLTexture> t = [mtl newTextureWithDescriptor:td];
            id<MTL4CommandAllocator> a = [mtl newCommandAllocator];
            id<MTL4CommandBuffer> cb = [mtl newCommandBuffer];
            [cb beginCommandBufferWithAllocator:a];
            MTL4RenderPassDescriptor* rp = [MTL4RenderPassDescriptor new];
            rp.colorAttachments[0].texture = t;
            id<MTL4RenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
            spi::probeLineWidth(e);
            [e endEncoding];
            [cb endCommandBuffer];
        }
        MGL_INFO("Metal 4 device: %s (apple9=%d, bc=%d, sparse=%d%s)", mtl.name.UTF8String, caps.hasFamilyApple9, caps.bcCompression,
                 caps.sparse,
                 caps.superStrict() ? ", super strict" : caps.strict() ? ", strict" : caps.speedDemon() ? ", SPEEDDEMON" : caps.turbo() ? ", TURBOHAX" : "");
    }
    return true;
}

int parseAccuracyLevel() {
    int strict = parseStrictLevel(getenv("METALGLDRV_STRICT"));
    if (strict > 0) return strict;  // accuracy wins over speed
    auto on = [](const char* n) {
        const char* v = getenv(n);
        return v && *v && strcmp(v, "0") && strcasecmp(v, "false") && strcasecmp(v, "no");
    };
    if (on("METALGLDRV_SPEEDDEMON")) return -2;
    if (on("METALGLDRV_TURBOHAX")) return -1;
    return 0;
}

int parseStrictLevel(const char* v) {
    if (!v || !*v || !strcmp(v, "0") || !strcasecmp(v, "false") || !strcasecmp(v, "no")) return 0;
    if (!strcasecmp(v, "super") || !strcasecmp(v, "superstrict") || !strcasecmp(v, "super-strict")) return 2;
    int n = atoi(v);
    return n >= 2 ? 2 : 1;
}

uint64_t Device::reserveSerial() {
    std::lock_guard<std::mutex> g(mutex_);
    return nextSerial_++;
}

uint64_t Device::lastReservedSerial() {
    std::lock_guard<std::mutex> g(mutex_);
    return nextSerial_ - 1;
}

double Device::takeMaxGpuSeconds() {
    std::lock_guard<std::mutex> g(mutex_);
    double m = maxGpuSeconds_;
    maxGpuSeconds_ = 0;
    return m;
}

double Device::takeGpuBusySeconds() {
    std::lock_guard<std::mutex> g(mutex_);
    double b = gpuBusySeconds_;
    gpuBusySeconds_ = 0;
    return b;
}

void Device::takeTessGpuSeconds(double out[2]) {
    std::lock_guard<std::mutex> g(mutex_);
    for (int i = 0; i < 2; i++) {
        out[i] = tessGpuSeconds_[i];
        tessGpuSeconds_[i] = 0;
    }
}

bool Device::gpuTimes(uint64_t serial, double* start, double* end) {
    std::lock_guard<std::mutex> g(mutex_);
    auto it = gpuTimes_.find(serial);
    if (it == gpuTimes_.end()) return false;
    *start = it->second.first;
    *end = it->second.second;
    return true;
}

void Device::completionState(uint64_t out[4]) {
    std::lock_guard<std::mutex> g(mutex_);
    out[0] = watermark_;
    out[1] = completedAbove_.size();
    out[2] = submitted_.empty() ? 0 : *submitted_.begin();
    out[3] = nextSerial_;
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
            // Command buffers near the GPU watchdog (a few seconds) get killed ("CDM Kill timeout"): report
            // long ones (at most every 10 s), which is where to look when that happens.
            double dt = gpuEnd - gpuStart;
            maxGpuSeconds_ = std::max(maxGpuSeconds_, dt);
            // Exclusive time: the part of this command buffer's interval after the previous ones ended (a
            // command buffer starting early mostly waits at its leading barrier).
            double exclusive = gpuEnd > gpuBusyUntil_ ? gpuEnd - std::max(gpuStart, gpuBusyUntil_) : 0;
            if (g_profilePasses) profileComplete(serial, exclusive);
            gpuBusySeconds_ += exclusive;
            gpuBusyUntil_ = std::max(gpuBusyUntil_, gpuEnd);
            if (dt > 0.5 && gpuEnd - lastLongWarning_ > 10) {
                lastLongWarning_ = gpuEnd;
                MGL_WARN("command buffer %llu ran %.0f ms on the GPU", (unsigned long long)serial, dt * 1000);
            }
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
    commitResidencyLocked();
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

id<MTLCommandQueue> Device::classicQueue() {
    std::lock_guard<std::mutex> g(mutex_);
    if (classicQueueTried_) return classicQueue_;
    classicQueueTried_ = true;
    @autoreleasepool {
        MTLCommandQueueDescriptor* qd = [MTLCommandQueueDescriptor new];
        qd.maxCommandBufferCount = 64;
        spi::markOpenGLQueue(qd);
        classicQueue_ = [mtl newCommandQueueWithDescriptor:qd];
        if (!classicQueue_) {
            MGL_WARN("no classic Metal command queue: tessellation stays emulated");
            return nil;
        }
        classicQueue_.label = @"metalgldrv tessellation";
        [classicQueue_ addResidencySet:residency];
    }
    return classicQueue_;
}

void Device::commitClassic(id<MTLCommandBuffer> cb, bool render, uint64_t serial) {
    waitForeground();
    if (!cb) {
        if (serial) markComplete(serial, nil, 0, 0);
        return;
    }
    [cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
        if (c.GPUEndTime > 0) {
            std::lock_guard<std::mutex> g(mutex_);
            tessGpuSeconds_[render] += c.GPUEndTime - c.GPUStartTime;
        }
        if (serial) this->markComplete(serial, c.error, 0, 0);
        else if (c.error) MGL_ERR("GPU command buffer (tessellation) failed: %s", c.error.description.UTF8String);
    }];
    std::lock_guard<std::mutex> g(mutex_);
    commitResidencyLocked();
    if (serial) submitted_.insert(serial);
    [cb commit];
}

void Device::queueSignal(id<MTLEvent> e, uint64_t value) {
    std::lock_guard<std::mutex> g(mutex_);
    [queue signalEvent:e value:value];
}

void Device::queueWait(id<MTLEvent> e, uint64_t value) {
    std::lock_guard<std::mutex> g(mutex_);
    [queue waitForEvent:e value:value];
}

void Device::commitResidencyLocked() {
    if (!residencyDirty_) return;
    // Allocations removed from the set are deallocated here, and their dealloc autoreleases.
    auto t0 = std::chrono::steady_clock::now();
    @autoreleasepool {
        [residency commit];
    }
    pendingResidentBytes_ = 0;
    pendingResidentCount_ = 0;
    if (g_hitchMs > 0) hitchCompile("residency commit", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    residencyDirty_ = false;
}

void Device::makeResident(id<MTLAllocation> a) {
    std::lock_guard<std::mutex> g(mutex_);
    [residency addAllocation:a];
    residencyDirty_ = true;
    // Commit on the allocating thread once a lot is pending: a streaming thread creating hundreds of MB of
    // textures would otherwise leave the whole commit (making them resident) to the next submit on the render
    // thread, a stall of 100+ ms.
    pendingResidentBytes_ += a.allocatedSize;
    if (++pendingResidentCount_ >= 256 || pendingResidentBytes_ >= (64u << 20)) commitResidencyLocked();
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

void Device::retainSparse(id obj, uint64_t serial) {
    if (!obj) return;
    std::lock_guard<std::mutex> g(mutex_);
    sparseRetired_.push_back({CFAbsoluteTimeGetCurrent() + kSparseSettleTime, {serial, obj, false}});
}

void Device::collect() {
    std::lock_guard<std::mutex> g(mutex_);
    // Dropping the last reference deallocates, and Metal objects autorelease in dealloc.
    @autoreleasepool {
        if (!sparseRetired_.empty()) {
            const double now = CFAbsoluteTimeGetCurrent();
            while (!sparseRetired_.empty() && sparseRetired_.front().first <= now && isCompleteLocked(sparseRetired_.front().second.serial))
                sparseRetired_.pop_front();
        }
        // deferred_ is not strictly sorted (different contexts), so scan; compact in one pass (erasing
        // entries one at a time is quadratic when a streaming thread frees thousands of objects at once).
        auto t0 = std::chrono::steady_clock::now();
        size_t freed = 0;
        auto out = deferred_.begin();
        for (auto it = deferred_.begin(); it != deferred_.end(); ++it) {
            if (isCompleteLocked(it->serial)) {
                if (it->removeResidency) {
                    [residency removeAllocation:(id<MTLAllocation>)it->obj];
                    residencyDirty_ = true;
                }
                it->obj = nil;  // drops the last reference
                freed++;
            } else {
                if (out != it) *out = std::move(*it);
                ++out;
            }
        }
        deferred_.erase(out, deferred_.end());
        if (freed && g_hitchMs > 0)
            hitchCompile("deferred releases", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
}

// --- sparse resources -------------------------------------------------------
// Returns tiles whose last use completed to their heaps; releases heaps that became empty
// (keeping one spare) so sparse resources don't pin memory after they are deleted.
void Device::reclaimSparseLocked() {
    const double now = CFAbsoluteTimeGetCurrent();
    for (auto it = sparsePendingFree_.begin(); it != sparsePendingFree_.end();) {
        if (it->notBefore > now || !isComplete(it->serial)) {
            ++it;
            continue;
        }
        const SparseTiles& t = it->tiles;
        SparseHeap& h = sparseHeaps_[t.heap];
        for (uint32_t i = 0; i < t.count; i++) h.used[t.first + i] = false;
        h.freeCount += t.count;
        h.hint = std::min(h.hint, t.first);
        it = sparsePendingFree_.erase(it);
    }
    bool spare = false;
    for (SparseHeap& h : sparseHeaps_) {
        if (!h.heap || h.freeCount != h.used.size()) continue;
        if (!spare) {
            spare = true;
            continue;
        }
        releaseAfter(h.heap, 0);
        h = SparseHeap{nil, {}, 0, 0};
    }
}

bool Device::allocSparseTiles(uint32_t count, SparseTiles& out) {
    if (!sparseTileBytes || !count) return false;
    std::lock_guard<std::mutex> g(sparseMutex_);
    reclaimSparseLocked();
    auto take = [&](uint32_t hi) {
        SparseHeap& h = sparseHeaps_[hi];
        if (!h.heap || h.freeCount < count) return false;
        uint32_t n = (uint32_t)h.used.size(), run = 0;
        for (uint32_t i = count == 1 ? h.hint : 0; i < n; i++) {
            run = h.used[i] ? 0 : run + 1;
            if (run < count) continue;
            uint32_t first = i + 1 - count;
            for (uint32_t k = first; k <= i; k++) h.used[k] = true;
            h.freeCount -= count;
            if (count == 1) h.hint = i + 1;
            out = {hi, first, count};
            return true;
        }
        return false;
    };
    for (uint32_t hi = 0; hi < sparseHeaps_.size(); hi++)
        if (take(hi)) return true;
    // New placement heap: 16 MB, or larger for a big contiguous run.
    const uint32_t tiles = std::max<uint32_t>(count, (uint32_t)((16u << 20) / sparseTileBytes));
    MTLHeapDescriptor* hd = [MTLHeapDescriptor new];
    hd.type = MTLHeapTypePlacement;
    hd.storageMode = MTLStorageModePrivate;
    hd.hazardTrackingMode = MTLHazardTrackingModeUntracked;
    hd.size = (NSUInteger)tiles * sparseTileBytes;
    hd.maxCompatiblePlacementSparsePageSize = kSparsePageSize;
    id<MTLHeap> heap = [mtl newHeapWithDescriptor:hd];
    if (!heap) {
        MGL_ERR("sparse placement heap allocation failed (%u tiles)", tiles);
        return false;
    }
    heap.label = @"metalgldrv sparse tiles";
    makeResident(heap);
    uint32_t n = (uint32_t)(heap.size / sparseTileBytes);
    uint32_t slot = 0;
    while (slot < sparseHeaps_.size() && sparseHeaps_[slot].heap) slot++;
    if (slot == sparseHeaps_.size()) sparseHeaps_.emplace_back();
    sparseHeaps_[slot] = {heap, std::vector<bool>(n, false), n, 0};
    return take(slot);
}

void Device::freeSparseTiles(const SparseTiles& t, uint64_t serial, bool unmapped) {
    if (!t.count) return;
    std::lock_guard<std::mutex> g(sparseMutex_);
    sparsePendingFree_.push_back({serial, unmapped ? CFAbsoluteTimeGetCurrent() + kSparseSettleTime : 0.0, t});
    reclaimSparseLocked();
}

// Issues the ops as queue mapping updates, one call per run of ops sharing a heap.
template <typename Res, typename Op>
void Device::submitSparseOps(Res res, const std::vector<SparseOp>& ops, Op makeOp) {
    std::vector<id<MTLHeap>> heaps(ops.size());
    {
        std::lock_guard<std::mutex> g(sparseMutex_);
        for (size_t i = 0; i < ops.size(); i++) heaps[i] = sparseHeaps_[ops[i].tiles.heap].heap;  // unmaps too: a nil heap leaves later maps of the resource broken
    }
    std::lock_guard<std::mutex> g(mutex_);
    commitResidencyLocked();
    for (const SparseOp& o : ops)
        MGL_DBG("sparse %s level %lu slice %lu tiles (%lu,%lu,%lu)+(%lu,%lu,%lu) -> heap %u tile %u", o.unmap ? "unmap" : "map",
                (unsigned long)o.level, (unsigned long)o.slice, (unsigned long)o.region.origin.x, (unsigned long)o.region.origin.y,
                (unsigned long)o.region.origin.z, (unsigned long)o.region.size.width, (unsigned long)o.region.size.height,
                (unsigned long)o.region.size.depth, o.tiles.heap, o.tiles.first);
    for (size_t i = 0; i < ops.size();) {
        size_t j = i;
        std::vector<decltype(makeOp(ops[0]))> batch;
        while (j < ops.size() && heaps[j] == heaps[i]) batch.push_back(makeOp(ops[j++]));
        if constexpr (std::is_same_v<decltype(makeOp(ops[0])), MTL4UpdateSparseTextureMappingOperation>)
            [queue updateTextureMappings:res heap:heaps[i] operations:batch.data() count:batch.size()];
        else
            [queue updateBufferMappings:res heap:heaps[i] operations:batch.data() count:batch.size()];
        i = j;
    }
}

void Device::updateSparseMappings(id<MTLTexture> tex, const std::vector<SparseOp>& ops) {
    submitSparseOps(tex, ops, [](const SparseOp& o) {
        MTL4UpdateSparseTextureMappingOperation m;
        m.mode = o.unmap ? MTLSparseTextureMappingModeUnmap : MTLSparseTextureMappingModeMap;
        m.textureRegion = o.region;
        m.textureLevel = o.level;
        m.textureSlice = o.slice;
        m.heapOffset = o.tiles.first;
        return m;
    });
}

void Device::updateSparseMappings(id<MTLBuffer> buf, const std::vector<SparseOp>& ops) {
    submitSparseOps(buf, ops, [](const SparseOp& o) {
        MTL4UpdateSparseBufferMappingOperation m;
        m.mode = o.unmap ? MTLSparseTextureMappingModeUnmap : MTLSparseTextureMappingModeMap;
        m.bufferRange = NSMakeRange(o.region.origin.x, o.region.size.width);
        m.heapOffset = o.tiles.first;
        return m;
    });
}

id<MTLBuffer> Device::newBuffer(size_t size, MTLResourceOptions opts) {
    if (size == 0) size = 16;
    if (g_statsEnabled) statAddCaller("newBuffer KB", __builtin_return_address(0), size >> 10);
    id<MTLBuffer> b = [mtl newBufferWithLength:size options:opts | MTLResourceHazardTrackingModeUntracked];
    if (b) makeResident(b);
    return b;
}

// Size class of newBufferRecycled: 256 B << c (c = 15: too large to recycle)
static int recycleClass(size_t size) {
    int c = 0;
    while (c < 15 && (size_t)256 << c < size) c++;
    return c;
}

id<MTLBuffer> Device::newBufferRecycled(size_t size) {
    int c = recycleClass(size);
    if (c >= kRecycleClasses) return newBuffer(size);
    {
        std::lock_guard<std::mutex> g(mutex_);
        auto& pend = recyclePending_[c];
        auto& free = recycleFree_[c];
        while (!pend.empty() && isCompleteLocked(pend.front().first)) {
            free.push_back(pend.front().second);
            pend.pop_front();
        }
        if (!free.empty()) {
            id<MTLBuffer> b = free.back();
            free.pop_back();
            recycleBytes_ -= b.length;
            return b;
        }
    }
    return newBuffer((size_t)256 << c);
}

void Device::recycleBufferAfter(id<MTLBuffer> b, uint64_t serial) {
    if (!b) return;
    NSUInteger len = b.length;
    int c = recycleClass(len);
    if (c < kRecycleClasses && len == (size_t)256 << c) {
        std::lock_guard<std::mutex> g(mutex_);
        if (recycleBytes_ + len <= kRecycleMaxBytes) {
            recycleBytes_ += len;
            recyclePending_[c].push_back({serial, b});
            return;
        }
    }
    releaseAfter(b, serial);
}

uint64_t Device::zeroBufferAddress() {
    static id<MTLBuffer> zero = [this] {
        id<MTLBuffer> b = newBuffer(kZeroBufferBytes);
        memset(bufferContents(b), 0, kZeroBufferBytes);
        return b;
    }();
    return zero.gpuAddress;
}

id<MTLLibrary> Device::compileLibrary(const std::string& source, std::string* errorOut, bool fastMath) {
    @autoreleasepool {  // the source string, options and errors autorelease
        MTL4LibraryDescriptor* ld = [MTL4LibraryDescriptor new];
        // GLSL lets compilers fuse a * b + c into fma() outside `precise` expressions (which turn it off for
        // their function, msl.cpp): allow it across statements too (relaxed math already does; this is for
        // strict's safe math). Super strict keeps Metal's default.
        bool fuse = caps.strictLevel < 2;
        if (const char* o = getenv("METALGLDRV_MSL_OPTS"); o && strstr(o, "nofuse")) fuse = false;
        ld.source = fuse ? @(("#pragma METAL fp contract(fast)\n" + source).c_str()) : @(source.c_str());
        MTLCompileOptions* co = [MTLCompileOptions new];
        co.languageVersion = MTLLanguageVersion4_0;
        // GLSL lets compilers reassociate and contract floating-point math except in `precise` expressions,
        // so by default shaders get relaxed math (which keeps NaN and infinity semantics: isnan / isinf still
        // work); it's worth ~10% in shader-heavy games (SuperTuxKart's deferred lighting). The strict
        // profiles and shaders with `precise` keep safe math.
        // TURBOHAX / SPEEDDEMON: fast math everywhere (reassociation, no NaN / infinity guarantees).
        bool precise = source.find("#pragma METAL fp contract(off)") != std::string::npos;
        co.mathMode = fastMath || caps.turbo()               ? MTLMathModeFast
                      : caps.strictLevel >= 1 || precise ? MTLMathModeSafe
                                                             : MTLMathModeRelaxed;
        // Safe arithmetic (no reassociation, IEEE NaN/Inf) but the fast sin/cos/exp/log/pow...: GLSL bounds
        // them loosely (sin/cos: 2^-11 absolute), and the precise versions cost several times as much in
        // shaders that use many (DOOM (2016)'s depth of field: 64 sin/cos per pixel). The strict profiles
        // keep the precise ones; METALGLDRV_MSL_OPTS=precisefn too.
        co.mathFloatingPointFunctions = caps.strictLevel >= 1 ? MTLMathFloatingPointFunctionsPrecise : MTLMathFloatingPointFunctionsFast;
        co.preserveInvariance = YES;
        // Debugging aid, read per compile: METALGLDRV_MSL_OPTS containing "noinv", "safemath", "nofuse", "size",
        // "precisefn".
        if (const char* o = getenv("METALGLDRV_MSL_OPTS")) {
            if (strstr(o, "noinv")) co.preserveInvariance = NO;
            if (strstr(o, "safemath")) co.mathMode = MTLMathModeSafe;
            if (strstr(o, "size")) co.optimizationLevel = MTLLibraryOptimizationLevelSize;
            if (strstr(o, "precisefn")) co.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        }
        ld.options = co;
        NSError* err = nil;
        id<MTLLibrary> lib = [compiler newLibraryWithDescriptor:ld error:&err];
        if (!lib && errorOut) *errorOut = err ? err.localizedDescription.UTF8String : "MSL compile failed";
        return lib;
    }
}

}  // namespace mgl
