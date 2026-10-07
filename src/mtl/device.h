// Process-wide Metal 4 device state shared by every GL context.
#pragma once
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include <atomic>
#include <cstdint>
#include <deque>
#include <condition_variable>
#include <mutex>
#include <map>
#include <set>
#include <vector>

namespace mgl {

struct DeviceCaps {
    bool meshShaders = true;
    uint32_t maxTextureSize = 16384;
    uint32_t maxColorAttachments = 8;
    uint32_t maxThreadsPerThreadgroup = 1024;
    uint32_t maxThreadgroupMemory = 32768;
    bool hasFamilyApple9 = false;
    bool bcCompression = false;
    bool samplerReduction = false;  // MTLSamplerDescriptor.reductionMode honored (ARB_texture_filter_minmax)
};

class Device {
public:
    static Device* get();  // nullptr if Metal 4 is unavailable

    id<MTLDevice> mtl;
    id<MTL4CommandQueue> queue;
    id<MTL4Compiler> compiler;
    id<MTLResidencySet> residency;
    DeviceCaps caps;

    // --- submission / completion tracking ---------------------------------
    // Every command buffer reserves a unique serial when it begins recording.
    // Resources record the serial of the command buffer they were last used
    // in. Completion is reported per serial by commit feedback handlers, so
    // contexts may submit out of order.
    uint64_t reserveSerial();
    bool isComplete(uint64_t serial);
    void waitFor(uint64_t serial);
    void waitIdle();
    // GPU execution interval (host time, seconds) of a completed serial; false if unknown.
    bool gpuTimes(uint64_t serial, double* start, double* end);
    // A reserved serial whose command buffer was never submitted.
    void abandonSerial(uint64_t serial) { markComplete(serial, nil); }

    // Commits a finished command buffer recorded under `serial`.
    // `drawable` (optional) is presented after the command buffer.
    void submit(id<MTL4CommandBuffer> cb, id<MTL4CommandAllocator> alloc, uint64_t serial,
                id<CAMetalDrawable> drawable = nil);

    id<MTL4CommandAllocator> acquireAllocator();
    id<MTL4CommandBuffer> newCommandBuffer();

    // --- app lifecycle (iOS) --------------------------------------------------
    // GPU work is not permitted while the app is in the background; GL threads
    // park here (at submits / swaps) until the app returns to the foreground.
    void waitForeground();
    bool inBackground() const { return background_.load(); }
    void setBackground(bool bg);

    // --- residency ----------------------------------------------------------
    void makeResident(id<MTLAllocation> a);
    // Removes from residency and drops the reference once the GPU is done with `serial`.
    void releaseAfter(id<MTLAllocation> a, uint64_t serial);
    // Keeps an arbitrary object alive until `serial` completes.
    void retainUntil(id obj, uint64_t serial);

    // --- helpers ------------------------------------------------------------
    id<MTLBuffer> newBuffer(size_t size, MTLResourceOptions opts = MTLResourceStorageModeShared);
    id<MTLLibrary> compileLibrary(NSString* source, NSString** errorOut, bool fastMath = false);

    // Called periodically (e.g. on submit) to reap finished work.
    void collect();

private:
    Device() = default;
    bool init();

    void markComplete(uint64_t serial, NSError* error, double gpuStart = 0, double gpuEnd = 0);
    std::map<uint64_t, std::pair<double, double>> gpuTimes_;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> background_{false};
    std::mutex bgMutex_;
    std::condition_variable bgCv_;
    uint64_t nextSerial_ = 1;
    uint64_t watermark_ = 0;              // all serials <= watermark_ are complete
    std::set<uint64_t> completedAbove_;   // completed serials > watermark_
    std::set<uint64_t> submitted_;        // submitted, not complete
    bool residencyDirty_ = false;
    bool isCompleteLocked(uint64_t serial) const {
        return serial <= watermark_ || completedAbove_.count(serial);
    }
    struct Pending {
        uint64_t serial;
        id<MTL4CommandAllocator> allocator;
    };
    std::deque<Pending> inflightAllocators_;
    std::vector<id<MTL4CommandAllocator>> freeAllocators_;
    struct Deferred {
        uint64_t serial;
        id obj;
        bool removeResidency;
    };
    std::deque<Deferred> deferred_;
};

}  // namespace mgl
