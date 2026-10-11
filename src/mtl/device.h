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
#include <string>
#include <vector>

namespace mgl {

// Autorelease pools: GL threads (glcts' main thread, the game's render thread) never drain an
// autorelease pool, so anything autoreleased there lives until the thread exits. Code that can
// autorelease (Metal factory methods, encoder creation, submits) runs inside a local
// @autoreleasepool, and buffer contents are read through bufferContents().
//
// MTLBuffer.contents is objc_returns_inner_pointer: on a local or parameter receiver ARC
// retains and autoreleases the buffer, leaking it (and its GPU memory). A precise-lifetime
// local is exempt and costs nothing.
inline void* bufferContents(id<MTLBuffer> b) {
    __attribute__((objc_precise_lifetime)) id<MTLBuffer> keep = b;
    return keep.contents;
}
// Same as [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:width:height:mipmapped:NO] and
// textureBufferDescriptorWithPixelFormat:..., which return autoreleased descriptors.
inline MTLTextureDescriptor* texture2DDescriptor(MTLPixelFormat pf, NSUInteger w, NSUInteger h) {
    MTLTextureDescriptor* td = [MTLTextureDescriptor new];
    td.pixelFormat = pf;
    td.width = w;
    td.height = h;
    return td;
}
inline MTLTextureDescriptor* textureBufferDescriptor(MTLPixelFormat pf, NSUInteger w, MTLResourceOptions opts, MTLTextureUsage usage) {
    MTLTextureDescriptor* td = [MTLTextureDescriptor new];
    td.textureType = MTLTextureTypeTextureBuffer;
    td.pixelFormat = pf;
    td.width = w;
    td.resourceOptions = opts;
    td.usage = usage;
    return td;
}

struct DeviceCaps {
    bool meshShaders = true;
    uint32_t maxTextureSize = 16384;
    uint32_t maxColorAttachments = 8;
    uint32_t maxThreadsPerThreadgroup = 1024;
    uint32_t maxThreadgroupMemory = 32768;
    bool hasFamilyApple9 = false;
    bool bcCompression = false;
    bool samplerReduction = false;  // MTLSamplerDescriptor.reductionMode honored (ARB_texture_filter_minmax)
    bool programmableBlending = false;  // fragment functions read color attachments (KHR_blend_equation_advanced)
    bool sparse = false;  // placement sparse textures/buffers (ARB_sparse_texture*, ARB_sparse_buffer)
    // METALGLDRV_STRICT, read once at startup: the accuracy profile, 0 default, 1 strict (accuracy features
    // that cost performance and aren't needed for conformance), 2 super strict (strict plus the most
    // expensive emulations). See docs/ARCHITECTURE.md, "Environment variables".
    // Below 0, the speed profiles trade accuracy for performance: -1 TURBOHAX, -2 SPEEDDEMON (TURBOHAX plus
    // more). The strict profiles take precedence (parseAccuracyLevel).
    int strictLevel = 0;
    bool strict() const { return strictLevel >= 1; }
    bool superStrict() const { return strictLevel >= 2; }
    bool turbo() const { return strictLevel <= -1; }
    bool speedDemon() const { return strictLevel <= -2; }
};

// METALGLDRV_STRICT value -> DeviceCaps::strictLevel: unset/"0"/"false"/"no" 0, "2"/"super"/"superstrict" 2,
// anything else ("1", "strict", ...) 1.
int parseStrictLevel(const char* v);
// The accuracy level from the environment: METALGLDRV_STRICT (1, 2) if set, else -2 with METALGLDRV_SPEEDDEMON,
// else -1 with METALGLDRV_TURBOHAX, else 0.
int parseAccuracyLevel();

// A run of consecutive tiles in one of the device's placement heaps (sparse resources).
struct SparseTiles {
    uint32_t heap = 0, first = 0, count = 0;
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
    // The most recently reserved serial: work recorded so far completes with it at the latest.
    uint64_t lastReservedSerial();
    // (watermark, completed serials above it, oldest submitted serial still pending, next serial): for stats
    void completionState(uint64_t out[4]);
    bool isComplete(uint64_t serial);
    void waitFor(uint64_t serial);
    void waitIdle();
    // GPU execution interval (host time, seconds) of a completed serial; false if unknown.
    bool gpuTimes(uint64_t serial, double* start, double* end);
    // The longest GPU time of a command buffer completed since the last call (seconds; mglTestMaxGpuSeconds).
    double takeMaxGpuSeconds();
    // Seconds the GPU was busy with this device's command buffers since the last call (overlaps counted once).
    double takeGpuBusySeconds();
    // GPU time of classic (tessellation) command buffers: [0] VS + TCS kernels, [1] render passes
    void takeTessGpuSeconds(double out[2]);
    // A reserved serial whose command buffer was never submitted.
    void abandonSerial(uint64_t serial) { markComplete(serial, nil); }

    // Commits a finished command buffer recorded under `serial`.
    // `drawable` (optional) is presented after the command buffer.
    void submit(id<MTL4CommandBuffer> cb, id<MTL4CommandAllocator> alloc, uint64_t serial,
                id<CAMetalDrawable> drawable = nil);

    id<MTL4CommandAllocator> acquireAllocator();
    id<MTL4CommandBuffer> newCommandBuffer();

    // --- classic Metal (tessellation) ------------------------------------------
    // Metal 4 has no tessellation; tessellated draws run on this classic queue between Metal 4 command buffers
    // (Context::openTessSegment). nil if it can't be created.
    id<MTLCommandQueue> classicQueue();
    // Commits a classic command buffer (after the residency set's pending changes); `serial` (0: none) completes
    // with it.
    void commitClassic(id<MTLCommandBuffer> cb, bool render, uint64_t serial);
    // On the Metal 4 queue: signal `e` with `value` once the work committed so far is done / make the work
    // committed from now on wait until `e` reaches `value`.
    void queueSignal(id<MTLEvent> e, uint64_t value);
    void queueWait(id<MTLEvent> e, uint64_t value);

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

    // --- sparse resources ---------------------------------------------------
    // Placement sparse textures and buffers are backed by kSparsePageSize tiles taken
    // from a pool of placement heaps. Mapping updates are queue operations: callers
    // submit the work that must see the old mapping first; later encoders start with
    // a queue barrier that covers MTLStageResourceState.
    static constexpr MTLSparsePageSize kSparsePageSize = MTLSparsePageSize16;
    size_t sparseTileBytes = 0;  // bytes per tile (0: sparse resources unsupported)
    std::atomic<int> sparseTextures{0};  // live sparse textures (draws only look for sparse images while > 0)
    bool allocSparseTiles(uint32_t count, SparseTiles& out);
    // The tiles may be reused once `serial` (the last work that could reference them) completed.
    // Tiles unmapped from a live resource are also held back for a while: the GPU keeps using
    // stale translations of an unmap until it idles, so a quick reuse could alias them.
    void freeSparseTiles(const SparseTiles& t, uint64_t serial, bool unmapped = false);
    // One mapping update: `region` (in tiles) of a texture level/slice, or `region.origin.x` /
    // `region.size.width` tiles of a buffer. Maps `tiles` (walked in x, y, z order) or unmaps
    // when `unmap` is set (`tiles` then names the tiles being unmapped).
    struct SparseOp {
        NSUInteger level = 0, slice = 0;
        MTLRegion region;
        SparseTiles tiles;
        bool unmap = false;
    };
    void updateSparseMappings(id<MTLTexture> tex, const std::vector<SparseOp>& ops);
    // The GPU keeps using stale translations of a sparse resource's address range for a while
    // after unmaps (until it idles): released sparse resources keep their range for
    // kSparseSettleTime so a new resource can't be placed there, and a remap of a range unmapped
    // less than kSparseSettleTime ago waits that out (sparse.mm).
    static constexpr double kSparseSettleTime = 0.5;
    void retainSparse(id obj, uint64_t serial);
    void updateSparseMappings(id<MTLBuffer> buf, const std::vector<SparseOp>& ops);

    // --- helpers ------------------------------------------------------------
    id<MTLBuffer> newBuffer(size_t size, MTLResourceOptions opts = MTLResourceStorageModeShared);
    // GL buffer storage, reallocated often (glBufferData orphaning, renames of busy buffers): buffers of power-of-two
    // sizes (256 B .. 4 MB) that return to a free list once the GPU is done with them (recycleBufferAfter), resident
    // all along. At least `size` bytes; the contents are undefined.
    id<MTLBuffer> newBufferRecycled(size_t size);
    void recycleBufferAfter(id<MTLBuffer> b, uint64_t serial);
    // GPU address of a zeroed, always resident buffer of kZeroBufferBytes for pointers a shader may follow
    // but that a draw has nothing for (never 0). Shaders that write through such a pointer by mistake
    // only change this scratch memory.
    static constexpr size_t kZeroBufferBytes = 64 << 10;
    uint64_t zeroBufferAddress();
    id<MTLLibrary> compileLibrary(const std::string& source, std::string* errorOut, bool fastMath = false);

    // Called periodically (e.g. on submit) to reap finished work.
    void collect();

private:
    Device() = default;
    bool init();

    void markComplete(uint64_t serial, NSError* error, double gpuStart = 0, double gpuEnd = 0);
    id<MTLCommandQueue> classicQueue_ = nil;
    bool classicQueueTried_ = false;
    std::map<uint64_t, std::pair<double, double>> gpuTimes_;
    double maxGpuSeconds_ = 0, lastLongWarning_ = 0;
    double gpuBusySeconds_ = 0, gpuBusyUntil_ = 0, tessGpuSeconds_[2] = {};

    std::mutex mutex_;
    static constexpr int kRecycleClasses = 15;           // 256 B << 0 .. 14
    static constexpr size_t kRecycleMaxBytes = 64u << 20;  // free + pending
    std::deque<std::pair<uint64_t, id<MTLBuffer>>> recyclePending_[kRecycleClasses];
    std::vector<id<MTLBuffer>> recycleFree_[kRecycleClasses];
    size_t recycleBytes_ = 0;
    std::condition_variable cv_;
    std::atomic<bool> background_{false};
    std::mutex bgMutex_;
    std::condition_variable bgCv_;
    uint64_t nextSerial_ = 1;
    uint64_t watermark_ = 0;              // all serials <= watermark_ are complete
    std::set<uint64_t> completedAbove_;   // completed serials > watermark_
    std::set<uint64_t> submitted_;        // submitted, not complete
    bool residencyDirty_ = false;
    uint64_t pendingResidentBytes_ = 0, pendingResidentCount_ = 0;  // added since the last commit
    void commitResidencyLocked();
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

    struct SparseHeap {
        id<MTLHeap> heap;
        std::vector<bool> used;  // per tile
        uint32_t freeCount;
        uint32_t hint;           // search start for single tiles
    };
    void reclaimSparseLocked();
    template <typename Res, typename Op>
    void submitSparseOps(Res res, const std::vector<SparseOp>& ops, Op makeOp);
    std::mutex sparseMutex_;
    std::vector<SparseHeap> sparseHeaps_;
    struct PendingTiles {
        uint64_t serial;
        double notBefore;  // CFAbsoluteTime
        SparseTiles tiles;
    };
    std::deque<PendingTiles> sparsePendingFree_;
    std::deque<std::pair<double, Deferred>> sparseRetired_;  // (release not before, object)
};

}  // namespace mgl
