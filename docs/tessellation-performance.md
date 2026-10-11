# Tessellation performance vs Apple's OpenGL

Findings from running Unigine Heaven 4.0 on both drivers (2026-10-10). Tessellation is where metalgldrv falls
furthest behind Apple's OpenGL; without it the two are close.

## Unigine Heaven 4.0

1920x1080 windowed, High quality, the benchmark's automatic mode (`-extern_define PHORONIX`); scripts in
`~/metalgldrv-deps/heaven` (`run.sh`, `bench.sh`). Before ed36142 / ae3dc5a, so it predates the tessellation work
below; re-run on a quiet machine before quoting it as current.

| Configuration | Apple GL | metalgldrv |
|---|---|---|
| Normal tessellation (2 interleaved rounds) | 102.5 / 103.0 fps | 34.4 / 35.2 fps |
| Same, `ARB_compute_shader` hidden (Apple's 4.1 lacks it) | — | 35.3 / 35.1 fps |
| Tessellation disabled (1 run each) | 123.6 fps | 111.0 fps |

* The first metalgldrv run (124.5 fps) is invalid: 176 shaders failed to link and most of the scene rendered black.
  Fixed in 812c143 and 4e100bc; check saved frames (`shot.sh`) before trusting a Heaven number.
* Per-pass GPU profiles (`METALGLDRV_PROFILE_PASSES=1`) put the cost in the tessellated shadow-map passes (about
  23 ms) and the tessellated G-buffer pass (about 17 ms).
* One frame has 848 draws, 224 of them tessellated, in 15 consecutive runs.

## displacebench

`tests/bench/displace_bench.cpp` runs Heaven's TCS / TES (frustum cull, levels from per-vertex weights times edge
length, `fractional_odd_spacing`, displacement by `textureLod` of a height map) on a 320 000-patch terrain grid.
It is built twice: `displacebench` links metalgldrv's CGL, `displacebench-apple` links OpenGL.framework.
`--weight W` sets the levels (0.5 gives one triangle per patch, 8 gives edges of up to 16 segments);
`--checksum` / `--dump` compare images. Reported numbers are GPU time per frame (GL_TIME_ELAPSED).

| GPU ms / frame | weight 0.5 | 2 | 4 | 8 |
|---|---|---|---|---|
| metalgldrv before a51164f | 16.5 | 66 | 68 | 179 |
| metalgldrv a51164f (parallel generation) | 17.5 | 39.3 | 40.8 | 63.4 |
| metalgldrv ed36142 (pieces, batching) | 11.4 | 26.8 | 26.9 | 47.7 |
| Apple GL | 5.7 – 5.9 | 6.3 | 5.0 – 7.4 | 9.4 |

The last three rows were measured back to back with nothing else of ours running but the Mac in use (WindowServer
busy), so expect noise; an earlier quiet measurement of Apple GL gave 1.7 / 2.5 / 3.0 / 7.0. Without
tessellation both drivers take about 0.25 ms. Apple's time barely grows with the level; ours is dominated by
fixed per-patch and per-threadgroup costs.

## Where the time went (and what was fixed)

Metal 4 has no tessellation (`MTL4RenderCommandEncoder` has no `drawPatches`, `MTL4RenderPipelineDescriptor` no
tessellation state; the concrete encoder class, `AGXG16XFamilyRenderContext_mtlnext`, doesn't implement the
classic encoder's patch methods either). metalgldrv emulates it with an object + mesh pipeline
(docs/ARCHITECTURE.md, "Tessellation").

Measured with `METALGLDRV_MSL_OVERRIDE` edits to the generated MSL:

1. **Serial domain generation** (fixed in a51164f): thread 0 of each mesh threadgroup generated every domain point
   and index while 31 threads waited, about 80% of the cost. Now each thread generates its vertices / primitives
   (`mgl_tess_unit_vertex` / `_prim`, a merge-path search for strip triangulation), bit-identical to the serial
   generator (`glslrun --tesscheck`).
2. **Mesh threadgroup occupancy** (ed36142): threadgroups sized for 130 vertices / 128 primitives left few in
   flight; capping them at 34 vertices took weight 2 from 41 to 25 ms. Units are now cut into pieces of at most
   32 primitives.
3. **Object stage utilisation** (ed36142): one object threadgroup of 3 threads per patch. Up to 3 patches now
   share one (when the VS / TCS have no side effects); larger batches were slower because of their payloads.
4. Not the cost: the TES itself (removing its texture fetch changed nothing), struct indexing, rasterisation.

Floors left: an empty mesh stage with one threadgroup per patch already costs about 12 ms at 320 000 patches;
larger object payloads reduce throughput.

## Correctness notes

* Primitive order across mesh threadgroups isn't guaranteed (true before this work too): without a depth test,
  two runs of the same build give different images. Depth ties and blending within one tessellated draw can
  resolve differently from GL's order; opaque, depth-tested scenes differ in a handful of pixels.
* With order-independent output (additive fragment counts), every build gives identical images, so the
  emulation's geometry is unchanged by these commits.
* ae3dc5a: the driver now keeps at most 3 frames queued on the GPU. Before, nextDrawable was the only throttle;
  when it didn't block (display asleep, window hidden), a GPU-bound Heaven queued command buffers without limit
  (20 GB of Metal allocations, 10 s waits, GPU faults).

## Reverse engineering of Apple's system OpenGL driver (2026-10-10)

Analysis of `AppleMetalOpenGLRenderer.bundle` (macOS 27.0, M4 Pro) and `libGLVMPlugin.dylib` via IDA Pro
decompilation. All references are to the arm64e slice.

### Architecture: classic Metal tessellation, not compute emulation

Apple's GL driver uses **classic Metal `drawPatches`** (from `Metal.framework`, not Metal 4). The binary links
`Metal.framework` and calls `setTessellationFactorBuffer:offset:instanceStride:`,
`setTessellationPartitionMode:`, `setMaxTessellationFactor:`, `drawPatches:patchStart:patchCount:...`.
Metal 4 has no `drawPatches` — Apple's GL driver simply doesn't use Metal 4.

### Pipeline structure

Two Metal pipeline objects per tessellated program:

1. **VS+TCS compute pipeline** (`copyVSCSComputeDescriptor`). The vertex shader and tessellation control shader
   are compiled into a single compute kernel by `glvmBuildCPTransformFunction` in libGLVMPlugin. The compute
   pipeline descriptor has:
   - `setOpenGLModeEnabled: YES` — private API that tells the Metal compiler to handle GL vertex fetch via
     `MTLStageInputOutputDescriptor` (stage-in).
   - Function constants embed the VS data, TCS data, and pipeline key into the Metal function variant.

2. **TES+FS render pipeline** (`copyESFSRenderDescriptor`). The TES runs as the **vertex function** of a
   `drawPatches` render pipeline. Configuration:
   - `setTessellationPartitionMode:` from the GL shader (`equal_spacing`, `fractional_even`, `fractional_odd`).
   - `setMaxTessellationFactor: 64` (hardcoded; GL's max is 64).
   - `setTessellationFactorFormat: MTLTessellationFactorFormatHalf` (0).
   - `setTessellationControlPointIndexType: MTLTessellationControlPointIndexTypeNone` (0).
   - `setTessellationFactorScaleEnabled: NO`.
   - `setTessellationOutputWindingOrder:` from the GL shader.
   - Vertex descriptor step functions: `MTLVertexStepFunctionPerPatchControlPoint` (3) for per-control-point
     data, `MTLVertexStepFunctionPerPatch` (4) for per-patch data (`fillIntermediateVertexDescriptor`).

### Draw path for tessellation (non-GS)

Flow in `gldRenderVertexArray` (39 KB, ~8800 lines of pseudocode):

1. **Allocate intermediate buffers** from a free-list pool (`GLRBufferResourceFreePool`):
   - TCS output buffer: `getBufferResourceFromFreePool(patches * controlPoints * stride, MTLStorageModePrivate)`.
   - Tessellation factor buffer: `getBufferResourceFromFreePool(8 or 12, MTLStorageModeShared)` for uniform
     levels, or a GPU-private buffer when a TCS writes per-patch factors.
   - GS buffers (if needed): intermediate vertices, prefix sum, counts.

2. **Dispatch TCS compute** on a **separate preprocessing command buffer**:
   - `startTransformFeedbackComputeCommandEncoderIfNeeded` creates a compute encoder on a preprocessing
     `MTLCommandBuffer` (created with `commandBufferWithUnretainedReferences`). This encoder is separate
     from the render encoder — calling it does NOT end the render pass (`endEncoding` only ends the
     preprocessing compute/blit encoders, not the render encoder at `GLDContextRec+64`).
   - `setStageInRegion:` sets the vertex fetch region.
   - `dispatchThreads:threadsPerThreadgroup:` dispatches the VS+TCS kernel.
   - **Threadgroup size**: `(0x20 / controlPointsPerPatch) * controlPointsPerPatch` = max 32, rounded down to
     a multiple of the control-point count. For 3-vertex triangles: 30 threads = 10 patches per threadgroup.
     For 4-vertex quads: 32 threads = 8 patches.
   - Total threads: `patches * controlPointsPerPatch * instances`.

3. **Set up the render encoder** (still active on the main command buffer):
   - `setVertexBuffer:offset:atIndex:` at index 1 (VS output) and 2 (TCS control-point output).
   - `setTessellationFactorBuffer:offset:instanceStride:` with the GPU-private factor buffer.

4. **Draw**: `drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:baseInstance:`.
   One call covers all patches. Patch count from `v1117 + 104` or `GLDContextRec + 18540`.

5. **Clear**: `setTessellationFactorBuffer:offset:instanceStride:` with `(nil, 0, 0)`.

6. **Separate back face**: if polygon mode differs front/back, `handleSeparateBackFill` + second `drawPatches`.

### For uniform tessellation (no TCS)

When tessellation levels are uniform (no TCS outputs per-patch factors), the driver writes half-precision
factors directly to a CPU-shared buffer:
- Triangles (v325 == 7): 12 bytes = 6 × half (inner[0], outer[0..2], plus two more).
- Quads (v325 == 4): 8 bytes = 4 × half (inner[0..1], outer[0..1]).
- Conversion: `vcvt_f16_f32` (NEON) and `FCVT H0, S0` from the GL state at `GLDContextRec + 17 * 8 + 4636 * 4`.
- No compute dispatch needed.

### Preprocessing command buffer model

- Created with `commandBufferWithUnretainedReferences` on the same `MTLCommandQueue` as the main CB.
- Method IMPs for `setBuffer:offset:atIndex:`, `setBytes:length:atIndex:`, etc. are pre-resolved and cached
  at context setup time to avoid per-call `objc_msgSend` dispatch overhead.
- Committed with **`commitAndHold`** (private Metal API) by `flushPreprocessingCommandEncoders`.
- Committed before the main command buffer, so Metal queue ordering guarantees the compute results are visible
  to the render pass.
- Multiple tessellated draws in a frame accumulate compute work on the same preprocessing encoder.
- No render pass breaks: the render encoder on the main CB stays alive while compute dispatches happen on the
  preprocessing CB.

### GS path (tessellation + geometry shader)

When a geometry shader follows tessellation, the compute path adds:
1. VS+TCS+GS combined compute kernel dispatch.
2. `parallelGSPrefixSumPass1` and `parallelGSPrefixSumPass2` — two built-in compute kernels for a GPU parallel
   prefix sum over GS output counts. Retrieved by `getParallelGSPrefixSum`, cached after first compilation.
3. GS output expansion kernel.
4. Buffers: intermediate vertex data buffer, prefix sum (2 × padded-to-1024 count × 2 bytes), index buffer
   (4 × primitives), and a 4-byte counter.
5. After the GS compute pass, `flushPreprocessingCommandEncoders` → `endRenderPass` → `flushContext`.
   The GS path DOES break the render pass and flush the context.

### Buffer management

- `GLRBufferResourceFreePool` is a free-list allocator. `getBufferResourceFromFreePool(size, storageMode)`
  grabs a buffer ≥ size; `releaseBufferResourceToFreePool` returns it.
- Command buffer completion handlers (`addCompletedHandler:`) return temporary buffers to the pool.
- Intermediate buffers are `MTLStorageModePrivate` (GPU-only). Tess factor buffers for uniform levels are
  `MTLStorageModeShared`.
- The GS path uses `allocBufferResource` from a different pool (the general buffer resource pool), and releases
  the prefix sum / index buffers immediately after the compute pass.

### Shader compilation

`libGLVMPlugin.dylib` compiles each GL stage independently:
- `glvmBuildVPTransformFunction` — vertex program (VS)
- `glvmBuildCPTransformFunction` — control point program (TCS, 46K chars decompiled)
- `glvmBuildEPTransformFunction` — evaluation point program (TES)
- `glvmBuildGPTransformFunction` — geometry program (GS)
- `glvmBuildFPTransformFunction` — fragment program (FS)
- `glpVertexStateToLLVMIR` / `glpFragmentStateToLLVMIR` compile to LLVM IR for Metal's toolchain.

### Why Apple's approach is faster: the key differences from metalgldrv

| Factor | Apple GL | metalgldrv |
|---|---|---|
| **Tessellator** | Metal's `drawPatches` (driver/HW) | Software in mesh shader (`mgl_tess_unit_vertex`, merge-path) |
| **TCS dispatch** | One compute dispatch, up to 10 patches/threadgroup | One compute dispatch, same |
| **Domain point generation** | Metal's tessellator, zero CPU/GPU work | Counted path: count kernel → **CPU wait** → prefix sum → generate kernel; mesh path: in mesh shader |
| **CPU-GPU sync** | None | `mpWait` blocks CPU to read back per-patch counts, builds serial prefix sum |
| **Draw call** | One `drawPatches` for all patches | `drawMeshThreadgroups` (chunked ≤ 65536 columns), with per-patch object threadgroups |
| **Output limits** | Metal tessellator has none | Mesh shader: MAXV/MAXP limits force splitting patches into pieces of ≤ 32 prims |
| **Render pass** | Never broken (preprocessing CB) | Not broken (mesh pipeline), but mesh overhead still applies |
| **Buffer allocation** | Free-list pool, buffers recycled | `newBuffer` per draw (`mpScratch`), no pool |
| **Primitive order** | Metal's tessellator preserves GL order | Mesh threadgroups give unordered output within a draw |

The three largest contributors to the performance gap:

1. **CPU-GPU sync** (`mpWait` + serial prefix sum): metalgldrv's counted tessellation path submits a count
   kernel, stalls the CPU to read back patch primitive counts, runs a serial prefix sum on the CPU, then
   submits the generate kernel. Apple's driver has no CPU-GPU sync at all — the tess factor buffer goes
   directly from the TCS compute to `drawPatches`.

2. **Per-patch threadgroup launch overhead**: metalgldrv's mesh pipeline launches one object threadgroup per
   1–3 patches, each of which generates mesh threadgroups for that patch's tessellation units. Apple calls
   `drawPatches` once for all 320 000 patches in displacebench. The per-threadgroup launch, payload transfer,
   and synchronization cost dominates at high patch counts (the "empty mesh stage with one threadgroup per
   patch already costs 12 ms" floor in the notes above).

3. **Software domain-point generation**: the merge-path search in `mgl_tess_unit_vertex` / `_prim` generates
   domain coordinates and triangulation indices in the mesh shader. Metal's tessellator does this internally,
   likely at lower cost (dedicated hardware or a highly optimized driver path).

## Ideas not yet tried (updated)

* **Classic Metal `drawPatches` on a separate command queue.** Apple's GL driver proves this works on current
  hardware and OS. Use a classic-Metal `MTLCommandQueue` for tessellated draws; synchronize with the Metal 4
  queue via `MTLSharedEvent`. Tessellated draws break the Metal 4 render pass once per frame (not per draw)
  to switch queues. The classic-Metal side runs VS+TCS compute → `drawPatches` in a single command buffer;
  this avoids the CPU-GPU sync, the mesh output limits, and the per-patch threadgroup overhead. All three of
  the dominant costs disappear.

  Caveats: Metal's tessellation follows Direct3D's rules (isolines/point-mode/strict profiles need the
  existing emulation); two command queues contend for GPU resources; and the classic-Metal render pass must
  render to the same textures as the Metal 4 pass (verify `MTLSharedEvent` ordering with textures attached
  to both queues).

* **GPU prefix sum to eliminate the CPU sync.** Replace `mpTessRecords`'s CPU readback and serial prefix sum
  with a GPU parallel prefix sum (two compute dispatches, like Apple's `parallelGSPrefixSumPass1/Pass2`),
  followed by `mgl_tes_gen` with the GPU-computed offsets. The generate kernel then feeds an indirect indexed
  draw (`drawIndexedPrimitives:indexType:indexBuffer:indexBufferOffset:indirectBuffer:indirectBufferOffset:`)
  or continues into the mesh pipeline. This removes the CPU stall but keeps the mesh overhead.

* **Buffer pool for tessellation scratch.** Replace `mpScratch`'s `newBuffer` with a free-list pool
  (ring buffer with fence-gated reuse) to avoid per-draw allocation overhead and reduce heap fragmentation.

* **Pattern instancing** (bin patches by integer levels, draw each pattern instanced): fast for equal spacing,
  too many patterns for fractional spacing.

## Private / SPI Metal APIs in Apple's GL driver

Apple's `AppleMetalOpenGLRenderer.bundle` uses ~30 private Metal selectors (System Programming Interfaces,
not in the public SDK headers). These implement GL features that Metal's public API either lacks entirely or
exposes differently. All identified by decompiling the arm64e slice in IDA Pro; arguments and semantics inferred
from the decompiled call sites.

A compile-time flag (`METALGLDRV_USE_PRIVATE_API` or similar) could gate use of these. They work on macOS with
SIP intact (Apple's own driver calls them), but would be rejected from App Store review.

### Queue and command buffer

| Selector | Target | Arguments | What it does |
|---|---|---|---|
| `setIsOpenGLQueue:` | `MTLCommandQueueDescriptor` | `BOOL` (always YES) | Marks the queue as OpenGL-mode. Set during context init alongside `setMaxCommandBufferCount:32`. Likely affects scheduling, priority, or enables other GL-mode selectors on encoders created from this queue. |
| `commitAndHold` | `MTLCommandBuffer` | none | Commits the command buffer but "holds" it — the GPU begins executing, but the CB's position in the queue is reserved so a subsequent CB on the same queue is guaranteed to run after. Apple uses this to commit the preprocessing (TCS compute) CB before the main render CB without breaking the render pass. The completed handler fires normally. |
| `commitAndWaitUntilSubmitted` | `MTLCommandBuffer` | none | Synchronous commit: blocks the CPU until the CB is submitted to the GPU (not until completed). Used in `glFlush`-like paths, texture/buffer upload synchronization, and context flush. Nine call sites in the driver. |

### Render pipeline descriptor (set at PSO creation time)

| Selector | Arguments | What it does |
|---|---|---|
| `setOpenGLModeEnabled:` | `BOOL` (always YES) | Set on render pipeline, compute pipeline, AND render pass descriptors. On the render pipeline, enables GL vertex fetch semantics through the vertex descriptor. On the compute pipeline, enables `MTLStageInputOutputDescriptor`-based vertex fetch for the VS+TCS kernel. On the render pass descriptor, enables GL rasterization mode (point coord flipping, dithering path, etc.). |
| `setLogicOperationEnabled:` | `BOOL` | Enables GL logic operations (AND, OR, XOR, NAND, etc.) in the fragment output. Pipeline key bit 56. |
| `setLogicOperation:` | `uint64_t` (4 bits, 0–15) | The logic operation enum, from bits 52–55 of the pipeline key. Maps to GL's 16 `glLogicOp` values (`GL_CLEAR`=0 through `GL_SET`=15). |
| `setPointCoordLowerLeft:` | `BOOL` | Whether `gl_PointCoord` has (0,0) at lower-left (GL convention) vs upper-left (Metal default). Pipeline key bit 5 at offset 301. |
| `setPointSizeOutputVS:` | `BOOL` | Whether the vertex shader outputs point size (`gl_PointSize`). Pipeline key bit 6 at offset 301. |
| `setDepthStencilWriteDisabled:` | `BOOL` | Disables depth and stencil writes at the pipeline level (optimization when the GL state has depth/stencil writes off). Pipeline key bit 40. |
| `setSampleCoverage:` | `float` | GL's `glSampleCoverage` value, passed as a float to the pipeline descriptor. |
| `setSampleCoverageInvert:` | `BOOL` | Whether sample coverage is inverted (`GL_SAMPLE_COVERAGE_INVERT`). Pipeline key bit 45. |
| `setSampleMask:` | `uint32_t` | GL's `glSampleMaski` value — a bitmask of which MSAA samples are active. |
| `setNeedsCustomBorderColorSamplers:` | `BOOL` | Tells the pipeline that at least one sampler uses a custom border color (from `setBorderColorSPI:`), enabling whatever internal fixup Metal needs for non-standard border colors. Set when pipeline key bits indicate custom border colors or `GL_CLAMP_TO_BORDER` wrapping. |

### Render command encoder (set per-draw)

| Selector | Arguments | What it does |
|---|---|---|
| `setProvokingVertexMode:` | `uint64_t` | Controls which vertex of a primitive provides flat-shaded attributes. Value from `GLDContextRec+5768*8`, mapping GL's `GL_FIRST_VERTEX_CONVENTION` / `GL_LAST_VERTEX_CONVENTION`. Protected by a device feature flag check (bit 3 at offset 130 of device caps) — if unsupported, warns once and falls back. |
| `setPrimitiveRestartEnabled:index:` | `BOOL enabled, uint32_t index` | Enables primitive restart with a configurable restart index. Checked with `gldMTLRespondsToSelector:` first. If unsupported and non-default (not enabled=YES,index=0xFFFFFFFF), warns once. Eliminates the need to split draw calls at restart indices. |
| `setViewportTransformEnabled:` | `BOOL` | Disables Metal's built-in viewport transform. Set to `NO` when `glClipControl(GL_ZERO_TO_ONE)` is active (Metal offset 249 == non-zero). Used in viewport setup and blit paths (3 call sites). Allows the shader to handle the NDC→viewport transform itself, needed for GL's clip control extension. |
| `setLineWidth:` | `float` | Sets line width for line primitives. Metal's public API only supports 1px lines. The value comes directly from `glLineWidth` state (GLDContextRec offset 11530). |

### Render pass descriptor

| Selector | Arguments | What it does |
|---|---|---|
| `setOpenGLModeEnabled:` | `BOOL` (always YES) | See above — also set on the render pass descriptor. |
| `setPointCoordYFlipEnabled:` | `BOOL` (always YES) | Flips point sprite Y coordinates to match GL's convention (lower-left origin for `gl_PointCoord`). Set alongside `setOpenGLModeEnabled:` in `buildRenderPassDescriptor`. |
| `setDitherEnabled:` | `BOOL` | GL's `glEnable(GL_DITHER)` state, from GLDContextRec offset 612. |
| `setDefaultRasterSampleCount:` | `uint64_t` | The MSAA sample count for the render pass when no explicit MSAA attachments are present. From GLDContextRec offset 29. |
| `setYInvert:` | `BOOL` (always 0) | Per-color-attachment Y inversion. Hardcoded to NO in `buildRenderPassDescriptor`, but set to YES in some `gldBlitFramebufferData` paths (3 call sites). Controls whether the attachment's content is Y-flipped during load/store. |
| `setDepthCleared` / `setStencilCleared` | (on attachment) | Marks depth/stencil attachments as pre-cleared, skipping the hardware clear. One call site each in `gldClearFramebufferData`. |

### Texture descriptor

| Selector | Arguments | What it does |
|---|---|---|
| `setWriteSwizzleEnabled:` | `BOOL` (always YES) | Set on `MTLTextureDescriptor` during texture creation (`GLDTextureRec::load`). Enables write-side component swizzling — when the GPU writes to this texture (render target, compute write), components are swizzled according to the texture's swizzle state. Six call sites across three texture load paths. Without this, texture writes go to the "wrong" channels for formats where GL and Metal disagree on component order (e.g. `GL_BGR`). |

### Sampler descriptor

| Selector | Arguments | What it does |
|---|---|---|
| `setBorderColorSPI:` | `uint64_t` (always 3 = custom) | Sets the border color mode to "custom", beyond Metal's public `MTLSamplerBorderColor` enum (which only offers transparent-black, opaque-black, opaque-white). Value 3 means "use the custom values below". Guarded by device feature flag `0x80`. |
| `setCustomBorderColorValue_0:` through `_3:` | `uint32_t` each (raw bits) | Four components of the custom border color, passed as raw `uint32_t` (reinterpreted as float). Called after `setBorderColorSPI:3`. Also calls `setPixelFormat:` on the sampler to tell Metal how to interpret the values. |

### Compute command encoder

| Selector | Arguments | What it does |
|---|---|---|
| `enableNullBufferBinds:` | `BOOL` | Allows binding nil buffers to the compute encoder without a validation error. Used in `executeKernel`. |

### Metal function creation

| Selector | Target | Arguments | What it does |
|---|---|---|---|
| `newFunctionWithGLCoreIR:inputsDescription:functionType:` | `MTLDevice` | `dispatch_data_t IR, id inputsDescription, MTLFunctionType type` | Creates a `MTLFunction` from pre-compiled GL Core IR (produced by `libGLVMPlugin.dylib`'s GLVM compiler). The IR is LLVM bitcode in Apple's internal format — not usable by metalgldrv since we produce MSL, not GLVM IR. |

### Usefulness for metalgldrv

**High value — eliminates shader emulation or unblocks features:**

1. **`setProvokingVertexMode:`** — metalgldrv currently emulates provoking vertex in the vertex shader by
   reordering attributes. The private API does it in hardware. Worth checking if it works on Metal 4 encoders.

2. **`setPrimitiveRestartEnabled:index:`** — metalgldrv emulates primitive restart by scanning index buffers
   on the CPU and splitting draws. Direct hardware support eliminates the scan and the extra draw calls.

3. **`setLogicOperation:` / `setLogicOperationEnabled:`** — metalgldrv either doesn't support GL logic ops or
   emulates them with shader-based blending. Hardware logic ops on the pipeline descriptor is the correct fix.

4. **`setLineWidth:`** — Metal's public API is 1px only. This enables `glLineWidth` > 1. Without it, apps that
   rely on thick lines (CAD, debug visualization) render incorrectly.

5. **`setBorderColorSPI:` + `setCustomBorderColorValue_0-3:`** — GL supports arbitrary RGBA border colors via
   `glTexParameterfv(GL_TEXTURE_BORDER_COLOR, ...)`. Metal's public API only has black/white/transparent.
   The SPI enables full GL border color support.

6. **`commitAndHold` + `setIsOpenGLQueue:`** — critical for the "classic Metal tessellation" path described in
   "Ideas not yet tried". The preprocessing command buffer model requires `commitAndHold` to commit the TCS
   compute CB without breaking the render pass.

**Medium value — correctness or minor optimization:**

7. **`setViewportTransformEnabled:`** — needed for correct `GL_ARB_clip_control` with `GL_ZERO_TO_ONE` depth
   range. Without it, the viewport transform double-applies.

8. **`setWriteSwizzleEnabled:`** — may be needed for correct `glTexSubImage2D` with swizzled formats (BGR, etc.)
   where Metal and GL disagree on component order.

9. **`setPointCoordLowerLeft:` / `setPointCoordYFlipEnabled:`** — correct `gl_PointCoord` Y direction without
   shader fixups.

10. **`setDitherEnabled:`** — trivial to use if available; without it, `glEnable(GL_DITHER)` is a no-op.

11. **`setSampleCoverage:` / `setSampleMask:`** — proper MSAA coverage control.

**Low value — not applicable or niche:**

12. **`newFunctionWithGLCoreIR:`** — requires Apple's GLVM IR format. We compile GLSL→MSL, not GLSL→GLVM IR.
13. **`commitAndWaitUntilSubmitted`** — synchronous, worse than our existing sync paths.
14. **`enableNullBufferBinds:`** — minor convenience.
15. **`setOpenGLModeEnabled:`** — unclear if it has any effect without the rest of Apple's GL plumbing.
