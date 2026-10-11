# Tessellation performance vs Apple's OpenGL

Findings from running Unigine Heaven 4.0 on both drivers (2026-10-10). Tessellation is where metalgldrv fell
furthest behind Apple's OpenGL; without it the two are close.

## Current state: Metal's tessellator (classic `drawPatches`)

Tessellated draws now go to classic Metal's tessellator, as Apple's GL driver does it (reverse engineering below):
the `mgl_tcs` kernel runs VS + TCS and writes Metal's half-precision factors, `drawPatches` runs the TES as the
post-tessellation vertex function. Metal 4 command buffers can't encode `drawPatches`, so these draws run in
segments on a classic `MTLCommandQueue` between Metal 4 command buffers, ordered by an `MTLEvent`
(docs/ARCHITECTURE.md, "Tessellation"). Metal's tessellator is implemented in Apple's driver, not in
fixed-function hardware (Apple GPUs have no hardware tessellator); it is still far faster than our mesh-shader
emulation, which stays for what it can't do (isolines, point_mode, geometry shaders, transform feedback, queries,
...). `METALGLDRV_TESS_METAL=0` turns it off.

Heaven, back to back on the same (busy) machine: Apple GL 78.4 fps, metalgldrv 73.9 fps (was 35 fps with the
emulation; Apple GL measured 103 fps on a quieter machine earlier).

displacebench, GPU ms per frame, back to back:

| | weight 0.5 | 2 | 4 | 8 |
|---|---|---|---|---|
| Metal's tessellator | 3.5 | 4.4 | 4.5 | 7.6 |
| emulation (`METALGLDRV_TESS_METAL=0`) | 8.5 | 23.0 | 23.3 | 48.8 |
| Apple GL | 1.7 | 2.5 | 2.9 | 6.2 |

The remaining gap is a fixed cost per patch (the VS + TCS kernel, the TES reading its control points from device
memory instead of stage-in). Differences from the emulation: Metal places domain points in fixed point (~1e-4 off
for equal spacing), chooses its own lengths for the short segments of fractional spacing (GL leaves them to the
implementation) and its own triangulation, so images differ in a few percent of pixels by small amounts;
`glslrun --tesscheck` and `mgltest tessellation_metal_tessellator` check the differences stay within that.

What it took besides the shaders:

* Metal reads factors in half precision: levels round *up* to a half (35.011 would otherwise become 35.0, one
  segment short).
* `tessellationFactorStepFunction` defaults to constant (every patch reads patch 0's factors).
* `upload()` kept its current 4 MB chunk only until the next flush, and every segment flushes: Heaven allocated
  ~4 GB/s of new buffers. The current chunk now lives across flushes, large requests get their own power-of-two
  buffers and the retired pool is bounded by bytes (256 MB).

The sections below describe the emulation as measured before this change.

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
   domain coordinates and triangulation indices in the mesh shader. Metal's tessellator does this internally, in
   Apple's driver (not fixed-function hardware), at much lower cost.

## Ideas not yet tried (updated)

* **Classic Metal `drawPatches` on a separate command queue.** Done (see "Current state" at the top). Apple's GL driver proves this works on current
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

metalgldrv gates the ones it uses (src/mtl/spi.h) behind the CMake option `MGL_PRIVATE_API` (on by default; turn
it off for App Store builds, as review rejects private API) and, at run time, `METALGLDRV_MORALLY_WRONG=0`.
Tested on macOS 27 / M4 Pro, most of them exist only on classic Metal objects, not on the Metal 4 encoders and
descriptors metalgldrv records with:

| Selector | On Metal 4 objects | Used |
|---|---|---|
| `setLineWidth:` | yes (`MTL4RenderCommandEncoder`) | yes: `glLineWidth`, 1..16 like Apple's GL (`GL_ALIASED_LINE_WIDTH_RANGE`), exact row counts in `mgltest wide_lines` |
| `setIsOpenGLQueue:` | no (classic queue descriptor only) | yes, on the tessellation queue, as Apple's driver; no measured effect |
| `setBorderColorSPI:` + `setCustomBorderColorValue_N:` + `setPixelFormat:` | yes (sampler descriptor) | no: the color only appears with directly bound samplers (`setSamplerState:`), never through argument buffers, which is how all metalgldrv samplers are bound; integer formats read 0 either way |
| `commitAndHold`, `commitAndWaitUntilSubmitted` | no (classic command buffers only) | no: the tessellation segments order their command buffers with an event |
| `setPrimitiveRestartEnabled:index:`, `setProvokingVertexMode:`, `setViewportTransformEnabled:` | no | no |
| pipeline / render pass `setOpenGLModeEnabled:`, `setLogicOperation*:`, `setPointCoordLowerLeft:`, `setSampleCoverage*:`, `setDitherEnabled:` | no (classic descriptors only) | no |

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

12. **`newFunctionWithGLCoreIR:`** — requires Apple's GLVM IR format. We compile GLSL→MSL, not GLVM IR.
13. **`commitAndWaitUntilSubmitted`** — synchronous, worse than our existing sync paths.
14. **`enableNullBufferBinds:`** — minor convenience.
15. **`setOpenGLModeEnabled:`** — unclear if it has any effect without the rest of Apple's GL plumbing.

## General private Metal APIs (not GL-specific)

Beyond the OpenGL-emulation SPIs above, the Metal runtime exposes private APIs on its descriptor, encoder,
command buffer, and command queue classes that could improve metalgldrv performance. Discovered by dumping
ObjC methods from the loaded Metal runtime classes (`MTL*Internal`, `AGXG16XFamily*`, `MTL3On4*`,
`MTL4*Descriptor`) after forcing the GPU plugin to load. All verified present on the M4 Pro (AGXG16S).

### Device feature queries (SPI)

`MTLDevice` (via `_MTLDevice` and `AGXG16XFamilyDevice`) exposes ~250 `supports*` selectors. Most correspond
to public `MTLGPUFamily` feature sets, but many are private. All return YES on M4 Pro.

Key private capabilities verified on M4 Pro:
- `supportsCustomBorderColor` — custom sampler border colors (see GL SPI section above)
- `supportsPrimitiveRestartOverride` — hardware primitive restart with arbitrary index
- `supportsDepthBoundsTesting` — depth bounds test (`GL_EXT_depth_bounds_test`)
- `supportsDepthClipModeClampExtended` — extended depth clamp behavior
- `supportsTessellation` — classic Metal tessellation (drawPatches)
- `supportsFP32TessFactors` — FP32 tessellation factors (vs the default FP16)
- `supportsIndirectTessellation` — indirect drawPatches
- `supportsConcurrentComputeDispatch` — concurrent compute execution
- `supportsDeadlineProfile` — deadline-based GPU scheduling
- `supportsCommandBufferJump` — command buffer jumping (conditional execution?)
- `supportsForkJoin` — fork/join in command buffers
- `supportsCommandQueueBarriers` — cross-queue barriers
- `supportsLateEvalEvent` — late-evaluated events
- `supportsOpenCLTextureWriteSwizzles` — texture write swizzles
- `supportsPartialRenderMemoryBarrier` — partial (non-full) render memory barriers
- `supportsStencilFeedback` — read stencil buffer in fragment shader
- `supportsSIMDGroupParallelForwardProgress` — forward progress for SIMD groups

### Compute pipeline descriptor (MTL4ComputePipelineDescriptor)

These are directly applicable to metalgldrv since it uses Metal 4 compute pipelines.

| Selector | Arguments | What it does |
|---|---|---|
| `setContentionRelief:` | `uint64_t` | Reduces GPU contention when multiple compute pipelines compete for resources. Likely a hint to the scheduler about how aggressively to timeslice. Applicable to metalgldrv's TCS and prefix-sum compute dispatches that run alongside rendering. |
| `setOptimizeForPersistentKernel:` | `BOOL` | Tells Metal that this kernel will be dispatched many times and should be kept resident in the GPU's instruction cache. Applicable to metalgldrv's TCS compute, which dispatches per tessellated draw. |
| `setForwardProgressUsage:` | `uint64_t` | Controls forward progress guarantees for the compute pipeline. A lower setting may allow more aggressive GPU scheduling at the cost of starvation risk. |

### Compute command encoder

| Selector | Metal ver. | Arguments | What it does |
|---|---|---|---|
| `setParallelExecution:` | Classic | `BOOL` | Enables concurrent execution of compute dispatches within the same encoder. When YES, dispatches that don't depend on each other can run simultaneously on different GPU shader cores. Could improve utilization for independent tessellation compute passes. |
| `setThreadgroupDistributionMode:` | Classic | `uint64_t` | Controls how threadgroups are distributed across GPU cores. Different modes may favor occupancy, locality, or load balancing. Could tune mesh shader / tessellation compute threadgroup scheduling. |
| `setThreadgroupDistributionModeWithClusterGroupIndex:` | Classic | `uint64_t` | Same, but targets a specific cluster group on multi-cluster GPUs (M4 Pro has multiple GPU clusters). |
| `enableNullBufferBinds:` | Classic | `BOOL` | Allows binding nil buffers without validation errors. |
| `beginVirtualSubstream` / `nextVirtualSubstream` / `endVirtualSubstream` | Classic | — | Virtual substreams for partitioning compute work within an encoder. |

### Command queue descriptor

| Selector | Metal ver. | Arguments | What it does |
|---|---|---|---|
| `setQosLevel:` | Both | `uint64_t` | Quality-of-service level for the command queue. Higher QoS gets priority GPU scheduling. Could prioritize the render queue over background compute. |
| `setDisableCrossQueueHazardTracking:` | Classic | `BOOL` | Disables Metal's automatic hazard tracking between queues. When metalgldrv manages queue synchronization manually (via events/fences), disabling the driver's tracking eliminates overhead. |
| `setDisableIOFencing:` | Classic | `BOOL` | Disables IO fencing on the queue. May reduce overhead when IO ordering isn't needed. |
| `setEnableLowLatencySignalSharedEvent:` | Classic | `BOOL` | Enables low-latency signaling of shared events from this queue. Reduces the latency when one queue signals another via `MTLSharedEvent`, relevant for the two-queue tessellation architecture. |
| `setEnableLowLatencyWaitSharedEvent:` | Classic | `BOOL` | Enables low-latency waiting on shared events. The wait-side counterpart of the above. |
| `setDevicePartition:` | Classic | `uint64_t` | Partitions GPU resources for this queue. Could isolate compute from render on multi-cluster GPUs. |

### Command queue (runtime)

| Selector | Arguments | What it does |
|---|---|---|
| `setGPUPriority:` | `int64_t` | Sets the GPU scheduling priority for this queue's work. Higher priority = more GPU time relative to other queues. Could ensure the render queue isn't starved by compute. |
| `setGPUPriority:offset:` | `int64_t, int64_t` | Sets priority with an offset — finer control than the base priority. |
| `resolveAndUpdateQueuePriority:` | `int64_t` | Resolves and updates the queue's effective priority. |

### Command buffer

| Selector | Arguments | What it does |
|---|---|---|
| `commitWithDeadline:` | `uint64_t` (Mach absolute time) | Commits the command buffer with a deadline — the GPU scheduler attempts to complete it by the specified time. For frame-paced rendering, set the deadline to the next VSync to prioritize the frame over background work. Requires `supportsDeadlineProfile`. |
| `setDeadline:` | `uint64_t` | Sets the deadline before commit. |
| `encodeCacheHintTag:resourceGroups:count:` | `uint64_t tag, id* groups, uint64_t count` | Tags resources with a cache hint for the GPU's L2/system-level cache. Tells the GPU which resources will be reused soon and should stay cached. |
| `encodeCacheHintFinalize:resourceGroups:count:` | `uint64_t tag, id* groups, uint64_t count` | Finalizes a cache hint — tells the GPU the tagged resources are no longer needed in cache and can be evicted. Useful after a render pass finishes reading tessellation output buffers. |
| `encodeWaitForEvent:value:timeout:` | `id event, uint64_t value, uint64_t timeout` | Wait-with-timeout variant of `encodeWaitForEvent:value:`. Avoids indefinite GPU hangs if the signal never comes. |
| `commitAndHold` | — | See GL SPI section. Also applicable to non-GL use: commit a compute CB whose results feed a render CB without breaking the render pass. |

### Render pipeline descriptor (classic Metal, non-GL)

| Selector | Arguments | What it does |
|---|---|---|
| `setAlphaTestEnabled:` / `setAlphaTestFunction:` | `BOOL` / `MTLCompareFunction` | Hardware alpha test — the GPU discards fragments below the threshold before the fragment shader runs. Eliminates shader-based `discard` for alpha-tested geometry (foliage, fences in games). Only on classic Metal pipeline descriptors; not available on MTL4. |
| `setClipDistanceEnableMask:` | `uint32_t` | Bitmask enabling `gl_ClipDistance[0..7]`. On classic Metal, clip distances aren't a public API — the shader outputs them and the pipeline must enable the mask. |
| `setPointSmoothEnabled:` | `BOOL` | Enables smooth (antialiased) point rendering. |
| `setTextureWriteRoundingMode:` | `uint64_t` | Controls rounding mode for floating-point-to-integer texture writes (round-to-nearest-even vs truncation). |
| `setColorSampleCount:` | `uint64_t` | Separate color sample count (can differ from `rasterSampleCount` for custom MSAA resolve). |
| `setProfileControl:` | `uint64_t` | Pipeline profiling mode — likely controls GPU performance counter collection. |
| `setDriverCompilerOptions:` / `setGpuCompilerSPIOptions:` | `id` / `id` | Pass private options to Metal's shader compiler (driver-level or GPU compiler backend). Could control optimization levels, register pressure, or enable experimental codegen. |

### Render command encoder

| Selector | Metal ver. | Arguments | What it does |
|---|---|---|---|
| `setLineWidth:` | **Both** | `float` | Wide lines. Confirmed present on both `AGXG16XFamilyRenderContext` (classic) and `AGXG16XFamilyRenderContext_mtlnext` (Metal 4). metalgldrv can call this directly on its Metal 4 render encoder. |
| `setDepthTestMinBound:maxBound:` | Classic | `float, float` | Depth bounds test — fragments outside [min, max] are discarded before the fragment shader. Useful for `GL_EXT_depth_bounds_test` and as a general early-Z optimization. Only on classic Metal encoder. |
| `setPointSize:` | Classic | `float` | Set point size from the CPU side (vs outputting `[[point_size]]` from the vertex shader). Classic Metal only. |
| `setAlphaTestReferenceValue:` | Classic | `float` | Alpha test reference value, companion to `setAlphaTestEnabled:` on the pipeline. |
| `barrierAfterQueueStages:beforeStages:options:` | **Metal 4** | `uint64_t, uint64_t, uint64_t` | Cross-queue barrier on the Metal 4 encoder. Could synchronize a classic Metal tessellation queue with the Metal 4 render queue without MTLSharedEvent overhead. |

### Potential performance wins for metalgldrv

**Scheduling (frame pacing):**
- `commitWithDeadline:` with the next VSync time gives the GPU scheduler a clear priority signal.
  Combined with `setQosLevel:` on the command queue, this could reduce frame jitter and ensure
  render work preempts background compute (shader compilation, etc.).

**Tessellation (two-queue architecture):**
- `setEnableLowLatencySignalSharedEvent:` / `setEnableLowLatencyWaitSharedEvent:` on the classic
  Metal queue reduces the sync latency between the TCS compute queue and the Metal 4 render queue.
- `setDisableCrossQueueHazardTracking:` eliminates Metal's per-resource tracking overhead when
  metalgldrv already handles synchronization via MTLSharedEvent.
- `barrierAfterQueueStages:beforeStages:options:` on the Metal 4 render encoder might replace
  MTLSharedEvent entirely for within-frame sync.
- `encodeCacheHintTag:` / `encodeCacheHintFinalize:` could keep tessellation output buffers hot
  in L2 between the compute and render passes.

**Compute (TCS / prefix sum):**
- `setOptimizeForPersistentKernel: YES` on the TCS compute pipeline, which dispatches hundreds
  of times per frame across tessellated draws.
- `setContentionRelief:` to reduce contention between TCS compute and concurrent render work.
- `setThreadgroupDistributionMode:` could improve occupancy for the mesh shader's small
  threadgroups (32 threads) by controlling how they spread across GPU clusters.

**Rendering features:**
- `setLineWidth:` on Metal 4 is the immediate win — enables `glLineWidth` with no fallback needed.
- `setAlphaTestEnabled:` (classic Metal only) avoids shader-based `discard` for alpha-tested
  geometry, which is a significant optimization since `discard` disables early-Z on Apple GPUs.
