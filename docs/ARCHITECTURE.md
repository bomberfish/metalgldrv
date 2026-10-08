# metalgldrv architecture

A standalone OpenGL 4.6 (core + compatibility) implementation on Metal 4, written
from scratch with no third-party code. Primary consumer: the Amethyst iOS
Minecraft launcher (LWJGL via EGL), targeting Sodium and Iris.

```
 app (LWJGL / GL)          EGL (src/egl)         GL frontend (src/gl)            Metal 4 (src/mtl + Metal.framework)
 gl* entry points  ──────► eglCreateWindowSurface ──► Context state, objects ────► MTL4CommandQueue / Allocator / ArgumentTable
                           CAMetalLayer                 draw/dispatch path          MTL4Compiler pipelines, residency set
                                                        GLSL compiler (src/glsl) ─► MSL source per pipeline variant
```

## Layout
| Path | Contents |
|---|---|
| `scripts/gen_glapi.py` | Generates `glapi.h`, weak stubs for every GL command, and the `*GetProcAddress` table from `registry/gl.xml`. Real implementations override the weak stubs at link time. |
| `src/egl` | EGL 1.5 over `CAMetalLayer` (window) and offscreen (pbuffer) surfaces. |
| `src/gl` | GL state machine, object model (`objects.h`), entry points (`api_*.mm`), draw path (`draw.mm`), internal shaders (clear/blit/present). |
| `src/mtl` | Metal 4 device (`device.mm`), GL↔Metal formats and pixel transfer conversion (`formats.mm`). |
| `src/glsl` | GLSL compiler: preprocessor → parser → semantic analysis → linker → MSL emitter. Public contract in `glsl.h`. |
| `src/glsl/runtime/fp64.metal` | Soft double-precision MSL library (embedded into shaders that use doubles). |
| `tests/host` | GL-level tests (run on macOS host and on the iPad through `tools/ios-runner`). |
| `tests/fp64` | Bit-exact GPU validation of the fp64 library against host IEEE doubles. |
| `demo/amethyst` | Patch set + build script for the Amethyst demo (Amethyst source lives outside the repo). |

## Coordinate conventions
* Every texture/renderbuffer stores GL row 0 in Metal row 0 (no data flipping
  on upload/readback). Vertex-processing stages negate `position.y`, so GL window
  y equals the Metal raster row: viewport/scissor rects pass through unchanged,
  `gl_FragCoord.y` is the Metal position y, and the only flip happens once when
  presenting the default framebuffer (`presentSurface`).
* The y negation inverts orientation, so the runtime maps `GL_CCW` front faces to
  `MTLWindingClockwise`.
* Clip z is remapped from [-1,1] to [0,1] in the shader unless
  `glClipControl(…, GL_ZERO_TO_ONE)`; `glDepthRange` maps onto `MTLViewport.znear/zfar`.

## Command recording & synchronization (Metal 4)
* One `MTL4CommandQueue` per process; each context records into a lazily begun
  `MTL4CommandBuffer` with a recycled `MTL4CommandAllocator`.
* Every command buffer reserves a **serial**; resources record the serial that
  last used/wrote them. Completion is reported per serial by `MTL4CommitFeedback`
  handlers (contexts may submit out of order), which also yields GPU timestamps
  used for timer queries.
* Metal 4 has no automatic hazard tracking. Every encoder starts with a
  conservative `barrierAfterQueueStages:All` barrier, which gives GL's in-order
  semantics; `glMemoryBarrier` maps to intra-encoder stage barriers.
* CPU writes to buffers the GPU may still read either **rename** the storage
  (small buffers; earlier draws keep the old `MTLBuffer`) or stage a GPU copy in
  stream order (large buffers). Texture uploads always stage through the upload
  ring and copy in stream order.
* All resources live in one `MTLResidencySet` attached to the queue.
* GL threads never drain an autorelease pool (glcts' main thread, the game's
  render thread), so driver code must not leave autoreleased objects behind:
  calls that autorelease (encoder creation, residency commits, pipeline and
  library compiles, object teardown) run in a local `@autoreleasepool`, buffer
  contents are read with `bufferContents()` (ARC autoreleases the receiver of
  `MTLBuffer.contents`), and texture descriptors come from
  `texture2DDescriptor()`/`textureBufferDescriptor()` (`mtl/device.h`).
* `glClear` is folded into render-pass load actions when nothing has been drawn
  yet; otherwise (scissor/masks/mid-pass) a quad clear is drawn.

## Binding model
Each linked program owns an *argument buffer* of 8-byte slots (layout chosen by
the linker, `glsl::Reflection::slots`): UBO/SSBO/atomic-counter GPU addresses,
texture & sampler `MTLResourceID`s per sampler uniform (element), image views,
the default-uniform-block address, SSBO sizes. Per draw the runtime fills a
fresh slot array from current GL bindings (`buildArguments`). Argument-table
buffer indices: 0 = slots, 1 = `MGLSys` (per-draw system values), 2..17 =
vertex buffer bindings (stage_in with dynamic strides), 30 = current generic
attribute values (constant step), 18..21 xfb, 22+ multi-pass scratch.

## Pipelines
`ProgramVariant` = MSL generated for a `glsl::VariantKey` (pipeline kind + data
the shader must bake in). `RenderPipelineKey` adds vertex formats, attachment
formats, blend state and sample count; depth/stencil formats are not part of
Metal 4 pipelines. All compiled through `MTL4Compiler`. Render pipelines are cached
on the variant they were built from, so they go away with the program (deleted
or relinked).

## Geometry shaders → mesh shaders (planned)
`PipelineKind::MeshGeometry`: no object stage; the mesh grid has one threadgroup
per (input primitive × instance × GS invocation). Threads fetch the primitive's
vertices (manual vertex pulling; formats baked into the variant) and run the VS
into threadgroup memory, then the GS body runs with `EmitVertex/EndPrimitive`
appending to the mesh output (≤256 vertices / 512 primitives per threadgroup,
which matches GL's limits); strips are converted to lists on the fly.
`gl_Layer` and `gl_ViewportIndex` map to per-primitive mesh outputs.

## Tessellation (planned, MoltenVK-style split without fixed function)
Metal 4 dropped `drawPatches`, so the fixed-function tessellator isn't
available. Following MoltenVK's stage decomposition: pass 1 is a compute kernel
running VS + TCS per patch, writing control points, patch constants and tess
levels to scratch buffers. Pass 2 is an object+mesh pipeline: the object stage
reads levels and sizes the work; mesh threadgroups generate domain points and
primitives with our own GL-rules tessellator (equal / fractional_even /
fractional_odd spacing, triangles / quads / isolines, point_mode, cw/ccw) and
run the TES per generated vertex.

## Sparse textures and buffers (ARB_sparse_texture(2), _clamp, ARB_sparse_buffer)
Advertised when the device supports placement sparse resources (`supportsPlacementSparse`;
M4 Mac and M5 iPad do). Sparse storage is a placement sparse texture/buffer; committing takes
16 KB tiles from a pool of placement heaps (`Device::allocSparseTiles`) and maps them with
`MTL4CommandQueue` mapping updates (`src/gl/sparse.mm`). Virtual pages are the standard 64 KB
shapes (whole Metal tiles); levels from `firstMipmapInTail` form the per-layer mip tail.
Sparse lookups use MSL `sparse_*` functions; `*ClampARB` maps to `min_lod_clamp`.
Metal quirks worked around:
* Writes to unmapped memory linger in the virtually tagged GPU caches: readbacks of sparse
  resources and accesses through aliases use `MTL4VisibilityOptionResourceAlias`; image
  loads/atomics on sparse textures check residency (`VariantKey::sparseImageSlots`).
* Residency of shader-writable textures with non-square tiles (2/8-byte texels) is wrong;
  sampling goes through a read-only twin mapped to the same tiles (`Texture::sparseRead`).
* The GPU keeps stale translations of an unmapped range until it idles (~0.5 s): released sparse
  resources keep their address range, unmapped tiles are quarantined, and recommitting a range
  right after decommitting it waits (`Device::kSparseSettleTime`). Decommitted texels may read
  old data during that window.
* Metal shaders can't write multisample textures, so sparse multisample textures are 2D arrays
  with a slice per sample, the layout of multisample image shadows: images bind them directly and
  samplers index them (`VariantKey::msArraySlots`). They can't be framebuffer attachments.
* Sparse buffers have no CPU mapping: CPU reads (index/indirect data, PBOs) copy just the range
  they use through the GPU (never the whole virtual size: flat-shaded indexed draws rotate the
  drawn indices, not the whole buffer), buffer textures sample a copy of the committed range.

## fp64
GLSL `double`/`dvecN` are `ulong`/`vec<ulong,N>` bit patterns in MSL; all
arithmetic is via `mgl_f64_*` (correctly rounded add/sub/mul/div/sqrt/fma,
subnormals supported). This follows metal-float64's idea of software IEEE
emulation on the integer ALUs. Planned: move heavy ops into an
`MTLDynamicLibrary`/binary functions to cut per-shader compile time.

## Testing
* `./build/tests/mgltest [filter]` – host GL tests (macOS, Metal 4).
* `scripts/ios-run.sh [filter]` – same suite on the connected iPad (builds,
  signs with the wildcard dev profile, installs, streams console output).
* `./build/tests/fp64test` – fp64 bit-exactness.
* `tools/glslc` / `tests/glsl` – compiler tests and real-world shader corpus.
* `METALGLDRV_STRICT=1 ./build/tests/mgltest`, `METALGLDRV_STRICT=2 ./build/tests/mgltest`
  – the same suite in strict and super strict mode (`tests/host/test_strict.cpp`
  and `test_fsr.cpp` cover the strict-only paths, `test_pdc.cpp` the super
  strict ones).

## Environment variables
Read by the driver at startup (`envFlag`/`envInt` in `src/util/log.mm`, or `getenv`).

* `METALGLDRV_STRICT=<level>` – the accuracy profile: accuracy features that
  are possible but cost performance, and that conformance doesn't need. Read
  once at device creation (`DeviceCaps::strictLevel`, copied to
  `Context::strictLevel`; `strict()` is level >= 1, `superStrict()` level >= 2;
  the compiler gets it through `CompileOptions` / `LinkInput::strictLevel`).
  * unset, `0` (or `false` / `no`) – default.
  * `1` (or `strict`, or any other value) – strict.
  * `2` (or `super`, `superstrict`) – super strict: strict plus the most
    expensive emulations.

  The default mode stays conformant (the public KHR lists pass at every
  level). Strict adds:
  * `texelFetch` outside the texture or mip range returns zero in every
    context. By default only robust-access contexts pay for the size checks
    (GL leaves these reads undefined otherwise); they cost ~35% on a
    fetch-bound shader.
  * Multisample images: `GL_MAX_IMAGE_SAMPLES` / `gl_MaxImageSamples` = 4
    (0 by default). `image2DMS[Array]` bind a per-sample 2D-array shadow
    (slice = layer * samples + sample). Each draw/dispatch copies the texture
    into the shadow, and writable bindings copy it back into every sample
    afterwards, which splits the render pass at each such draw.
  * EXT_fragment_shading_rate (+ `_primitive`, `_attachment`): coarse shading
    emulated in the fragment shader (docs/glsl-compiler.md). Supported rates
    1x1, 1x2, 2x1, 2x2 (larger requests clamp); attachment texels 2..16 pixels,
    one layer, not on the default framebuffer. Draws whose rate can only be 1x1
    use the regular pipeline; coarse draws shade every pixel lane anyway (no
    speedup), so the feature is about semantics, not performance.

  Super strict adds:
  * ARB_post_depth_coverage: `layout(post_depth_coverage) in;` (implies early
    fragment tests) makes `gl_SampleMaskIn` hold only the samples that passed
    the depth/stencil tests. Metal's `[[sample_mask]]` input doesn't reflect
    them, so draws whose fragment shader reads `gl_SampleMaskIn` with a depth or
    stencil test in effect first render a pre-pass (same vertex function with an
    invariant position, same rasterizer and depth/stencil tests, writes off) into
    a transient R8 coverage attachment, then the draw ANDs `gl_SampleMaskIn` with
    its pixel's marked samples (`draw.mm` executePdc, `VariantKey::pdc`). Draws
    that write depth or stencil are split into a pre-pass + draw per primitive
    and instance (GL order, two render passes each). Pre-passes don't count for
    queries / transform feedback and run the vertex stages on copies of their
    writable storage. Indirect draws are read back on the CPU. Limitation:
    geometry/tessellation draws (and vertex shaders past the classic varying
    budget, which run on the mesh pipeline) get one pre-pass per draw, so their
    primitives don't see each other's depth/stencil writes in the coverage.
* `METALGLDRV_LOG=<0-4>`, `METALGLDRV_LOG_FILE=<path|documents>` – log level / file.
* `METALGLDRV_LOG_UNSUPPORTED=1` – warns once per unique unsupported thing: an entry point that
  is a stub (when called) or unknown (GetProcAddress), an `INVALID_ENUM` (with the entry point
  and, for most, the enum), any other GL error, a GLSL `#extension` the compiler doesn't support.
  Failed compiles and links also write their sources and info log to `METALGLDRV_DUMP_DIR`
  (default `$TMPDIR/metalgldrv-dumps`, one file per failure).
* Application profiles (next section): `METALGLDRV_APP_PROFILE`, `METALGLDRV_VENDOR`,
  `METALGLDRV_RENDERER`, `METALGLDRV_VERSION_STRING`, `METALGLDRV_GLSL_VERSION_STRING`,
  `METALGLDRV_HIDE_EXTENSIONS`, `METALGLDRV_EXTRA_EXTENSIONS`, `METALGLDRV_VRAM_MB`,
  `METALGLDRV_GLSL_QUIRKS`.
* `METALGLDRV_STATS=1` – per-function event counters, logged every 5 s.
* `METALGLDRV_PROFILE=core` – EGL creates core instead of compatibility contexts.
* `METALGLDRV_DUMP_GLSL`, `METALGLDRV_DUMP_MSL`, `METALGLDRV_DUMP_SPIRV`,
  `METALGLDRV_DUMP_FAILED_GLSL`, `METALGLDRV_DUMP_FAILED_MSL` – shader dumps.
* `METALGLDRV_MAX_FPS`, `METALGLDRV_UPSCALER=metalfx`, `METALGLDRV_CAPTURE*`,
  `METALGLDRV_MSL_OPTS`, `METALGLDRV_MSL_OVERRIDE`, `METALGLDRV_GLSL_PATCH`,
  `METALGLDRV_DEBUG_*` – presentation and debugging switches (see the
  `getenv` sites).

Accuracy features that stay on in the default mode because the CTS needs them:
the non-seamless cube clamp (`mgl_cube_noseam`, texture_view), integer border
colors (`mgl_int_border`, texture_border_clamp), feedback-loop snapshots
(texture_barrier), and the varying spill (> 124 components; it backs the
128-component GL_MAX_FRAGMENT_INPUT_COMPONENTS minimum and costs nothing below
that).

## Application profiles
`src/gl/app_profiles.cpp` holds one table of per-application quirks, matched
case-insensitively against the executable name. Under Wine the process is the Wine
loader, so the name is the last path component of the first argv entry ending in
`.exe` (Wine passes the Windows image path); otherwise it is the process name. The
first context creation logs one line with the profile in effect, e.g.
`app profile 'doom2016' (exe DOOMx64.exe), vendor "ATI Technologies Inc.", ...`.
Without a match (and without the variables below) nothing changes: the default
profile is the conformant driver.

| Variable | Effect |
|---|---|
| `METALGLDRV_APP_PROFILE=<name>` | force a profile by its name or one of its executable names; `none` disables matching |
| `METALGLDRV_VENDOR`, `METALGLDRV_RENDERER`, `METALGLDRV_VERSION_STRING`, `METALGLDRV_GLSL_VERSION_STRING` | replace `glGetString(GL_VENDOR / GL_RENDERER / GL_VERSION / GL_SHADING_LANGUAGE_VERSION)`; `GL_MAJOR/MINOR_VERSION` and the context stay 4.6 |
| `METALGLDRV_HIDE_EXTENSIONS="A B C*"` | remove extensions from `GL_EXTENSIONS` / `glGetStringi` and from GLSL (`#extension` rejects them, no predefined macro); a `*` suffix matches a prefix; space, comma or semicolon separated |
| `METALGLDRV_EXTRA_EXTENSIONS="A B"` | advertise extra names (a testing aid for apps that refuse to start; nothing implements them, their entry points still report unimplemented) |
| `METALGLDRV_VRAM_MB=<n>` | total video memory reported by `GL_NVX_gpu_memory_info` / `GL_ATI_meminfo` (default: Metal's `recommendedMaxWorkingSetSize`, capped by the profile) |
| `METALGLDRV_GLSL_QUIRKS="q1,-q2"` | GLSL compiler leniency, Mesa driconf names (`-name` turns off one the profile sets): `force_glsl_extensions_warn` (an unsupported `#extension X : require` only warns), `vs_position_always_invariant` (`gl_Position` is invariant in every program) |

A profile entry sets the same options; environment variables override it.
`tools/glslc --quirk <name> --hide <ext>` compiles with the GLSL side of them.

| Profile | Executables | Options | Why |
|---|---|---|---|
| `doom2016` | `DOOMx64.exe` | vendor `ATI Technologies Inc.`, VRAM at most 8192 MB, `vs_position_always_invariant` | docs/doom2016.md |
