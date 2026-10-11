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
| `src/gl/winsys.*` | Window-system glue shared by EGL and CGL: default-framebuffer surfaces, `CAMetalLayer` setup (incl. the sublayer added over a view's layer), presentation, swap interval. |
| `src/macos` | macOS only: system-GL replacement. `cgl.mm` implements CGL (Apple's names/signatures, OpenGL.framework never linked); `nsgl.mm` reroutes `NSOpenGLContext` / `NSOpenGLPixelFormat` onto it; `hooks.mm` decides when any of this is active (CrossOver's `winemac.so`, the `Agent_OnLoad` JVMTI agent, `METALGLDRV_HOOK_NSOPENGL=1`) and rebinds imports with `macho_rebind.cpp`. See docs/minecraft-macos.md and docs/crossover.md. |
| `src/gl` | GL state machine, object model (`objects.h`), entry points (`api_*.mm`), draw path (`draw.mm`), internal shaders (clear/blit/present). |
| `src/mtl` | Metal 4 device (`device.mm`), GL↔Metal formats and pixel transfer conversion (`formats.mm`). |
| `src/glsl` | GLSL compiler: preprocessor → parser → semantic analysis → linker → MSL emitter. Public contract in `glsl.h`. |
| `src/glsl/runtime/fp64.metal` | Soft double-precision MSL library (embedded into shaders that use doubles). |
| `tests/host` | GL-level tests (run on macOS host and on the iPad through `tools/ios-runner`). |
| `tests/fp64` | Bit-exact GPU validation of the fp64 library against host IEEE doubles. |
| `demo/amethyst` | Patch set + build script for the Amethyst demo (Amethyst source lives outside the repo). |

## macOS system-GL replacement (src/macos)
libmetalgldrv.dylib exports, besides EGL and every GL entry point in the
`*GetProcAddress` table (so `dlsym` on our handle resolves them all), CGL under
Apple's names and `gluCheckExtension`, plus `Agent_OnLoad`. None of the hooks run
unless activated:

* **Wine mode**: `scripts/crossover-install.sh` rewrites CrossOver's `winemac.so`
  to link libmetalgldrv in place of OpenGL.framework (CGL binds to us); when
  `winemac.so` loads, `hooks.mm` rebinds its `dlopen` import so
  `dlopen(".../OpenGL.framework/OpenGL")` returns our handle, and installs the
  NSOpenGL hooks (`WineOpenGLContext` subclasses `NSOpenGLContext`).
* **Replacement mode** (`java -agentpath:libmetalgldrv.dylib`, or
  `METALGLDRV_HOOK_NSOPENGL=1` with the dylib already loaded, e.g. via
  `DYLD_INSERT_LIBRARIES` in a non-hardened app): NSOpenGL hooks, plus in every
  non-system image (loaded now or later) OpenGL.framework imports, `dlopen` of
  OpenGL.framework and `CFBundleGetFunctionPointerForName` on `com.apple.opengl`
  resolve to libmetalgldrv. The JVMTI agent also sets `org.lwjgl.opengl.libname`
  to our path at VMInit unless already set (`=nolibname` option disables).

CGL model: one accelerated renderer (RGB888/ARGB8888, depth 0/16/24/32, stencil
0/8, 2x/4x MSAA), every context GL 4.6 (core for `kCGLOGLPVersion_3_2_Core` /
`GL4_Core` pixel formats, forward-compatible; compatibility otherwise). A
context's drawable is a window surface or a pbuffer surface; changes requested
from another thread (AppKit's main thread) take effect at the context's next
make-current / flush / update on its own thread. `kCGLCPSurfaceBackingSize`
fixes the render size (`Surface::fixedWidth/Height`); the layer scales it.
NSOpenGL view surfaces are a `CAMetalLayer` sublayer of the view's layer, one per
view, shared by the contexts attached to it; window default framebuffers are
sRGB-capable. Not supported: accumulation/aux/stereo/float pixel formats,
`kCGLCPSurfaceOrder` below the window, rendering into pbuffer mip levels > 0,
`NSOpenGLPixelBuffer`, `CAOpenGLLayer` / `NSOpenGLLayer`.

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

## Tessellation (Metal's tessellator, object + mesh emulation, fused fast path)
Metal 4 dropped `drawPatches`. Most rasterized tessellation goes to classic Metal's tessellator on a second queue
(below); what it can't do falls back to the emulation, which runs VS + TCS per patch, then an object+mesh pipeline:
the object stage reads the levels and sizes the work; mesh
threadgroups generate domain points and primitives with our own GL-rules
tessellator (equal / fractional_even / fractional_odd spacing, triangles / quads
/ isolines, point_mode, cw/ccw) and run the TES per generated vertex.
* **Metal's tessellator** (`VariantKey::tessMetal`, draw.mm `executeTessMetal`), as Apple's own GL driver does it:
  the `mgl_tcs` kernel runs VS + TCS (up to 32 threads, several patches per threadgroup) and also converts the
  levels to Metal's half-precision factors (`mgl_tess_metal_factors`: GL's culling, levels rounded up so that a
  level like 35.01 keeps its 36 segments); `drawPatches` then runs the TES as the post-tessellation vertex function
  (`mgl_tes_vs`) in front of a classic fragment function. Metal's tessellator is implemented by Apple's driver, not
  fixed-function hardware, and it is 4-5x faster than the emulation (docs/tessellation-performance.md). Metal 4
  command buffers can't encode it, so tessellated draws run in *segments* on the device's classic
  `MTLCommandQueue` (`Context::openTessSegment`): the Metal 4 work so far is committed, a pre-pass command buffer
  takes the draws' VS + TCS kernels and a classic render pass over the same attachments takes their
  `drawPatches`; one `MTLEvent` per context orders Metal 4 work, pre-pass, render pass and the next Metal 4
  command buffer, whose serial covers everything the segment used. Consecutive tessellated draws into the same
  pass share a segment; any Metal 4 recording (`Context::commandBuffer`), a flush, `glMemoryBarrier` or
  `glTextureBarrier` ends it. Classic encoders bind `MTLBuffer`s, so the draw's argument / system / draw-info
  uploads are found by address (`Context::uploadBuffer`); resources stay resident through the device's residency
  set, which the classic queue shares. Domain points match the emulation's for equal spacing up to Metal's fixed
  point (~1e-4); for fractional spacing Metal (Direct3D's rules) picks other lengths for the two short segments,
  which GL leaves to the implementation; triangulation and the provoking vertex of generated triangles differ too
  (`glslrun --tesscheck` compares the points of 783 level sets, `mgltest tessellation_metal_tessellator` the
  images). Not for isolines, point_mode, a geometry shader, a TES with memory writes or subgroup operations or that
  writes gl_Layer / gl_ViewportIndex, a fragment shader that reads gl_PrimitiveID (`Reflection::tessMetal`),
  program pipelines, indirect draws, transform feedback, primitive / occlusion / statistics queries, rasterizer
  discard, polygon mode points, coarse shading or post-depth-coverage pre-passes. `METALGLDRV_TESS_METAL=0` turns it off.
* **Pass 1 (VS + TCS)** runs in the object stage itself (one threadgroup per
  patch, control points and patch data passed in the payload), so tessellated
  draws stay inside the render pass. Patches too big for Metal's 16 KB payload,
  and the counted paths (geometry shader after tessellation, transform feedback,
  primitive queries), run it as the `mgl_tcs` compute kernel instead, which ends
  the render pass. `METALGLDRV_TESS_PASS1=compute` forces the kernel.
* **Pieces and batches**: with a TCS in the object stage, units are cut into pieces of <= 32 primitives and
  packed into small mesh threadgroups (34 vertices), up to 3 patches share an object threadgroup when the VS / TCS
  have no side effects, and domain points / primitives are generated in parallel. `build/tests/displacebench`
  (Heaven's displacement TCS / TES; `displacebench-apple` runs it on Apple's OpenGL) measures it. Primitive order
  across mesh threadgroups isn't guaranteed: depth ties and blending within one tessellated draw may resolve
  differently from GL's order.
* **Multi-draws** (`glMultiDraw*`, also with `METALGLDRV_IMPLICIT_PATCHES`) run
  all their sub-draws in one emulation pass: a sub-draw table
  (`MGLDrawInfo::subDraws`) gives each patch its gl_DrawID, gl_BaseVertex and
  draw-relative gl_PrimitiveID. Not with a geometry shader, transform feedback,
  primitive queries or post-depth-coverage pre-passes (`tessBatchable`).
  `METALGLDRV_TESS_BATCH=0` draws the sub-draws one by one. Indexed patch lists
  read the GL index buffer on the GPU (16/32-bit, no primitive restart);
  otherwise the CPU decomposes them. The object/mesh grid of such a draw (one
  object threadgroup per patch) is drawn in chunks: Apple GPUs never finish
  grids whose payloads add up to a few hundred MB (docs/glsl-compiler.md).
* **Fast path** (`VariantKey::tessFused`): when the levels make every patch
  exactly one primitive at the domain corners (triangles or quads; outer levels
  in (0, 1], inner levels <= 1; equal or fractional_odd spacing; no point_mode),
  a classic vertex function runs, per output vertex, the VS for the patch's
  vertices, the TCS invocations in sequence and the TES at its corner, in the
  emulation's vertex order (same winding and provoking vertex). The linker
  decides whether a program qualifies and how the TCS computes its levels from
  constants and default-block uniforms (`Reflection::tessFusable`,
  `tessLevels`); the draw evaluates them with the current uniform values (or the
  GL_PATCH_DEFAULT_* levels without a TCS). When the TES is affine in
  gl_TessCoord (`Reflection::tesAffine`: interpolation of the inputs, as
  Shrimple's terrain and water do), the fast path applies at any levels: the
  tessellated patch covers the same triangle with the same perspective-correct
  values; the fused vertex function drops patches whose outer levels cull them.
  Shader packs that tessellate every terrain draw (Shrimple) render at close to
  the cost of no tessellation. The affine case is off in the strict profiles
  (equal only up to rounding); the trivial-levels case stays. `METALGLDRV_TESS_FAST=0`
  disables both. Conditions
  in docs/glsl-compiler.md.
* **Watchdog budget**: emulated geometry (tessellation, geometry shaders, and
  fast-path patches at a lower weight) is weighed in expected output triangles;
  a command buffer is committed before it holds more than ~250 ms of it, also
  between the chunks of one large draw (`meshWork*` in draw.mm). The ns per
  triangle comes from the GPU times of completed command buffers. A command
  buffer that runs longer than 0.5 s on the GPU is logged as a warning.

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
* `tests/host/test_macos.mm` (`mgltest cgl`, `mgltest macho`, `mgltest gl_exports`)
  – CGL, import rebinding and GL exports. The universal build runs them on the
  x86_64 slice too: `arch -x86_64 ./build-universal/tests/mgltest` (configure it
  with `-DMGL_BUILD_TESTS=ON`).
* `./build/tests/nsgltest` – replacement mode against real AppKit (it turns the
  hooks on itself): NSOpenGL pixel formats / contexts / a subclass like Wine's,
  view layer and Retina scale, resize, backing size, swap interval, GL lookups
  through `CFBundle` and `dlopen` of OpenGL.framework, and `Agent_OnLoad` with fake
  JVMTI/JNI tables. `./build/tests/winemodetest` – Wine mode: a fake `winemac.so`
  linked against the driver (dlopen redirect, `-initWithCGLContextObj:`, setView
  from another thread). Neither shows a window. Both also run as
  `arch -x86_64 ./build-universal/tests/...`.
* Universal (arm64 + x86_64, for CrossOver/Rosetta) driver:
  `cmake -G Ninja -B build-universal -DMGL_UNIVERSAL=ON -DMGL_BUILD_TESTS=OFF && ninja -C build-universal metalgldrv`.
* `tools/glslc` / `tests/glsl` – compiler tests and real-world shader corpus.
* `./build/tests/tessbench` – benchmark of Sodium + Iris terrain with a tessellating shader pack
  (multi-draws of tessellated triangles; `--level`, `--notess`, `--single`, see the file header).
  Not part of mgltest.
* `./build/tests/packreplay PREFIX...` – draws real shader pack programs (stage files, e.g. assembled
  with `tests/glsl/corpus.py`'s include resolution) with Minecraft + Sodium + Iris's draw calls
  (multi-draws, indirect, instanced, 8-bit indices, a depth-only pass), for reproducing driver
  problems under `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`. Not part of mgltest.
* `METALGLDRV_STRICT=1 ./build/tests/mgltest`, `METALGLDRV_STRICT=2 ./build/tests/mgltest`
  – the same suite in strict and super strict mode (`tests/host/test_strict.cpp`
  and `test_fsr.cpp` cover the strict-only paths, `test_pdc.cpp` the super
  strict ones).

## Private Metal API
src/mtl/spi.h wraps the few private Metal selectors that help GL and work on the objects metalgldrv records with
(Apple's GL driver uses them; docs/tessellation-performance.md lists what was tried): `setLineWidth:` on render
encoders for `glLineWidth` (1..16, as Apple's GL; otherwise lines are 1 pixel and the range is 1..1) and
`setIsOpenGLQueue:` on the tessellation queue, and `setForceSeamsOnCubemapFiltering:` on samplers of cube maps
without `GL_TEXTURE_CUBE_MAP_SEAMLESS` (otherwise the shader keeps those lookups inside the face). Built in with the CMake option `MGL_PRIVATE_API` (default ON; App
Store review rejects private API, so apps shipped there turn it off), used unless `METALGLDRV_MORALLY_WRONG=0`, and
each call checks that the object implements the selector.

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
* `METALGLDRV_TURBOHAX=1`, `METALGLDRV_SPEEDDEMON=1` – the speed profiles, the other
  direction from strict: they give up accuracy, and conformance, for performance. They are
  `DeviceCaps::strictLevel` -1 and -2 (`turbo()`, `speedDemon()`); `METALGLDRV_STRICT` takes
  precedence over both, and SPEEDDEMON over TURBOHAX.
  * TURBOHAX: shaders compile with `MTLMathModeFast` (reassociation, no NaN / infinity
    guarantees); anisotropic filtering is capped at 4x.
  * SPEEDDEMON, TURBOHAX plus: `GL_QUERY_RESULT` never waits for the GPU (an unfinished occlusion
    query reads as visible, other queries give their previous result); `glFinish` submits without
    waiting; robust-access contexts get no texelFetch bounds checks; anisotropy is capped at 2x.
* macOS: `METALGLDRV_*=value` lines in `~/Library/Application Support/metalgldrv/metalgldrv.conf`
  apply to every process that loads the driver, unless its environment sets them (for hosts whose
  environment is hard to reach, such as Steam → NotProton → CrossOver). `METALGLDRV_CONF=<path>`
  reads another file and `METALGLDRV_CONF=none` none; tests/cts/run.sh uses `none`.
* `METALGLDRV_LOG=<0-4>`, `METALGLDRV_LOG_FILE=<path|documents>` – log level / file.
* `METALGLDRV_LOG_UNSUPPORTED=1` – warns once per unique unsupported thing: an entry point that
  is a stub (when called) or unknown (GetProcAddress), an `INVALID_ENUM` (with the entry point
  and, for most, the enum), any other GL error, a GLSL `#extension` the compiler doesn't support.
  Failed compiles and links also write their sources and info log to `METALGLDRV_DUMP_DIR`
  (default `$TMPDIR/metalgldrv-dumps`, one file per failure).
* Application profiles (next section): `METALGLDRV_APP_PROFILE`, `METALGLDRV_VENDOR`,
  `METALGLDRV_RENDERER`, `METALGLDRV_VERSION_STRING`, `METALGLDRV_GLSL_VERSION_STRING`,
  `METALGLDRV_HIDE_EXTENSIONS`, `METALGLDRV_EXTRA_EXTENSIONS`, `METALGLDRV_EXTENSION_YEAR`, `METALGLDRV_VRAM_MB`,
  `METALGLDRV_GLSL_QUIRKS`, `METALGLDRV_SHARE_FRAMEBUFFERS`.
* `METALGLDRV_STATS=1` – per-function event counters and the GPU's busy time, logged every 5 s.
  `METALGLDRV_PROFILE_PASSES=1` adds GPU time per render pass / compute dispatch (each is committed
  in its own command buffer, so it changes performance; labels name the framebuffer, the entry
  point that opened the pass, its draw count and first program).
* `METALGLDRV_TRACE_CALLS=1|<n>` – calls per GL entry point, the 30 (or n) most frequent logged
  every 5 s from a background thread (also while the app draws nothing);
  `METALGLDRV_TRACE_SEQ=<n>` also logs each thread's first n calls in order.
* `METALGLDRV_WINE_SHARE_PATCH=0` – leave CrossOver's `winemac.so` check that stops core and
  compatibility contexts from sharing (patched in memory by default; see docs/doom2016.md).
* `METALGLDRV_IMPLICIT_PATCHES=1` – draws whose primitives have GL_PATCH_VERTICES vertices run as
  patches while tessellation is active (Iris + Sodium; set by the Fabric mod).
* `METALGLDRV_TESS_METAL=0`, `METALGLDRV_TESS_FAST=0`, `METALGLDRV_TESS_BATCH=0`, `METALGLDRV_TESS_PASS1=compute` –
  turn off Metal's tessellator, the tessellation fast path or multi-draw batching, or run the emulation's pass 1 as a
  compute kernel (see Tessellation; for comparisons and debugging).
* `METALGLDRV_MORALLY_WRONG=0` – don't use private Metal API (src/mtl/spi.h) in a build that has it (the CMake
  option `MGL_PRIVATE_API`, on by default; turn it off for apps that go through App Store review). See "Private
  Metal API".
* `METALGLDRV_PROFILE=core` – EGL and CGL create core instead of compatibility contexts.
* `METALGLDRV_HOOK_NSOPENGL=1` – macOS replacement mode (NSOpenGL hooks + import
  rebinding, see above) without the JVMTI agent. It also turns on by itself when the process's
  main executable links libmetalgldrv (an app ported by linking it in place of OpenGL.framework,
  docs/porting.md); `METALGLDRV_HOOK_NSOPENGL=0` prevents that. Apps can call
  `metalgldrv_activate()` (include/metalgldrv/metalgldrv.h) instead.
* `METALGLDRV_DUMP_GLSL=<substring|#program>`, `METALGLDRV_DUMP_MSL=<1|substring|#program>`, `METALGLDRV_DUMP_SPIRV`,
  `METALGLDRV_DUMP_FAILED_GLSL`, `METALGLDRV_DUMP_FAILED_MSL` – shader dumps.
* `METALGLDRV_USE_METALFX=1` – window surfaces present at the display's native resolution, upscaled
  with MetalFX's spatial scaler when the render size is smaller (`metalfxUpscale` in
  internal_shaders.mm).
* `METALGLDRV_RENDER_SCALE=<0.25..1>` – macOS replacement mode (Minecraft): views carrying our
  surfaces report and render at that fraction of their backing scale (`ourBackingScale` in nsgl.mm),
  so GLFW / SDL size the framebuffer smaller; ignored under Wine, which sizes windows itself.
* `METALGLDRV_MAX_FPS`, `METALGLDRV_CAPTURE*` (frame and
  per-program texture captures, written to `METALGLDRV_DUMP_DIR` when set),
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
| `METALGLDRV_EXTENSION_YEAR=<year>` | the `glGetString(GL_EXTENSIONS)` string lists only extensions published up to that year, oldest first (`src/gl/ext_years.cpp`, years from Mesa's table), for games from around 1999–2005 that copy it into a fixed-size buffer and crash on a long string (like NVIDIA's `__GL_ExtensionStringVersion`); `glGetStringi` still lists everything |
| `METALGLDRV_VRAM_MB=<n>` | total video memory reported by `GL_NVX_gpu_memory_info` / `GL_ATI_meminfo` (default: Metal's `recommendedMaxWorkingSetSize`, capped by the profile) |
| `METALGLDRV_GLSL_QUIRKS="q1,-q2"` | GLSL compiler leniency, Mesa driconf names (`-name` turns off one the profile sets): `force_glsl_extensions_warn` (an unsupported `#extension X : require` only warns), `vs_position_always_invariant` (`gl_Position` is invariant in every program) |

A profile entry sets the same options; environment variables override it.
`tools/glslc --quirk <name> --hide <ext>` compiles with the GLSL side of them.

| Profile | Executables | Options | Why |
|---|---|---|---|
| `doom2016` | `DOOMx64.exe` | vendor `ATI Technologies Inc.`, VRAM at most 8192 MB, `vs_position_always_invariant` | docs/doom2016.md |
