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

## Ideas not yet tried

* **Compute tessellation into buffers + an ordinary indexed draw.** Apple GPUs reportedly have no
  fixed-function tessellator (Asahi Linux emulates tessellation and geometry shaders with compute; not verified
  here), so Apple's GL probably wins with a better software design rather than hardware. Plan: VS + TCS and the
  tessellator in compute kernels writing domain points and indices into a buffer at offsets from a GPU prefix sum
  over the patches, then one indirect indexed draw whose vertex function runs the TES. That avoids mesh shader
  output limits and keeps GL's primitive order. The transform feedback path (`mgl_tcs`, `mgl_tes_count`,
  `mgl_tes_gen`) already does most of this, with a CPU prefix sum.
* **Hoisting that compute into a pre-pass command buffer** committed just before the main one, so tessellated
  draws don't break render passes; draws reading data the GPU writes earlier in the same command buffer would
  keep the current path.
* **Classic Metal queue with `drawPatches`** for tessellated draws, synchronised with events: about 30 extra pass
  breaks per Heaven frame; Metal's tessellator follows Direct3D's rules, so the strict profiles, isolines, point
  mode, geometry shaders and transform feedback would keep the emulation.
* **Pattern instancing** (bin patches by integer levels, draw each pattern instanced): fast for equal spacing,
  too many patterns for fractional spacing.
