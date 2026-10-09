# GLSL compiler (src/glsl)

From-scratch GLSL 1.10–4.60 → MSL 4.0 compiler. Public interface: `src/glsl/glsl.h`.

| file | role |
|---|---|
| `pp.cpp` | preprocessor (comments, line splicing, #version/#extension/#line/#pragma/#error, object + function macros, hide-set rescanning, `#`/`##` tolerated, `#if` evaluator) |
| `parser.cpp` | recursive-descent parser with integrated semantic analysis (types, conversions, overload resolution, constant folding, layout qualifiers, interface blocks) |
| `builtins.cpp` | builtin function table (signature mini-language + generated texture/image overloads) and per-stage builtin preambles (incl. compatibility state) |
| `fold.cpp` | constant folding |
| `link.cpp` | linking: reachability, attribute/output locations, uniform merging, std140/std430/default-block layout, argument slots, reflection |
| `msl.cpp` | MSL emission |
| `spirv.cpp` | SPIR-V front end (ARB_gl_spirv): parses modules, applies specialization constants, decompiles the entry point to GLSL 4.60 (structured CF -> if/for/switch, hoisted SSA locals, phi copies, interfaces keyed by location/binding); `METALGLDRV_DUMP_SPIRV=1` prints the GLSL, `glslc -S <stage> x.spv` runs it on the host |
| `msl_prelude.cpp` | demand-driven MSL helper prelude (+ fp64 library hook) |

Tools: `build/tools/glslc` (compile/link/emit/validate, `--time`, `--xfb a,b`), `build/tools/glslrun` (executes self-checking compute
tests with known test textures, `--draw` graphics tests, `--pipeline` GS/tess/xfb with the runtime's dispatch, `--tesscheck`).
Tests: `tests/glsl/run_cases.py build/tools/glslc`, `tests/glsl/run_exec.py build/tools/glslrun`, `tests/glsl/corpus.py` (real-world corpora kept outside the repo in `~/metalgldrv-deps/corpus`).

## Runtime ABI

### Entry points
```
vertex   vs::Out mgl_vs([vs::In in [[stage_in]],] constant MGLArgs& A [[buffer(0)]], constant MGLSys& sys [[buffer(1)]],
                        uint [[vertex_id]], uint [[instance_id]], uint [[base_instance]] [, uint [[base_vertex]]])
fragment fs::Out|void mgl_fs([fs::In in [[stage_in]],] constant MGLArgs& A [[buffer(0)]], constant MGLSys& sys [[buffer(1)]], <builtins>)
kernel   void mgl_cs(constant MGLArgs& A [[buffer(0)]], constant MGLSys& sys [[buffer(1)]], <thread position builtins>)
```
`MSLOutput::entryPoints` lists them; compute threadgroup size = `Reflection::localSize`.
Buffer 0 must always be bound (MGLArgs has a dummy field when a program has no slots).

### Argument buffer (buffer 0)
8-byte slots laid out exactly as `Reflection::slots` (slot i at byte 8*i):
* `DefaultUniforms`: GPU address of the default-uniform block (`Reflection::defaultUniformSize` bytes, layout = std430 rules, bools stored as 32-bit 0/1, offsets/strides in `Uniform`).
* `Ubo` / `Ssbo` (index = block index): GPU address of the bound range (binding offset applied). Block arrays occupy consecutive slots.
* `BufferSize` (index = SSBO block index): byte size of the bound range (runtime-sized `.length()`).
* `AtomicCounter` (index = atomic counter buffer index): GPU address of the bound atomic counter buffer range.
* `Texture` / `Sampler` (index = uniform index, element = array element): `MTLResourceID` of the texture view bound to the unit in the sampler uniform's value / of the sampler state for that unit. A sampler uniform of size N has N consecutive Texture slots followed by N consecutive Sampler slots.
* `Image` (index = uniform index, element): `MTLResourceID` of the image unit's view.

Texture expectations (`Slot::textureTarget` gives the GL target the shader expects):
* 1D textures as 2D textures with height 1 (1D arrays as 2D arrays); rectangle textures are normal 2D textures (the shader normalizes coordinates).
* Shadow samplers are `depth2d`/`depth2d_array`/`depthcube`/`depthcube_array`; bind a sampler state with the GL compare function (compare mode COMPARE_REF_TO_TEXTURE).
* Cube *images* (`imageCube`, `imageCubeArray`) are declared `texture2d_array` — bind a 2D-array view (6 faces per layer).
* Buffer textures/images: `texture_buffer`.
* Image atomics need `MTLTextureUsageShaderAtomic` and r32ui/r32i formats.

### Vertex inputs
`[[stage_in]]` struct; `[[attribute(L)]]` = GL location L (matrices: one attribute per column, consecutive locations; arrays: per element).
Types follow the GLSL declaration (float/int/uint vectors). Doubles: `double`/`dvec2` → `uint2`/`uint4` at L; `dvec3`/`dvec4` → `uint4` at L plus `uint2`/`uint4` at L+1 (raw 32-bit pairs of the IEEE binary64 data — use UInt2/UInt4 vertex formats).
Compatibility attributes (`gl_Vertex`, `gl_Normal`, `gl_Color`, `gl_SecondaryColor`, `gl_MultiTexCoord0..7`, `gl_FogCoord`) appear in `Reflection::attributes` under those names with assigned locations (after user attributes). Hide `gl_` names from glGetActiveAttrib if desired.

### Varyings
Every varying is flattened to scalar/vector leaves named `<key>_<n>` (`[[user(<key>_<n>)]]`) where key is `l<location>[c<component>]` for explicit locations, `b_<BlockName>` for interface blocks, `n_<name>` otherwise, and `c0`/`c1`/`tc`/`fog` for compat builtins (gl_FrontColor↔gl_Color, gl_FrontSecondaryColor↔gl_SecondaryColor, gl_TexCoord, gl_FogFragCoord). Names are deterministic, so separable programs match. Double leaves travel as flat `uint2` per component.
**Provoking vertex:** Metal uses the first vertex for flat varyings; GL defaults to the last — the runtime must emulate `GL_LAST_VERTEX_CONVENTION` (e.g. rotate triangle indices).

### Position / rasterization conventions
* `gl_Position.y` is negated unless `MGLSys::flags & 4` (GL_UPPER_LEFT clip origin); z remapped `(z + w) / 2` when `depthMode == 0`.
* `[[point_size]]` is always written: `gl_PointSize` when written and `flags & 1`, else `MGLSys::pointSize`.
* `gl_PointCoord.y = 1 - point_coord.y` unless `flags & 2` (LOWER_LEFT sprite origin).
* `gl_FragCoord`: `[[position]]` directly (GL window y == Metal row); `origin_upper_left` uses `MGLSys::renderTargetHeight`.
* Clip distances: `[[clip_distance]]` array sized by the shader's gl_ClipDistance (or 8 for gl_ClipVertex, using `gl_ClipPlane`), masked by `MGLSys::clipDistanceMask` (disabled planes output 1.0).
* gl_InstanceID = instance_id − base_instance; gl_BaseVertex = [[base_vertex]]; gl_DrawID = `MGLSys::drawID`.
* Fragment outputs: `[[color(location)]]` (+`index(1)` for dual-source), always 4-component (vec3 outputs padded with alpha 1). gl_FragColor → color(0); gl_FragData[i] → color(i) for the indices used.
* Compatibility-profile fragment shaders with a float color(0) apply the alpha test from `MGLSys::alphaFunc/alphaRef` (GL_ALWAYS disables it).

### Compatibility uniforms
Fed by the runtime through the default block, reported in `Reflection::uniforms` with `location == -1`:
`gl_ModelViewMatrix, gl_ProjectionMatrix, gl_ModelViewProjectionMatrix, gl_TextureMatrix[8], gl_NormalMatrix, gl_ModelViewMatrixInverse, gl_ProjectionMatrixInverse, gl_ModelViewProjectionMatrixInverse, gl_TextureMatrixInverse[8], gl_ModelViewMatrixTranspose, gl_ProjectionMatrixTranspose, gl_ModelViewProjectionMatrixTranspose, gl_TextureMatrixTranspose[8], gl_ModelViewMatrixInverseTranspose, gl_ProjectionMatrixInverseTranspose, gl_ModelViewProjectionMatrixInverseTranspose, gl_TextureMatrixInverseTranspose[8], gl_NormalScale, gl_ClipPlane[8], gl_Point.{size,sizeMin,sizeMax,fadeThresholdSize,distanceConstantAttenuation,distanceLinearAttenuation,distanceQuadraticAttenuation}, gl_FrontMaterial/gl_BackMaterial.{emission,ambient,diffuse,specular,shininess}, gl_LightSource[8].{ambient,diffuse,specular,position,halfVector,spotDirection,spotExponent,spotCutoff,spotCosCutoff,constantAttenuation,linearAttenuation,quadraticAttenuation}, gl_LightModel.ambient, gl_FrontLightModelProduct/gl_BackLightModelProduct.sceneColor, gl_FrontLightProduct[8]/gl_BackLightProduct[8].{ambient,diffuse,specular}, gl_TextureEnvColor[8], gl_EyePlaneS/T/R/Q[8], gl_ObjectPlaneS/T/R/Q[8], gl_Fog.{color,density,start,end,scale}`.
`gl_DepthRange` comes from `MGLSys`.

### fp64
`double` → `ulong` (IEEE bit pattern), `dvecN` → `ulongN`, `dmatCxR` → `mgl_dmat<C,R>` (C columns of `vec<ulong,R>`). All arithmetic/comparison/conversion goes through the `mgl_f64_*` library (src/glsl/runtime/fp64.metal, embedded via `fp64_msl.inc` when that header is on the include path; it must expand to a string-literal expression). Matrix helpers (`mgl_dmat_*`) are composed on the library in the prelude.

### Multi-pass pipelines (MGLDrawInfo at buffer(22) = kBufStageIO)
All non-classic kinds fetch vertices manually: attribute at location L uses `VariantKey::attribs[L]` (binding, glType, size (5 = BGRA),
normalized, integer, isDouble, relativeOffset) with `MGLDrawInfo::vertexBuffer/vertexStride/vertexDivisor[binding]` (stride must be the
effective stride; instanced: element = instance / divisor + baseInstance). Disabled arrays (`enabled = false`) read `MGLSys::currentAttrib[L]`.
Vertices whose data would start past `vertexBufferSize[binding]` read the default value (out-of-range indices never read unmapped memory).
Vertex ids come from the uint32 list `indexBuffer` (+ `baseVertex`). Pointers a draw has nothing for hold the address of a zeroed
buffer (`Device::zeroBufferAddress`), never 0; only `stats` and unbound `vertexBuffer` entries are null, and shaders test those.
Mesh grids (`mgl_tes_obj`, `mgl_gs`, `mgl_vsm`) are drawn in chunks along x: each chunk's draw info has `gridBase`, which the shaders
add to their threadgroup x position. Chunks are at most 65536 threadgroups and 64 MB of object payload: Apple GPUs never finish an
object grid whose payloads add up to a few hundred MB (the watchdog kills it or the GPU faults).

* **MeshGeometry** (`mgl_gs`, EntryPoint::Mesh): no object function. Grid (primitiveCount, instanceCount, gsInvocations),
  threads = vertices per input primitive (1/2/4/3/6). maxMeshVertices = max_vertices (<= 256), maxMeshPrimitives = strip-adjusted.
  Strips are converted to lists with GL's last-vertex provoking order (flat varyings follow GL's default). gl_PrimitiveIDIn = list
  primitive index within the instance, gl_InvocationID = grid z.
* **Tessellation**: patches come from the sub-draw table `MGLDrawInfo::subDraws` (`MGLSubDraw`: first patch, gl_DrawID,
  gl_BaseVertex, list offset / first vertex / index address, as `subDrawIndexType` says): grid patch p belongs to the last sub-draw
  whose `firstPatch` <= p (`mgl_sd_find`), its control point j is list position (p - firstPatch) * patchVertices + j of that
  sub-draw (`mgl_sd_vertex`), and gl_PrimitiveID counts from the sub-draw's first patch. One draw has one sub-draw; a batched
  multi-draw all of them (`primitiveCount` = patches over all sub-draws).
  Pass 1 (VS + TCS per patch) writes the control point records (`multipass.tcsVertexOutBytes` each; cps = `multipass.tcsOutputVertices`,
  or `patchVertices` when it is 0) and the patch record (`multipass.tcsPatchOutBytes`: float outer[4], inner[2] at offset 0, patch
  varyings from byte 32). Without a TCS the levels come from `MGLSys::tessDefaultOuter/Inner`. It runs
  * in the object stage `mgl_tes_obj` (threads = TCS output vertices, 32 without a TCS), which passes the records in the payload
    (`uint4 rec[]`: cps records, then the patch record) when they fit 16 KB (`EntryPoint::payloadBytes`); or
  * as the kernel `mgl_tcs` (`VariantKey::tessComputePass`, payloads over 16 KB, the counted paths below): dispatchThreadgroups(patches,
    instances, 1), records at `tcsVertexOut[(instance*patches + patch) * cps + cp]` / `tcsPatchOut[...]`; `mgl_tes_obj` has 1 thread.
  The variant has `mgl_tcs` exactly when the runtime must dispatch it. Pass 2 `mgl_tes_obj` (Object) + `mgl_tes` (Mesh, 64 threads,
  <= 130 vertices / 128 primitives): grid (patches, instances, 1); the object stage sizes the mesh grid itself. Tessellation follows
  GL 4.6 11.2.2 (clamping/rounding per spacing, culling on outer <= 0 or NaN, all domains, point_mode, cw/ccw), verified watertight
  by `glslrun --tesscheck`.
* **Tessellation fast path** (`VariantKey::tessFused` = patch vertices): `mgl_vs` (Vertex) for a non-indexed triangle list of 3
  (quads: 6) vertices per grid patch; vertex k of patch p runs the VS for the patch's input vertices, the TCS invocations one after
  another, packs their outputs into the same records as pass 1 and runs the TES at domain corner k (mgl_tess_unit's special-case
  order: triangles (0,0,1) (1,0,0) (0,1,0), quads (0,0) (1,0) (1,1) / (0,0) (1,1) (0,1), cw swapping the last two), with the
  tessellator's fp pragmas; the TCS / TES shared state lives in thread memory (`sharedSpace_`). Equivalent to the emulation when
  the levels give one segment per edge: the outer levels the domain uses in (0, 1] and its inner levels <= 1, with equal or
  fractional_odd spacing (fractional_even clamps to 2). The linker checks the rest (`Reflection::tessFusable`): VS + TES (+ TCS
  with at most 4 output vertices and no barrier()), no GS, triangles or quads, no point_mode, no memory writes (buffer stores,
  buffer lvalues passed to out parameters, atomics, image stores) or subgroup operations in VS / TCS / TES, no gl_Layer /
  gl_ViewportIndex from the TES, a fragment shader that doesn't read gl_PrimitiveID and has <= 124 input scalars. The TCS must
  write each level the domain uses exactly once, in main, as `gl_TessLevel*[c] = expr;` (chains allowed) at top level or in
  `if (gl_InvocationID == c)` for an existing invocation, after no return, with expr made of constants, default-block uniforms
  (float scalars or components, int/uint converted), unary minus, + - * /, min, max and clamp; `Reflection::tessLevels` holds them
  as postfix programs that the runtime evaluates (in float) at each draw. Without barrier(), running the TCS invocations in sequence
  is a valid order (reads of other invocations' outputs are undefined without it).
* **XfbCapture** (`mgl_xfb`, Kernel, 64 threads): dispatchThreads(count * instances). Writes captured varyings at
  (xfbVertexBase + t) * `multipass.xfbStride[b]` + offset into `xfbBuffer[b]`, primitives `t / vpp < xfbPrimLimit[0]` only.
  Captured layout comes from `Reflection::xfbVaryings` (glTransformFeedbackVaryings incl. gl_NextBuffer / gl_SkipComponentsN, or
  xfb_buffer/xfb_offset/xfb_stride qualifiers, which take precedence); `multipass.xfbBufferStream[b]` is the vertex stream of buffer b.
  GS and tessellation output sizes are only known on the GPU, so capture is counted (the runtime reads the counts back):
  * GS: `mgl_gs_count` (one thread per work item = input primitive x invocation, `itemCount`) runs VS+GS and writes the
    primitives emitted per stream to `itemCounts[t]`; with `gsRecord` set it also stores the item's captured primitives there
    (`scratchWordsPerItem` uint32 per item, see the `mgl_gs_copy` entry point). The runtime writes per-stream prefix sums to
    `itemOffsets` and `xfbPrimLimit[s]` (primitives that fit), then runs `mgl_gs_copy` (places the records, no shader re-run) or
    `mgl_gs_write` (runs the GS again and writes at the offsets).
  * Tessellation: `mgl_tcs`, then `mgl_tes_count` (one thread per patch: itemCounts = primitives, vertices, units), CPU prefix
    sums into itemOffsets (first primitive, first vertex), then `mgl_tes_gen` (threadgroup per (patch, unit)) writes TES output
    records (`tesVertexOutBytes` each) to `tesVertexOut` and the GL-ordered primitive list (vertex indices) to `tesPrimOut`.
    Without a GS, `mgl_tes_xfb` (one thread per primitive) captures from the records; with a GS, the GS kernels and the GS mesh
    pipeline (`PipelineKind::Tessellation` with a GS) read their inputs from the records (grid = tesPrimCount x invocations).
  Primitive queries (PRIMITIVES_GENERATED / TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN per stream, overflow) are counted on the CPU.
* Program pipelines with stages from different separable programs: `generateMSL(stages[])` emits one `MGLArgs` per stage
  namespace; buffer 0 then holds a table of 6 argument buffer addresses indexed by Stage.
* Fragment shaders paired with mesh pipelines read gl_PrimitiveID from the per-primitive `[[user(mgl_primid)]]`; gl_Layer /
  gl_ViewportIndex are per-primitive outputs. `Reflection::usesFlatShading` tells the classic path when index rotation for
  GL_LAST_VERTEX_CONVENTION is needed; `writesLayer`/`writesViewportIndex` describe the last pre-raster stage.
* **Varying spill** (mesh producers: GS, TES): Metal caps a mesh vertex + primitive at 124 unique scalars (position
  included) and a whole mesh at 32 KB. When the fragment inputs do not fit, the trailing varyings are spilled: each
  emitted vertex takes a record slot from `MGLDrawInfo::spillCounter` and writes them (one uint4 per leaf,
  `EntryPoint::spillRecordBytes` per vertex) to `spillBuffer`; every primitive carries its vertices' slots in the flat
  per-primitive `uint3 mgl_spill` (index order: vertex 0 = provoking vertex); the fragment shader reads the records
  (also bound at buffer 22) and interpolates with `[[barycentric_coord, <center|centroid|sample>_<perspective|
  no_perspective>]]` (flat/integer/double inputs take vertex 0). The runtime allocates `spillBuffer` for the worst
  case of the grid (capped at 64 MB; vertices past `spillCapacity` read zeros). Programs within the budgets emit no
  spill code. This is what lets GL_MAX_FRAGMENT_INPUT_COMPONENTS and GL_MAX_VERTEX_OUTPUT_COMPONENTS be 128: a
  VS + FS program whose fragment inputs exceed 124 scalars (`Reflection::vsMeshPassthrough`) draws through
  PipelineKind::MeshGeometry without a GS: `mgl_vsm` (Mesh, one threadgroup per input primitive of the decomposed
  list, one thread per vertex) runs the vertex shader and emits the primitive with the provoking vertex first.
  interpolateAt*() on a spilled input is not supported.
* The tessellator chunk sets `#pragma METAL fp math_mode(safe)` / `fp contract(off)` (shared vertices must be bit-identical); this
  applies to the rest of the generated source.

### Mesh shaders (EXT_mesh_shader, PipelineKind::Mesh)
* The task shader is the Metal object function `mgl_ts`, the mesh shader the mesh function `mgl_ms`, both with the
  GL local size as threadgroup size (`EntryPoint::threadsPerThreadgroup`); `glDrawMeshTasks*EXT` draw
  `drawMeshThreadgroups` (indirect: `drawMeshThreadgroupsWithIndirectBuffer`, whose arguments match
  DrawMeshTasksIndirectCommandEXT). gl_DrawID = `MGLSys::drawID` (one draw per multi-draw command).
* `taskPayloadSharedEXT` lives in the Metal payload (`object_data`, `EntryPoint::payloadBytes` = MSL size of the
  variable), reached through `G::PL`; `mgl_array` gets object_data accessors in shaders that use a payload.
  EmitMeshTasksEXT() records the counts (and returns from main); invocation 0 calls `set_threadgroups_per_grid`.
* Mesh outputs (user arrays, one array per used member of gl_MeshVerticesEXT / gl_MeshPrimitivesEXT, the index
  arrays) live in threadgroup memory; after main and a barrier the workgroup copies the SetMeshOutputsEXT()
  counts into the Metal mesh (vertex: position, point size for points, clip/cull distances, the per-vertex
  varyings the fragment shader reads; primitive: perprimitiveEXT varyings, `mgl_primid`, gl_Layer,
  gl_ViewportIndex). gl_CullPrimitiveEXT compacts the primitive list. Per-primitive fragment inputs are flat.
  Flat per-vertex inputs follow GL's provoking vertex (indices rotated for GL_LAST_VERTEX_CONVENTION).
* Limits that follow from Metal: a threadgroup has 32 KB, an object threadgroup 16 bytes less (Metal keeps the mesh
  grid properties there), so GL_MAX_TASK_SHARED_MEMORY_SIZE_EXT is 32752; a mesh threadgroup has 60 KB of
  threadgroup memory + mesh output together, and the outputs are held twice (staging + Metal mesh), so
  GL_MAX_MESH_OUTPUT_MEMORY_SIZE_EXT is 30720. Both are below the EXT minimums (32768). The 124-scalar vertex +
  primitive interface limit applies (no varying spill for mesh shaders).
* Metal compiler workaround: an object function whose only buffer accesses go through argument-buffer pointers
  and that writes its payload crashes the pipeline compiler (XPC_ERROR_CONNECTION_INTERRUPTED); `mgl_ts` reads
  `sys` directly to avoid it.

### Subroutines
`Reflection::subroutines[stage]`: `functions` (GL subroutine index = position), `uniforms` (location, arraySize, compatible indices),
`numLocations`. Stages that use subroutine uniforms get a `SlotKind::SubroutineIndices` slot (index = stage) holding the GPU address of
`uint selected[numLocations]` (glUniformSubroutinesuiv values). Unselected/incompatible indices make the call a no-op returning zero.

### Uniform initializers
`Reflection::defaultUniformInit` holds the initial default-block bytes (initializers in GLSL, zero elsewhere). Copy it into the
program's default-uniform storage at link time.

### Compatibility attribute locations
gl_Vertex = 0, gl_Normal = 2, gl_Color = 3, gl_SecondaryColor = 4, gl_FogCoord = 5, gl_MultiTexCoordN = 8 + N. Generic attributes may
alias them (explicit locations), as on NVIDIA.

### Images
Stores and atomics on read-write images are followed by `texture.fence()` so an invocation observes its own writes.

### Coarse shading (EXT_fragment_shading_rate, VariantKey::fsr, METALGLDRV_STRICT)
Metal has no per-draw/per-primitive/attachment shading rates (rasterization rate maps are a different model), so the
fragment shader emulates coarse fragments. The supported rates are 1x1, 1x2, 2x1 and 2x2, so a coarse fragment lies
inside one Metal quad (quads are aligned to even pixels; lane = (x & 1) | (y & 1) << 1). `mgl_fsr_setup` (entry point,
uniform control flow):
* computes the effective rate from `MGLFsr` (buffer(23): draw rate, combiner ops, attachment `texture2d<uint>` with
  log2 texel size >= 1, so one texel per quad) and the flat varying `psr_0` (gl_PrimitiveShadingRateEXT of the
  provoking vertex, 0 if the last pre-rasterization stage doesn't write it; always passed by fsr variants), clamped
  to <= 2 per axis;
* assigns the quad's four lanes to a 2x2 block of coarse fragments aligned to even coarse coordinates (`mgl_bl`:
  lane of each block position, `mgl_bp`: this lane's). Each of the quad's fragments that has coverage is shaded by
  one of its covered lanes (helper lanes can't write memory); the remaining positions are helper invocations
  (`mgl_fl` false), there for derivatives and quad operations. With 2x1/1x2 the lanes need not sit in a 2x2
  arrangement (rows covered in different columns), so derivatives and quad ops shuffle by block position.
* Inputs are re-evaluated at the lane's fragment centre (`mgl_fe`) from planes: v·(1/w) and 1/w are affine in window
  coordinates, so the lane's value and its quad neighbours' differences extrapolate exactly (helper lanes interpolate
  on the plane). Centroid inputs use the centre of a covered pixel (the spec allows any covered pixel);
  interpolateAt* keep the plane (`mgl_pa_/px_/py_`). Inputs are declared center-interpolated in the In struct.
* gl_FragCoord = fragment centre (z, w from planes), gl_HelperInvocation = !mgl_fl, gl_SampleMaskIn = the coverage
  of all the fragment's pixels (lane order, `samples` bits each), gl_ShadingRateEXT = the rate.
* dFdx/dFdy/fwidth: difference to the neighbouring block position / fragment width (`mgl_ddx`); implicit-LOD
  lookups use explicit gradients of the difference (`mgl_gdx`, i.e. the LOD scaled by the fragment size);
  textureQueryLod feeds calculate_*_lod a synthetic coordinate whose quad differences are each lane's gradients.
* Stores and atomics are predicated on `mgl_fl`. `discard` sets `mgl_disc`, clears `mgl_fl` and returns (each loop
  iteration also leaves a discarded lane); the entry point discards the pixel when the lane that shaded its fragment
  (`mgl_home`) discarded, and copies that lane's color outputs.
Fragment shaders that write gl_FragDepth / gl_SampleMask, shade per sample or read gl_FragCoord without enabling the
extension (`Reflection::fsrForce1x1`), and draws with sample shading or a partial SAMPLE_MASK_VALUE use the regular
variant (rate 1x1), as do draws whose rate can only be 1x1 (`fsrWanted`).

### Post-depth coverage (ARB_post_depth_coverage, VariantKey::pdc, METALGLDRV_STRICT=2)
`layout(post_depth_coverage) in;` is parsed in super strict mode only (elsewhere it stays an unknown, ignored
qualifier) and implies `[[early_fragment_tests]]`. `Reflection::postDepthCoverage` is set when the shader also reads
gl_SampleMaskIn; such draws with a depth/stencil test get a `pdc` variant (runtime: draw.mm, executePdc):
* gl_SampleMaskIn = `[[sample_mask]]` & `mgl_pdc_mask`: the samples of the pixel (`[[position]]`, and
  `[[render_target_array_index]]` when `MGLPdc::layered`) that the pre-pass marked in the coverage attachment
  (`MGLPdc` at buffer(24): `texture2d_ms_array<float>` or, single-sampled, `texture2d_array<float>`);
* gl_PrimitiveID (classic) adds `MGLPdc::primBase`, since draws that write depth/stencil are split per primitive;
* the extra fragment function `mgl_fs_pdc` (writes 1, early tests) is the pre-pass's;
* the vertex position is `[[invariant]]` so that both pipelines rasterize the same samples, and the classic vertex
  function takes gl_BaseInstance from `MGLSys` instead of `[[base_instance]]` (split draws pass the instance as
  Metal's base instance). Fragment shaders that declare the qualifier force the 1x1 shading rate.

## Not yet supported
Mesh pipelines (GS / tessellation rasterization) carry at most 124 output scalars per vertex (including gl_Position and
point size), below GL's 128 MAX_GEOMETRY_OUTPUT_COMPONENTS. gl_CullDistance is emulated with extra clip distances (-1 when
negative, 1e20 otherwise). Float image atomics (Metal has none), whole-value use of structs containing opaque types, dynamic indexing of
uniform struct arrays containing opaque members, interpolateAtSample on multisample-less targets uses the sample position Metal reports.
