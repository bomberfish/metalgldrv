# CTS receipts: run matrix and NotSupported audit

## Current receipts (commit a9bd463)

| platform | mode | runs | result |
|---|---|---|---|
| macOS 27.0 / Apple M4 Pro | default | 14 (full runner matrix) | 0 Fail/Crash/Timeout. KHR-GL46 18061 pass, 1652 NotSupported (18043/1670 on config 2); KHR-Single-GL46 11340 pass, 8 NotSupported; KHR-NoContext 8/8 |
| macOS 27.0 / Apple M4 Pro | METALGLDRV_STRICT=1 | 14 (`*-strict`) | 0 Fail/Crash/Timeout. KHR-GL46 18100 pass, 1613 NotSupported (18082/1631 on config 2); Single and NoContext as in default mode |
| iPadOS 27.0.1 / Apple M5 | default | 3 (`*-ipad`) | 0 Fail/Crash/Timeout. KHR-GL46 18066 pass, 1647 NotSupported; Single 11340/8; NoContext 8/8 |

On the Mac, NotSupported is the iPad breakdown below plus the 5 texture_filter_minmax cases: the M4 Pro (Apple9)
ignores the sampler reduction mode, so ARB_texture_filter_minmax is only exposed on Apple10 GPUs. Strict mode
exposes multisample images (GL_MAX_IMAGE_SAMPLES 4), which turns those 39 cases into passes. Config 2 adds the 18
fragdepth cases that need a depth buffer.

The sections below are the earlier audit for commit 926bb9c, from before the optional extensions, followed by the
iPad breakdown. That iPad breakdown matches the current receipts.

## Earlier receipts (commit 926bb9c)

Khronos OpenGL CTS 4.6.8.1, public case lists, run on macOS / Apple M4 Pro by `tests/cts/run_receipts.sh`.
Per-run results are in `index.csv`, `LATEST.md` and the per-run folders (`results.csv` lists every case,
and `results.qpa.gz` is the raw glcts log).

## Run matrix

The configurations match the Khronos runner's tables (`external/openglcts/modules/runner/glcKhronosMustpassGl*.hpp`):

| run | case list | surface | EGL config | size | seed |
|---|---|---|---|---|---|
| KHR-GL46 | gl46-main | pbuffer | rgba8888d24s8 (id 1) | 64x64 | 1 |
| KHR-GL46-113x47 | gl46-main | pbuffer | id 1 | 113x47 | 2 |
| KHR-GL46-fbo-64xD | gl46-main | fbo | rgba8888d24s8 | 64 x default | 3 |
| KHR-GL46-fbo-Dx64 | gl46-main | fbo | rgba8888d24s8 | default x 64 | 3 |
| KHR-GL46-cfgN, -cfgN-113x47 | gl46-main | pbuffer | ids 2, 3, 5, 6 | 64x64 / 113x47 | 1 / 2 |
| KHR-Single-GL46 | gl46-khr-single | pbuffer | id 1 | 64x64 | 1 |
| KHR-NoContext | gl45-khr-main | window | id 1 | 64x64 | 1 |

EGL configs (`src/egl/egl.mm`): 1 = RGBA8 D24S8, 2 = RGBA8 without depth/stencil, 3 = RGBA8 D24, 5 = RGB8 D24S8,
6 = RGBA8 D32S8. Config 4 (4x MSAA) isn't part of the matrix: the runner only accepts configs with
`EGL_SAMPLES == 0` (`glcConfigListEGL.cpp`). On that config, 15 cases that count fragment-shader side
effects differ, because Apple GPUs run memory-writing fragment shaders once per sample under MSAA and
Metal can't request once per pixel.

A conformant result has no Fail, Crash or Timeout. NotSupported is allowed only for optional
functionality, for limits above the spec minimum, or where the test itself doesn't apply.

## NotSupported in KHR-GL46 (4495 per run; 4513 on config 2)

| group | cases | reason |
|---|---|---|
| sparse_texture_tests, sparse_texture2_tests, sparse_texture_clamp_tests, sparse_buffer_tests | 2950 | ARB_sparse_texture/texture2/texture_clamp/buffer are optional extensions that metalgldrv doesn't expose |
| negative_texture_lookup_functions_with_bias_tests (sparse variants) | 30 | need ARB_sparse_texture2 |
| fragment_shading_rate | 1067 | EXT_fragment_shading_rate is optional; since the FSR change below it is exposed with METALGLDRV_STRICT=1, where 1052 pass and only the 15 `render_target.*` cases stay NotSupported (they need GL_OVR_multiview) |
| blend_equation_advanced | 88 | KHR_blend_equation_advanced is optional |
| texture_filter_minmax_tests | 5 | ARB_texture_filter_minmax is optional |
| ext_texture_shadow_lod | 15 | EXT_texture_shadow_lod is optional |
| shader_ballot_tests | 4 | ARB_shader_ballot is optional |
| post_depth_coverage_tests | 2 | ARB_post_depth_coverage is optional; it is emulated and exposed with METALGLDRV_STRICT=2 (super strict), where both pass (see "ARB_post_depth_coverage" below) |
| shader_viewport_layer_array | 3 | ARB_shader_viewport_layer_array is optional |
| shaders.shader_integer_mix.prototypes | 1 | EXT_shader_integer_mix is optional |
| sample_variables, es_31_compatibility.sample_variables | 76 | 8-sample variants; GL_MAX_SAMPLES is 4, the GL 4.6 minimum (the GPU supports 1/2/4). 8x emulation was evaluated and rejected, see "8x MSAA" below |
| shader_image_size, shader_image_load_store, shader_texture_image_samples_tests (multisample variants) | 39 | GL_MAX_IMAGE_SAMPLES is 0, the GL 4.6 minimum |
| internalformat.renderbuffer.rgb9_e5 | 1 | RGB9_E5 isn't a required renderable format |
| texture_swizzle | 192 | the test's own static format/target exclusions (depth or RGB9_E5 with multisample/3D targets); no driver query is involved |
| cull_distance item 8 (lines/triangles) | 16 | the test reads gl_CullDistance only for points and skips the other primitive modes by design |
| texture_query_lod | 3 | "Too small viewport" at 64x64 and 113x47; all three pass at 256x256 |
| texture_cube_map_array.generate_mip_map_non_filterable_* | 2 | ES-only test |
| draw_elements_base_vertex_tests.valid_active_tf | 1 | ES-only test |
| shaders.fragdepth.* (config 2 only) | 18 | need a depth buffer; config 2 has none |

## NotSupported in KHR-Single-GL46 (6292)

| group | cases | reason |
|---|---|---|
| subgroups | 6053 | KHR_shader_subgroup is optional |
| meshShader.propertyTests.max_mesh_output_size_*_view_index_* | 8 | multiview variants: GL_MAX_MESH_MULTIVIEW_VIEW_COUNT_EXT is 1, the EXT minimum (GL_OVR_multiview isn't exposed) |

## Warnings

- `KHR-GL46.direct_state_access.framebuffers_check_status`: CompatibilityWarning, which counts as a pass. The test expects
  INCOMPLETE_LAYER_TARGETS for a 3D texture attached with FramebufferTexture3D next to a 2D texture. By the spec that
  attachment isn't layered, so metalgldrv reports COMPLETE, as Mesa does.

# iPad receipts (iPadOS 27.0.1 / Apple M5, driver commit 90f9b17)

The `*-ipad` runs use the shipping device and a newer driver than the Mac matrix above. Since 926bb9c
the driver gained KHR_shader_subgroup, ARB_shader_ballot, ARB_gpu_shader_int64, EXT_mesh_shader, the
ARB_sparse_texture/texture2/texture_clamp/buffer family, KHR_blend_equation_advanced(_coherent),
ARB_texture_filter_minmax (Apple10 GPUs only), EXT_texture_shadow_lod, ARB_shader_viewport_layer_array,
renderable RGB9_E5, METALGLDRV_STRICT, and fixes for several Metal resource leaks.

## NotSupported in KHR-GL46-ipad (1647)

| group | cases | reason |
|---|---|---|
| fragment_shading_rate | 1067 | EXT_fragment_shading_rate is optional (Metal's rasterization rate maps don't match GL's per-draw/per-primitive rates) |
| sparse_texture_tests (SparseDSATextureCommitment, TextureParameterQueries) | 248 | 240 need GL_EXT_direct_state_access, which isn't exposed; 8 are skipped by the test when sparse_texture2 is present |
| texture_swizzle | 192 | the test's own static exclusions (see above) |
| sample_variables, es_31_compatibility.sample_variables | 76 | 8-sample variants; GL_MAX_SAMPLES is 4 (the GPU supports 1/2/4) |
| shader_image_size, shader_image_load_store, shader_texture_image_samples_tests (multisample) | 39 | GL_MAX_IMAGE_SAMPLES is 0 in the default mode; all pass with METALGLDRV_STRICT=1 |
| cull_distance item 8 | 16 | test design (see above) |
| texture_query_lod | 3 | viewport too small at 64x64 |
| post_depth_coverage_tests | 2 | ARB_post_depth_coverage is optional. Metal's sample-mask input doesn't reflect depth/stencil results, so it is emulated with depth/stencil pre-passes, exposed in super strict mode only (METALGLDRV_STRICT=2); feasible, not exposed by default because of its cost |
| texture_cube_map_array (2), draw_elements_base_vertex_tests (1) | 3 | ES-only tests |
| shaders.shader_integer_mix.prototypes | 1 | the test skips GLSL < 4.50 by design |

KHR-Single-GL46-ipad: the 8 NotSupported are `meshShader` multiview variants (GL_MAX_MESH_MULTIVIEW_VIEW_COUNT_EXT is 1,
the minimum).

## Known deviations

- EXT_mesh_shader: two limits sit just below the extension's required minimums because of Metal's threadgroup-memory
  cap. GL_MAX_TASK_SHARED_MEMORY_SIZE_EXT is 32752 (minimum 32768), and GL_MAX_MESH_OUTPUT_MEMORY_SIZE_EXT is 30720
  (minimum 32768). The CTS passes with these values.
- ARB_sparse_texture: after a page is decommitted, the GPU may read its old contents for about 0.2–0.5 s.

# EXT_fragment_shading_rate (METALGLDRV_STRICT=1)

Exposed with `_primitive` and `_attachment` in strict mode only. Metal's rasterization rate maps can't express GL's
per-draw, per-primitive and attachment rates, so coarse shading is emulated in the fragment shader
(docs/glsl-compiler.md, "Coarse shading"). The supported rates are 1x1, 1x2, 2x1 and 2x2, which are the required
ones for 1 and 4 samples. Larger requests clamp as the spec describes, so a coarse fragment always lies inside one
Metal quad. Limits: attachment texels are 2..16 pixels with aspect ratio 1, the attachment has one layer, and it
isn't allowed on the default framebuffer. All five combiner ops are supported. Shaders that write gl_FragDepth or
gl_SampleMask get the 1x1 rate (both `..._SUPPORTED_EXT` queries are FALSE), as do shaders that shade per sample or
read gl_FragCoord without enabling the extension, and draws with sample shading or a partial SAMPLE_MASK_VALUE.
Draws whose rate can only be 1x1 use the regular pipelines, so default mode and non-FSR draws are unchanged.

`KHR-GL46.fragment_shading_rate.*`: 1052 pass, and 15 `render_target.*` are NotSupported because they need
GL_OVR_multiview. The CTS only checks gl_ShadingRateEXT. A shader that failed to compile also passed, because
those cases skip pixels that weren't rendered. The emulation itself (one invocation per coarse fragment,
inputs at the fragment centre, perspective-correct values, derivatives, implicit and queried LOD, discard,
side effects, scissor, partial coverage, the primitive rate from VS/GS/separable/mesh, attachment rates, MSAA
coverage masks, interpolateAt*, quad subgroup operations) is covered by `tests/host/test_fsr.cpp`, which passes on
the M4 Pro and on the iPad M5.

Full runs on the M4 Pro with this change (gl46-main at 64x64 config 1, and gl46-khr-single), 0 Fail/Crash/Timeout:
default KHR-GL46 18061 pass / 1652 NotSupported / 1 CompatibilityWarning (unchanged), strict 19152 / 561 / 1
(previously 18100 / 1613); KHR-Single-GL46 11340 / 8 in both modes. `tests/host` passes in both modes on the Mac and
the iPad.

Known deviations:
- Metal splits polygons that cross the clip volume into several triangles. A coarse fragment that straddles an
  internal edge of that fan is shaded once per triangle, and each invocation covers only its own pixels. Every
  pixel still gets exactly one value, and the values agree because the triangles share the attribute planes.
  The extra invocation and the partial gl_SampleMaskIn are visible, though. At 1x1 the same split only affects
  multisampled pixels, where GL allows several invocations per fragment.
- Every pixel lane still runs the shader. The emulation provides the semantics, not a speedup.

# 8x MSAA (GL_MAX_SAMPLES = 8): not exposed

Both GPUs render 1, 2 or 4 samples. The probes behind this analysis are on the M4 Pro: custom sample positions
(`setSamplePositions`) do move coverage, helper-lane writes are dropped, the vertex function runs once per view
under vertex amplification, and so does a mesh function.

What an emulation with two 4x halves (samples 0-3 and 4-7 at different programmable positions) can keep within
GL's rules:
- Fragment shaders running once per covered half. GL 4.6 7.13.1 allows 1..N invocations per fragment when
  multisampled, gl_SampleMaskIn holds the samples "covered for this invocation", and every covered sample is in
  exactly one invocation.
- Interpolation at the half's pixel centre ("anywhere within the pixel", 14.3.1).
- gl_SampleID / gl_SampleMask / gl_SamplePosition with a +4 / <<4 offset in the second half.
- Alpha-to-coverage giving twice the 4-sample count (only proportionality is required).
- Occlusion queries summed over both halves, and vertex or tessellation-evaluation shaders running twice (both
  "may be executed more than once").

What it breaks:
1. Geometry, task and mesh shader side effects run twice. GL fixes those invocation counts, unlike VS/TES. Both
   ways to feed two halves re-run these stages: drawing twice, or one draw with vertex amplification into two
   slices (a shifted viewport per slice gives the second sample set). The probes showed the vertex and mesh
   functions running once per view. Masking the stores in the second run doesn't fix it either, because atomics
   return different values and geometry that depends on them diverges between the halves. Only capturing the
   post-GS / mesh output once and rasterizing it twice is correct.
2. With two passes, the second half of draw N runs after the first half of draw N+1. A later draw's buffer, image
   or atomic writes then reach an earlier draw's replayed vertex fetch, uniforms or indirect arguments, and
   glMemoryBarrier / glTextureBarrier ordering breaks too. Metal 4 allows one open render encoder per command
   buffer, so the second half has to be a recorded replay of every encoder and argument-table call. That replay
   must split at every side-effecting draw and barrier. Transform feedback, primitive queries and VS-captured
   feedback must run in one half only.
3. The amplification variant needs a viewport per half. That is 2x GL_MAX_VIEWPORTS (16, the GL minimum) Metal
   viewports, and Metal has 16, so shaders that write gl_ViewportIndex would need amplification-dependent
   position shifts, which re-run the pre-rasterization stages again (point 1). A double-width 4x layout avoids
   the second pass. But then lines, points and polygon-offset slopes are distorted along one axis, 8x surfaces
   are limited to 8192 pixels against a MAX_TEXTURE_SIZE of 16384, and every fragment shader needs the coarse
   emulation.

Beyond these, every multisample path (storage, texelFetch, textureSamples, resolves, blits, CopyImageSubData,
views, clears, invalidation, readback, sparse multisample storage, internalformat queries) would need an 8x form.
A version without the fixes in 1 and 2 deviates in ways GL forbids, and the full version means capturing
GS/mesh output and replaying render passes across the whole draw path, for the 76 cases above. GL_MAX_SAMPLES
therefore stays 4, the GL 4.6 minimum, which is conformant.

# ARB_post_depth_coverage (METALGLDRV_STRICT=2, super strict)

Metal's `[[sample_mask]]` fragment input doesn't reflect the depth/stencil tests, even with
`[[early_fragment_tests]]` (probed on the M4 Pro and the M5), so the extension is emulated, and exposed only in
super strict mode because of its cost. `layout(post_depth_coverage) in;` implies early fragment tests. A draw whose
fragment shader reads gl_SampleMaskIn, with a depth or stencil test in effect, first renders a pre-pass: the same
vertex function (invariant position), rasterizer and depth/stencil tests, with depth/stencil writes off, into a
transient R8 coverage attachment that a fragment function sets to 1. The draw then ANDs gl_SampleMaskIn with its
pixel's marked samples, which are exactly the samples that pass the early tests. Without depth/stencil writes one
pre-pass per draw is exact. With writes, primitives of the draw affect each other in order, so the draw is split
into a pre-pass and a draw per primitive and instance, in GL order (strips, fans, loops, adjacency and primitive
restart decomposed on the CPU, provoking vertices kept, gl_PrimitiveID / gl_InstanceID / per-instance attributes
preserved). That costs two render passes per primitive. Pre-passes don't count for occlusion or primitive queries
or transform feedback, and bind copies of the storage buffers, atomic counters and writable images that the
pre-rasterization stages use, so their side effects happen once. Indirect draws are read back on the CPU.
Single-sampled targets get the same logic with a one-bit mask. `tests/host/test_pdc.cpp` covers occluded samples,
intra-draw occlusion (depth and stencil), restart strips/fans, instancing, 1/2/4 samples, layered targets, a
geometry shader and the side-effect copies.

Limitations: geometry/tessellation draws, and vertex shaders that run on the mesh pipeline because their outputs
exceed the classic varying budget, get one pre-pass per draw. Their rasterized primitives are produced on the GPU,
so the CPU can't split the draw per primitive. When such a draw writes depth or stencil and its own primitives
overlap, the later ones' masks include samples that an earlier primitive of the same draw occluded. Vertex
processing whose positions depend on atomic results can differ between the pre-pass and the draw.

`KHR-GL46.post_depth_coverage_tests.*`: both pass in super strict mode (previously NotSupported). Full KHR-GL46
runs on the M4 Pro (64x64, config 1) with this change, 0 Fail/Crash/Timeout: super strict 19154 pass / 559
NotSupported / 1 CompatibilityWarning, strict 19152 / 561 / 1 and default 18061 / 1652 / 1 (both unchanged). `tests/host` passes at all three levels.
