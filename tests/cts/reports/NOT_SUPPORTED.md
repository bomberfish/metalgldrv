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
| fragment_shading_rate | 1067 | EXT_fragment_shading_rate is optional |
| blend_equation_advanced | 88 | KHR_blend_equation_advanced is optional |
| texture_filter_minmax_tests | 5 | ARB_texture_filter_minmax is optional |
| ext_texture_shadow_lod | 15 | EXT_texture_shadow_lod is optional |
| shader_ballot_tests | 4 | ARB_shader_ballot is optional |
| post_depth_coverage_tests | 2 | ARB_post_depth_coverage is optional |
| shader_viewport_layer_array | 3 | ARB_shader_viewport_layer_array is optional |
| shaders.shader_integer_mix.prototypes | 1 | EXT_shader_integer_mix is optional |
| sample_variables, es_31_compatibility.sample_variables | 76 | 8-sample variants; GL_MAX_SAMPLES is 4, the GL 4.6 minimum (the GPU supports 1/2/4) |
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
| post_depth_coverage_tests | 2 | not implementable on Metal: the fragment sample-mask input doesn't reflect depth/stencil results |
| texture_cube_map_array (2), draw_elements_base_vertex_tests (1) | 3 | ES-only tests |
| shaders.shader_integer_mix.prototypes | 1 | the test skips GLSL < 4.50 by design |

KHR-Single-GL46-ipad: the 8 NotSupported are `meshShader` multiview variants (GL_MAX_MESH_MULTIVIEW_VIEW_COUNT_EXT is 1,
the minimum).

## Known deviations

- EXT_mesh_shader: two limits sit just below the extension's required minimums because of Metal's threadgroup-memory
  cap. GL_MAX_TASK_SHARED_MEMORY_SIZE_EXT is 32752 (minimum 32768), and GL_MAX_MESH_OUTPUT_MEMORY_SIZE_EXT is 30720
  (minimum 32768). The CTS passes with these values.
- ARB_sparse_texture: after a page is decommitted, the GPU may read its old contents for about 0.2–0.5 s.
