# CTS receipts: run matrix and NotSupported audit (commit 926bb9c)

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
