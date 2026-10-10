# Vendor extensions

Besides OpenGL 4.6 and the ARB / EXT / KHR extensions in `src/gl/api_get.mm`, metalgldrv implements
Apple's own extensions, so code written against Apple's OpenGL runs unchanged, and the vendor
extensions that games from the fixed-function and early-shader era look for. Each has a test in
`tests/host/` (`test_extensions.cpp`, `test_compat.cpp`, `test_arb_program.cpp`).

## Apple (`GL_APPLE_*`)

All of Apple's OpenGL extensions, including two older ones its current driver no longer lists:

| Extension | How |
|---|---|
| `aux_depth_stencil` | the query answers FALSE (there are no aux buffers) |
| `client_storage` | the pixel store flag; uploads copy, which the extension allows |
| `container_object_shareable` | CGL / NSOpenGL contexts share framebuffers, vertex arrays and program pipelines across their share group, as Apple's does (not EGL contexts, nor Windows programs under Wine) |
| `element_array` | a client index array drawn with `glDrawElements` |
| `fence` | command buffer serials, like sync objects |
| `float_pixels` | ARB_texture_float's formats, including the alpha / luminance / intensity ones |
| `flush_buffer_range` | unsynchronized and explicitly flushed `glMapBuffer` |
| `flush_render` | flush / finish without presenting, `glSwapAPPLE` |
| `object_purgeable` | Metal purgeable states |
| `packed_pixels` | the GL 1.2 packed pixel types |
| `pixel_buffer` | CGL pbuffers and their minimum viewport query |
| `rgb_422`, `ycbcr_422` | 4:2:2 data unpacked at upload (Y'CbCr kept / converted to RGB with BT.601) |
| `row_bytes` | byte row strides for pixel transfers |
| `specular_vector` | Phong specular in fixed-function lighting |
| `texture_range` | recorded hints |
| `transform_hint` | a hint |
| `vertex_array_object` | VAOs created on first bind |
| `vertex_array_range` | recorded (client arrays are read when drawing) |
| `vertex_point_size` | per-vertex point sizes in fixed-function drawing |
| `vertex_program_evaluators` | evaluator maps for generic attributes (on the GL 1.0 evaluators) |

## Other vendors

NVIDIA: `fence`, `fog_distance`, `half_float`, `light_max_exponent`, `occlusion_query`,
`packed_depth_stencil`, `primitive_restart`, `texgen_reflection`, `texture_env_combine4`,
`texture_rectangle` (plus `blend_square`, `conditional_render`, `depth_clamp`, `texture_barrier`).
ATI: `draw_buffers`, `separate_stencil`, `texture_compression_3dc`, `texture_env_combine3`,
`texture_float`, `texture_mirror_once`, `meminfo`. HP: `occlusion_test`. IBM: `rasterpos_clip`.
SGI: `SGIS_generate_mipmap`, `SGIS_texture_edge_clamp`, `SGIS_texture_border_clamp`, `SGIS_texture_lod`,
`SGIX_depth_texture`, `SGIX_shadow`. Multi-vendor EXT ones of the same era: `compiled_vertex_array`,
`debug_label`, `debug_marker`, `fog_coord`, `framebuffer_multisample_blit_scaled`, `geometry_shader4`,
`point_parameters`, `polygon_offset`, `secondary_color`, `texture_compression_latc`,
`texture_cube_map`, `texture_env_add`, `texture_env_combine`, `texture_env_dot3`, `texture_object`,
`vertex_array`, `clip_volume_hint`.

`GL_ARB_vertex_program` / `GL_ARB_fragment_program` (and `_shadow`) run assembly programs by translating
them to GLSL (`src/gl/arb_program.cpp`).

### Not implemented

These would each need a separate emulation layer for little use today, so they stay out:

* `GL_NV_register_combiners`, `GL_NV_texture_shader`, `GL_ATI_fragment_shader`: pre-shader fragment
  pipelines; games that have these paths also have ARB_fragment_program or GLSL ones.
* `GL_NV_vertex_program*`, `GL_NV_fragment_program*` and the NV options for ARB programs: NVIDIA's
  assembly dialects (branching, condition codes).
* `GL_ARB_vertex_blend`, `GL_ARB_matrix_palette`, program matrices (`state.matrix.program[n]`):
  fixed-function skinning.
* `GL_EXT_paletted_texture`, `GL_ARB_imaging`, pixel maps, polygon stipple, feedback / selection.
* `GL_NV_vertex_array_range` (it needs `wglAllocateMemoryNV`), `GL_ARB_shadow_ambient`.

## Old games and the extension string

Games from around 1999–2005 copy `glGetString(GL_EXTENSIONS)` into a fixed-size buffer and can crash
on a long one (ours is about 6 KB). `METALGLDRV_EXTENSION_YEAR=<year>` (or `extensionYear` in an
app profile, `src/gl/app_profiles.cpp`) lists only extensions published up to that year, oldest
first, like NVIDIA's `__GL_ExtensionStringVersion`; `glGetStringi` still lists everything. When a
game of that age fails at startup, try the year it was released.
