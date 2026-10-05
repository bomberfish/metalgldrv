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
| `msl_prelude.cpp` | demand-driven MSL helper prelude (+ fp64 library hook) |

Tools: `build/tools/glslc` (compile/link/emit/validate, `--time`), `build/tools/glslrun` (executes self-checking compute tests and `--draw` graphics tests on the GPU).
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

## Not yet supported
Geometry/tessellation shaders and transform-feedback capture (generateMSL returns an error; reflection is produced), subroutines, opaque types inside structs, arrays of samplers/images as function parameters, uniform initializer values in reflection, interpolateAt* (returns the interpolated input), gl_CullDistance, fragment-stage gl_ClipDistance reads, float image atomics, textureGatherOffsets.
