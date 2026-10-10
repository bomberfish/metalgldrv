// glGet* state queries, glGetString(i).
#import "gl/texture.h"
#include "mgl_version.h"
#include <cmath>
#include "gl/app_profile.h"
#include "gl/ext_years.h"
#include "util/log.h"

using namespace mgl;
#define API extern "C" __attribute__((visibility("default")))

namespace {

const char* kExtensions[] = {
    "GL_APPLE_aux_depth_stencil", "GL_APPLE_client_storage", "GL_APPLE_element_array", "GL_APPLE_fence", "GL_APPLE_float_pixels", "GL_APPLE_flush_buffer_range", "GL_APPLE_flush_render", "GL_APPLE_object_purgeable", "GL_APPLE_packed_pixels", "GL_APPLE_pixel_buffer", "GL_APPLE_rgb_422", "GL_APPLE_row_bytes", "GL_APPLE_specular_vector", "GL_APPLE_texture_range", "GL_APPLE_transform_hint", "GL_APPLE_vertex_point_size", "GL_APPLE_vertex_program_evaluators", "GL_APPLE_ycbcr_422", "GL_APPLE_vertex_array_object", "GL_APPLE_vertex_array_range", "GL_ARB_ES2_compatibility", "GL_ARB_ES3_compatibility", "GL_ARB_ES3_1_compatibility", "GL_ARB_arrays_of_arrays",
    "GL_ARB_base_instance", "GL_ARB_blend_func_extended", "GL_ARB_buffer_storage", "GL_ARB_clear_buffer_object",
    "GL_ARB_clear_texture", "GL_ARB_clip_control", "GL_ARB_color_buffer_float", "GL_ARB_compressed_texture_pixel_storage",
    "GL_ARB_compute_shader", "GL_ARB_conditional_render_inverted", "GL_ARB_conservative_depth", "GL_ARB_copy_buffer",
    "GL_ARB_copy_image", "GL_ARB_cull_distance", "GL_ARB_debug_output", "GL_ARB_depth_buffer_float", "GL_ARB_depth_clamp",
    "GL_ARB_depth_texture", "GL_ARB_derivative_control", "GL_ARB_direct_state_access", "GL_ARB_draw_buffers",
    "GL_ARB_draw_buffers_blend", "GL_ARB_draw_elements_base_vertex", "GL_ARB_draw_indirect", "GL_ARB_draw_instanced",
    "GL_ARB_enhanced_layouts", "GL_ARB_explicit_attrib_location", "GL_ARB_explicit_uniform_location",
    "GL_ARB_fragment_coord_conventions", "GL_ARB_fragment_layer_viewport", "GL_ARB_fragment_program", "GL_ARB_fragment_program_shadow", "GL_ARB_fragment_shader",
    "GL_ARB_framebuffer_no_attachments", "GL_ARB_framebuffer_object", "GL_ARB_framebuffer_sRGB", "GL_ARB_geometry_shader4",
    "GL_ARB_get_program_binary", "GL_ARB_get_texture_sub_image", "GL_ARB_gl_spirv", "GL_ARB_gpu_shader5", "GL_ARB_gpu_shader_fp64", "GL_ARB_gpu_shader_int64",
    "GL_ARB_half_float_pixel", "GL_ARB_half_float_vertex", "GL_ARB_indirect_parameters", "GL_ARB_instanced_arrays",
    "GL_ARB_internalformat_query", "GL_ARB_internalformat_query2", "GL_ARB_invalidate_subdata", "GL_ARB_map_buffer_alignment",
    "GL_ARB_map_buffer_range", "GL_ARB_multi_bind", "GL_ARB_multi_draw_indirect", "GL_ARB_multisample",
    "GL_ARB_multitexture", "GL_ARB_occlusion_query", "GL_ARB_occlusion_query2", "GL_ARB_pipeline_statistics_query", "GL_ARB_pixel_buffer_object",
    "GL_ARB_point_parameters", "GL_ARB_point_sprite", "GL_ARB_polygon_offset_clamp", "GL_ARB_program_interface_query",
    "GL_ARB_provoking_vertex", "GL_ARB_query_buffer_object", "GL_ARB_robust_buffer_access_behavior", "GL_ARB_robustness",
    "GL_ARB_sample_shading", "GL_ARB_sampler_objects", "GL_ARB_seamless_cube_map", "GL_ARB_seamless_cubemap_per_texture",
    "GL_ARB_separate_shader_objects", "GL_ARB_shader_atomic_counter_ops", "GL_ARB_shader_ballot", "GL_ARB_shader_atomic_counters",
    "GL_ARB_shader_bit_encoding", "GL_ARB_shader_draw_parameters", "GL_ARB_shader_group_vote",
    "GL_ARB_shader_image_load_store", "GL_ARB_shader_image_size", "GL_ARB_shader_objects", "GL_ARB_shader_precision",
    "GL_ARB_shader_storage_buffer_object", "GL_ARB_shader_subroutine", "GL_ARB_shader_texture_image_samples", "GL_ARB_shader_viewport_layer_array",
    "GL_ARB_shader_texture_lod", "GL_ARB_shading_language_100", "GL_ARB_shading_language_420pack",
    "GL_ARB_shading_language_packing", "GL_ARB_shadow", "GL_ARB_spirv_extensions", "GL_ARB_stencil_texturing", "GL_ARB_sync",
    "GL_ARB_tessellation_shader", "GL_ARB_texture_barrier", "GL_ARB_texture_border_clamp", "GL_ARB_texture_buffer_object",
    "GL_ARB_texture_buffer_object_rgb32", "GL_ARB_texture_buffer_range", "GL_ARB_texture_compression",
    "GL_ARB_texture_compression_bptc", "GL_ARB_texture_compression_rgtc", "GL_ARB_texture_cube_map",
    "GL_ARB_texture_cube_map_array", "GL_ARB_texture_env_add", "GL_ARB_texture_env_combine", "GL_ARB_texture_env_crossbar",
    "GL_ARB_texture_env_dot3", "GL_ARB_texture_filter_anisotropic", "GL_ARB_texture_float", "GL_ARB_texture_gather",
    "GL_ARB_texture_mirror_clamp_to_edge", "GL_ARB_texture_mirrored_repeat", "GL_ARB_texture_multisample",
    "GL_ARB_texture_non_power_of_two", "GL_ARB_texture_query_levels", "GL_ARB_texture_query_lod", "GL_ARB_texture_rectangle",
    "GL_ARB_texture_rg", "GL_ARB_texture_rgb10_a2ui", "GL_ARB_texture_stencil8", "GL_ARB_texture_storage",
    "GL_ARB_texture_storage_multisample", "GL_ARB_texture_swizzle", "GL_ARB_texture_view", "GL_ARB_timer_query",
    "GL_ARB_transform_feedback2", "GL_ARB_transform_feedback3", "GL_ARB_transform_feedback_instanced",
    "GL_ARB_transform_feedback_overflow_query", "GL_ARB_transpose_matrix", "GL_ARB_uniform_buffer_object",
    "GL_ARB_vertex_array_bgra", "GL_ARB_vertex_array_object", "GL_ARB_vertex_attrib_64bit", "GL_ARB_vertex_attrib_binding",
    "GL_ARB_vertex_buffer_object", "GL_ARB_vertex_program", "GL_ARB_vertex_shader", "GL_ARB_vertex_type_10f_11f_11f_rev",
    "GL_ARB_vertex_type_2_10_10_10_rev", "GL_ARB_viewport_array", "GL_ARB_window_pos", "GL_EXT_abgr", "GL_EXT_bgra",
    "GL_EXT_blend_color", "GL_EXT_blend_equation_separate", "GL_EXT_blend_func_separate", "GL_EXT_blend_minmax",
    "GL_EXT_blend_subtract", "GL_EXT_clip_volume_hint", "GL_EXT_debug_label", "GL_EXT_debug_marker", "GL_EXT_compiled_vertex_array", "GL_EXT_draw_buffers2", "GL_EXT_draw_instanced", "GL_EXT_draw_range_elements", "GL_EXT_fog_coord",
    "GL_EXT_framebuffer_blit", "GL_EXT_framebuffer_multisample", "GL_EXT_framebuffer_multisample_blit_scaled", "GL_EXT_framebuffer_object", "GL_EXT_framebuffer_sRGB",
    "GL_EXT_geometry_shader4", "GL_EXT_gpu_program_parameters", "GL_EXT_gpu_shader4", "GL_EXT_mesh_shader", "GL_EXT_multi_draw_arrays", "GL_EXT_packed_depth_stencil",
    "GL_EXT_packed_float", "GL_EXT_packed_pixels", "GL_EXT_pixel_buffer_object", "GL_EXT_point_parameters", "GL_EXT_polygon_offset", "GL_EXT_polygon_offset_clamp",
    "GL_EXT_provoking_vertex", "GL_EXT_rescale_normal", "GL_EXT_secondary_color", "GL_EXT_separate_specular_color", "GL_EXT_shader_integer_mix",
    "GL_EXT_shadow_funcs", "GL_EXT_stencil_two_side", "GL_EXT_stencil_wrap", "GL_EXT_texture3D", "GL_EXT_texture_cube_map", "GL_EXT_texture_array",
    "GL_EXT_texture_compression_dxt1", "GL_EXT_texture_compression_latc", "GL_EXT_texture_compression_rgtc", "GL_EXT_texture_compression_s3tc",
    "GL_EXT_texture_edge_clamp", "GL_EXT_texture_env_add", "GL_EXT_texture_env_combine", "GL_EXT_texture_env_dot3", "GL_EXT_texture_filter_anisotropic", "GL_EXT_texture_integer", "GL_EXT_texture_lod_bias",
    "GL_EXT_texture_mirror_clamp", "GL_EXT_texture_object", "GL_EXT_texture_rectangle", "GL_EXT_texture_sRGB", "GL_EXT_texture_sRGB_decode",
    "GL_EXT_texture_shadow_lod", "GL_EXT_texture_shared_exponent", "GL_EXT_texture_snorm", "GL_EXT_texture_swizzle", "GL_EXT_timer_query",
    "GL_EXT_transform_feedback", "GL_EXT_vertex_array", "GL_EXT_vertex_array_bgra", "GL_HP_occlusion_test", "GL_IBM_rasterpos_clip", "GL_IBM_texture_mirrored_repeat", "GL_KHR_context_flush_control",
    "GL_KHR_debug", "GL_KHR_no_error", "GL_KHR_parallel_shader_compile", "GL_KHR_robust_buffer_access_behavior",
    "GL_KHR_robustness", "GL_KHR_shader_subgroup", "GL_KHR_texture_compression_astc_ldr", "GL_NV_blend_square", "GL_NV_conditional_render",
    "GL_NV_depth_clamp", "GL_NV_fence", "GL_NV_fog_distance", "GL_NV_half_float", "GL_NV_light_max_exponent", "GL_NV_occlusion_query", "GL_NV_packed_depth_stencil", "GL_NV_primitive_restart", "GL_NV_texgen_reflection", "GL_NV_texture_rectangle", "GL_NV_shader_subgroup_partitioned", "GL_NV_texture_barrier", "GL_NV_texture_env_combine4", "GL_SGIS_generate_mipmap", "GL_SGIS_texture_edge_clamp", "GL_SGIS_texture_border_clamp", "GL_SGIS_texture_lod", "GL_SGIX_depth_texture",
    "GL_ATI_draw_buffers", "GL_ATI_meminfo", "GL_ATI_separate_stencil", "GL_ATI_texture_compression_3dc", "GL_ATI_texture_env_combine3", "GL_ATI_texture_float", "GL_ATI_texture_mirror_once", "GL_NVX_gpu_memory_info",
};
struct DeviceExtension {
    const char* name;
    bool glsl;  // also a GLSL #extension
    bool (*supported)(const DeviceCaps&);
};
const DeviceExtension kDeviceExtensions[] = {
    {"GL_ARB_texture_filter_minmax", false, [](const DeviceCaps& c) { return c.samplerReduction; }},
    {"GL_KHR_blend_equation_advanced", true, [](const DeviceCaps& c) { return c.programmableBlending; }},
    {"GL_KHR_blend_equation_advanced_coherent", false, [](const DeviceCaps& c) { return c.programmableBlending; }},
    // The GLSL side of sparse_texture2/_clamp is gated by CompileOptions::sparseTexture.
    {"GL_ARB_sparse_buffer", false, [](const DeviceCaps& c) { return c.sparse; }},
    {"GL_ARB_sparse_texture", false, [](const DeviceCaps& c) { return c.sparse; }},
    {"GL_ARB_sparse_texture2", false, [](const DeviceCaps& c) { return c.sparse; }},
    {"GL_ARB_sparse_texture_clamp", false, [](const DeviceCaps& c) { return c.sparse; }},
    // coarse shading emulated in the fragment shader (glsl msl.cpp, emitFS)
    {"GL_EXT_fragment_shading_rate", true, [](const DeviceCaps& c) { return c.strict(); }},
    {"GL_EXT_fragment_shading_rate_attachment", false, [](const DeviceCaps& c) { return c.strict(); }},
    {"GL_EXT_fragment_shading_rate_primitive", false, [](const DeviceCaps& c) { return c.strict(); }},
    // depth/stencil pre-passes into a coverage attachment (draw.mm, executePdc)
    {"GL_ARB_post_depth_coverage", true, [](const DeviceCaps& c) { return c.superStrict(); }},
};

// kExtensions plus the ones that depend on the GPU (mglDeviceExtensions), sorted.
struct ExtensionList {
    std::vector<const char*> names;
    std::string joined;  // glGetString(GL_EXTENSIONS): limited by AppProfile::extensionYear, oldest first
};

// GL_NVX_gpu_memory_info (not in gl.xml)
constexpr GLenum GL_GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX = 0x9047;
constexpr GLenum GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX = 0x9048;
constexpr GLenum GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX = 0x9049;
constexpr GLenum GL_GPU_MEMORY_INFO_EVICTION_COUNT_NVX = 0x904A;
constexpr GLenum GL_GPU_MEMORY_INFO_EVICTED_MEMORY_NVX = 0x904B;

// Video memory for GL_NVX_gpu_memory_info / GL_ATI_meminfo, in KiB: Metal's recommended working set
// (the memory the GPU can use without paging; unified memory has no dedicated pool), capped or replaced
// by the app profile (vramCapMB / vramMB), and what the device has not allocated of it.
void videoMemoryKB(Context* ctx, int64_t* total, int64_t* avail) {
    const AppProfile& app = appProfile();
    uint64_t bytes = (uint64_t)ctx->dev->mtl.recommendedMaxWorkingSetSize;
    if (app.vramCapMB) bytes = std::min<uint64_t>(bytes, app.vramCapMB << 20);
    if (app.vramMB) bytes = app.vramMB << 20;
    uint64_t used = (uint64_t)ctx->dev->mtl.currentAllocatedSize;
    *total = (int64_t)std::min<uint64_t>(bytes >> 10, INT32_MAX);
    *avail = (int64_t)std::min<uint64_t>((bytes > used ? bytes - used : 0) >> 10, INT32_MAX);
}

// Two lists: contexts made through CGL / NSOpenGL also have Apple's container object sharing.
const ExtensionList& extensions(Context* ctx) {
    auto build = [ctx](bool appleSharing) {
        ExtensionList r;
        const AppProfile& app = appProfile();
        r.names.assign(std::begin(kExtensions), std::end(kExtensions));
        if (appleSharing) r.names.push_back("GL_APPLE_container_object_shareable");
        for (const char* e : mglDeviceExtensions(ctx->dev->caps)) r.names.push_back(e);
        std::erase_if(r.names, [&](const char* e) { return app.hidden(e); });
        for (const auto& e : app.extraExtensions)
            if (std::find_if(r.names.begin(), r.names.end(), [&](const char* x) { return e == x; }) == r.names.end())
                r.names.push_back(e.c_str());
        std::sort(r.names.begin(), r.names.end(), [](const char* a, const char* b) { return strcmp(a, b) < 0; });
        std::vector<const char*> legacy = r.names;
        if (app.extensionYear) {
            // Unknown years count as new. Oldest first, so a truncating copy keeps the basics.
            auto year = [](const char* e) { int y = extensionYear(e); return y ? y : 9999; };
            std::erase_if(legacy, [&](const char* e) { return year(e) > app.extensionYear; });
            std::stable_sort(legacy.begin(), legacy.end(), [&](const char* a, const char* b) { return year(a) < year(b); });
        }
        for (auto e : legacy) {
            if (!r.joined.empty()) r.joined += ' ';
            r.joined += e;
        }
        return r;
    };
    if (ctx->share && ctx->share->shareContainers) {
        static const ExtensionList apple = build(true);
        return apple;
    }
    static const ExtensionList l = build(false);
    return l;
}

enum class VT { Bool, Int, Int64, Float, Double, Enum };

struct Value {
    VT type = VT::Int;
    int n = 1;
    union {
        int64_t i[16];
        double d[16];
    };
    Value() { memset(i, 0, sizeof i); }
};

Value I(int64_t a) { Value v; v.type = VT::Int; v.i[0] = a; return v; }
Value I64(int64_t a) { Value v; v.type = VT::Int64; v.i[0] = a; return v; }
Value E(GLenum a) { Value v; v.type = VT::Enum; v.i[0] = a; return v; }
Value B(bool a) { Value v; v.type = VT::Bool; v.i[0] = a; return v; }
Value F(double a) { Value v; v.type = VT::Float; v.d[0] = a; return v; }
Value I2(int64_t a, int64_t b) { Value v; v.type = VT::Int; v.n = 2; v.i[0] = a; v.i[1] = b; return v; }
Value I3(int64_t a, int64_t b, int64_t c) { Value v; v.type = VT::Int; v.n = 3; v.i[0] = a; v.i[1] = b; v.i[2] = c; return v; }
Value I4(int64_t a, int64_t b, int64_t c, int64_t d) { Value v; v.type = VT::Int; v.n = 4; v.i[0] = a; v.i[1] = b; v.i[2] = c; v.i[3] = d; return v; }
Value F2(double a, double b) { Value v; v.type = VT::Float; v.n = 2; v.d[0] = a; v.d[1] = b; return v; }
Value F4(const float* f) { Value v; v.type = VT::Float; v.n = 4; for (int k = 0; k < 4; k++) v.d[k] = f[k]; return v; }
Value B4(bool a, bool b, bool c, bool d) { Value v; v.type = VT::Bool; v.n = 4; v.i[0] = a; v.i[1] = b; v.i[2] = c; v.i[3] = d; return v; }

GLuint nameOf(Object* o) { return o ? o->name : 0; }

}  // namespace

std::vector<const char*> mgl::mglDeviceExtensions(const DeviceCaps& caps) {
    std::vector<const char*> r;
    for (const auto& e : kDeviceExtensions)
        if (e.supported(caps)) r.push_back(e.name);
    return r;
}

const std::vector<std::string>& mgl::mglDeviceGlslExtensions(const DeviceCaps& caps) {
    static const std::vector<std::string> r = [&caps] {
        std::vector<std::string> v;
        for (const auto& e : kDeviceExtensions)
            if (e.glsl && e.supported(caps)) v.push_back(e.name);
        return v;
    }();
    return r;
}

bool mglIsEnabled(Context* ctx, GLenum cap, int index, bool* out);
bool mglCompatGet(Context* ctx, GLenum pname, double* out, int* n, bool* isFloat);  // compat.mm
bool mglCompatGetPointer(Context* ctx, GLenum pname, void** out);                    // compat.mm

static bool getState(Context* ctx, GLenum pname, int index, Value& v) {
    const int idx = index < 0 ? 0 : index;
    // Limits that GLSL also exposes as gl_Max* constants come from the shared table.
    static const std::pair<GLenum, int> kShaderLimits[] = {
#define X(name, glenum, value) {glenum, value},
#include "glsl/limits.def"
#undef X
    };
    if (pname == GL_MAX_IMAGE_SAMPLES && ctx->strict()) {
        v = I(glsl::kStrictMaxImageSamples);
        return true;
    }
    if (ctx->strict()) {
        // EXT_fragment_shading_rate (+ _primitive, _attachment)
        switch (pname) {
        case GL_SHADING_RATE_EXT: v = E(ctx->shadingRate); return true;
        case GL_MIN_FRAGMENT_SHADING_RATE_ATTACHMENT_TEXEL_WIDTH_EXT:
        case GL_MIN_FRAGMENT_SHADING_RATE_ATTACHMENT_TEXEL_HEIGHT_EXT: v = I(2); return true;
        case GL_MAX_FRAGMENT_SHADING_RATE_ATTACHMENT_TEXEL_WIDTH_EXT:
        case GL_MAX_FRAGMENT_SHADING_RATE_ATTACHMENT_TEXEL_HEIGHT_EXT: v = I(16); return true;
        case GL_MAX_FRAGMENT_SHADING_RATE_ATTACHMENT_TEXEL_ASPECT_RATIO_EXT: v = I(1); return true;
        case GL_MAX_FRAGMENT_SHADING_RATE_ATTACHMENT_LAYERS_EXT: v = I(1); return true;
        case GL_FRAGMENT_SHADING_RATE_WITH_SHADER_DEPTH_STENCIL_WRITES_SUPPORTED_EXT: v = B(false); return true;
        case GL_FRAGMENT_SHADING_RATE_WITH_SAMPLE_MASK_SUPPORTED_EXT: v = B(false); return true;
        case GL_FRAGMENT_SHADING_RATE_ATTACHMENT_WITH_DEFAULT_FRAMEBUFFER_SUPPORTED_EXT: v = B(false); return true;
        case GL_FRAGMENT_SHADING_RATE_NON_TRIVIAL_COMBINERS_SUPPORTED_EXT: v = B(true); return true;
        case GL_FRAGMENT_SHADING_RATE_PRIMITIVE_RATE_WITH_MULTI_VIEWPORT_SUPPORTED_EXT: v = B(true); return true;
        default: break;
        }
    }
    if (pname != 0)
        for (auto& [e, val] : kShaderLimits)
            if (e == pname) {
                if (pname == GL_MAX_CLIP_PLANES && ctx->coreProfile) break;  // compatibility-only
                v = I(val);
                return true;
            }
    switch (pname) {
    // --- GL_NVX_gpu_memory_info, GL_ATI_meminfo (KiB) ---
    case GL_GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX:
    case GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX:
    case GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX:
    case GL_VBO_FREE_MEMORY_ATI:
    case GL_TEXTURE_FREE_MEMORY_ATI:
    case GL_RENDERBUFFER_FREE_MEMORY_ATI: {
        if (index >= 0) return false;
        int64_t total, avail;
        videoMemoryKB(ctx, &total, &avail);
        if (pname == GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX) v = I(avail);
        else if (pname == GL_GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX || pname == GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX) v = I(total);
        else v = I4(avail, avail, 0, 0);  // free total, largest free block, free auxiliary (none), largest auxiliary
        return true;
    }
    case GL_GPU_MEMORY_INFO_EVICTION_COUNT_NVX:
    case GL_GPU_MEMORY_INFO_EVICTED_MEMORY_NVX:
        if (index >= 0) return false;
        v = I(0);
        return true;
    // --- implementation limits ---
    case GL_MAJOR_VERSION: v = I(4); return true;
    case GL_MINOR_VERSION: v = I(6); return true;
    case GL_CONTEXT_FLAGS: {
        int f = 0;
        if (ctx->debugContext) f |= GL_CONTEXT_FLAG_DEBUG_BIT;
        if (ctx->forwardCompatible) f |= GL_CONTEXT_FLAG_FORWARD_COMPATIBLE_BIT;
        if (ctx->robustAccess) f |= GL_CONTEXT_FLAG_ROBUST_ACCESS_BIT;
        if (ctx->noErrorMode) f |= GL_CONTEXT_FLAG_NO_ERROR_BIT;
        v = I(f);
        return true;
    }
    case GL_CONTEXT_PROFILE_MASK: v = I(ctx->coreProfile ? GL_CONTEXT_CORE_PROFILE_BIT : GL_CONTEXT_COMPATIBILITY_PROFILE_BIT); return true;
    case GL_NUM_EXTENSIONS: v = I((int64_t)extensions(ctx).names.size()); return true;
    case GL_NUM_SHADING_LANGUAGE_VERSIONS: v = I(17); return true;
    case GL_NUM_SPIR_V_EXTENSIONS: v = I((int)glsl::supportedSpirvExtensions().size()); return true;
    case GL_CLAMP_READ_COLOR: v = E(ctx->clampReadColor); return true;
    case GL_PRIMITIVE_RESTART_FOR_PATCHES_SUPPORTED: v = B(false); return true;  // tess path ignores restart
    case GL_RESET_NOTIFICATION_STRATEGY: v = E(ctx->loseContextOnReset ? GL_LOSE_CONTEXT_ON_RESET : GL_NO_RESET_NOTIFICATION); return true;
    case GL_CONTEXT_RELEASE_BEHAVIOR: v = E(GL_CONTEXT_RELEASE_BEHAVIOR_FLUSH); return true;
    case GL_MAX_TEXTURE_SIZE: case GL_MAX_CUBE_MAP_TEXTURE_SIZE: case GL_MAX_RENDERBUFFER_SIZE: case GL_MAX_RECTANGLE_TEXTURE_SIZE:
        v = I(16384); return true;
    case GL_MAX_3D_TEXTURE_SIZE: v = I(2048); return true;
    case GL_MAX_ARRAY_TEXTURE_LAYERS: v = I(2048); return true;
    case GL_MAX_SPARSE_TEXTURE_SIZE_ARB: v = I(kMaxSparseTextureSize); return ctx->dev->caps.sparse;
    case GL_MAX_SPARSE_3D_TEXTURE_SIZE_ARB: v = I(kMaxSparse3DTextureSize); return ctx->dev->caps.sparse;
    case GL_MAX_SPARSE_ARRAY_TEXTURE_LAYERS_ARB: v = I(kMaxSparseArrayTextureLayers); return ctx->dev->caps.sparse;
    case GL_SPARSE_TEXTURE_FULL_ARRAY_CUBE_MIPMAPS_ARB: v = B(true); return ctx->dev->caps.sparse;  // Metal mip tails are per slice
    case GL_SPARSE_BUFFER_PAGE_SIZE_ARB: v = I((int64_t)ctx->dev->sparseTileBytes); return ctx->dev->caps.sparse;
    case GL_MAX_TEXTURE_BUFFER_SIZE: v = I(1 << 27); return true;
    case GL_MAX_TEXTURE_LOD_BIAS: v = F(16); return true;
    case GL_MAX_TEXTURE_MAX_ANISOTROPY: v = F(16); return true;
    case GL_MAX_VIEWPORT_DIMS: v = I2(16384, 16384); return true;
    // GL_APPLE_pixel_buffer (CGL pbuffers, macos/cgl.mm): the smallest pbuffer viewport, 1x1 as on Apple's GL.
    case 0x8A10 /* GL_MIN_PBUFFER_VIEWPORT_DIMS_APPLE */: v = I2(1, 1); return true;
    case GL_MAX_VIEWPORTS: v = I(kMaxViewports); return true;
    case GL_VIEWPORT_SUBPIXEL_BITS: v = I(0); return true;
    case GL_VIEWPORT_BOUNDS_RANGE: v = F2(-32768, 32767); return true;
    case GL_VIEWPORT_INDEX_PROVOKING_VERTEX: case GL_LAYER_PROVOKING_VERTEX: v = E(GL_UNDEFINED_VERTEX); return true;
    case GL_MAX_COLOR_ATTACHMENTS: case GL_MAX_DRAW_BUFFERS: v = I(kMaxDrawBuffers); return true;
    case GL_MAX_DUAL_SOURCE_DRAW_BUFFERS: v = I(1); return true;
    case GL_MAX_SAMPLES: case GL_MAX_COLOR_TEXTURE_SAMPLES: case GL_MAX_DEPTH_TEXTURE_SAMPLES: case GL_MAX_INTEGER_SAMPLES:
    case GL_MAX_FRAMEBUFFER_SAMPLES:
        v = I(4); return true;
    case GL_MAX_SAMPLE_MASK_WORDS: v = I(1); return true;
    case GL_MAX_FRAMEBUFFER_WIDTH: case GL_MAX_FRAMEBUFFER_HEIGHT: v = I(16384); return true;
    case GL_MAX_FRAMEBUFFER_LAYERS: v = I(2048); return true;
    case GL_MAX_VERTEX_ATTRIBS: v = I(kMaxVertexAttribs); return true;
    case GL_MAX_VERTEX_ATTRIB_BINDINGS: v = I(kMaxVertexBindings); return true;
    case GL_MAX_VERTEX_ATTRIB_STRIDE: v = I(2048); return true;
    case GL_MAX_VERTEX_ATTRIB_RELATIVE_OFFSET: v = I(2047); return true;
    case GL_MAX_ELEMENTS_VERTICES: case GL_MAX_ELEMENTS_INDICES: v = I(1 << 24); return true;
    case GL_MAX_ELEMENT_INDEX: v = I64(0xffffffffll); return true;
    case GL_MAX_TEXTURE_IMAGE_UNITS: case GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS: case GL_MAX_GEOMETRY_TEXTURE_IMAGE_UNITS:
    case GL_MAX_TESS_CONTROL_TEXTURE_IMAGE_UNITS: case GL_MAX_TESS_EVALUATION_TEXTURE_IMAGE_UNITS:
    case GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS:
        v = I(32); return true;
    case GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS: v = I(kMaxTextureUnits); return true;
    case GL_MAX_TEXTURE_UNITS: v = I(8); return true;  // compat fixed-function
    case GL_MAX_TEXTURE_COORDS: v = I(8); return true;
    // ARB_vertex_program / ARB_fragment_program
    case GL_PROGRAM_ERROR_POSITION_ARB: v = I(ctx->arbErrorPos); return true;
    case GL_MAX_PROGRAM_MATRICES_ARB: v = I(8); return true;
    case GL_MAX_PROGRAM_MATRIX_STACK_DEPTH_ARB: v = I(1); return true;
    case GL_MAX_IMAGE_UNITS: v = I(kMaxImageUnits); return true;
    case GL_MAX_VERTEX_IMAGE_UNIFORMS: case GL_MAX_FRAGMENT_IMAGE_UNIFORMS: case GL_MAX_COMPUTE_IMAGE_UNIFORMS:
    case GL_MAX_GEOMETRY_IMAGE_UNIFORMS: case GL_MAX_TESS_CONTROL_IMAGE_UNIFORMS: case GL_MAX_TESS_EVALUATION_IMAGE_UNIFORMS:
    case GL_MAX_COMBINED_IMAGE_UNIFORMS:
        v = I(kMaxImageUnits); return true;
    case GL_MAX_IMAGE_SAMPLES: v = I(0); return true;
    case GL_MAX_UNIFORM_BUFFER_BINDINGS: v = I(kMaxUniformBufferBindings); return true;
    case GL_MAX_VERTEX_UNIFORM_BLOCKS: case GL_MAX_FRAGMENT_UNIFORM_BLOCKS: case GL_MAX_GEOMETRY_UNIFORM_BLOCKS:
    case GL_MAX_TESS_CONTROL_UNIFORM_BLOCKS: case GL_MAX_TESS_EVALUATION_UNIFORM_BLOCKS: case GL_MAX_COMPUTE_UNIFORM_BLOCKS:
        v = I(14); return true;
    case GL_MAX_COMBINED_UNIFORM_BLOCKS: v = I(84); return true;
    case GL_MAX_UNIFORM_BLOCK_SIZE: v = I(65536); return true;
    case GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT: v = I(16); return true;
    case GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS: v = I(kMaxSSBOBindings); return true;
    case GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS: case GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS: case GL_MAX_GEOMETRY_SHADER_STORAGE_BLOCKS:
    case GL_MAX_TESS_CONTROL_SHADER_STORAGE_BLOCKS: case GL_MAX_TESS_EVALUATION_SHADER_STORAGE_BLOCKS:
    case GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS: case GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS:
        v = I(kMaxSSBOBindings); return true;
    case GL_MAX_SHADER_STORAGE_BLOCK_SIZE: v = I64(1ll << 30); return true;
    case GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT: v = I(16); return true;
    case GL_MAX_COMBINED_SHADER_OUTPUT_RESOURCES: v = I(kMaxSSBOBindings + kMaxImageUnits + kMaxDrawBuffers); return true;
    case GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS: v = I(kMaxAtomicCounterBindings); return true;
    case GL_MAX_VERTEX_ATOMIC_COUNTER_BUFFERS: case GL_MAX_FRAGMENT_ATOMIC_COUNTER_BUFFERS: case GL_MAX_GEOMETRY_ATOMIC_COUNTER_BUFFERS:
    case GL_MAX_TESS_CONTROL_ATOMIC_COUNTER_BUFFERS: case GL_MAX_TESS_EVALUATION_ATOMIC_COUNTER_BUFFERS:
    case GL_MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS: case GL_MAX_COMBINED_ATOMIC_COUNTER_BUFFERS:
        v = I(kMaxAtomicCounterBindings); return true;
    case GL_MAX_VERTEX_ATOMIC_COUNTERS: case GL_MAX_FRAGMENT_ATOMIC_COUNTERS: case GL_MAX_GEOMETRY_ATOMIC_COUNTERS:
    case GL_MAX_TESS_CONTROL_ATOMIC_COUNTERS: case GL_MAX_TESS_EVALUATION_ATOMIC_COUNTERS: case GL_MAX_COMPUTE_ATOMIC_COUNTERS:
    case GL_MAX_COMBINED_ATOMIC_COUNTERS:
        v = I(1024); return true;
    case GL_MAX_ATOMIC_COUNTER_BUFFER_SIZE: v = I(65536); return true;
    case GL_MAX_VERTEX_UNIFORM_COMPONENTS: case GL_MAX_FRAGMENT_UNIFORM_COMPONENTS: case GL_MAX_GEOMETRY_UNIFORM_COMPONENTS:
    case GL_MAX_TESS_CONTROL_UNIFORM_COMPONENTS: case GL_MAX_TESS_EVALUATION_UNIFORM_COMPONENTS: case GL_MAX_COMPUTE_UNIFORM_COMPONENTS:
        v = I(16384); return true;
    case GL_MAX_VERTEX_UNIFORM_VECTORS: case GL_MAX_FRAGMENT_UNIFORM_VECTORS: v = I(4096); return true;
    case GL_MAX_COMBINED_VERTEX_UNIFORM_COMPONENTS: case GL_MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS:
    case GL_MAX_COMBINED_GEOMETRY_UNIFORM_COMPONENTS: case GL_MAX_COMBINED_TESS_CONTROL_UNIFORM_COMPONENTS:
    case GL_MAX_COMBINED_TESS_EVALUATION_UNIFORM_COMPONENTS: case GL_MAX_COMBINED_COMPUTE_UNIFORM_COMPONENTS:
        v = I(16384 + 14 * 65536 / 4); return true;
    case GL_MAX_UNIFORM_LOCATIONS: v = I(16384); return true;
    case GL_MAX_VARYING_COMPONENTS: v = I(124); return true;
    case GL_MAX_VARYING_VECTORS: v = I(31); return true;
    case GL_MAX_VERTEX_OUTPUT_COMPONENTS: v = I(128); return true;
    case GL_MAX_FRAGMENT_INPUT_COMPONENTS: v = I(128); return true;
    case GL_MAX_GEOMETRY_INPUT_COMPONENTS: case GL_MAX_GEOMETRY_OUTPUT_COMPONENTS: v = I(128); return true;
    case GL_MAX_TESS_CONTROL_INPUT_COMPONENTS: case GL_MAX_TESS_CONTROL_OUTPUT_COMPONENTS:
    case GL_MAX_TESS_EVALUATION_INPUT_COMPONENTS: case GL_MAX_TESS_EVALUATION_OUTPUT_COMPONENTS:
        v = I(128); return true;
    case GL_MAX_TESS_PATCH_COMPONENTS: v = I(120); return true;
    case GL_MAX_TESS_CONTROL_TOTAL_OUTPUT_COMPONENTS: v = I(4096); return true;
    case GL_MAX_TESS_GEN_LEVEL: v = I(64); return true;
    case GL_MAX_PATCH_VERTICES: v = I(32); return true;
    case GL_MAX_GEOMETRY_OUTPUT_VERTICES: v = I(256); return true;
    case GL_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS: v = I(1024); return true;
    case GL_MAX_GEOMETRY_SHADER_INVOCATIONS: v = I(32); return true;
    case GL_MAX_VERTEX_STREAMS: v = I(4); return true;
    case GL_MAX_COMPUTE_WORK_GROUP_COUNT: v = I(index < 0 ? 65535 : 65535); return true;
    case GL_MAX_COMPUTE_WORK_GROUP_SIZE: { int64_t s[3] = {1024, 1024, 64}; v = I(s[std::min(idx, 2)]); return true; }
    case GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS: v = I(1024); return true;
    // KHR_shader_subgroup: SIMD-groups (32 threads on every Apple GPU)
    case GL_SUBGROUP_SIZE_KHR: v = I(32); return true;
    case GL_SUBGROUP_SUPPORTED_STAGES_KHR:
        v = I(GL_VERTEX_SHADER_BIT | GL_TESS_CONTROL_SHADER_BIT | GL_TESS_EVALUATION_SHADER_BIT | GL_GEOMETRY_SHADER_BIT |
              GL_FRAGMENT_SHADER_BIT | GL_COMPUTE_SHADER_BIT);
        return true;
    case GL_SUBGROUP_SUPPORTED_FEATURES_KHR:
        v = I(GL_SUBGROUP_FEATURE_BASIC_BIT_KHR | GL_SUBGROUP_FEATURE_VOTE_BIT_KHR | GL_SUBGROUP_FEATURE_ARITHMETIC_BIT_KHR |
              GL_SUBGROUP_FEATURE_BALLOT_BIT_KHR | GL_SUBGROUP_FEATURE_SHUFFLE_BIT_KHR | GL_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT_KHR |
              GL_SUBGROUP_FEATURE_CLUSTERED_BIT_KHR | GL_SUBGROUP_FEATURE_QUAD_BIT_KHR | GL_SUBGROUP_FEATURE_PARTITIONED_BIT_NV);
        return true;
    case GL_SUBGROUP_QUAD_ALL_STAGES_KHR: v = B(true); return true;
    case GL_MAX_COMPUTE_SHARED_MEMORY_SIZE: v = I(32768); return true;
    // EXT_mesh_shader: task shaders run as Metal object functions, mesh shaders as mesh functions whose outputs
    // are staged in threadgroup memory before they are copied into the Metal mesh. Apple GPUs give a threadgroup
    // 32 KB (an object threadgroup 16 bytes less: Metal reserves them for the mesh grid properties) and a mesh
    // threadgroup 60 KB of threadgroup memory + mesh output together; the payload is separate (at most 16 KB).
    // Two limits are below the EXT minimums for that reason (see docs/glsl-compiler.md, mesh shaders).
    case GL_MAX_TASK_WORK_GROUP_TOTAL_COUNT_EXT: case GL_MAX_MESH_WORK_GROUP_TOTAL_COUNT_EXT: v = I(1 << 22); return true;
    case GL_MAX_TASK_WORK_GROUP_COUNT_EXT: case GL_MAX_MESH_WORK_GROUP_COUNT_EXT: v = I(65535); return true;
    case GL_MAX_TASK_WORK_GROUP_INVOCATIONS_EXT: case GL_MAX_MESH_WORK_GROUP_INVOCATIONS_EXT: v = I(128); return true;
    case GL_MAX_TASK_WORK_GROUP_SIZE_EXT: case GL_MAX_MESH_WORK_GROUP_SIZE_EXT: v = I(128); return true;
    case GL_MAX_TASK_PAYLOAD_SIZE_EXT: v = I(16384); return true;
    case GL_MAX_TASK_SHARED_MEMORY_SIZE_EXT: v = I(32752); return true;  // EXT minimum 32768
    case GL_MAX_TASK_PAYLOAD_AND_SHARED_MEMORY_SIZE_EXT: v = I(32768); return true;
    case GL_MAX_MESH_SHARED_MEMORY_SIZE_EXT: case GL_MAX_MESH_PAYLOAD_AND_SHARED_MEMORY_SIZE_EXT: v = I(28672); return true;
    case GL_MAX_MESH_OUTPUT_MEMORY_SIZE_EXT: v = I(30720); return true;  // EXT minimum 32768: outputs are held twice
    case GL_MAX_MESH_PAYLOAD_AND_OUTPUT_MEMORY_SIZE_EXT: v = I(48128); return true;
    case GL_MAX_MESH_OUTPUT_COMPONENTS_EXT: v = I(128); return true;
    case GL_MAX_MESH_OUTPUT_VERTICES_EXT: case GL_MAX_MESH_OUTPUT_PRIMITIVES_EXT: v = I(256); return true;
    case GL_MAX_MESH_OUTPUT_LAYERS_EXT: v = I(8); return true;
    case GL_MAX_MESH_MULTIVIEW_VIEW_COUNT_EXT: v = I(1); return true;
    case GL_MESH_OUTPUT_PER_VERTEX_GRANULARITY_EXT: case GL_MESH_OUTPUT_PER_PRIMITIVE_GRANULARITY_EXT: v = I(1); return true;
    case GL_MAX_PREFERRED_TASK_WORK_GROUP_INVOCATIONS_EXT: case GL_MAX_PREFERRED_MESH_WORK_GROUP_INVOCATIONS_EXT: v = I(32); return true;
    case GL_MESH_PREFERS_LOCAL_INVOCATION_VERTEX_OUTPUT_EXT: case GL_MESH_PREFERS_LOCAL_INVOCATION_PRIMITIVE_OUTPUT_EXT:
    case GL_MESH_PREFERS_COMPACT_VERTEX_OUTPUT_EXT: case GL_MESH_PREFERS_COMPACT_PRIMITIVE_OUTPUT_EXT:
        v = B(false); return true;
    case GL_MAX_TASK_UNIFORM_BLOCKS_EXT: case GL_MAX_MESH_UNIFORM_BLOCKS_EXT: v = I(14); return true;
    case GL_MAX_TASK_TEXTURE_IMAGE_UNITS_EXT: case GL_MAX_MESH_TEXTURE_IMAGE_UNITS_EXT: v = I(32); return true;
    case GL_MAX_TASK_IMAGE_UNIFORMS_EXT: case GL_MAX_MESH_IMAGE_UNIFORMS_EXT: v = I(kMaxImageUnits); return true;
    case GL_MAX_TASK_UNIFORM_COMPONENTS_EXT: case GL_MAX_MESH_UNIFORM_COMPONENTS_EXT: v = I(16384); return true;
    case GL_MAX_COMBINED_TASK_UNIFORM_COMPONENTS_EXT: case GL_MAX_COMBINED_MESH_UNIFORM_COMPONENTS_EXT: v = I(16384 + 14 * 65536 / 4); return true;
    case GL_MAX_TASK_ATOMIC_COUNTER_BUFFERS_EXT: case GL_MAX_MESH_ATOMIC_COUNTER_BUFFERS_EXT: v = I(kMaxAtomicCounterBindings); return true;
    case GL_MAX_TASK_ATOMIC_COUNTERS_EXT: case GL_MAX_MESH_ATOMIC_COUNTERS_EXT: v = I(1024); return true;
    case GL_MAX_TASK_SHADER_STORAGE_BLOCKS_EXT: case GL_MAX_MESH_SHADER_STORAGE_BLOCKS_EXT: v = I(kMaxSSBOBindings); return true;
    case GL_MAX_CLIP_DISTANCES: case GL_MAX_CULL_DISTANCES: case GL_MAX_COMBINED_CLIP_AND_CULL_DISTANCES: v = I(8); return true;
    case GL_MAX_SUBROUTINES: v = I(256); return true;
    case GL_MAX_SUBROUTINE_UNIFORM_LOCATIONS: v = I(1024); return true;
    case GL_MAX_TRANSFORM_FEEDBACK_BUFFERS: v = I(kMaxXfbBuffers); return true;
    case GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS: v = I(128); return true;
    case GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS: v = I(4); return true;
    case GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS: v = I(64); return true;
    case GL_MIN_PROGRAM_TEXEL_OFFSET: v = I(-8); return true;
    case GL_MAX_PROGRAM_TEXEL_OFFSET: v = I(7); return true;
    case GL_MIN_PROGRAM_TEXTURE_GATHER_OFFSET: v = I(-32); return true;
    case GL_MAX_PROGRAM_TEXTURE_GATHER_OFFSET: v = I(31); return true;
    case GL_MIN_FRAGMENT_INTERPOLATION_OFFSET: v = F(-0.5); return true;
    case GL_MAX_FRAGMENT_INTERPOLATION_OFFSET: v = F(0.4375); return true;
    case GL_FRAGMENT_INTERPOLATION_OFFSET_BITS: v = I(4); return true;
    case GL_MAX_SERVER_WAIT_TIMEOUT: v = I64(1000000000ll * 60); return true;
    case GL_MAX_LABEL_LENGTH: v = I(256); return true;
    case GL_MAX_DEBUG_MESSAGE_LENGTH: v = I(1024); return true;
    case GL_MAX_DEBUG_LOGGED_MESSAGES: v = I(64); return true;
    case GL_MAX_DEBUG_GROUP_STACK_DEPTH: v = I(64); return true;
    case GL_DEBUG_GROUP_STACK_DEPTH: v = I((int)ctx->debugGroups.size() + 1); return true;
    case GL_DEBUG_LOGGED_MESSAGES: v = I((int)ctx->debugLog.size()); return true;
    case GL_DEBUG_NEXT_LOGGED_MESSAGE_LENGTH: v = I(ctx->debugLog.empty() ? 0 : (int)ctx->debugLog.front().text.size() + 1); return true;
    case GL_TEXTURE_BUFFER_OFFSET_ALIGNMENT: v = I(64); return true;
    case GL_MIN_MAP_BUFFER_ALIGNMENT: v = I(64); return true;
    case GL_SUBPIXEL_BITS: v = I(8); return true;
    case GL_ALIASED_LINE_WIDTH_RANGE: v = F2(1, 1); return true;
    case GL_SMOOTH_LINE_WIDTH_RANGE: v = F2(1, 1); return true;
    case GL_SMOOTH_LINE_WIDTH_GRANULARITY: v = F(0.0); return true;
    case GL_POINT_SIZE_RANGE: v = F2(1, 511); return true;
    case GL_ALIASED_POINT_SIZE_RANGE: v = F2(1, 511); return true;
    case GL_POINT_SIZE_GRANULARITY: v = F(0.125); return true;
    case GL_NUM_COMPRESSED_TEXTURE_FORMATS: v = I(0); return true;
    case GL_COMPRESSED_TEXTURE_FORMATS: v = I(0); v.n = 0; return true;
    case GL_NUM_PROGRAM_BINARY_FORMATS: v = I(0); return true;
    case GL_PROGRAM_BINARY_FORMATS: v = I(0); v.n = 0; return true;
    case GL_NUM_SHADER_BINARY_FORMATS: v = I(1); return true;
    case GL_SHADER_BINARY_FORMATS: v = E(GL_SHADER_BINARY_FORMAT_SPIR_V); return true;
    case GL_SHADER_COMPILER: v = B(true); return true;
    case GL_IMPLEMENTATION_COLOR_READ_FORMAT: v = E(GL_RGBA); return true;
    case GL_IMPLEMENTATION_COLOR_READ_TYPE: v = E(GL_UNSIGNED_BYTE); return true;
    case GL_TIMESTAMP: v = I64((int64_t)(CACurrentMediaTime() * 1e9)); return true;
    case GL_MAX_SHADER_COMPILER_THREADS_KHR: v = I(0); return true;
    case GL_SAMPLE_BUFFERS: {
        PassDesc p;
        v = I(ctx->resolvePass(p) && p.samples > 1 ? 1 : 0);
        return true;
    }
    case GL_SAMPLES: {
        PassDesc p;
        v = I(ctx->resolvePass(p) && p.samples > 1 ? p.samples : 0);
        return true;
    }
    case GL_COLOR_FLOAT_APPLE:  // APPLE_float_pixels / ARB_color_buffer_float: a floating-point color buffer
    case GL_RGBA_FLOAT_MODE_ARB: {
        PassDesc p;
        const FormatInfo* f = ctx->resolvePass(p) && p.color[0].tex ? formatInfoForMTL(p.color[0].tex.pixelFormat) : nullptr;
        v = B(f && f->kind == FmtKind::Float);
        return true;
    }
    case GL_ACTIVE_STENCIL_FACE_EXT: v = E(ctx->activeStencilFaceEXT); return true;
    case 0x8166: {  // GL_OCCLUSION_TEST_RESULT_HP
        bool mglHPOcclusionResult(Context*);
        v = B(mglHPOcclusionResult(ctx));
        return true;
    }
    case GL_POLYGON_OFFSET_BIAS_EXT: v = F(ctx->polygonOffsetUnits / 16777216.0); return true;  // EXT_polygon_offset
    case GL_PIXEL_COUNTER_BITS_NV: v = I(64); return true;  // GL_NV_occlusion_query
    case GL_CURRENT_OCCLUSION_QUERY_ID_NV: {
        GLint id = 0;
        glGetQueryiv(GL_SAMPLES_PASSED, GL_CURRENT_QUERY, &id);
        v = I(id);
        return true;
    }
    case GL_DOUBLEBUFFER: v = B(true); return true;
    // GL_APPLE_aux_depth_stencil: whether aux buffers have their own depth/stencil. There are no aux buffers
    // (GL_AUX_BUFFERS is 0), even with kCGLPFAAuxDepthStencil, which pixel formats accept.
    case GL_AUX_DEPTH_STENCIL_APPLE: v = B(false); return true;
    case GL_STEREO: v = B(false); return true;

    // --- bindings ---
    case GL_ARRAY_BUFFER_BINDING: v = I(nameOf(ctx->arrayBuffer.get())); return true;
    case GL_ELEMENT_ARRAY_BUFFER_BINDING: v = I(nameOf(ctx->bufferBindingRef(GL_ELEMENT_ARRAY_BUFFER)->get())); return true;
    case GL_COPY_READ_BUFFER_BINDING: v = I(nameOf(ctx->copyReadBuffer.get())); return true;
    case GL_COPY_WRITE_BUFFER_BINDING: v = I(nameOf(ctx->copyWriteBuffer.get())); return true;
    case GL_PIXEL_PACK_BUFFER_BINDING: v = I(nameOf(ctx->pixelPackBuffer.get())); return true;
    case GL_PIXEL_UNPACK_BUFFER_BINDING: v = I(nameOf(ctx->pixelUnpackBuffer.get())); return true;
    case GL_DRAW_INDIRECT_BUFFER_BINDING: v = I(nameOf(ctx->drawIndirectBuffer.get())); return true;
    case GL_DISPATCH_INDIRECT_BUFFER_BINDING: v = I(nameOf(ctx->dispatchIndirectBuffer.get())); return true;
    case GL_QUERY_BUFFER_BINDING: v = I(nameOf(ctx->queryBuffer.get())); return true;
    case GL_TEXTURE_BUFFER_BINDING: v = I(nameOf(ctx->textureBufferBinding.get())); return true;
    case GL_PARAMETER_BUFFER_BINDING: v = I(nameOf(ctx->parameterBuffer.get())); return true;
    case GL_UNIFORM_BUFFER_BINDING:
        if (index >= 0) {
            if (index >= kMaxUniformBufferBindings) return false;
            v = I(nameOf(ctx->uniformBuffers[index].buffer.get()));
        } else v = I(nameOf(ctx->uniformBufferGeneric.get()));
        return true;
    case GL_UNIFORM_BUFFER_START: if (index < 0 || index >= kMaxUniformBufferBindings) return false; v = I64(ctx->uniformBuffers[index].offset); return true;
    case GL_UNIFORM_BUFFER_SIZE: if (index < 0 || index >= kMaxUniformBufferBindings) return false; v = I64(ctx->uniformBuffers[index].size); return true;
    case GL_SHADER_STORAGE_BUFFER_BINDING:
        if (index >= 0) {
            if (index >= kMaxSSBOBindings) return false;
            v = I(nameOf(ctx->storageBuffers[index].buffer.get()));
        } else v = I(nameOf(ctx->ssboGeneric.get()));
        return true;
    case GL_SHADER_STORAGE_BUFFER_START: if (index < 0 || index >= kMaxSSBOBindings) return false; v = I64(ctx->storageBuffers[index].offset); return true;
    case GL_SHADER_STORAGE_BUFFER_SIZE: if (index < 0 || index >= kMaxSSBOBindings) return false; v = I64(ctx->storageBuffers[index].size); return true;
    case GL_ATOMIC_COUNTER_BUFFER_BINDING:
        if (index >= 0) {
            if (index >= kMaxAtomicCounterBindings) return false;
            v = I(nameOf(ctx->atomicCounterBuffers[index].buffer.get()));
        } else v = I(nameOf(ctx->atomicCounterGeneric.get()));
        return true;
    case GL_ATOMIC_COUNTER_BUFFER_START: if (index < 0 || index >= kMaxAtomicCounterBindings) return false; v = I64(ctx->atomicCounterBuffers[index].offset); return true;
    case GL_ATOMIC_COUNTER_BUFFER_SIZE: if (index < 0 || index >= kMaxAtomicCounterBindings) return false; v = I64(ctx->atomicCounterBuffers[index].size); return true;
    case GL_TRANSFORM_FEEDBACK_BUFFER_BINDING:
        if (index >= 0) {
            if (index >= kMaxXfbBuffers) return false;
            v = I(nameOf(ctx->xfb->buffers[index].buffer.get()));
        } else v = I(nameOf(ctx->xfbGeneric.get()));
        return true;
    case GL_TRANSFORM_FEEDBACK_BUFFER_START: if (index < 0 || index >= kMaxXfbBuffers) return false; v = I64(ctx->xfb->buffers[index].offset); return true;
    case GL_TRANSFORM_FEEDBACK_BUFFER_SIZE: if (index < 0 || index >= kMaxXfbBuffers) return false; v = I64(ctx->xfb->buffers[index].size); return true;
    case GL_TRANSFORM_FEEDBACK_BINDING: v = I(ctx->xfb == ctx->defaultXfb ? 0 : nameOf(ctx->xfb.get())); return true;
    case GL_TRANSFORM_FEEDBACK_BUFFER_ACTIVE: v = B(ctx->xfb->active); return true;
    case GL_TRANSFORM_FEEDBACK_BUFFER_PAUSED: v = B(ctx->xfb->paused); return true;
    case GL_VERTEX_ARRAY_BINDING: v = I(ctx->vao && ctx->vao != ctx->defaultVao ? nameOf(ctx->vao.get()) : 0); return true;
    case GL_CURRENT_PROGRAM: v = I(nameOf(ctx->program.get())); return true;
    case GL_PROGRAM_PIPELINE_BINDING: v = I(nameOf(ctx->pipeline.get())); return true;
    case GL_DRAW_FRAMEBUFFER_BINDING: v = I(ctx->drawFb && !ctx->drawFb->isDefault ? nameOf(ctx->drawFb.get()) : 0); return true;
    case GL_READ_FRAMEBUFFER_BINDING: v = I(ctx->readFb && !ctx->readFb->isDefault ? nameOf(ctx->readFb.get()) : 0); return true;
    case GL_RENDERBUFFER_BINDING: v = I(nameOf(ctx->renderbuffer.get())); return true;
    case GL_ACTIVE_TEXTURE: v = E(GL_TEXTURE0 + ctx->activeTexture); return true;
    case GL_SAMPLER_BINDING: {  // indexed form (ARB_multi_bind) selects the texture unit
        int unit = index >= 0 ? index : ctx->activeTexture;
        if (unit >= kMaxTextureUnits) return false;
        v = I(nameOf(ctx->samplers[unit].get()));
        return true;
    }
    case GL_TEXTURE_BINDING_1D: case GL_TEXTURE_BINDING_2D: case GL_TEXTURE_BINDING_3D: case GL_TEXTURE_BINDING_1D_ARRAY:
    case GL_TEXTURE_BINDING_2D_ARRAY: case GL_TEXTURE_BINDING_RECTANGLE: case GL_TEXTURE_BINDING_CUBE_MAP:
    case GL_TEXTURE_BINDING_CUBE_MAP_ARRAY: case GL_TEXTURE_BINDING_BUFFER: case GL_TEXTURE_BINDING_2D_MULTISAMPLE:
    case GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY: {
        static const GLenum map[][2] = {
            {GL_TEXTURE_BINDING_1D, GL_TEXTURE_1D}, {GL_TEXTURE_BINDING_2D, GL_TEXTURE_2D}, {GL_TEXTURE_BINDING_3D, GL_TEXTURE_3D},
            {GL_TEXTURE_BINDING_1D_ARRAY, GL_TEXTURE_1D_ARRAY}, {GL_TEXTURE_BINDING_2D_ARRAY, GL_TEXTURE_2D_ARRAY},
            {GL_TEXTURE_BINDING_RECTANGLE, GL_TEXTURE_RECTANGLE}, {GL_TEXTURE_BINDING_CUBE_MAP, GL_TEXTURE_CUBE_MAP},
            {GL_TEXTURE_BINDING_CUBE_MAP_ARRAY, GL_TEXTURE_CUBE_MAP_ARRAY}, {GL_TEXTURE_BINDING_BUFFER, GL_TEXTURE_BUFFER},
            {GL_TEXTURE_BINDING_2D_MULTISAMPLE, GL_TEXTURE_2D_MULTISAMPLE},
            {GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY, GL_TEXTURE_2D_MULTISAMPLE_ARRAY}};
        for (auto& m : map)
            if (m[0] == pname) {
                int unit = index >= 0 ? index : ctx->activeTexture;
                if (unit >= kMaxTextureUnits) return false;
                v = I(nameOf(ctx->textures[unit][texTargetIndex(m[1])].get()));
            }
        return true;
    }
    case GL_IMAGE_BINDING_NAME: if (index < 0 || index >= kMaxImageUnits) return false; v = I(nameOf(ctx->imageUnits[index].tex.get())); return true;
    case GL_IMAGE_BINDING_LEVEL: if (index < 0 || index >= kMaxImageUnits) return false; v = I(ctx->imageUnits[index].level); return true;
    case GL_IMAGE_BINDING_LAYERED: if (index < 0 || index >= kMaxImageUnits) return false; v = B(ctx->imageUnits[index].layered); return true;
    case GL_IMAGE_BINDING_LAYER: if (index < 0 || index >= kMaxImageUnits) return false; v = I(ctx->imageUnits[index].layer); return true;
    case GL_IMAGE_BINDING_ACCESS: if (index < 0 || index >= kMaxImageUnits) return false; v = E(ctx->imageUnits[index].access); return true;
    case GL_IMAGE_BINDING_FORMAT: if (index < 0 || index >= kMaxImageUnits) return false; v = E(ctx->imageUnits[index].format); return true;
    case GL_VERTEX_BINDING_DIVISOR: case GL_VERTEX_BINDING_OFFSET: case GL_VERTEX_BINDING_STRIDE: case GL_VERTEX_BINDING_BUFFER: {
        if (index < 0 || index >= kMaxVertexBindings || !ctx->vao) return false;
        auto& b = ctx->vao->bindings[index];
        if (pname == GL_VERTEX_BINDING_DIVISOR) v = I(b.divisor);
        else if (pname == GL_VERTEX_BINDING_OFFSET) v = I64(b.offset);
        else if (pname == GL_VERTEX_BINDING_STRIDE) v = I(b.stride);
        else v = I(nameOf(b.buffer.get()));
        return true;
    }

    // --- framebuffer state ---
    case GL_DRAW_BUFFER: case GL_DRAW_BUFFER0: case GL_DRAW_BUFFER1: case GL_DRAW_BUFFER2: case GL_DRAW_BUFFER3:
    case GL_DRAW_BUFFER4: case GL_DRAW_BUFFER5: case GL_DRAW_BUFFER6: case GL_DRAW_BUFFER7: {
        int i = pname == GL_DRAW_BUFFER ? 0 : pname - GL_DRAW_BUFFER0;
        v = E(ctx->drawFb ? ctx->drawFb->drawBuffers[i] : GL_NONE);
        return true;
    }
    case GL_READ_BUFFER: v = E(ctx->readFb ? ctx->readFb->readBuffer : GL_NONE); return true;

    // --- fixed function ---
    case GL_VIEWPORT: {
        if (idx >= kMaxViewports) return false;
        Value r; r.type = VT::Float; r.n = 4;
        for (int k = 0; k < 4; k++) r.d[k] = ctx->viewport[idx][k];
        v = r;  // convertOut rounds for integer getters; float getters keep fractions
        return true;
    }
    case GL_SCISSOR_BOX: if (idx >= kMaxViewports) return false; v = I4(ctx->scissor[idx][0], ctx->scissor[idx][1], ctx->scissor[idx][2], ctx->scissor[idx][3]); return true;
    case GL_DEPTH_RANGE: { if (idx >= kMaxViewports) return false; Value r; r.type = VT::Double; r.n = 2; r.d[0] = ctx->depthRange[idx][0]; r.d[1] = ctx->depthRange[idx][1]; v = r; return true; }
    case GL_COLOR_CLEAR_VALUE: v = F4(ctx->clearColor); return true;
    case GL_DEPTH_CLEAR_VALUE: { Value r; r.type = VT::Double; r.d[0] = ctx->clearDepth; v = r; return true; }
    case GL_STENCIL_CLEAR_VALUE: v = I(ctx->clearStencil); return true;
    case GL_COLOR_WRITEMASK: { uint8_t m = ctx->blend[idx].colorMask; v = B4(m & 1, m & 2, m & 4, m & 8); return true; }
    case GL_DEPTH_WRITEMASK: v = B(ctx->depthWrite); return true;
    case GL_DEPTH_FUNC: v = E(ctx->depthFunc); return true;
    case GL_BLEND_SRC_RGB: case GL_BLEND_SRC: v = E(ctx->blend[idx].srcRGB); return true;
    case GL_BLEND_DST_RGB: case GL_BLEND_DST: v = E(ctx->blend[idx].dstRGB); return true;
    case GL_BLEND_SRC_ALPHA: v = E(ctx->blend[idx].srcA); return true;
    case GL_BLEND_DST_ALPHA: v = E(ctx->blend[idx].dstA); return true;
    case GL_BLEND_EQUATION_RGB: v = E(ctx->blend[idx].eqRGB); return true;
    case GL_BLEND_EQUATION_ALPHA: v = E(ctx->blend[idx].eqA); return true;
    case GL_BLEND_COLOR: v = F4(ctx->blendColor); return true;
    case GL_STENCIL_FUNC: v = E(ctx->stencil[0].func); return true;
    case GL_STENCIL_REF: v = I(ctx->stencil[0].ref); return true;
    case GL_STENCIL_VALUE_MASK: v = I(ctx->stencil[0].valueMask); return true;
    case GL_STENCIL_WRITEMASK: v = I(ctx->stencil[0].writeMask); return true;
    case GL_STENCIL_FAIL: v = E(ctx->stencil[0].sfail); return true;
    case GL_STENCIL_PASS_DEPTH_FAIL: v = E(ctx->stencil[0].dpfail); return true;
    case GL_STENCIL_PASS_DEPTH_PASS: v = E(ctx->stencil[0].dppass); return true;
    case GL_STENCIL_BACK_FUNC: v = E(ctx->stencil[1].func); return true;
    case GL_STENCIL_BACK_REF: v = I(ctx->stencil[1].ref); return true;
    case GL_STENCIL_BACK_VALUE_MASK: v = I(ctx->stencil[1].valueMask); return true;
    case GL_STENCIL_BACK_WRITEMASK: v = I(ctx->stencil[1].writeMask); return true;
    case GL_STENCIL_BACK_FAIL: v = E(ctx->stencil[1].sfail); return true;
    case GL_STENCIL_BACK_PASS_DEPTH_FAIL: v = E(ctx->stencil[1].dpfail); return true;
    case GL_STENCIL_BACK_PASS_DEPTH_PASS: v = E(ctx->stencil[1].dppass); return true;
    case GL_CULL_FACE_MODE: v = E(ctx->cullMode); return true;
    case GL_FRONT_FACE: v = E(ctx->frontFace); return true;
    case GL_POLYGON_MODE: { Value r; r.type = VT::Enum; r.n = 2; r.i[0] = r.i[1] = ctx->polygonMode; v = r; return true; }
    case GL_POLYGON_OFFSET_FACTOR: v = F(ctx->polygonOffsetFactor); return true;
    case GL_POLYGON_OFFSET_UNITS: v = F(ctx->polygonOffsetUnits); return true;
    case GL_POLYGON_OFFSET_CLAMP: v = F(ctx->polygonOffsetClamp); return true;
    case GL_LINE_WIDTH: v = F(ctx->lineWidth); return true;
    case GL_POINT_SIZE: v = F(ctx->pointSize); return true;
    case GL_POINT_FADE_THRESHOLD_SIZE: v = F(ctx->pointFadeThreshold); return true;
    case GL_POINT_SPRITE_COORD_ORIGIN: v = E(ctx->pointSpriteOrigin); return true;
    case GL_PRIMITIVE_RESTART_INDEX: case GL_PRIMITIVE_RESTART_INDEX_NV: v = I(ctx->primitiveRestartIndex); return true;
    case GL_PROVOKING_VERTEX: v = E(ctx->provokingVertex); return true;
    case GL_SAMPLE_COVERAGE_VALUE: v = F(ctx->sampleCoverageValue); return true;
    case GL_SAMPLE_COVERAGE_INVERT: v = B(ctx->sampleCoverageInvert); return true;
    case GL_SAMPLE_MASK_VALUE: v = I(ctx->sampleMaskValue); return true;
    case GL_MIN_SAMPLE_SHADING_VALUE: v = F(ctx->minSampleShading); return true;
    case GL_LOGIC_OP_MODE: v = E(ctx->logicOp); return true;
    case GL_CLIP_ORIGIN: v = E(ctx->clipOrigin); return true;
    case GL_CLIP_DEPTH_MODE: v = E(ctx->clipDepthMode); return true;
    case GL_PATCH_VERTICES: v = I(ctx->patchVertices); return true;
    case GL_PATCH_DEFAULT_OUTER_LEVEL: v = F4(ctx->patchDefaultOuter); return true;
    case GL_PATCH_DEFAULT_INNER_LEVEL: v = F2(ctx->patchDefaultInner[0], ctx->patchDefaultInner[1]); return true;
    case GL_LINE_SMOOTH_HINT: v = E(ctx->hints[0]); return true;
    case GL_POLYGON_SMOOTH_HINT: v = E(ctx->hints[1]); return true;
    case GL_TEXTURE_COMPRESSION_HINT: v = E(ctx->hints[2]); return true;
    case GL_FRAGMENT_SHADER_DERIVATIVE_HINT: v = E(ctx->hints[3]); return true;
    case GL_TRANSFORM_HINT_APPLE: v = E(ctx->hints[5]); return true;
    case GL_CLIP_VOLUME_CLIPPING_HINT_EXT: v = E(ctx->hints[6]); return true;
    case GL_PACK_SWAP_BYTES: v = B(ctx->pack.swapBytes); return true;
    case GL_PACK_LSB_FIRST: v = B(ctx->pack.lsbFirst); return true;
    case GL_PACK_ROW_LENGTH: v = I(ctx->pack.rowLength); return true;
    case GL_PACK_ROW_BYTES_APPLE: v = I(ctx->pack.rowBytes); return true;
    case GL_UNPACK_ROW_BYTES_APPLE: v = I(ctx->unpack.rowBytes); return true;
    case GL_UNPACK_CLIENT_STORAGE_APPLE: v = B(ctx->unpack.clientStorage); return true;
    case GL_PACK_IMAGE_HEIGHT: v = I(ctx->pack.imageHeight); return true;
    case GL_PACK_SKIP_ROWS: v = I(ctx->pack.skipRows); return true;
    case GL_PACK_SKIP_PIXELS: v = I(ctx->pack.skipPixels); return true;
    case GL_PACK_SKIP_IMAGES: v = I(ctx->pack.skipImages); return true;
    case GL_PACK_ALIGNMENT: v = I(ctx->pack.alignment); return true;
    case GL_PACK_COMPRESSED_BLOCK_WIDTH: v = I(ctx->pack.compressedBlockWidth); return true;
    case GL_PACK_COMPRESSED_BLOCK_HEIGHT: v = I(ctx->pack.compressedBlockHeight); return true;
    case GL_PACK_COMPRESSED_BLOCK_DEPTH: v = I(ctx->pack.compressedBlockDepth); return true;
    case GL_PACK_COMPRESSED_BLOCK_SIZE: v = I(ctx->pack.compressedBlockSize); return true;
    case GL_UNPACK_SWAP_BYTES: v = B(ctx->unpack.swapBytes); return true;
    case GL_UNPACK_LSB_FIRST: v = B(ctx->unpack.lsbFirst); return true;
    case GL_UNPACK_ROW_LENGTH: v = I(ctx->unpack.rowLength); return true;
    case GL_UNPACK_IMAGE_HEIGHT: v = I(ctx->unpack.imageHeight); return true;
    case GL_UNPACK_SKIP_ROWS: v = I(ctx->unpack.skipRows); return true;
    case GL_UNPACK_SKIP_PIXELS: v = I(ctx->unpack.skipPixels); return true;
    case GL_UNPACK_SKIP_IMAGES: v = I(ctx->unpack.skipImages); return true;
    case GL_UNPACK_ALIGNMENT: v = I(ctx->unpack.alignment); return true;
    case GL_UNPACK_COMPRESSED_BLOCK_WIDTH: v = I(ctx->unpack.compressedBlockWidth); return true;
    case GL_UNPACK_COMPRESSED_BLOCK_HEIGHT: v = I(ctx->unpack.compressedBlockHeight); return true;
    case GL_UNPACK_COMPRESSED_BLOCK_DEPTH: v = I(ctx->unpack.compressedBlockDepth); return true;
    case GL_UNPACK_COMPRESSED_BLOCK_SIZE: v = I(ctx->unpack.compressedBlockSize); return true;
    default: break;
    }
    bool en;
    if (mglIsEnabled(ctx, pname, index, &en)) {
        v = B(en);
        return true;
    }
    if (!ctx->coreProfile && index < 0) {
        double out[16];
        int n = 0;
        bool isFloat = false;
        if (mglCompatGet(ctx, pname, out, &n, &isFloat)) {
            Value r;
            r.type = isFloat ? VT::Float : VT::Int;
            r.n = n;
            for (int k = 0; k < n; k++) {
                if (isFloat) r.d[k] = out[k];
                else r.i[k] = (int64_t)out[k];
            }
            v = r;
            return true;
        }
    }
    return false;
}

template <typename T>
static void convertOut(const Value& v, T* out) {
    for (int k = 0; k < v.n; k++) {
        bool isFloat = v.type == VT::Float || v.type == VT::Double;
        if constexpr (std::is_same_v<T, GLboolean>) {
            out[k] = isFloat ? (v.d[k] != 0) : (v.i[k] != 0);
        } else if constexpr (std::is_floating_point_v<T>) {
            out[k] = isFloat ? (T)v.d[k] : (T)v.i[k];
        } else {
            if (isFloat) {
                // GL: color/depth-like normalized floats map to the full int range; others round.
                out[k] = (T)llround(v.d[k]);
            } else {
                out[k] = (T)v.i[k];
            }
        }
    }
}

static bool isNormalizedFloatState(GLenum pname) {
    switch (pname) {
    case GL_COLOR_CLEAR_VALUE: case GL_BLEND_COLOR: case GL_DEPTH_CLEAR_VALUE: case GL_DEPTH_RANGE:
        return true;
    default: return false;
    }
}

template <typename T>
static void getv(GLenum pname, int index, T* data) {
    MGL_CTX();
    Value v;
    if (!getState(ctx, pname, index, v)) {
        // an indexed state whose index is out of range is INVALID_VALUE, an unknown name INVALID_ENUM
        Value probe;
        bool indexedName = index > 0 && getState(ctx, pname, 0, probe);
        return ctx->setError(indexedName ? GL_INVALID_VALUE : GL_INVALID_ENUM, "glGet(0x%x, %d)", pname, index);
    }
    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, GLboolean>) {
        if (isNormalizedFloatState(pname) && (v.type == VT::Float || v.type == VT::Double)) {
            // GL 4.6 table 2.x: normalized fixed-point conversion
            for (int k = 0; k < v.n; k++) {
                double d = std::clamp(v.d[k], -1.0, 1.0);
                double r = d * 2147483647.0;
                data[k] = (T)(int64_t)llround(r);
            }
            return;
        }
    }
    convertOut(v, data);
}

API void APIENTRY glGetBooleanv(GLenum pname, GLboolean* data) { getv(pname, -1, data); }
API void APIENTRY glGetIntegerv(GLenum pname, GLint* data) { getv(pname, -1, data); }
API void APIENTRY glGetInteger64v(GLenum pname, GLint64* data) { getv(pname, -1, data); }
API void APIENTRY glGetFloatv(GLenum pname, GLfloat* data) { getv(pname, -1, data); }
API void APIENTRY glGetDoublev(GLenum pname, GLdouble* data) { getv(pname, -1, data); }
API void APIENTRY glGetBooleani_v(GLenum pname, GLuint index, GLboolean* data) { getv(pname, (int)index, data); }
API void APIENTRY glGetIntegeri_v(GLenum pname, GLuint index, GLint* data) { getv(pname, (int)index, data); }
API void APIENTRY glGetInteger64i_v(GLenum pname, GLuint index, GLint64* data) { getv(pname, (int)index, data); }
API void APIENTRY glGetFloati_v(GLenum pname, GLuint index, GLfloat* data) { getv(pname, (int)index, data); }
API void APIENTRY glGetDoublei_v(GLenum pname, GLuint index, GLdouble* data) { getv(pname, (int)index, data); }
API void APIENTRY glGetFloatIndexedvEXT(GLenum pname, GLuint index, GLfloat* data) { getv(pname, (int)index, data); }
API void APIENTRY glGetDoubleIndexedvEXT(GLenum pname, GLuint index, GLdouble* data) { getv(pname, (int)index, data); }
API void APIENTRY glGetIntegerIndexedvEXT(GLenum pname, GLuint index, GLint* data) { getv(pname, (int)index, data); }
API void APIENTRY glGetBooleanIndexedvEXT(GLenum pname, GLuint index, GLboolean* data) { getv(pname, (int)index, data); }

static const char* rendererString() {
    static std::string s = [] {
        Device* d = Device::get();
        return std::string("metalgldrv (") + (d ? d->mtl.name.UTF8String : "no device") + ", Metal 4)";
    }();
    return s.c_str();
}

API const GLubyte* APIENTRY glGetString(GLenum name) {
    MGL_CTX_RET(nullptr);
    const AppProfile& app = appProfile();
    switch (name) {
    case GL_VENDOR: return (const GLubyte*)(app.vendor.empty() ? "metalgldrv" : app.vendor.c_str());
    case GL_RENDERER: return (const GLubyte*)(app.renderer.empty() ? rendererString() : app.renderer.c_str());
    case GL_VERSION:
        if (!app.version.empty()) return (const GLubyte*)app.version.c_str();
        return (const GLubyte*)(ctx->coreProfile ? "4.6.0 Core Profile metalgldrv " MGL_VERSION_STRING
                                                 : "4.6.0 Compatibility Profile metalgldrv " MGL_VERSION_STRING);
    case GL_PROGRAM_ERROR_STRING_ARB: return (const GLubyte*)ctx->arbErrorString.c_str();  // ARB programs' last load error
    case GL_SHADING_LANGUAGE_VERSION: return (const GLubyte*)(app.glslVersion.empty() ? "4.60 metalgldrv" : app.glslVersion.c_str());
    case GL_EXTENSIONS:
        // Removed from core profiles (use glGetStringi). Clients whose GLFW shim still calls this,
        // like Amethyst's, should create a compatibility context.
        if (ctx->coreProfile) {
            ctx->setError(GL_INVALID_ENUM);
            return nullptr;
        }
        return (const GLubyte*)extensions(ctx).joined.c_str();
    default: ctx->setError(GL_INVALID_ENUM); return nullptr;
    }
}

API const GLubyte* APIENTRY glGetStringi(GLenum name, GLuint index) {
    MGL_CTX_RET(nullptr);
    static const char* glslVersions[] = {"460", "450", "440", "430", "420", "410", "400", "330", "150", "140", "130", "120", "110", "100", "300 es", "310 es", ""};
    if (name == GL_EXTENSIONS) {
        const auto& names = extensions(ctx).names;
        if (index >= names.size()) {
            ctx->setError(GL_INVALID_VALUE);
            return nullptr;
        }
        return (const GLubyte*)names[index];
    }
    if (name == GL_SHADING_LANGUAGE_VERSION) {
        if (index >= 17) {
            ctx->setError(GL_INVALID_VALUE);
            return nullptr;
        }
        return (const GLubyte*)glslVersions[index];
    }
    if (name == GL_SPIR_V_EXTENSIONS) {
        const auto& exts = glsl::supportedSpirvExtensions();
        if (index >= exts.size()) {
            ctx->setError(GL_INVALID_VALUE);
            return nullptr;
        }
        return (const GLubyte*)exts[index].c_str();
    }
    ctx->setError(GL_INVALID_ENUM);
    return nullptr;
}

API void APIENTRY glGetPointerv(GLenum pname, void** params) {
    MGL_CTX();
    switch (pname) {
    case GL_DEBUG_CALLBACK_FUNCTION: *params = (void*)ctx->debugCallback; break;
    case GL_DEBUG_CALLBACK_USER_PARAM: *params = (void*)ctx->debugUserParam; break;
    default:
        if (!ctx->coreProfile && mglCompatGetPointer(ctx, pname, params)) return;
        ctx->setError(GL_INVALID_ENUM);
    }
}
