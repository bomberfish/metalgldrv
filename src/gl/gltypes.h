// GLSL type enum helpers (GL_FLOAT_VEC3 -> base type / shape).
#pragma once
#include "glapi.h"

namespace mgl {

enum class BaseType : uint8_t { Float, Int, UInt, Bool, Double, Int64, UInt64, Sampler, Image, AtomicCounter, Unknown };

struct TypeShape {
    BaseType base;
    int cols;  // 1 for vectors/scalars
    int rows;  // components per column
    int components() const { return cols * rows; }
};

inline TypeShape typeShape(GLenum t) {
    switch (t) {
    case GL_FLOAT: return {BaseType::Float, 1, 1};
    case GL_FLOAT_VEC2: return {BaseType::Float, 1, 2};
    case GL_FLOAT_VEC3: return {BaseType::Float, 1, 3};
    case GL_FLOAT_VEC4: return {BaseType::Float, 1, 4};
    case GL_DOUBLE: return {BaseType::Double, 1, 1};
    case GL_DOUBLE_VEC2: return {BaseType::Double, 1, 2};
    case GL_DOUBLE_VEC3: return {BaseType::Double, 1, 3};
    case GL_DOUBLE_VEC4: return {BaseType::Double, 1, 4};
    case GL_INT: return {BaseType::Int, 1, 1};
    case GL_INT_VEC2: return {BaseType::Int, 1, 2};
    case GL_INT_VEC3: return {BaseType::Int, 1, 3};
    case GL_INT_VEC4: return {BaseType::Int, 1, 4};
    case GL_UNSIGNED_INT: return {BaseType::UInt, 1, 1};
    case GL_UNSIGNED_INT_VEC2: return {BaseType::UInt, 1, 2};
    case GL_UNSIGNED_INT_VEC3: return {BaseType::UInt, 1, 3};
    case GL_UNSIGNED_INT_VEC4: return {BaseType::UInt, 1, 4};
    case GL_BOOL: return {BaseType::Bool, 1, 1};
    case GL_BOOL_VEC2: return {BaseType::Bool, 1, 2};
    case GL_BOOL_VEC3: return {BaseType::Bool, 1, 3};
    case GL_BOOL_VEC4: return {BaseType::Bool, 1, 4};
    case GL_FLOAT_MAT2: return {BaseType::Float, 2, 2};
    case GL_FLOAT_MAT3: return {BaseType::Float, 3, 3};
    case GL_FLOAT_MAT4: return {BaseType::Float, 4, 4};
    case GL_FLOAT_MAT2x3: return {BaseType::Float, 2, 3};
    case GL_FLOAT_MAT2x4: return {BaseType::Float, 2, 4};
    case GL_FLOAT_MAT3x2: return {BaseType::Float, 3, 2};
    case GL_FLOAT_MAT3x4: return {BaseType::Float, 3, 4};
    case GL_FLOAT_MAT4x2: return {BaseType::Float, 4, 2};
    case GL_FLOAT_MAT4x3: return {BaseType::Float, 4, 3};
    case GL_DOUBLE_MAT2: return {BaseType::Double, 2, 2};
    case GL_DOUBLE_MAT3: return {BaseType::Double, 3, 3};
    case GL_DOUBLE_MAT4: return {BaseType::Double, 4, 4};
    case GL_DOUBLE_MAT2x3: return {BaseType::Double, 2, 3};
    case GL_DOUBLE_MAT2x4: return {BaseType::Double, 2, 4};
    case GL_DOUBLE_MAT3x2: return {BaseType::Double, 3, 2};
    case GL_DOUBLE_MAT3x4: return {BaseType::Double, 3, 4};
    case GL_DOUBLE_MAT4x2: return {BaseType::Double, 4, 2};
    case GL_DOUBLE_MAT4x3: return {BaseType::Double, 4, 3};
    case GL_UNSIGNED_INT_ATOMIC_COUNTER: return {BaseType::AtomicCounter, 1, 1};
    case GL_INT64_ARB: return {BaseType::Int64, 1, 1};
    case GL_INT64_VEC2_ARB: return {BaseType::Int64, 1, 2};
    case GL_INT64_VEC3_ARB: return {BaseType::Int64, 1, 3};
    case GL_INT64_VEC4_ARB: return {BaseType::Int64, 1, 4};
    case GL_UNSIGNED_INT64_ARB: return {BaseType::UInt64, 1, 1};
    case GL_UNSIGNED_INT64_VEC2_ARB: return {BaseType::UInt64, 1, 2};
    case GL_UNSIGNED_INT64_VEC3_ARB: return {BaseType::UInt64, 1, 3};
    case GL_UNSIGNED_INT64_VEC4_ARB: return {BaseType::UInt64, 1, 4};
    default: break;
    }
    if ((t >= GL_SAMPLER_1D && t <= GL_SAMPLER_2D_SHADOW) || t == GL_SAMPLER_1D_ARRAY || t == GL_SAMPLER_2D_ARRAY ||
        t == GL_SAMPLER_1D_ARRAY_SHADOW || t == GL_SAMPLER_2D_ARRAY_SHADOW || t == GL_SAMPLER_CUBE_SHADOW ||
        (t >= GL_INT_SAMPLER_1D && t <= GL_UNSIGNED_INT_SAMPLER_BUFFER) || t == GL_SAMPLER_BUFFER || t == GL_SAMPLER_2D_RECT ||
        t == GL_SAMPLER_2D_RECT_SHADOW || t == GL_SAMPLER_CUBE_MAP_ARRAY || t == GL_SAMPLER_CUBE_MAP_ARRAY_SHADOW ||
        t == GL_INT_SAMPLER_CUBE_MAP_ARRAY || t == GL_UNSIGNED_INT_SAMPLER_CUBE_MAP_ARRAY || t == GL_SAMPLER_2D_MULTISAMPLE ||
        t == GL_INT_SAMPLER_2D_MULTISAMPLE || t == GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE || t == GL_SAMPLER_2D_MULTISAMPLE_ARRAY ||
        t == GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY || t == GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY)
        return {BaseType::Sampler, 1, 1};
    if ((t >= GL_IMAGE_1D && t <= GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_ARRAY)) return {BaseType::Image, 1, 1};
    return {BaseType::Unknown, 1, 1};
}

inline bool isOpaque(GLenum t) {
    auto b = typeShape(t).base;
    return b == BaseType::Sampler || b == BaseType::Image || b == BaseType::AtomicCounter;
}

}  // namespace mgl
