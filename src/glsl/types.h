#pragma once
#include "glsl/ast.h"

namespace glsl {
const Type* typeKeyword(const std::string& name);  // null if not a builtin type keyword
uint32_t textureTargetFor(const Type& samplerType);
}  // namespace glsl
