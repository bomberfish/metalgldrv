#pragma once
#include <string>
#include <vector>

#include "glsl/ast.h"

namespace glsl {

enum class BIOp : uint16_t { Generic };

// All builtin overloads of a function name (null if none).
const std::vector<const Builtin*>* findBuiltins(const std::string& name);

// GLSL source declaring builtin variables/constants/compat uniforms for a stage.
const std::string& builtinPreamble(Stage stage);

// Map a builtin variable name (as declared by the preamble) to its id.
BuiltinVar builtinVarId(const std::string& name, Stage stage);

}  // namespace glsl
