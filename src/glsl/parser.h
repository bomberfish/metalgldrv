#pragma once
#include "glsl/ast.h"
#include "glsl/pp.h"

namespace glsl {

// Parse + type-check a preprocessed token stream into `sh`. Returns false on error.
bool parseShader(Shader& sh, const PPResult& pp, const CompileOptions& opts, std::string& log);

// Expression helpers shared with the linker / emitter
bool exprIsLvalue(const Expr* e);

}  // namespace glsl
