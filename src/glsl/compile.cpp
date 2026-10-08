#include "glsl/ast.h"
#include "glsl/glsl.h"
#include "glsl/parser.h"
#include "glsl/pp.h"

namespace glsl {

CompileResult compile(Stage stage, const std::vector<std::string>& sources, const CompileOptions& opts) {
    CompileResult r;
    PPResult pp = preprocess(sources, stage, opts);
    r.infoLog = pp.log;
    r.unsupportedExtensions = pp.unsupportedExtensions;
    if (!pp.ok) return r;
    auto sh = std::make_shared<Shader>();
    sh->stage = stage;
    std::string log;
    bool ok = parseShader(*sh, pp, opts, log);
    r.infoLog += log;
    if (!ok) return r;
    r.shader = sh;
    return r;
}

}  // namespace glsl
