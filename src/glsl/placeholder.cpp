// Temporary stubs until the compiler lands (replaced by the real implementation).
#include "glsl/glsl.h"
namespace glsl {
__attribute__((weak)) CompileResult compile(Stage, const std::vector<std::string>&, const CompileOptions&) {
    CompileResult r;
    r.infoLog = "compiler not available";
    return r;
}
__attribute__((weak)) LinkResult link(const LinkInput&) {
    LinkResult r;
    r.infoLog = "compiler not available";
    return r;
}
__attribute__((weak)) MSLOutput generateMSL(const LinkedProgram&, const VariantKey&) {
    MSLOutput o;
    o.error = "compiler not available";
    return o;
}
}  // namespace glsl
