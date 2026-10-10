// GL_ARB_vertex_program / GL_ARB_fragment_program: assembly programs translated to compatibility GLSL
// (arb_program.cpp), which compiles like any other shader. state.* bindings become the gl_* built-in uniforms
// the fixed-function emulation feeds; program.env / program.local become uniform arrays the draw fills in.
#pragma once
#include <cstdint>
#include <string>

namespace mgl {

// Most of a resource a program may use (the GL_MAX_PROGRAM_*_ARB answers).
constexpr int kArbMaxEnvParams = 256, kArbMaxLocalParams = 256, kArbMaxInstructions = 16384, kArbMaxTemps = 256,
              kArbMaxParams = 1024, kArbMaxAttribs = 16, kArbMaxAddressRegs = 1, kArbMaxTexUnits = 16;

struct ArbTranslation {
    bool ok = false;
    std::string error;  // PROGRAM_ERROR_STRING_ARB
    int errorPos = -1;  // PROGRAM_ERROR_POSITION_ARB: byte offset of the error
    std::string glsl;   // the shader (GLSL 1.20 compatibility)
    // resources (PROGRAM_*_ARB queries)
    int instructions = 0, aluInstructions = 0, texInstructions = 0, texIndirections = 0, temps = 0, params = 0, attribs = 0,
        addressRegs = 0;
    int envUsed = 0, localUsed = 0;  // highest program.env / program.local index used + 1 (uniforms mgl_{vp,fp}{Env,Local})
    uint32_t genericAttribs = 0;     // vertex.attrib[n] used (n >= 1): attributes mgl_attrib<n> at location n
};

// target: GL_VERTEX_PROGRAM_ARB or GL_FRAGMENT_PROGRAM_ARB.
ArbTranslation translateArbProgram(unsigned target, const std::string& source);

}  // namespace mgl
