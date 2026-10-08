// GLSL preprocessor: comments, line splicing, directives and macro expansion.
#pragma once

#include <map>
#include <string>
#include <vector>

#include "glsl/glsl.h"

namespace glsl {

enum class PK : uint8_t { Ident, Number, Punct, End };

struct PPToken {
    PK kind = PK::End;
    std::string text;
    int line = 0;
    int file = 0;
    bool space = false;  // whitespace precedes this token
    std::vector<int> hide;  // macro hide set (sorted macro ids)
};

struct PPResult {
    bool ok = true;
    std::string log;
    std::vector<PPToken> tokens;
    int version = 110;
    bool versionSpecified = false;
    bool es = false;
    bool core = false;  // "#version NNN core" or >=150 without profile
    std::map<std::string, std::string> extensions;  // name -> behavior
    std::vector<std::string> pragmas;
    std::vector<std::string> unsupportedExtensions;  // #extension names not supported (or hidden)
};

PPResult preprocess(const std::vector<std::string>& sources, Stage stage, const CompileOptions& opts);

// Extensions we advertise (shared with the runtime's GL_EXTENSIONS)
const std::vector<std::string>& supportedExtensions();

}  // namespace glsl
