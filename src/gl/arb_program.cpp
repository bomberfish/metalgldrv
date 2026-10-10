// GL_ARB_vertex_program / GL_ARB_fragment_program assembly to GLSL 1.20 (compatibility profile).
//
// Each register is a vec4. Instructions become GLSL statements on vec4 values: the result is computed whole,
// clamped for _SAT, and stored through the destination's write mask. Bindings:
//   vertex.*            gl_Vertex, gl_Normal, gl_Color, gl_MultiTexCoordN, ... ; vertex.attrib[n] -> mgl_attrib<n>
//                       (attribute n; attribute 0 is the position)
//   fragment.*          gl_Color, gl_SecondaryColor, gl_TexCoord[n], gl_FogFragCoord, gl_FragCoord
//   state.*             the gl_* built-in uniforms (matrices, lights, materials, fog, texgen planes, ...)
//   program.env/local   uniform vec4 mgl_vpEnv[] / mgl_vpLocal[] (mgl_fpEnv / mgl_fpLocal in fragment programs)
//   result.*            gl_Position, gl_FrontColor, gl_TexCoord[n], ... ; fragment results go through locals so
//                       the fog options can apply at the end
//   texture[n]          uniform samplerXX texN (bound to unit n)
#include "gl/arb_program.h"
#include <cctype>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <vector>

namespace mgl {
namespace {

constexpr unsigned kVertexProgram = 0x8620, kFragmentProgram = 0x8804;

struct Token {
    enum Kind { Ident, Number, Punct, End } kind = End;
    std::string text;
    int pos = 0;
};

struct Error {
    std::string msg;
    int pos;
};

struct Lexer {
    const std::string& s;
    size_t i = 0;
    explicit Lexer(const std::string& src) : s(src) {}
    std::vector<Token> run() {
        std::vector<Token> out;
        while (true) {
            while (i < s.size()) {
                if (isspace((unsigned char)s[i])) i++;
                else if (s[i] == '#') {
                    while (i < s.size() && s[i] != '\n') i++;
                } else break;
            }
            Token t;
            t.pos = (int)i;
            if (i >= s.size()) {
                t.kind = Token::End;
                out.push_back(t);
                return out;
            }
            char c = s[i];
            if (isalpha((unsigned char)c) || c == '_' || c == '$' || c == '!') {
                size_t b = i;
                if (c == '!') {  // the !!ARBvp1.0 / !!ARBfp1.0 header
                    while (i < s.size() && !isspace((unsigned char)s[i])) i++;
                } else {
                    while (i < s.size() && (isalnum((unsigned char)s[i]) || s[i] == '_' || s[i] == '$')) i++;
                }
                t.kind = Token::Ident;
                t.text = s.substr(b, i - b);
            } else if (isdigit((unsigned char)c) || (c == '.' && i + 1 < s.size() && isdigit((unsigned char)s[i + 1]))) {
                size_t b = i;
                while (i < s.size() && isdigit((unsigned char)s[i])) i++;
                if (i < s.size() && s[i] == '.' && !(i + 1 < s.size() && s[i + 1] == '.')) {
                    i++;
                    while (i < s.size() && isdigit((unsigned char)s[i])) i++;
                }
                if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
                    size_t save = i++;
                    if (i < s.size() && (s[i] == '+' || s[i] == '-')) i++;
                    if (i < s.size() && isdigit((unsigned char)s[i])) {
                        while (i < s.size() && isdigit((unsigned char)s[i])) i++;
                    } else {
                        i = save;
                    }
                }
                t.kind = Token::Number;
                t.text = s.substr(b, i - b);
            } else if (c == '.' && i + 1 < s.size() && s[i + 1] == '.') {
                t.kind = Token::Punct;
                t.text = "..";
                i += 2;
            } else {
                t.kind = Token::Punct;
                t.text = std::string(1, c);
                i++;
            }
            out.push_back(t);
        }
    }
};

std::string fnum(double v) {
    char b[48];
    snprintf(b, sizeof b, "%.9g", v);
    std::string r = b;
    if (r.find_first_of(".eEn") == std::string::npos) r += ".0";
    return r;
}

enum class SymKind { Temp, Attrib, Param, ParamArray, Output, Address };

struct Symbol {
    SymKind kind;
    std::string expr;                // GLSL lvalue / rvalue (vec4), or the array name
    int arraySize = 0;               // ParamArray
    std::string outputScalar;        // Output whose GLSL target is a float (fog coordinate, point size, depth): its name
    int outputScalarComp = 0;        // the component written to it (x, or z for depth)
};

struct Translator {
    unsigned target;
    const std::string& src;
    std::vector<Token> toks;
    size_t p = 0;
    bool vp;
    ArbTranslation r;
    std::map<std::string, Symbol> syms;
    std::string decls;     // global declarations
    std::string prologue;  // start of main
    std::string body;
    std::string epilogue;
    std::set<std::string> globals;  // uniforms / attributes declared
    std::map<int, std::string> texTargets;  // unit -> GLSL sampler type
    bool positionInvariant = false;
    int fogMode = 0;  // ARB_fog_*: 1 linear, 2 exp, 3 exp2
    bool shadowOption = false;
    bool usesColorLocal = false, usesDepth = false;
    std::set<int> colorOutputs;  // result.color[n] used (fragment)
    int tempCount = 0, paramSlots = 0, nameCounter = 0;
    std::set<std::string> attribsUsed;
    std::set<int> texUnitsSampled;
    int lastTexDepth = 0;

    Translator(unsigned t, const std::string& s) : target(t), src(s), vp(t == kVertexProgram) {}

    [[noreturn]] void fail(const std::string& m, int pos = -1) { throw Error{m, pos >= 0 ? pos : cur().pos}; }
    const Token& cur() const { return toks[p]; }
    const Token& peek(int k = 1) const { return toks[std::min(p + k, toks.size() - 1)]; }
    bool isP(const char* t) const { return cur().kind == Token::Punct && cur().text == t; }
    bool isI(const char* t) const { return cur().kind == Token::Ident && cur().text == t; }
    void expectP(const char* t) {
        if (!isP(t)) fail(std::string("expected '") + t + "'");
        p++;
    }
    std::string ident() {
        if (cur().kind != Token::Ident) fail("expected an identifier");
        return toks[p++].text;
    }
    int integer() {
        if (cur().kind != Token::Number) fail("expected an integer");
        int v = atoi(cur().text.c_str());
        p++;
        return v;
    }
    double number() {
        bool neg = false;
        if (isP("-")) {
            neg = true;
            p++;
        } else if (isP("+")) {
            p++;
        }
        if (cur().kind != Token::Number) fail("expected a number");
        double v = atof(cur().text.c_str());
        p++;
        return neg ? -v : v;
    }
    // "." then the identifier; false (nothing consumed) when no "." follows
    bool dotIdent(const char* name) {
        if (isP(".") && peek().kind == Token::Ident && peek().text == name) {
            p += 2;
            return true;
        }
        return false;
    }
    std::string afterDot() {
        expectP(".");
        return ident();
    }
    int bracketIndex() {
        expectP("[");
        int v = integer();
        expectP("]");
        return v;
    }
    void global(const std::string& decl) {
        if (globals.insert(decl).second) decls += decl + "\n";
    }

    // ---------------- bindings ----------------
    std::string matrixName(const std::string& base) {
        // state.matrix.<which>[.inverse|.transpose|.invtrans]
        std::string mod;
        if (isP(".") && peek().kind == Token::Ident && (peek().text == "inverse" || peek().text == "transpose" || peek().text == "invtrans")) {
            p++;
            mod = ident();
        }
        std::string suffix = mod == "inverse" ? "Inverse" : mod == "transpose" ? "Transpose" : mod == "invtrans" ? "InverseTranspose" : "";
        if (base.compare(0, 17, "gl_TextureMatrix[") == 0) return "gl_TextureMatrix" + suffix + base.substr(16);  // the array index last
        return base + suffix;
    }

    // state.matrix.X...: the matrix's GLSL name and the rows bound (all four unless .row[a(..b)]).
    void stateMatrix(std::string& m, int& r0, int& r1) {
        std::string which = afterDot();
        std::string base;
        if (which == "mvp") base = "gl_ModelViewProjectionMatrix";
        else if (which == "projection") base = "gl_ProjectionMatrix";
        else if (which == "modelview") {
            if (isP("[")) {
                int n = bracketIndex();
                if (n != 0) fail("vertex blending matrices (state.matrix.modelview[n], n > 0) are not supported");
            }
            base = "gl_ModelViewMatrix";
        } else if (which == "texture") {
            int n = isP("[") ? bracketIndex() : 0;
            base = "gl_TextureMatrix[" + std::to_string(n) + "]";
        } else if (which == "palette" || which == "program") {
            fail("state.matrix." + which + " is not supported");
        } else {
            fail("unknown matrix '" + which + "'");
        }
        m = matrixName(base);
        r0 = 0;
        r1 = 3;
        if (dotIdent("row")) {
            expectP("[");
            r0 = r1 = integer();
            if (isP("..")) {
                p++;
                r1 = integer();
            }
            expectP("]");
            if (r0 < 0 || r1 > 3 || r0 > r1) fail("bad matrix row range");
        }
    }

    static std::string row(const std::string& m, int i) {
        std::string k = std::to_string(i);
        return "vec4(" + m + "[0][" + k + "], " + m + "[1][" + k + "], " + m + "[2][" + k + "], " + m + "[3][" + k + "])";
    }

    // A state.* binding after "state": GLSL vec4 expressions (several for matrices).
    std::vector<std::string> stateBinding() {
        std::string what = afterDot();
        if (what == "matrix") {
            std::string m;
            int r0, r1;
            stateMatrix(m, r0, r1);
            std::vector<std::string> rows;
            for (int i = r0; i <= r1; i++) rows.push_back(row(m, i));
            return rows;
        }
        auto face = [&]() {
            if (dotIdent("front")) return std::string("Front");
            if (dotIdent("back")) return std::string("Back");
            return std::string("Front");
        };
        if (what == "material") {
            std::string f = face();
            std::string prop = afterDot();
            std::string mat = "gl_" + f + "Material";
            if (prop == "shininess") return {"vec4(" + mat + ".shininess, 0.0, 0.0, 1.0)"};
            if (prop == "ambient" || prop == "diffuse" || prop == "specular" || prop == "emission") return {mat + "." + prop};
            fail("unknown material property '" + prop + "'");
        }
        if (what == "light") {
            int n = bracketIndex();
            std::string L = "gl_LightSource[" + std::to_string(n) + "]";
            std::string prop = afterDot();
            if (prop == "ambient" || prop == "diffuse" || prop == "specular" || prop == "position") return {L + "." + prop};
            if (prop == "half") return {L + ".halfVector"};
            if (prop == "attenuation")
                return {"vec4(" + L + ".constantAttenuation, " + L + ".linearAttenuation, " + L + ".quadraticAttenuation, " + L + ".spotExponent)"};
            if (prop == "spot") {
                if (!dotIdent("direction")) fail("expected 'direction'");
                return {"vec4(" + L + ".spotDirection, " + L + ".spotCosCutoff)"};
            }
            fail("unknown light property '" + prop + "'");
        }
        if (what == "lightmodel") {
            if (dotIdent("ambient")) return {"gl_LightModel.ambient"};
            std::string f = face();
            if (!dotIdent("scenecolor")) fail("expected 'ambient' or 'scenecolor'");
            return {"gl_" + f + "LightModelProduct.sceneColor"};
        }
        if (what == "lightprod") {
            int n = bracketIndex();
            std::string f = face();
            std::string prop = afterDot();
            if (prop != "ambient" && prop != "diffuse" && prop != "specular") fail("unknown light product '" + prop + "'");
            return {"gl_" + f + "LightProduct[" + std::to_string(n) + "]." + prop};
        }
        if (what == "texgen") {
            int n = isP("[") ? bracketIndex() : 0;
            std::string kind = afterDot();
            if (kind != "eye" && kind != "object") fail("expected 'eye' or 'object'");
            std::string c = afterDot();
            if (c.size() != 1 || std::string("strq").find(c) == std::string::npos) fail("expected s, t, r or q");
            return {std::string(kind == "eye" ? "gl_EyePlane" : "gl_ObjectPlane") + (char)toupper(c[0]) + "[" + std::to_string(n) + "]"};
        }
        if (what == "fog") {
            if (dotIdent("color")) return {"gl_Fog.color"};
            if (dotIdent("params")) return {"vec4(gl_Fog.density, gl_Fog.start, gl_Fog.end, gl_Fog.scale)"};
            fail("expected 'color' or 'params'");
        }
        if (what == "clip") {
            int n = bracketIndex();
            if (!dotIdent("plane")) fail("expected 'plane'");
            return {"gl_ClipPlane[" + std::to_string(n) + "]"};
        }
        if (what == "point") {
            if (dotIdent("size")) return {"vec4(gl_Point.size, gl_Point.sizeMin, gl_Point.sizeMax, gl_Point.fadeThresholdSize)"};
            if (dotIdent("attenuation"))
                return {"vec4(gl_Point.distanceConstantAttenuation, gl_Point.distanceLinearAttenuation, gl_Point.distanceQuadraticAttenuation, 1.0)"};
            fail("expected 'size' or 'attenuation'");
        }
        if (what == "depth") {
            if (!dotIdent("range")) fail("expected 'range'");
            return {"vec4(gl_DepthRange.near, gl_DepthRange.far, gl_DepthRange.diff, 1.0)"};
        }
        if (what == "texenv") {
            int n = isP("[") ? bracketIndex() : 0;
            if (!dotIdent("color")) fail("expected 'color'");
            return {"gl_TextureEnvColor[" + std::to_string(n) + "]"};
        }
        fail("unknown state binding 'state." + what + "'");
    }

    // program.env[a(..b)] / program.local[a(..b)]
    std::vector<std::string> programBinding(bool allowRange) {
        std::string which = afterDot();
        if (which != "env" && which != "local") fail("expected 'env' or 'local'");
        expectP("[");
        int a = integer(), b = a;
        if (allowRange && isP("..")) {
            p++;
            b = integer();
        }
        expectP("]");
        int max = which == "env" ? kArbMaxEnvParams : kArbMaxLocalParams;
        if (a < 0 || b >= max || a > b) fail("program." + which + " index out of range");
        // per stage: a vertex and a fragment program linked together each have their own
        std::string arr = std::string(vp ? "mgl_vp" : "mgl_fp") + (which == "env" ? "Env" : "Local");
        global("uniform vec4 " + arr + "[" + std::to_string(max) + "];");
        int& used = which == "env" ? r.envUsed : r.localUsed;
        used = std::max(used, b + 1);
        std::vector<std::string> out;
        for (int i = a; i <= b; i++) out.push_back(arr + "[" + std::to_string(i) + "]");
        return out;
    }

    std::string constVector() {  // { x (, y (, z (, w))) }
        expectP("{");
        double v[4] = {0, 0, 0, 1};
        int n = 0;
        while (true) {
            if (n >= 4) fail("too many components");
            v[n++] = number();
            if (isP(",")) {
                p++;
                continue;
            }
            break;
        }
        expectP("}");
        return "vec4(" + fnum(v[0]) + ", " + fnum(v[1]) + ", " + fnum(v[2]) + ", " + fnum(v[3]) + ")";
    }

    bool atNumber() const { return cur().kind == Token::Number || ((cur().text == "-" || cur().text == "+") && peek().kind == Token::Number); }

    // One item of a PARAM initializer (several vec4s for matrices and ranges).
    std::vector<std::string> paramItems(bool allowRange) {
        if (isI("state")) {
            p++;
            return stateBinding();
        }
        if (isI("program")) {
            p++;
            return programBinding(allowRange);
        }
        if (isP("{")) return {constVector()};
        if (atNumber()) {
            double v = number();
            return {"vec4(" + fnum(v) + ")"};
        }
        fail("expected a parameter binding");
    }

    // vertex.* (vertex programs) / fragment.* (fragment programs) attribute bindings
    std::string attribBinding() {
        std::string src0 = ident();
        if (vp) {
            if (src0 != "vertex") fail("expected 'vertex'");
            std::string a = afterDot();
            if (a == "position") return "gl_Vertex";
            if (a == "normal") return "vec4(gl_Normal, 1.0)";
            if (a == "color") {
                if (dotIdent("secondary")) return "gl_SecondaryColor";
                dotIdent("primary");
                return "gl_Color";
            }
            if (a == "fogcoord") return "vec4(gl_FogCoord, 0.0, 0.0, 1.0)";
            if (a == "texcoord") {
                int n = isP("[") ? bracketIndex() : 0;
                if (n < 0 || n > 7) fail("texture coordinate set out of range");
                return "gl_MultiTexCoord" + std::to_string(n);
            }
            if (a == "weight") {
                if (isP("[")) bracketIndex();
                return "vec4(1.0, 0.0, 0.0, 0.0)";
            }
            if (a == "matrixindex") {
                if (isP("[")) bracketIndex();
                return "vec4(0.0)";
            }
            if (a == "attrib") {
                int n = bracketIndex();
                if (n < 0 || n >= kArbMaxAttribs) fail("generic attribute out of range");
                if (n == 0) return "gl_Vertex";  // attribute 0 is the position
                global("attribute vec4 mgl_attrib" + std::to_string(n) + ";");
                r.genericAttribs |= 1u << n;
                return "mgl_attrib" + std::to_string(n);
            }
            fail("unknown vertex attribute '" + a + "'");
        }
        if (src0 != "fragment") fail("expected 'fragment'");
        std::string a = afterDot();
        if (a == "color") {
            if (dotIdent("secondary")) return "gl_SecondaryColor";
            dotIdent("primary");
            return "gl_Color";
        }
        if (a == "texcoord") {
            int n = isP("[") ? bracketIndex() : 0;
            if (n < 0 || n > 7) fail("texture coordinate set out of range");
            return "gl_TexCoord[" + std::to_string(n) + "]";
        }
        if (a == "fogcoord") return "vec4(gl_FogFragCoord, 0.0, 0.0, 1.0)";
        if (a == "position") return "gl_FragCoord";
        fail("unknown fragment attribute '" + a + "'");
    }

    // result.* bindings: a Symbol with the GLSL lvalue (vec4 or float).
    Symbol resultBinding() {
        if (ident() != "result") fail("expected 'result'");
        std::string a = afterDot();
        Symbol s{SymKind::Output, ""};
        if (vp) {
            if (a == "position") {
                if (positionInvariant) fail("result.position can't be written with ARB_position_invariant");
                s.expr = "gl_Position";
            } else if (a == "color") {
                bool back = false, secondary = false;
                for (;;) {
                    if (dotIdent("front")) back = false;
                    else if (dotIdent("back")) back = true;
                    else if (dotIdent("primary")) secondary = false;
                    else if (dotIdent("secondary")) secondary = true;
                    else break;
                }
                s.expr = std::string("gl_") + (back ? "Back" : "Front") + (secondary ? "SecondaryColor" : "Color");
            } else if (a == "texcoord") {
                int n = isP("[") ? bracketIndex() : 0;
                if (n < 0 || n > 7) fail("texture coordinate set out of range");
                s.expr = "gl_TexCoord[" + std::to_string(n) + "]";
            } else if (a == "fogcoord") {
                s.outputScalar = "gl_FogFragCoord";
            } else if (a == "pointsize") {
                s.outputScalar = "gl_PointSize";
            } else {
                fail("unknown vertex result '" + a + "'");
            }
            return s;
        }
        if (a == "color") {
            int n = isP("[") ? bracketIndex() : 0;
            if (n < 0 || n > 7) fail("color output out of range");
            colorOutputs.insert(n);
            s.expr = "mgl_color" + std::to_string(n);
            global("vec4 mgl_color" + std::to_string(n) + " = vec4(0.0);");
        } else if (a == "depth") {
            usesDepth = true;
            s.outputScalar = "gl_FragDepth";
            s.outputScalarComp = 2;
        } else {
            fail("unknown fragment result '" + a + "'");
        }
        return s;
    }

    // ---------------- declarations ----------------
    std::string newName(const char* prefix, const std::string& n) { return std::string(prefix) + n; }

    void declare(const std::string& name, Symbol s, int pos) {
        if (syms.count(name)) fail("'" + name + "' is already declared", pos);
        syms[name] = s;
    }

    void paramDecl() {
        int pos = cur().pos;
        std::string name = ident();
        if (isP("[")) {  // array
            p++;
            int size = -1;
            if (!isP("]")) size = integer();
            expectP("]");
            expectP("=");
            expectP("{");
            std::vector<std::string> items;
            while (true) {
                auto v = paramItems(true);
                items.insert(items.end(), v.begin(), v.end());
                if (isP(",")) {
                    p++;
                    continue;
                }
                break;
            }
            expectP("}");
            if (size >= 0 && size != (int)items.size()) fail("array size doesn't match its initializers", pos);
            std::string g = newName("P_", name);
            prologue += "  vec4 " + g + "[" + std::to_string(items.size()) + "];\n";
            for (size_t i = 0; i < items.size(); i++) prologue += "  " + g + "[" + std::to_string(i) + "] = " + items[i] + ";\n";
            paramSlots += (int)items.size();
            Symbol s{SymKind::ParamArray, g};
            s.arraySize = (int)items.size();
            declare(name, s, pos);
            return;
        }
        expectP("=");
        auto v = paramItems(false);
        if (v.size() != 1) fail("a single PARAM needs a single binding (use an array for matrices)", pos);
        std::string g = newName("P_", name);
        prologue += "  vec4 " + g + " = " + v[0] + ";\n";
        paramSlots++;
        declare(name, Symbol{SymKind::Param, g}, pos);
    }

    void statement() {
        int pos = cur().pos;
        std::string kw = cur().text;
        if (cur().kind != Token::Ident) fail("expected a statement");
        p++;
        if (kw == "OPTION") {
            std::string opt = ident();
            if (vp && opt == "ARB_position_invariant") {
                positionInvariant = true;
            } else if (!vp && opt == "ARB_fog_linear") {
                fogMode = 1;
            } else if (!vp && opt == "ARB_fog_exp") {
                fogMode = 2;
            } else if (!vp && opt == "ARB_fog_exp2") {
                fogMode = 3;
            } else if (!vp && (opt == "ARB_precision_hint_fastest" || opt == "ARB_precision_hint_nicest")) {
            } else if (!vp && opt == "ARB_fragment_program_shadow") {
                shadowOption = true;
            } else {
                fail("unsupported option '" + opt + "'", pos);
            }
        } else if (kw == "ATTRIB") {
            std::string name = ident();
            expectP("=");
            std::string e = attribBinding();
            attribsUsed.insert(e);
            declare(name, Symbol{SymKind::Attrib, e}, pos);
        } else if (kw == "PARAM") {
            paramDecl();
        } else if (kw == "TEMP" || kw == "ADDRESS") {
            while (true) {
                int npos = cur().pos;
                std::string name = ident();
                if (kw == "TEMP") {
                    std::string g = newName("T_", name);
                    prologue += "  vec4 " + g + " = vec4(0.0);\n";
                    tempCount++;
                    declare(name, Symbol{SymKind::Temp, g}, npos);
                } else {
                    if (!vp) fail("ADDRESS registers are for vertex programs", npos);
                    std::string g = newName("A_", name);
                    prologue += "  ivec4 " + g + " = ivec4(0);\n";
                    r.addressRegs++;
                    declare(name, Symbol{SymKind::Address, g}, npos);
                }
                if (isP(",")) {
                    p++;
                    continue;
                }
                break;
            }
        } else if (kw == "OUTPUT") {
            std::string name = ident();
            expectP("=");
            declare(name, resultBinding(), pos);
        } else if (kw == "ALIAS") {
            std::string name = ident();
            expectP("=");
            int apos = cur().pos;
            std::string target = ident();
            auto it = syms.find(target);
            if (it == syms.end()) fail("'" + target + "' isn't declared", apos);
            declare(name, it->second, pos);
        } else {
            p--;
            instruction();
            return;
        }
        expectP(";");
    }

    // ---------------- operands ----------------
    static std::string swizzleChars(const std::string& sw) {
        std::string o;
        for (char c : sw) {
            switch (c) {
            case 'x': case 'r': o += 'x'; break;
            case 'y': case 'g': o += 'y'; break;
            case 'z': case 'b': o += 'z'; break;
            case 'w': case 'a': o += 'w'; break;
            default: return "";
            }
        }
        return o;
    }

    // A source register as a vec4 expression (before the swizzle).
    std::string srcRegister() {
        if (isP("{")) return constVector();
        if (atNumber()) return "vec4(" + fnum(number()) + ")";
        int pos = cur().pos;
        if (isI("vertex") || isI("fragment")) {
            std::string e = attribBinding();
            attribsUsed.insert(e);
            return e;
        }
        if (isI("state")) {
            p++;
            auto v = stateBinding();
            if (v.size() != 1) fail("a matrix binding needs .row[n] here", pos);
            paramSlots++;
            return v[0];
        }
        if (isI("program")) {
            p++;
            auto v = programBinding(false);
            paramSlots++;
            return v[0];
        }
        std::string name = ident();
        auto it = syms.find(name);
        if (it == syms.end()) fail("'" + name + "' isn't declared", pos);
        const Symbol& s = it->second;
        switch (s.kind) {
        case SymKind::Temp: case SymKind::Attrib: case SymKind::Param: return s.expr;
        case SymKind::ParamArray: {
            expectP("[");
            std::string idx;
            if (cur().kind == Token::Number) {
                int i = integer();
                if (i < 0 || i >= s.arraySize) fail("array index out of range", pos);
                idx = std::to_string(i);
            } else {  // relative: A0.x (+|- offset)
                int apos = cur().pos;
                std::string an = ident();
                auto a = syms.find(an);
                if (a == syms.end() || a->second.kind != SymKind::Address) fail("expected an address register", apos);
                expectP(".");
                if (ident() != "x") fail("address registers have only .x");
                std::string off;
                if (isP("+") || isP("-")) {
                    std::string sign = toks[p++].text;
                    off = " " + sign + " " + std::to_string(integer());
                }
                idx = "clamp(" + a->second.expr + ".x" + off + ", 0, " + std::to_string(s.arraySize - 1) + ")";
            }
            expectP("]");
            return s.expr + "[" + idx + "]";
        }
        default: fail("'" + name + "' can't be read", pos);
        }
    }

    // [-] register [.swizzle]
    std::string srcOperand() {
        bool neg = false;
        if (isP("-")) {
            neg = true;
            p++;
        } else if (isP("+")) {
            p++;
        }
        std::string e = srcRegister();
        if (isP(".") && peek().kind == Token::Ident) {
            int pos = peek().pos;
            std::string sw = swizzleChars(peek().text);
            if (sw.empty() || (sw.size() != 1 && sw.size() != 4)) fail("bad swizzle '" + peek().text + "'", pos);
            p += 2;
            if (sw.size() == 1) sw = std::string(4, sw[0]);
            if (sw != "xyzw") e = "(" + e + ")." + sw;
        }
        return neg ? "(-(" + e + "))" : e;
    }

    // A scalar source (.x of the operand unless one component is selected)
    std::string scalarOperand() {
        std::string e = srcOperand();
        return "(" + e + ").x";
    }

    struct Dst {
        Symbol sym;
        std::string mask;  // components written, "xyzw" when none given
    };

    Dst dstOperand() {
        Dst d;
        int pos = cur().pos;
        if (isI("result")) {
            d.sym = resultBinding();
        } else {
            std::string name = ident();
            auto it = syms.find(name);
            if (it == syms.end()) fail("'" + name + "' isn't declared", pos);
            d.sym = it->second;
            if (d.sym.kind != SymKind::Temp && d.sym.kind != SymKind::Output && d.sym.kind != SymKind::Address)
                fail("'" + name + "' can't be written", pos);
        }
        d.mask = "xyzw";
        if (isP(".") && peek().kind == Token::Ident) {
            int mpos = peek().pos;
            std::string m = swizzleChars(peek().text);
            p += 2;
            // a write mask: components in order, no repeats
            bool ok = !m.empty();
            for (size_t i = 1; i < m.size() && ok; i++) ok = std::string("xyzw").find(m[i]) > std::string("xyzw").find(m[i - 1]);
            if (!ok) fail("bad write mask", mpos);
            d.mask = m;
        }
        return d;
    }

    void store(const Dst& d, const std::string& value, bool sat) {
        std::string v = sat ? "clamp(" + value + ", 0.0, 1.0)" : value;
        body += "  { vec4 mgl_r = " + v + ";\n";
        if (!d.sym.outputScalar.empty()) {
            char c = "xyzw"[d.sym.outputScalarComp];
            if (d.mask.find(c) != std::string::npos) body += "    " + d.sym.outputScalar + " = mgl_r." + c + ";\n";
        } else if (d.mask == "xyzw") {
            body += "    " + d.sym.expr + " = mgl_r;\n";
        } else {
            body += "    " + d.sym.expr + "." + d.mask + " = mgl_r." + d.mask + ";\n";
        }
        body += "  }\n";
    }

    // texture[n], TARGET
    void texOperands(int& unit, std::string& type, std::string& kind) {
        int pos = cur().pos;
        if (ident() != "texture") fail("expected texture[n]", pos);
        unit = isP("[") ? bracketIndex() : 0;
        if (unit < 0 || unit >= kArbMaxTexUnits) fail("texture unit out of range", pos);
        expectP(",");
        int tpos = cur().pos;
        kind = cur().kind == Token::Number ? toks[p++].text : ident();
        if (kind == "1" || kind == "2" || kind == "3") {  // "1D" lexes as a number then an identifier
            if (!isI("D")) fail("expected a texture target", tpos);
            p++;
            kind += "D";
        }
        static const std::map<std::string, std::string> types = {
            {"1D", "sampler1D"}, {"2D", "sampler2D"}, {"3D", "sampler3D"}, {"CUBE", "samplerCube"}, {"RECT", "sampler2DRect"},
            {"SHADOW1D", "sampler1DShadow"}, {"SHADOW2D", "sampler2DShadow"}, {"SHADOWRECT", "sampler2DRectShadow"}};
        auto it = types.find(kind);
        if (it == types.end()) fail("unknown texture target '" + kind + "'", tpos);
        if (kind.compare(0, 6, "SHADOW") == 0 && !shadowOption) fail("shadow targets need OPTION ARB_fragment_program_shadow", tpos);
        type = it->second;
        auto prev = texTargets.find(unit);
        if (prev != texTargets.end() && prev->second != type) fail("texture unit used with two targets", tpos);
        texTargets[unit] = type;
        global("uniform " + type + " tex" + std::to_string(unit) + ";");
    }

    std::string texLookup(const std::string& op, int unit, const std::string& kind, const std::string& c) {
        std::string t = "tex" + std::to_string(unit);
        bool proj = op == "TXP", bias = op == "TXB";
        std::string b = bias ? ", (" + c + ").w" : "";
        if (kind == "1D") return proj ? "texture1DProj(" + t + ", " + c + ")" : "texture1D(" + t + ", (" + c + ").x" + b + ")";
        if (kind == "2D") return proj ? "texture2DProj(" + t + ", " + c + ")" : "texture2D(" + t + ", (" + c + ").xy" + b + ")";
        if (kind == "3D") return proj ? "texture3DProj(" + t + ", " + c + ")" : "texture3D(" + t + ", (" + c + ").xyz" + b + ")";
        if (kind == "CUBE") return "textureCube(" + t + ", (" + c + ").xyz" + b + ")";
        if (kind == "RECT") return proj ? "texture2DRectProj(" + t + ", " + c + ")" : "texture2DRect(" + t + ", (" + c + ").xy)";
        if (kind == "SHADOW1D") return proj ? "shadow1DProj(" + t + ", " + c + ")" : "shadow1D(" + t + ", (" + c + ").xyz" + b + ")";
        if (kind == "SHADOW2D") return proj ? "shadow2DProj(" + t + ", " + c + ")" : "shadow2D(" + t + ", (" + c + ").xyz" + b + ")";
        return proj ? "shadow2DRectProj(" + t + ", " + c + ")" : "shadow2DRect(" + t + ", (" + c + ").xyz)";
    }

    // ---------------- instructions ----------------
    void instruction() {
        int pos = cur().pos;
        std::string op = ident();
        bool sat = false;
        if (!vp && op.size() > 4 && op.compare(op.size() - 4, 4, "_SAT") == 0) {
            sat = true;
            op.resize(op.size() - 4);
        }
        auto vec1 = [](const std::string& s) { return "vec4(" + s + ")"; };
        r.instructions++;
        static const std::set<std::string> vpOnly = {"ARL", "EXP", "LOG"};
        static const std::set<std::string> fpOnly = {"CMP", "COS", "KIL", "LRP", "SCS", "SIN", "TEX", "TXB", "TXP"};
        if (vp && fpOnly.count(op)) fail(op + " is a fragment program instruction", pos);
        if (!vp && vpOnly.count(op)) fail(op + " is a vertex program instruction", pos);
        if (op == "KIL") {
            std::string a = srcOperand();
            body += "  if (any(lessThan(" + a + ", vec4(0.0)))) discard;\n";
            r.texInstructions++;
            expectP(";");
            return;
        }
        Dst d = dstOperand();
        auto comma = [&] { expectP(","); };
        std::string v;
        if (op == "ARL") {
            if (d.sym.kind != SymKind::Address) fail("ARL writes an address register", pos);
            comma();
            std::string a = scalarOperand();
            body += "  " + d.sym.expr + ".x = int(floor(" + a + "));\n";
            r.aluInstructions++;
            expectP(";");
            return;
        }
        if (d.sym.kind == SymKind::Address) fail("only ARL writes address registers", pos);
        if (op == "TEX" || op == "TXP" || op == "TXB") {
            comma();
            std::string c = srcOperand();
            comma();
            int unit;
            std::string type, kind;
            texOperands(unit, type, kind);
            v = texLookup(op, unit, kind, c);
            r.texInstructions++;
            texUnitsSampled.insert(unit);
        } else {
            static const std::map<std::string, int> arity = {
                {"ABS", 1}, {"ADD", 2}, {"CMP", 3}, {"COS", 1}, {"DP3", 2}, {"DP4", 2}, {"DPH", 2}, {"DST", 2}, {"EX2", 1},
                {"EXP", 1}, {"FLR", 1}, {"FRC", 1}, {"LG2", 1}, {"LIT", 1}, {"LOG", 1}, {"LRP", 3}, {"MAD", 3}, {"MAX", 2},
                {"MIN", 2}, {"MOV", 1}, {"MUL", 2}, {"POW", 2}, {"RCP", 1}, {"RSQ", 1}, {"SCS", 1}, {"SGE", 2}, {"SIN", 1},
                {"SLT", 2}, {"SUB", 2}, {"SWZ", 1}, {"XPD", 2}};
            auto it = arity.find(op);
            if (it == arity.end()) fail("unknown instruction '" + op + "'", pos);
            std::vector<std::string> a;
            static const std::set<std::string> scalar = {"COS", "EX2", "EXP", "LG2", "LOG", "RCP", "RSQ", "SCS", "SIN"};
            if (op == "SWZ") {
                comma();
                // SWZ dst, src, extended swizzle (4 of [-](0|1|x|y|z|w|r|g|b|a))
                bool neg = false;
                if (isP("-")) {
                    neg = true;
                    p++;
                }
                std::string base = srcRegister();
                std::string comps[4];
                for (int k = 0; k < 4; k++) {
                    comma();
                    bool cn = false;
                    if (isP("-")) {
                        cn = true;
                        p++;
                    } else if (isP("+")) {
                        p++;
                    }
                    std::string c;
                    if (cur().kind == Token::Number) c = toks[p++].text;
                    else c = ident();
                    std::string e;
                    if (c == "0") e = "0.0";
                    else if (c == "1") e = "1.0";
                    else {
                        std::string sc = swizzleChars(c);
                        if (sc.size() != 1) fail("bad extended swizzle component '" + c + "'");
                        e = "(" + base + ")." + sc;
                    }
                    comps[k] = (cn != neg) ? "-(" + e + ")" : e;
                }
                v = "vec4(" + comps[0] + ", " + comps[1] + ", " + comps[2] + ", " + comps[3] + ")";
            } else {
                for (int k = 0; k < it->second; k++) {
                    comma();
                    a.push_back(scalar.count(op) ? scalarOperand() : srcOperand());
                }
            }
            if (op == "ABS") v = "abs(" + a[0] + ")";
            else if (op == "ADD") v = a[0] + " + " + a[1];
            else if (op == "SUB") v = a[0] + " - " + a[1];
            else if (op == "MUL") v = a[0] + " * " + a[1];
            else if (op == "MAD") v = a[0] + " * " + a[1] + " + " + a[2];
            else if (op == "MOV") v = a[0];
            else if (op == "MAX") v = "max(" + a[0] + ", " + a[1] + ")";
            else if (op == "MIN") v = "min(" + a[0] + ", " + a[1] + ")";
            else if (op == "DP3") v = vec1("dot((" + a[0] + ").xyz, (" + a[1] + ").xyz)");
            else if (op == "DP4") v = vec1("dot(" + a[0] + ", " + a[1] + ")");
            else if (op == "DPH") v = vec1("dot((" + a[0] + ").xyz, (" + a[1] + ").xyz) + (" + a[1] + ").w");
            else if (op == "DST") v = "vec4(1.0, (" + a[0] + ").y * (" + a[1] + ").y, (" + a[0] + ").z, (" + a[1] + ").w)";
            else if (op == "EX2") v = vec1("exp2(" + a[0] + ")");
            else if (op == "LG2") v = vec1("log2(" + a[0] + ")");
            else if (op == "EXP") v = "vec4(exp2(floor(" + a[0] + ")), fract(" + a[0] + "), exp2(" + a[0] + "), 1.0)";
            else if (op == "LOG")
                v = "vec4(floor(log2(abs(" + a[0] + "))), abs(" + a[0] + ") / exp2(floor(log2(abs(" + a[0] + ")))), log2(abs(" + a[0] + ")), 1.0)";
            else if (op == "FLR") v = "floor(" + a[0] + ")";
            else if (op == "FRC") v = "fract(" + a[0] + ")";
            else if (op == "LIT")
                v = "mgl_lit(" + a[0] + ")";
            else if (op == "POW") v = vec1("pow((" + a[0] + ").x, (" + a[1] + ").x)");
            else if (op == "RCP") v = vec1("1.0 / " + a[0]);
            else if (op == "RSQ") v = vec1("inversesqrt(abs(" + a[0] + "))");
            else if (op == "SGE") v = "vec4(greaterThanEqual(" + a[0] + ", " + a[1] + "))";
            else if (op == "SLT") v = "vec4(lessThan(" + a[0] + ", " + a[1] + "))";
            else if (op == "XPD") v = "vec4(cross((" + a[0] + ").xyz, (" + a[1] + ").xyz), 1.0)";
            else if (op == "CMP") v = "mix(" + a[2] + ", " + a[1] + ", vec4(lessThan(" + a[0] + ", vec4(0.0))))";
            else if (op == "COS") v = vec1("cos(" + a[0] + ")");
            else if (op == "SIN") v = vec1("sin(" + a[0] + ")");
            else if (op == "SCS") v = "vec4(cos(" + a[0] + "), sin(" + a[0] + "), 0.0, 0.0)";
            else if (op == "LRP") v = "mix(" + a[2] + ", " + a[1] + ", " + a[0] + ")";
            if (op == "LIT") global("vec4 mgl_lit(vec4 s) { float d = max(s.x, 0.0); float sp = s.x > 0.0 ? pow(max(s.y, 0.0), clamp(s.w, -128.0, 128.0)) : 0.0; return vec4(1.0, d, sp, 1.0); }");
            r.aluInstructions++;
        }
        store(d, v, sat);
        expectP(";");
    }

    void run() {
        Lexer lx(src);
        toks = lx.run();
        const char* header = vp ? "!!ARBvp1.0" : "!!ARBfp1.0";
        if (cur().kind != Token::Ident || cur().text != header) fail(std::string("the program must start with ") + header, 0);
        p++;
        while (!isI("END")) {
            if (cur().kind == Token::End) fail("missing END");
            statement();
        }
        // GLSL
        std::string out = "#version 120\n" + decls + "void main() {\n" + prologue;
        if (positionInvariant) out += "  gl_Position = ftransform();\n";
        out += body;
        if (!vp) {
            std::string c0 = colorOutputs.count(0) || colorOutputs.empty() ? "mgl_color0" : "vec4(0.0)";
            if (colorOutputs.empty()) out = "#version 120\nvec4 mgl_color0 = vec4(0.0);\n" + out.substr(13);
            if (fogMode) {
                const char* f = fogMode == 1 ? "(gl_Fog.end - gl_FogFragCoord) * gl_Fog.scale"
                                : fogMode == 2 ? "exp(-gl_Fog.density * gl_FogFragCoord)"
                                               : "exp(-pow(gl_Fog.density * gl_FogFragCoord, 2.0))";
                out += "  " + c0 + ".rgb = mix(gl_Fog.color.rgb, " + c0 + ".rgb, clamp(" + f + ", 0.0, 1.0));\n";
            }
            bool mrt = colorOutputs.size() > 1 || (colorOutputs.size() == 1 && !colorOutputs.count(0));
            if (mrt) {
                for (int n : colorOutputs) out += "  gl_FragData[" + std::to_string(n) + "] = mgl_color" + std::to_string(n) + ";\n";
            } else {
                out += "  gl_FragColor = mgl_color0;\n";
            }
        }
        out += "}\n";
        r.glsl = out;
        r.temps = tempCount;
        r.params = paramSlots;
        r.attribs = (int)attribsUsed.size();
        r.texIndirections = texUnitsSampled.empty() ? 0 : 1;
        if (r.instructions > kArbMaxInstructions || r.temps > kArbMaxTemps || r.params > kArbMaxParams || r.attribs > kArbMaxAttribs)
            fail("program exceeds the implementation's limits", 0);
        r.ok = true;
    }
};

}  // namespace

ArbTranslation translateArbProgram(unsigned target, const std::string& source) {
    Translator t(target, source);
    try {
        t.run();
    } catch (const Error& e) {
        ArbTranslation r;
        r.error = e.msg;
        r.errorPos = e.pos;
        // the line, for the message
        int line = 1;
        for (int i = 0; i < e.pos && i < (int)source.size(); i++)
            if (source[i] == '\n') line++;
        r.error = "line " + std::to_string(line) + ": " + e.msg;
        return r;
    }
    return t.r;
}

}  // namespace mgl
