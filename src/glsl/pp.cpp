#include "glsl/pp.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <unordered_map>

namespace glsl {

const std::vector<std::string>& supportedExtensions() {
    static const std::vector<std::string> exts = {
        "GL_ARB_arrays_of_arrays",
        "GL_ARB_compute_shader",
        "GL_ARB_conservative_depth",
        "GL_ARB_derivative_control",
        "GL_ARB_draw_instanced",
        "GL_ARB_enhanced_layouts",
        "GL_ARB_explicit_attrib_location",
        "GL_ARB_explicit_uniform_location",
        "GL_ARB_fragment_coord_conventions",
        "GL_ARB_fragment_layer_viewport",
        "GL_ARB_gpu_shader5",
        "GL_ARB_gpu_shader_fp64",
        "GL_ARB_shader_draw_parameters",
        "GL_ARB_shader_atomic_counters",
        "GL_ARB_shader_atomic_counter_ops",
        "GL_ARB_shader_bit_encoding",
        "GL_ARB_shader_image_load_store",
        "GL_ARB_shader_image_size",
        "GL_ARB_shader_storage_buffer_object",
        "GL_ARB_shader_texture_lod",
        "GL_ARB_shader_texture_image_samples",
        "GL_ARB_shading_language_420pack",
        "GL_ARB_shading_language_packing",
        "GL_ARB_separate_shader_objects",
        "GL_ARB_texture_cube_map_array",
        "GL_ARB_texture_gather",
        "GL_ARB_texture_multisample",
        "GL_ARB_texture_query_levels",
        "GL_ARB_texture_query_lod",
        "GL_ARB_texture_rectangle",
        "GL_ARB_uniform_buffer_object",
        "GL_ARB_sample_shading",
        "GL_ARB_shader_group_vote",
        "GL_ARB_cull_distance",
        "GL_ARB_tessellation_shader",
        "GL_ARB_geometry_shader4",
        "GL_ARB_viewport_array",
        "GL_ARB_shader_viewport_layer_array",
        "GL_ARB_texture_buffer_object",
        "GL_ARB_vertex_attrib_64bit",
        "GL_ARB_shader_subroutine",
        "GL_ARB_shader_precision",
        "GL_EXT_shader_integer_mix",
        "GL_EXT_gpu_shader4",
        "GL_EXT_texture_array",
        "GL_EXT_geometry_shader4",
        "GL_AMD_vertex_shader_layer",
        "GL_AMD_vertex_shader_viewport_index",
    };
    return exts;
}

namespace {

struct Line {
    std::string text;
    int file, line;
};

struct Macro {
    int id;
    bool function = false;
    bool variadic = false;
    std::vector<std::string> params;
    std::vector<PPToken> body;
    bool builtin = false;  // __LINE__ etc
};

bool isIdStart(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; }
bool isIdChar(char c) { return isIdStart(c) || (c >= '0' && c <= '9'); }
bool isDigit(char c) { return c >= '0' && c <= '9'; }

const char* kPuncts3[] = {"<<=", ">>=", "...", nullptr};
const char* kPuncts2[] = {"++", "--", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||", "^^", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "##", nullptr};

void lexLine(const std::string& s, int file, int line, std::vector<PPToken>& out) {
    size_t i = 0, n = s.size();
    bool space = true;
    while (i < n) {
        char c = s[i];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v') { space = true; i++; continue; }
        PPToken t;
        t.file = file;
        t.line = line;
        t.space = space;
        space = false;
        if (isIdStart(c)) {
            size_t j = i;
            while (j < n && isIdChar(s[j])) j++;
            t.kind = PK::Ident;
            t.text = s.substr(i, j - i);
            i = j;
        } else if (isDigit(c) || (c == '.' && i + 1 < n && isDigit(s[i + 1]))) {
            size_t j = i;
            while (j < n) {
                char d = s[j];
                if (isIdChar(d) || d == '.') {
                    j++;
                } else if ((d == '+' || d == '-') && j > i && (s[j - 1] == 'e' || s[j - 1] == 'E') &&
                           !(s[i] == '0' && i + 1 < n && (s[i + 1] == 'x' || s[i + 1] == 'X'))) {
                    j++;
                } else {
                    break;
                }
            }
            t.kind = PK::Number;
            t.text = s.substr(i, j - i);
            i = j;
        } else {
            t.kind = PK::Punct;
            bool matched = false;
            for (const char** p = kPuncts3; *p && !matched; p++)
                if (s.compare(i, 3, *p) == 0) { t.text = *p; i += 3; matched = true; }
            for (const char** p = kPuncts2; *p && !matched; p++)
                if (s.compare(i, 2, *p) == 0) { t.text = *p; i += 2; matched = true; }
            if (!matched) { t.text = std::string(1, c); i++; }
        }
        out.push_back(std::move(t));
    }
}

class Preprocessor {
public:
    Preprocessor(Stage stage, const CompileOptions& opts) : stage_(stage), opts_(opts) {}

    PPResult run(const std::vector<std::string>& sources);

private:
    Stage stage_;
    const CompileOptions& opts_;
    PPResult res_;
    std::unordered_map<std::string, Macro> macros_;
    std::vector<std::string> macroNames_;  // id -> name
    int curFile_ = 0, curLine_ = 0;
    int lineOffset_ = 0;  // #line adjustment: reported = physical + offset
    int fileOverride_ = -1;
    bool sawNonDirective_ = false;

    void error(int line, const std::string& msg) {
        res_.ok = false;
        res_.log += "ERROR: " + std::to_string(curFile_) + ":" + std::to_string(line) + ": " + msg + "\n";
    }
    void warn(int line, const std::string& msg) {
        res_.log += "WARNING: " + std::to_string(curFile_) + ":" + std::to_string(line) + ": " + msg + "\n";
    }

    int macroId(const std::string& name) {
        auto it = macros_.find(name);
        if (it != macros_.end()) return it->second.id;
        return -1;
    }
    void define(const std::string& name, const std::string& value) {
        Macro m;
        m.id = (int)macroNames_.size();
        macroNames_.push_back(name);
        lexLine(value, 0, 0, m.body);
        macros_[name] = std::move(m);
    }

    void splitLines(const std::vector<std::string>& sources, std::vector<Line>& lines);
    void expand(std::vector<PPToken> input, std::vector<PPToken>& out, bool inIf);
    bool collectArgs(std::vector<PPToken>& stack, std::vector<std::vector<PPToken>>& args, PPToken& rparen);
    std::vector<PPToken> substitute(const Macro& m, const std::vector<std::vector<PPToken>>& args,
                                    const std::vector<int>& hs, int line);
    long long evalIf(std::vector<PPToken> toks, int line);
    void directive(const Line& ln, std::vector<PPToken>& toks, std::vector<bool>& condStack,
                   std::vector<bool>& condTaken, std::vector<bool>& condElse, bool& active);
};

void Preprocessor::splitLines(const std::vector<std::string>& sources, std::vector<Line>& lines) {
    // Remove comments and line continuations; keep line numbering.
    for (size_t f = 0; f < sources.size(); f++) {
        const std::string& s = sources[f];
        std::string cur;
        int line = 1, startLine = 1;
        size_t i = 0, n = s.size();
        bool inBlock = false;
        while (i < n) {
            char c = s[i];
            if (inBlock) {
                if (c == '*' && i + 1 < n && s[i + 1] == '/') { inBlock = false; i += 2; cur += ' '; continue; }
                if (c == '\n') {
                    // a block comment spanning lines: end current logical line here
                    lines.push_back({cur, (int)f, startLine});
                    cur.clear();
                    line++;
                    startLine = line;
                }
                i++;
                continue;
            }
            if (c == '\\' && i + 1 < n && (s[i + 1] == '\n' || (s[i + 1] == '\r' && i + 2 < n && s[i + 2] == '\n'))) {
                i += (s[i + 1] == '\r') ? 3 : 2;
                line++;
                continue;
            }
            if (c == '/' && i + 1 < n && s[i + 1] == '/') {
                while (i < n && s[i] != '\n') {
                    // line continuation inside // comment continues the comment
                    if (s[i] == '\\' && i + 1 < n && s[i + 1] == '\n') { i += 2; line++; continue; }
                    i++;
                }
                continue;
            }
            if (c == '/' && i + 1 < n && s[i + 1] == '*') { inBlock = true; i += 2; continue; }
            if (c == '\n') {
                lines.push_back({cur, (int)f, startLine});
                cur.clear();
                line++;
                startLine = line;
                i++;
                continue;
            }
            cur += c;
            i++;
        }
        lines.push_back({cur, (int)f, startLine});
    }
}

bool Preprocessor::collectArgs(std::vector<PPToken>& stack, std::vector<std::vector<PPToken>>& args, PPToken& rparen) {
    // stack.back() is '('
    stack.pop_back();
    int depth = 0;
    args.clear();
    args.emplace_back();
    while (!stack.empty()) {
        PPToken t = std::move(stack.back());
        stack.pop_back();
        if (t.kind == PK::Punct && t.text == "(") depth++;
        if (t.kind == PK::Punct && t.text == ")") {
            if (depth == 0) { rparen = t; return true; }
            depth--;
        }
        if (t.kind == PK::Punct && t.text == "," && depth == 0) { args.emplace_back(); continue; }
        args.back().push_back(std::move(t));
    }
    return false;
}

static void hsAdd(std::vector<int>& hs, int id) {
    auto it = std::lower_bound(hs.begin(), hs.end(), id);
    if (it == hs.end() || *it != id) hs.insert(it, id);
}
static bool hsHas(const std::vector<int>& hs, int id) { return std::binary_search(hs.begin(), hs.end(), id); }

std::vector<PPToken> Preprocessor::substitute(const Macro& m, const std::vector<std::vector<PPToken>>& args,
                                              const std::vector<int>& hs, int line) {
    std::vector<PPToken> out;
    auto paramIndex = [&](const PPToken& t) -> int {
        if (t.kind != PK::Ident) return -1;
        for (size_t i = 0; i < m.params.size(); i++)
            if (m.params[i] == t.text) return (int)i;
        if (m.variadic && t.text == "__VA_ARGS__") return (int)m.params.size();
        return -1;
    };
    auto argTokens = [&](int idx) -> std::vector<PPToken> {
        if (m.variadic && idx == (int)m.params.size()) {
            std::vector<PPToken> v;
            for (size_t a = m.params.size(); a < args.size(); a++) {
                if (a > m.params.size()) { PPToken c; c.kind = PK::Punct; c.text = ","; v.push_back(c); }
                v.insert(v.end(), args[a].begin(), args[a].end());
            }
            return v;
        }
        if (idx < (int)args.size()) return args[idx];
        return {};
    };
    const auto& body = m.body;
    for (size_t i = 0; i < body.size(); i++) {
        const PPToken& t = body[i];
        // stringification (not GLSL, tolerated)
        if (t.kind == PK::Punct && t.text == "#" && m.function && i + 1 < body.size() && paramIndex(body[i + 1]) >= 0) {
            auto a = argTokens(paramIndex(body[i + 1]));
            std::string s = "\"";
            for (size_t k = 0; k < a.size(); k++) { if (k && a[k].space) s += ' '; s += a[k].text; }
            s += "\"";
            PPToken st;
            st.kind = PK::Punct;
            st.text = s;
            st.line = line;
            out.push_back(st);
            i++;
            continue;
        }
        bool nextIsPaste = i + 1 < body.size() && body[i + 1].kind == PK::Punct && body[i + 1].text == "##";
        bool prevIsPaste = i > 0 && body[i - 1].kind == PK::Punct && body[i - 1].text == "##";
        if (t.kind == PK::Punct && t.text == "##") {
            // paste: merge last output token with next token (first of arg)
            if (i + 1 < body.size() && !out.empty()) {
                const PPToken& nt = body[i + 1];
                std::vector<PPToken> rhs;
                int pi = paramIndex(nt);
                if (pi >= 0) rhs = argTokens(pi); else rhs.push_back(nt);
                if (!rhs.empty()) {
                    std::string merged = out.back().text + rhs[0].text;
                    std::vector<PPToken> relexed;
                    lexLine(merged, out.back().file, line, relexed);
                    out.pop_back();
                    for (auto& r : relexed) { r.hide = hs; r.line = line; out.push_back(r); }
                    for (size_t k = 1; k < rhs.size(); k++) { PPToken r = rhs[k]; r.line = line; out.push_back(r); }
                }
                i++;
            }
            continue;
        }
        int pi = paramIndex(t);
        if (pi >= 0) {
            auto a = argTokens(pi);
            if (nextIsPaste || prevIsPaste) {
                for (auto& x : a) { PPToken y = x; y.line = line; out.push_back(y); }
            } else {
                std::vector<PPToken> ex;
                expand(a, ex, false);
                bool first = true;
                for (auto& x : ex) { PPToken y = x; y.line = line; if (first) { y.space = t.space; first = false; } out.push_back(y); }
            }
            continue;
        }
        PPToken y = t;
        y.line = line;
        out.push_back(y);
    }
    for (auto& t : out) for (int h : hs) hsAdd(t.hide, h);
    return out;
}

void Preprocessor::expand(std::vector<PPToken> input, std::vector<PPToken>& out, bool inIf) {
    std::vector<PPToken> stack(input.rbegin(), input.rend());
    while (!stack.empty()) {
        PPToken t = std::move(stack.back());
        stack.pop_back();
        if (t.kind != PK::Ident) { out.push_back(std::move(t)); continue; }
        if (inIf && t.text == "defined") {
            // handled before expansion; but if produced by expansion, evaluate here
            out.push_back(std::move(t));
            continue;
        }
        if (t.text == "__LINE__") {
            PPToken n = t;
            n.kind = PK::Number;
            n.text = std::to_string(t.line + lineOffset_);
            out.push_back(n);
            continue;
        }
        if (t.text == "__FILE__") {
            PPToken n = t;
            n.kind = PK::Number;
            n.text = std::to_string(fileOverride_ >= 0 ? fileOverride_ : t.file);
            out.push_back(n);
            continue;
        }
        auto it = macros_.find(t.text);
        if (it == macros_.end() || hsHas(t.hide, it->second.id)) { out.push_back(std::move(t)); continue; }
        const Macro& m = it->second;
        if (!m.function) {
            std::vector<int> hs = t.hide;
            hsAdd(hs, m.id);
            auto rep = substitute(m, {}, hs, t.line);
            if (!rep.empty()) rep[0].space = t.space;
            for (auto r = rep.rbegin(); r != rep.rend(); ++r) stack.push_back(*r);
            continue;
        }
        if (stack.empty() || stack.back().kind != PK::Punct || stack.back().text != "(") {
            out.push_back(std::move(t));
            continue;
        }
        std::vector<std::vector<PPToken>> args;
        PPToken rparen;
        if (!collectArgs(stack, args, rparen)) {
            error(t.line, "unterminated macro invocation of '" + t.text + "'");
            return;
        }
        if (args.size() == 1 && args[0].empty() && m.params.empty()) args.clear();
        if (!m.variadic && args.size() != m.params.size()) {
            if (!(m.params.size() == 1 && args.empty())) {
                error(t.line, "macro '" + t.text + "' expects " + std::to_string(m.params.size()) + " arguments, got " +
                                  std::to_string(args.size()));
                return;
            }
        }
        std::vector<int> hs;
        std::set_intersection(t.hide.begin(), t.hide.end(), rparen.hide.begin(), rparen.hide.end(), std::back_inserter(hs));
        hsAdd(hs, m.id);
        auto rep = substitute(m, args, hs, t.line);
        if (!rep.empty()) rep[0].space = t.space;
        for (auto r = rep.rbegin(); r != rep.rend(); ++r) stack.push_back(*r);
    }
}

// --- #if expression evaluation -------------------------------------------------
namespace {
struct IfEval {
    const std::vector<PPToken>& t;
    size_t p = 0;
    bool err = false;
    explicit IfEval(const std::vector<PPToken>& toks) : t(toks) {}
    bool is(const char* s) const { return p < t.size() && t[p].kind == PK::Punct && t[p].text == s; }
    long long primary() {
        if (p >= t.size()) { err = true; return 0; }
        const PPToken& k = t[p];
        if (k.kind == PK::Punct) {
            if (k.text == "(") { p++; long long v = expr(0); if (!is(")")) err = true; else p++; return v; }
            if (k.text == "-") { p++; return -primary(); }
            if (k.text == "+") { p++; return primary(); }
            if (k.text == "!") { p++; return !primary(); }
            if (k.text == "~") { p++; return ~primary(); }
            err = true;
            p++;
            return 0;
        }
        p++;
        if (k.kind == PK::Number) {
            std::string s = k.text;
            while (!s.empty() && (s.back() == 'u' || s.back() == 'U' || s.back() == 'l' || s.back() == 'L')) s.pop_back();
            try {
                if (s.find('.') != std::string::npos || ((s.find('e') != std::string::npos || s.find('E') != std::string::npos) && s.find('x') == std::string::npos))
                    return (long long)std::stod(s);
                return std::stoll(s, nullptr, 0);
            } catch (...) { return 0; }
        }
        // identifier: true/false or undefined -> 0
        if (k.text == "true") return 1;
        return 0;
    }
    static int prec(const std::string& op) {
        static const std::map<std::string, int> P = {
            {"*", 10}, {"/", 10}, {"%", 10}, {"+", 9}, {"-", 9}, {"<<", 8}, {">>", 8}, {"<", 7}, {">", 7}, {"<=", 7}, {">=", 7},
            {"==", 6}, {"!=", 6}, {"&", 5}, {"^", 4}, {"|", 3}, {"&&", 2}, {"||", 1}};
        auto it = P.find(op);
        return it == P.end() ? -1 : it->second;
    }
    long long expr(int minPrec) {
        long long lhs = primary();
        for (;;) {
            if (p >= t.size() || t[p].kind != PK::Punct) break;
            if (t[p].text == "?" && minPrec == 0) {
                p++;
                long long a = expr(0);
                if (!is(":")) { err = true; return 0; }
                p++;
                long long b = expr(0);
                lhs = lhs ? a : b;
                continue;
            }
            int pr = prec(t[p].text);
            if (pr < 0 || pr < minPrec || (minPrec > 0 && pr < minPrec)) break;
            if (pr <= minPrec - 1) break;
            std::string op = t[p].text;
            p++;
            long long rhs = expr(pr + 1);
            if (op == "*") lhs *= rhs;
            else if (op == "/") lhs = rhs ? lhs / rhs : 0;
            else if (op == "%") lhs = rhs ? lhs % rhs : 0;
            else if (op == "+") lhs += rhs;
            else if (op == "-") lhs -= rhs;
            else if (op == "<<") lhs <<= rhs;
            else if (op == ">>") lhs >>= rhs;
            else if (op == "<") lhs = lhs < rhs;
            else if (op == ">") lhs = lhs > rhs;
            else if (op == "<=") lhs = lhs <= rhs;
            else if (op == ">=") lhs = lhs >= rhs;
            else if (op == "==") lhs = lhs == rhs;
            else if (op == "!=") lhs = lhs != rhs;
            else if (op == "&") lhs &= rhs;
            else if (op == "^") lhs ^= rhs;
            else if (op == "|") lhs |= rhs;
            else if (op == "&&") lhs = lhs && rhs;
            else if (op == "||") lhs = lhs || rhs;
        }
        return lhs;
    }
};
}  // namespace

long long Preprocessor::evalIf(std::vector<PPToken> toks, int line) {
    // resolve defined first
    std::vector<PPToken> pre;
    for (size_t i = 0; i < toks.size(); i++) {
        if (toks[i].kind == PK::Ident && toks[i].text == "defined") {
            size_t j = i + 1;
            bool paren = false;
            if (j < toks.size() && toks[j].kind == PK::Punct && toks[j].text == "(") { paren = true; j++; }
            if (j >= toks.size() || toks[j].kind != PK::Ident) { error(line, "invalid 'defined' expression"); return 0; }
            bool d = macros_.count(toks[j].text) || toks[j].text == "__LINE__" || toks[j].text == "__FILE__";
            j++;
            if (paren) {
                if (j >= toks.size() || toks[j].text != ")") { error(line, "missing ')' after defined"); return 0; }
                j++;
            }
            PPToken n = toks[i];
            n.kind = PK::Number;
            n.text = d ? "1" : "0";
            pre.push_back(n);
            i = j - 1;
            continue;
        }
        pre.push_back(toks[i]);
    }
    std::vector<PPToken> ex;
    expand(pre, ex, true);
    if (ex.empty()) { error(line, "empty #if expression"); return 0; }
    IfEval e(ex);
    long long v = e.expr(0);
    if (e.err || e.p != ex.size()) { error(line, "invalid #if expression"); return 0; }
    return v;
}

void Preprocessor::directive(const Line& ln, std::vector<PPToken>& toks, std::vector<bool>& condStack,
                             std::vector<bool>& condTaken, std::vector<bool>& condElse, bool& active) {
    // toks[0] is '#'
    if (toks.size() == 1) return;  // null directive
    const std::string d = toks[1].text;
    int line = ln.line;
    std::vector<PPToken> rest(toks.begin() + 2, toks.end());
    auto parentActive = [&]() {
        for (size_t i = 0; i + 1 < condStack.size(); i++) if (!condStack[i]) return false;
        return true;
    };
    if (d == "if" || d == "ifdef" || d == "ifndef") {
        bool v = false;
        if (active) {
            if (d == "if") v = evalIf(rest, line) != 0;
            else {
                if (rest.empty() || rest[0].kind != PK::Ident) { error(line, "#" + d + " requires an identifier"); }
                else {
                    bool def = macros_.count(rest[0].text) || rest[0].text == "__LINE__" || rest[0].text == "__FILE__";
                    v = (d == "ifdef") ? def : !def;
                }
            }
        }
        condStack.push_back(active && v);
        condTaken.push_back(active && v);
        condElse.push_back(false);
        active = active && v;
        return;
    }
    if (d == "elif") {
        if (condStack.empty()) { error(line, "#elif without #if"); return; }
        if (condElse.back()) { error(line, "#elif after #else"); return; }
        bool pa = parentActive();
        if (!pa || condTaken.back()) { condStack.back() = false; active = false; return; }
        bool v = evalIf(rest, line) != 0;
        condStack.back() = v;
        if (v) condTaken.back() = true;
        active = v;
        return;
    }
    if (d == "else") {
        if (condStack.empty()) { error(line, "#else without #if"); return; }
        if (condElse.back()) { error(line, "#else after #else"); return; }
        condElse.back() = true;
        bool pa = parentActive();
        bool v = pa && !condTaken.back();
        condStack.back() = v;
        if (v) condTaken.back() = true;
        active = v;
        return;
    }
    if (d == "endif") {
        if (condStack.empty()) { error(line, "#endif without #if"); return; }
        condStack.pop_back();
        condTaken.pop_back();
        condElse.pop_back();
        active = true;
        for (bool b : condStack) active = active && b;
        return;
    }
    if (!active) return;

    if (d == "define") {
        if (rest.empty() || rest[0].kind != PK::Ident) { error(line, "#define requires a macro name"); return; }
        const std::string name = rest[0].text;
        if (name.compare(0, 3, "GL_") == 0 && macros_.count(name)) {
            warn(line, "redefinition of reserved macro '" + name + "'");
        }
        Macro m;
        size_t i = 1;
        if (rest.size() > 1 && rest[1].kind == PK::Punct && rest[1].text == "(" && !rest[1].space) {
            m.function = true;
            i = 2;
            while (i < rest.size() && !(rest[i].kind == PK::Punct && rest[i].text == ")")) {
                if (rest[i].kind == PK::Ident) m.params.push_back(rest[i].text);
                else if (rest[i].text == "...") m.variadic = true;
                else if (rest[i].text != ",") { error(line, "invalid macro parameter list"); return; }
                i++;
            }
            if (i >= rest.size()) { error(line, "unterminated macro parameter list"); return; }
            i++;
        }
        m.body.assign(rest.begin() + i, rest.end());
        auto it = macros_.find(name);
        if (it != macros_.end()) m.id = it->second.id;
        else { m.id = (int)macroNames_.size(); macroNames_.push_back(name); }
        macros_[name] = std::move(m);
        return;
    }
    if (d == "undef") {
        if (rest.empty() || rest[0].kind != PK::Ident) { error(line, "#undef requires a macro name"); return; }
        macros_.erase(rest[0].text);
        return;
    }
    if (d == "version") {
        if (sawNonDirective_ || res_.versionSpecified) {
            if (res_.versionSpecified) error(line, "#version must occur only once");
            else error(line, "#version must occur before anything else");
            return;
        }
        if (rest.empty() || rest[0].kind != PK::Number) { error(line, "#version requires a number"); return; }
        res_.version = std::atoi(rest[0].text.c_str());
        res_.versionSpecified = true;
        if (rest.size() > 1) {
            const std::string& prof = rest[1].text;
            if (prof == "es") res_.es = true;
            else if (prof == "core") res_.core = true;
            else if (prof == "compatibility") res_.core = false;
            else error(line, "unknown profile '" + prof + "'");
        } else {
            res_.core = res_.version >= 150 && res_.version != 300 && res_.version != 310 && res_.version != 320;
            if (res_.version == 300 || res_.version == 310 || res_.version == 320) res_.es = false;  // desktop GLSL has no 300
        }
        return;
    }
    if (d == "extension") {
        if (rest.size() < 3 || rest[0].kind != PK::Ident || rest[1].text != ":" || rest[2].kind != PK::Ident) {
            error(line, "malformed #extension directive");
            return;
        }
        const std::string& name = rest[0].text;
        const std::string& beh = rest[2].text;
        if (beh != "require" && beh != "enable" && beh != "warn" && beh != "disable") {
            error(line, "unknown extension behavior '" + beh + "'");
            return;
        }
        if (name != "all") {
            auto& exts = supportedExtensions();
            bool known = std::find(exts.begin(), exts.end(), name) != exts.end();
            if (!known && beh == "require") { error(line, "extension '" + name + "' is not supported"); return; }
            if (!known) warn(line, "extension '" + name + "' is not supported");
        }
        res_.extensions[name] = beh;
        return;
    }
    if (d == "line") {
        std::vector<PPToken> ex;
        expand(rest, ex, false);
        if (ex.empty() || ex[0].kind != PK::Number) { error(line, "#line requires a line number"); return; }
        int n = std::atoi(ex[0].text.c_str());
        // the line after the directive gets number n
        lineOffset_ = n - (line + 1);
        if (ex.size() > 1 && ex[1].kind == PK::Number) fileOverride_ = std::atoi(ex[1].text.c_str());
        return;
    }
    if (d == "pragma") {
        std::string s;
        for (auto& t : rest) { if (!s.empty() && t.space) s += ' '; s += t.text; }
        res_.pragmas.push_back(s);
        return;
    }
    if (d == "error") {
        std::string s;
        for (auto& t : rest) { if (!s.empty()) s += ' '; s += t.text; }
        error(line, "#error " + s);
        return;
    }
    if (d == "include") {
        error(line, "#include is not supported (GL_ARB_shading_language_include not exposed)");
        return;
    }
    error(line, "unknown preprocessor directive '#" + d + "'");
}

PPResult Preprocessor::run(const std::vector<std::string>& sources) {
    std::vector<Line> lines;
    splitLines(sources, lines);

    // Peek #version to set up predefined macros before anything else.
    // (#version must be the first non-comment, non-whitespace line.)
    std::vector<bool> condStack, condTaken, condElse;
    bool active = true;
    std::vector<PPToken> pending;
    bool predefined = false;

    auto setupPredefined = [&]() {
        if (predefined) return;
        predefined = true;
        define("__VERSION__", std::to_string(res_.version));
        if (res_.es) define("GL_ES", "1"), define("GL_FRAGMENT_PRECISION_HIGH", "1");
        else if (res_.core) define("GL_core_profile", "1");
        else define("GL_compatibility_profile", "1");
        if (!res_.es && res_.version >= 150 && res_.core) {
            // still accept compat if asked: nothing
        }
        for (auto& e : supportedExtensions()) define(e, "1");
        (void)stage_;
    };

    auto flush = [&]() {
        if (pending.empty()) return;
        std::vector<PPToken> ex;
        expand(std::move(pending), ex, false);
        pending.clear();
        for (auto& t : ex) {
            t.line += lineOffset_;
            t.hide.clear();
            res_.tokens.push_back(std::move(t));
        }
    };

    for (const Line& ln : lines) {
        curFile_ = ln.file;
        curLine_ = ln.line;
        std::vector<PPToken> toks;
        lexLine(ln.text, ln.file, ln.line, toks);
        if (toks.empty()) continue;
        if (toks[0].kind == PK::Punct && toks[0].text == "#") {
            if (toks.size() > 1 && toks[1].text == "version") {
                if (active) {
                    std::vector<bool> d1, d2, d3;
                    directive(ln, toks, condStack, condTaken, condElse, active);
                }
                continue;
            }
            setupPredefined();
            flush();
            directive(ln, toks, condStack, condTaken, condElse, active);
            continue;
        }
        if (!active) continue;
        setupPredefined();
        sawNonDirective_ = true;
        for (auto& t : toks) pending.push_back(std::move(t));
    }
    setupPredefined();
    flush();
    if (!condStack.empty()) error(curLine_, "unterminated #if");
    PPToken end;
    end.kind = PK::End;
    end.line = curLine_;
    res_.tokens.push_back(end);
    (void)opts_;
    return std::move(res_);
}

}  // namespace

PPResult preprocess(const std::vector<std::string>& sources, Stage stage, const CompileOptions& opts) {
    Preprocessor pp(stage, opts);
    return pp.run(sources);
}

}  // namespace glsl
