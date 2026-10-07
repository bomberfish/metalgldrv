// Internal AST, type system and symbol definitions for the GLSL compiler.
#pragma once

#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "glsl/glsl.h"

namespace glsl {

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
enum class BT : uint8_t { Void, Bool, Int, UInt, Float, Double, Struct, Sampler, Image, AtomicUint, Subroutine, Error };
enum class Dim : uint8_t { None, D1, D2, D3, Cube, Rect, Buffer };

struct StructDef;
struct Function;

struct Type {
    BT base = BT::Void;
    uint8_t vec = 1;   // components; for matrices: rows
    uint8_t cols = 0;  // matrix columns, 0 when not a matrix
    // samplers / images
    Dim dim = Dim::None;
    bool arrayed = false, shadow = false, ms = false;
    BT sampled = BT::Float;  // component type of sampler/image results
    StructDef* sdef = nullptr;
    Function* subr = nullptr;  // BT::Subroutine: the subroutine type (prototype)
    std::vector<int> dims;   // array dimensions, outermost first; 0 = unsized

    static Type scalar(BT b) { Type t; t.base = b; return t; }
    static Type vector(BT b, int n) { Type t; t.base = b; t.vec = (uint8_t)n; return t; }
    static Type matrix(BT b, int c, int r) { Type t; t.base = b; t.cols = (uint8_t)c; t.vec = (uint8_t)r; return t; }
    static Type voidT() { return Type(); }
    static Type error() { Type t; t.base = BT::Error; return t; }

    bool isArray() const { return !dims.empty(); }
    bool isMatrix() const { return cols != 0 && dims.empty(); }
    bool isVector() const { return cols == 0 && vec > 1 && dims.empty() && isNumericBase(); }
    bool isScalar() const { return cols == 0 && vec == 1 && dims.empty() && isNumericBase(); }
    bool isNumericBase() const { return base == BT::Bool || base == BT::Int || base == BT::UInt || base == BT::Float || base == BT::Double; }
    bool isNumeric() const { return dims.empty() && isNumericBase(); }  // scalar, vector or matrix
    bool isOpaque() const { return base == BT::Sampler || base == BT::Image || base == BT::AtomicUint; }
    bool containsOpaque() const;
    bool isStruct() const { return base == BT::Struct && dims.empty(); }
    bool isError() const { return base == BT::Error; }
    bool isVoid() const { return base == BT::Void && dims.empty(); }
    bool isFloatLike() const { return base == BT::Float || base == BT::Double; }
    bool isIntLike() const { return base == BT::Int || base == BT::UInt; }
    int components() const { return cols ? cols * vec : vec; }  // of a non-array numeric type
    Type elem() const { Type t = *this; if (!t.dims.empty()) t.dims.erase(t.dims.begin()); return t; }  // strip outer array
    Type arrayOf(int n) const { Type t = *this; t.dims.insert(t.dims.begin(), n); return t; }
    Type stripArrays() const { Type t = *this; t.dims.clear(); return t; }
    Type column() const { Type t = *this; t.cols = 0; return t; }  // matrix column type
    Type scalarType() const { Type t; t.base = base; return t; }
    Type withBase(BT b) const { Type t = *this; t.base = b; return t; }
    int arraySize() const { return dims.empty() ? 1 : dims[0]; }
    int totalArrayElements() const { int n = 1; for (int d : dims) n *= d ? d : 1; return n; }
    bool operator==(const Type& o) const;
    bool operator!=(const Type& o) const { return !(*this == o); }
    std::string str() const;  // GLSL spelling
};

enum class Packing : uint8_t { None, Shared, Packed, Std140, Std430 };

struct Layout {
    int location = -1, component = -1, binding = -1, offset = -1, align = -1, index = -1, set = -1;
    int localSize[3] = {-1, -1, -1};
    int maxVertices = -1, invocations = -1, vertices = -1, stream = -1;
    int xfbBuffer = -1, xfbOffset = -1, xfbStride = -1;
    Packing packing = Packing::None;
    int matrix = 0;           // 0 unspecified, 1 column_major, 2 row_major
    static constexpr uint32_t kNoPrimitive = 0xFFFFFFFFu;
    uint32_t primitive = kNoPrimitive;  // GS in/out primitive, tess primitive (GL enum; GL_POINTS == 0)
    uint32_t spacing = 0, order = 0;
    bool pointMode = false;
    bool originUpperLeft = false, pixelCenterInteger = false, earlyFragmentTests = false;
    uint32_t depthLayout = 0;  // 0 none, 1 any, 2 greater, 3 less, 4 unchanged
    std::string format;        // image format qualifier
    bool blendSupportAll = false;
};

enum class Storage : uint8_t { Temp, Const, In, Out, InOut, Uniform, Buffer, Shared, Attribute, Varying };
enum class Interp : uint8_t { Smooth, Flat, NoPerspective };

enum : uint8_t { MemCoherent = 1, MemVolatile = 2, MemRestrict = 4, MemReadOnly = 8, MemWriteOnly = 16 };

struct Qualifiers {
    Storage storage = Storage::Temp;
    Interp interp = Interp::Smooth;
    bool interpExplicit = false;
    bool centroid = false, sample = false, patch = false, invariant = false, precise = false;
    uint8_t memory = 0;
    bool subroutine = false;
    std::vector<std::string> subroutineTypes;  // subroutine(type, ...) on function definitions
    bool hasLayout = false;
    Layout layout;
};

struct Variable;

struct Member {
    std::string name;
    Type type;
    Qualifiers q;  // layout(offset/align/row_major/location...), interpolation for block members
    int line = 0;
    Variable* builtinVar = nullptr;  // for redeclared builtin blocks (gl_PerVertex)
};

struct StructDef {
    std::string name;
    std::vector<Member> members;
    bool isBlock = false;
    Qualifiers blockQ;          // for interface blocks
    std::string instanceName;   // interface block instance name ("" if anonymous)
    int id = 0;
    int findMember(const std::string& n) const {
        for (size_t i = 0; i < members.size(); i++) if (members[i].name == n) return (int)i;
        return -1;
    }
};

// ---------------------------------------------------------------------------
// Constant values (flattened components)
// ---------------------------------------------------------------------------
struct Scalar {
    union { double f; int64_t i; uint64_t u; };
    bool b = false;
    Scalar() : i(0) {}
};
struct ConstVal {
    std::vector<Scalar> c;  // flattened in GLSL component order
};

// ---------------------------------------------------------------------------
// Symbols
// ---------------------------------------------------------------------------
enum class BuiltinVar : uint16_t {
    None,
    // vertex
    VertexID, InstanceID, BaseVertex, BaseInstance, DrawID, Position, PointSize, ClipDistance, CullDistance, ClipVertex,
    // compat vertex inputs
    Vertex, Normal, Color, SecondaryColor, MultiTexCoord, FogCoord,
    // compat varyings
    FrontColor, BackColor, FrontSecondaryColor, BackSecondaryColor, TexCoord, FogFragCoord,
    // geometry / tess
    In, Out, PrimitiveIDIn, PrimitiveID, InvocationID, Layer, ViewportIndex, PatchVerticesIn, TessLevelOuter, TessLevelInner,
    TessCoord,
    // fragment
    FragCoord, FrontFacing, PointCoord, FragColor, FragData, FragDepth, SampleID, SamplePosition, SampleMaskIn, SampleMask,
    HelperInvocation, FragColorIn /*gl_Color in FS*/, SecondaryColorIn,
    // compute
    NumWorkGroups, WorkGroupID, LocalInvocationID, GlobalInvocationID, LocalInvocationIndex, WorkGroupSize,
    // uniforms
    DepthRange, NumSamples, CompatUniform,
};

struct Expr;
struct Stmt;

struct Variable {
    std::string name;
    Type type;
    Qualifiers q;
    BuiltinVar builtin = BuiltinVar::None;
    bool global = false;
    bool param = false;
    bool used = false;           // statically used by reachable code
    bool written = false;
    bool redeclared = false;  // builtin redeclared by the shader
    int id = 0;
    Expr* init = nullptr;
    std::shared_ptr<ConstVal> constVal;  // for const variables with constant initializers
    StructDef* block = nullptr;  // interface block this variable instantiates (named or hidden anon instance)
    bool anonBlock = false;      // hidden instance of an anonymous block
    int line = 0;
};

struct Builtin;

struct Function {
    std::string name;
    Type ret;
    std::vector<Variable*> params;
    Stmt* body = nullptr;
    const Builtin* builtin = nullptr;
    std::string mangled;  // name + param type signature
    std::vector<std::string> subroutineTypes;  // subroutine function: compatible subroutine types
    int subroutineIndex = -1;                  // layout(index = N)
    bool reachable = false;
    int line = 0;
};

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------
enum class EK : uint8_t { Const, Var, Binary, Unary, Assign, Ternary, Call, Ctor, Index, Swizzle, Field, Comma, Length, Builtin, SubroutineCall };

enum class Op : uint8_t {
    None,
    Add, Sub, Mul, Div, Mod, Shl, Shr, BitAnd, BitOr, BitXor, LogAnd, LogOr, LogXor,
    Lt, Gt, Le, Ge, Eq, Ne,
    Neg, Plus, Not, BitNot, PreInc, PreDec, PostInc, PostDec,
    Assign, AddAssign, SubAssign, MulAssign, DivAssign, ModAssign, ShlAssign, ShrAssign, AndAssign, OrAssign, XorAssign,
};

struct Expr {
    EK kind;
    Type type;
    int line = 0;
    Op op = Op::None;
    std::vector<Expr*> args;
    Variable* var = nullptr;
    Function* fn = nullptr;       // Call: user function; Builtin: builtin signature (fn->builtin set)
    int field = -1;               // Field: member index
    uint8_t swz[4] = {0, 0, 0, 0};
    uint8_t nswz = 0;
    std::shared_ptr<ConstVal> cv; // set when the expression is a compile-time constant
    Expr* aux = nullptr;          // SubroutineCall: array index of the subroutine uniform (or null)
    bool isLvalue() const;
};

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------
enum class SK : uint8_t { Block, Expr, Decl, If, For, While, DoWhile, Switch, Case, Default, Break, Continue, Return, Discard, Empty };

struct Stmt {
    SK kind;
    int line = 0;
    std::vector<Stmt*> body;       // Block, Switch body
    Expr* expr = nullptr;          // Expr, Return value, If/While/DoWhile/Switch cond, Case label
    Stmt* init = nullptr;          // For
    Expr* cond = nullptr;          // For condition (may be null)
    Stmt* condDecl = nullptr;      // For/While: condition declaring a variable
    Expr* incr = nullptr;          // For
    Stmt* then = nullptr;          // If then / loop body
    Stmt* els = nullptr;
    std::vector<Variable*> vars;   // Decl
    bool newScope = true;          // Block
};

// ---------------------------------------------------------------------------
// Compiled shader object
// ---------------------------------------------------------------------------
struct GlobalDecl {  // in declaration order, used by the emitter
    enum Kind : uint8_t { Var, StructType, FunctionDef, InvariantDecl } kind;
    Variable* var = nullptr;
    StructDef* sdef = nullptr;
    Function* fn = nullptr;
};

struct Shader {
    Stage stage;
    int version = 110;
    bool es = false;
    bool compat = true;
    std::vector<GlobalDecl> globals;
    std::vector<Function*> functions;  // user functions (with and without bodies)
    std::vector<Variable*> globalVars;  // all global variables incl. builtins referenced
    std::vector<StructDef*> structs;
    std::map<std::string, std::string> extensions;
    Layout inLayout, outLayout;          // stage-level layout(...) in/out; declarations
    // Transform feedback layout qualifiers (GLSL 4.60 §4.4.2.1). Output variables / blocks carry their resolved
    // xfb_buffer; block members their resolved xfb_offset (-1 = not captured).
    bool xfbMode = false;                // any xfb_* qualifier used
    int xfbStride[4] = {-1, -1, -1, -1}; // explicit xfb_stride per buffer
    bool hasMain = false;
    // redeclared gl_PerVertex members ("type name" each) for the in / out interface; empty = not redeclared
    std::vector<std::string> perVertexIn, perVertexOut;
    bool spirv = false;                  // decompiled from a SPIR-V module (interfaces match by location)
    // arenas
    std::deque<Expr> exprs;
    std::deque<Stmt> stmts;
    std::deque<Variable> vars;
    std::deque<Function> funcs;
    std::deque<StructDef> sdefs;
};

// Builtin function descriptor
enum class BIOp : uint16_t;  // defined in builtins.h

struct Builtin {
    std::string name;
    BIOp op;
    Type ret;
    std::vector<Type> params;
    std::vector<uint8_t> paramOut;  // 1 = out, 2 = inout
    uint32_t stageMask = 0xff;
    const char* ext = nullptr;  // shader extension that must be enabled to call it
};

// helpers (types.cpp)
uint32_t glTypeEnum(const Type& t);
std::string typeMangle(const Type& t);
bool isIntegerType(const Type& t);

}  // namespace glsl
