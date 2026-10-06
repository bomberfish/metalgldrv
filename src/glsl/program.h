// Linked program representation shared by the linker and the MSL emitter.
#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "glsl/ast.h"

namespace glsl {

// Memory layout of a type inside a buffer (std140 / std430 / default-block rules).
struct TypeLayout {
    uint32_t size = 0, align = 0;
    uint32_t arrayStride = 0;    // arrays: stride between elements
    uint32_t matrixStride = 0;   // matrices: stride between columns (or rows when rowMajor)
    bool rowMajor = false;
    std::shared_ptr<TypeLayout> elem;                  // arrays
    std::vector<uint32_t> offsets;                     // structs
    std::vector<std::shared_ptr<TypeLayout>> members;  // structs
};
using TL = std::shared_ptr<TypeLayout>;

// packing: Std140 (also used for Shared/Packed), Std430 (also the default uniform block)
TL computeLayout(const Type& t, Packing packing, bool rowMajor);
// Layout of an interface block's members
TL computeBlockLayout(const StructDef& sd, Packing packing, bool rowMajor);
uint32_t scalarSize(BT b);

struct StageCode {
    std::vector<std::shared_ptr<const Shader>> shaders;
    const Shader* primary = nullptr;  // shader holding main()
    Function* main = nullptr;
    std::set<const Function*> reachable;              // function definitions reachable from main
    std::map<std::string, Function*> defs;            // mangled -> definition
    std::set<const Variable*> used;                   // global variables referenced by reachable code
    std::set<const Variable*> activeBlocks;           // shared/std140/std430 blocks declared here (active even if unused)
    std::set<std::string> usedNames;                  // names of used globals
    Layout inLayout, outLayout;
    bool writesFragDepth = false, usesDiscard = false;
    std::vector<const Variable*> globals;             // de-duplicated user globals in declaration order
};

struct OpaqueBinding {
    SlotKind kind;   // Texture (sampler uniforms), Image, AtomicCounter
    int slot = -1;   // first texture/image slot
    int samplerSlot = -1;
    int count = 1;   // array elements
    bool isArray = false;  // declared as an array (also when count == 1)
    int uniformIndex = -1;
    int acbIndex = -1;
    uint32_t acOffset = 0;
    Type type;
    uint8_t memory = 0;
};

struct BlockBinding {
    int firstBlock = -1;  // index into Reflection::uniformBlocks / storageBlocks
    int count = 1;
    bool isArray = false;  // declared as an array of blocks (even with one element)
    int slot = -1;        // first argument slot
    int sizeSlot = -1;
    Packing packing = Packing::Std140;
    bool rowMajor = false;
    TL layout;            // layout of the block struct
    bool storage = false;
};

struct LinkedProgram {
    StageCode stages[(int)Stage::Count];
    uint32_t stageMask = 0;
    bool separable = false;
    Reflection refl;
    std::map<std::string, uint32_t> defaultOffsets;        // default-block uniform var name -> byte offset
    std::map<std::string, OpaqueBinding> opaque;           // opaque uniform var name -> binding
    std::map<std::string, BlockBinding> blocks;            // "U:Name" / "B:Name" -> binding
    std::map<std::string, int> attribLocations;            // VS input var name -> location
    std::map<std::string, std::pair<int, int>> fragOutputs;  // FS output var name -> (location, index)
    std::set<std::string> vsOutputNames;                   // flattened varying names produced by the last pre-raster stage
    int defaultSlot = -1;
    std::map<std::pair<int, std::string>, int> subroutineLocation;  // (stage, uniform) -> location
    std::map<std::pair<int, std::string>, int> subroutineIndex;     // (stage, function) -> index
};

// Varying naming shared by producer/consumer stages.
std::string varyingKey(const Variable* v, Stage stage);

}  // namespace glsl
