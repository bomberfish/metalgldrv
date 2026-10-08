// Rebinding of imported symbol pointers in loaded Mach-O images (fishhook-style).
// Pure C++ over <mach-o/*.h>; also compiled into mgltest for its unit tests.
#pragma once
#include <cstddef>
#include <cstdint>

struct mach_header;

namespace mgl::macho {

// Called for every imported symbol pointer slot of an image. `symbol` is the C name
// (no leading underscore); `library` is the install name of the dylib the symbol
// binds from, or nullptr (flat namespace / dynamic lookup / main executable). Return
// the new pointer value, or nullptr to leave the slot alone.
using RebindFn = void* (*)(const char* symbol, const char* library, void* current, void* user);

// Rewrites imported-symbol pointers (the __got / __la_symbol_ptr / __nl_symbol_ptr
// sections, located through the indirect symbol table, whether the image uses
// chained fixups or the classic dyld info) of one loaded 64-bit image. Read-only
// pages (__DATA_CONST after dyld has applied fixups) are made writable for the
// write and restored. Pointer-authenticated slots (__auth_got) are left alone.
// Returns the number of slots changed. Callers serialize calls per image.
int rebindImage(const mach_header* header, intptr_t slide, RebindFn fn, void* user);

// Whether a dlopen path names Apple's OpenGL framework (or its libGL / libGLU).
bool isOpenGLFrameworkPath(const char* path);

// Whether the path is part of the OS (left alone in replacement mode).
bool isSystemImagePath(const char* path);

}  // namespace mgl::macho
