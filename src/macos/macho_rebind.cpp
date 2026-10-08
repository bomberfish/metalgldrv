// Rebinding of imported symbol pointers in loaded Mach-O images (see macho_rebind.h).
#include "macos/macho_rebind.h"
#include <mach-o/loader.h>
#include <mach-o/nlist.h>
#include <mach/mach.h>
#include <cstring>
#include <vector>

namespace mgl::macho {

namespace {

// Writes one pointer slot, temporarily making its page writable if needed.
bool writePointer(void** slot, void* value) {
    vm_address_t addr = (vm_address_t)slot;
    vm_size_t size = 0;
    vm_region_basic_info_data_64_t info;
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    vm_prot_t prot = VM_PROT_READ | VM_PROT_WRITE;
    if (vm_region_64(mach_task_self(), &addr, &size, VM_REGION_BASIC_INFO_64, (vm_region_info_t)&info, &count, &object) ==
        KERN_SUCCESS)
        prot = info.protection;
    if (prot & VM_PROT_WRITE) {
        __atomic_store_n(slot, value, __ATOMIC_RELEASE);
        return true;
    }
    vm_size_t page = vm_page_size;
    vm_address_t start = (vm_address_t)slot & ~(page - 1);
    vm_size_t len = ((vm_address_t)slot + sizeof(void*) - start + page - 1) & ~(page - 1);
    if (vm_protect(mach_task_self(), start, len, false, VM_PROT_READ | VM_PROT_WRITE | VM_PROT_COPY) != KERN_SUCCESS)
        return false;
    __atomic_store_n(slot, value, __ATOMIC_RELEASE);
    vm_protect(mach_task_self(), start, len, false, prot);
    return true;
}

bool isPointerSection(const section_64& s) {
    uint32_t type = s.flags & SECTION_TYPE;
    if (type != S_LAZY_SYMBOL_POINTERS && type != S_NON_LAZY_SYMBOL_POINTERS) return false;
    return strncmp(s.sectname, "__auth", 6) != 0;  // arm64e: signed pointers
}

}  // namespace

int rebindImage(const mach_header* header, intptr_t slide, RebindFn fn, void* user) {
    if (!header || header->magic != MH_MAGIC_64) return 0;
    auto* h = (const mach_header_64*)header;
    const segment_command_64* linkedit = nullptr;
    const symtab_command* symtab = nullptr;
    const dysymtab_command* dysymtab = nullptr;
    std::vector<const segment_command_64*> segments;
    std::vector<const char*> dylibs;  // library ordinal - 1 -> install name
    const uint8_t* p = (const uint8_t*)(h + 1);
    for (uint32_t i = 0; i < h->ncmds; i++) {
        auto* lc = (const load_command*)p;
        switch (lc->cmd) {
        case LC_SEGMENT_64: {
            auto* seg = (const segment_command_64*)lc;
            if (strcmp(seg->segname, SEG_LINKEDIT) == 0) linkedit = seg;
            else segments.push_back(seg);
            break;
        }
        case LC_SYMTAB: symtab = (const symtab_command*)lc; break;
        case LC_DYSYMTAB: dysymtab = (const dysymtab_command*)lc; break;
        case LC_LOAD_DYLIB:
        case LC_LOAD_WEAK_DYLIB:
        case LC_REEXPORT_DYLIB:
        case LC_LOAD_UPWARD_DYLIB:
        case LC_LAZY_LOAD_DYLIB: {
            auto* dc = (const dylib_command*)lc;
            dylibs.push_back((const char*)dc + dc->dylib.name.offset);
            break;
        }
        default: break;
        }
        p += lc->cmdsize;
    }
    if (!linkedit || !symtab || !dysymtab || dysymtab->nindirectsyms == 0) return 0;
    uintptr_t linkeditBase = (uintptr_t)slide + linkedit->vmaddr - linkedit->fileoff;
    auto* syms = (const nlist_64*)(linkeditBase + symtab->symoff);
    auto* strtab = (const char*)(linkeditBase + symtab->stroff);
    auto* indirect = (const uint32_t*)(linkeditBase + dysymtab->indirectsymoff);
    int changed = 0;
    for (const segment_command_64* seg : segments) {
        auto* sects = (const section_64*)(seg + 1);
        for (uint32_t j = 0; j < seg->nsects; j++) {
            const section_64& s = sects[j];
            if (!isPointerSection(s)) continue;
            void** slots = (void**)((uintptr_t)slide + s.addr);
            uint64_t n = s.size / sizeof(void*);
            for (uint64_t k = 0; k < n; k++) {
                uint64_t ii = (uint64_t)s.reserved1 + k;
                if (ii >= dysymtab->nindirectsyms) break;
                uint32_t idx = indirect[ii];
                if (idx & (INDIRECT_SYMBOL_ABS | INDIRECT_SYMBOL_LOCAL)) continue;
                if (idx >= symtab->nsyms) continue;
                const nlist_64& sym = syms[idx];
                const char* name = strtab + sym.n_un.n_strx;
                if (name[0] != '_') continue;
                int ordinal = GET_LIBRARY_ORDINAL(sym.n_desc);
                const char* lib = ordinal >= 1 && ordinal <= (int)dylibs.size() ? dylibs[ordinal - 1] : nullptr;
                void* cur = slots[k];
                void* repl = fn(name + 1, lib, cur, user);
                if (!repl || repl == cur) continue;
                if (writePointer(&slots[k], repl)) changed++;
            }
        }
    }
    return changed;
}

bool isOpenGLFrameworkPath(const char* path) {
    if (!path) return false;
    static const char kPrefix[] = "/System/Library/Frameworks/OpenGL.framework/";
    if (strncmp(path, kPrefix, sizeof kPrefix - 1) != 0) return false;
    const char* leaf = strrchr(path, '/') + 1;
    return strcmp(leaf, "OpenGL") == 0 || strcmp(leaf, "libGL.dylib") == 0 || strcmp(leaf, "libGLU.dylib") == 0;
}

bool isSystemImagePath(const char* path) {
    if (!path) return true;
    static const char* const kPrefixes[] = {"/System/", "/usr/lib/", "/usr/libexec/", "/Library/Apple/"};
    for (const char* pre : kPrefixes)
        if (strncmp(path, pre, strlen(pre)) == 0) return true;
    return false;
}

}  // namespace mgl::macho
