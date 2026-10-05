#pragma once
#ifdef __cplusplus
extern "C" {
#endif
typedef struct { const char* name; void (*fn)(void); } mgl_proc_entry;
extern const mgl_proc_entry mgl_gl_procs[];
extern const unsigned mgl_gl_procs_count;
void (*mgl_lookup_proc(const char* name))(void);
#ifdef __cplusplus
}
#endif
