/*
 * metalgldrv: OpenGL 4.6 on Metal 4. Public C API for applications (docs/porting.md).
 *
 * The OpenGL and EGL entry points themselves come from the standard headers in this include directory
 * (GL/glcorearb.h, EGL/egl.h); on macOS the library also exports CGL under Apple's names. This header only
 * adds what is specific to metalgldrv.
 */
#ifndef METALGLDRV_H
#define METALGLDRV_H

#ifdef __cplusplus
extern "C" {
#endif

/* Whether this machine can run metalgldrv: a GPU with Metal 4 (Apple silicon) on macOS / iOS 26 or later.
   Without one, keep using the system's OpenGL. */
int metalgldrv_is_supported(void);

/* macOS: routes the process's system OpenGL to metalgldrv: NSOpenGLContext / NSOpenGLPixelFormat (as used by
   GLFW, SDL and NSOpenGLView), and libraries loaded in the process that link or dlopen OpenGL.framework.
   Call it before creating any window or GL context. Apps whose main executable links libmetalgldrv
   directly get this automatically at load time; apps that dlopen the library call it themselves.
   Returns 1 when active (also when it already was), 0 when the machine isn't supported (see
   metalgldrv_is_supported). iOS has no system OpenGL to replace: use EGL; returns 0. */
int metalgldrv_activate(void);

/* Whether metalgldrv_activate (or the automatic activation) is in effect. */
int metalgldrv_is_active(void);

/* The driver version, e.g. "0.1.0+3f4fa34". */
const char* metalgldrv_version(void);

#ifdef __cplusplus
}
#endif

#endif /* METALGLDRV_H */
