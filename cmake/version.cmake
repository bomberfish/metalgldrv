# Writes ${OUT} (mgl_version.h) from git describe, run on every build (see CMakeLists.txt) so the
# driver's version always names the commit it was built from. The file is only rewritten when the
# version changes, so unchanged builds don't recompile what includes it.
execute_process(COMMAND git describe --tags --always --dirty WORKING_DIRECTORY ${SRC}
  OUTPUT_VARIABLE MGL_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT MGL_VERSION)
  set(MGL_VERSION unknown)
endif()
configure_file(${SRC}/src/util/version.h.in ${OUT} @ONLY)
