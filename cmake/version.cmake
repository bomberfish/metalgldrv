# Writes ${OUT} (mgl_version.h) from git describe, run on every build (see CMakeLists.txt) so the
# driver's version always names the commit it was built from. The file is only rewritten when the
# version changes, so unchanged builds don't recompile what includes it.
execute_process(COMMAND git describe --tags --always --dirty WORKING_DIRECTORY ${SRC}
  OUTPUT_VARIABLE MGL_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT MGL_VERSION)
  set(MGL_VERSION unknown)
endif()
# GL_VERSION's vendor part: the release tag's version (v0.1.0 -> 0.1.0), plus what git describe adds after
# it on later commits (0.1.0+3-gabc1234, -dirty). Before any tag: 0.1.0+<commit>.
if(MGL_VERSION MATCHES "^v([0-9]+\\.[0-9]+\\.[0-9]+)(-(.+))?$")
  set(MGL_VERSION_STRING "${CMAKE_MATCH_1}")
  if(CMAKE_MATCH_3)
    string(APPEND MGL_VERSION_STRING "+${CMAKE_MATCH_3}")
  endif()
else()
  set(MGL_VERSION_STRING "0.1.0+${MGL_VERSION}")
endif()
configure_file(${SRC}/src/util/version.h.in ${OUT} @ONLY)
