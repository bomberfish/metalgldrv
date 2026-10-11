cmake_minimum_required(VERSION 3.24)

# MGL_UNIVERSAL=ON builds macOS arm64 + x86_64 (x86_64 is what CrossOver/Wine loads
# under Rosetta). Use a separate build dir (docs/crossover.md):
#   cmake -G Ninja -B build-universal -DMGL_UNIVERSAL=ON && ninja -C build-universal metalgldrv
option(MGL_UNIVERSAL "Build a universal (arm64 + x86_64) macOS library" OFF)
# Private Metal API (src/mtl/spi.h): selectors Apple's own GL driver uses, for wide lines, exact border colors and
# its GL-mode queue. On by default; turn it off for apps that go through App Store review. At run time
# METALGLDRV_MORALLY_WRONG=0 disables them in a build that has them.
option(MGL_PRIVATE_API "Use private Metal API where it helps GL (not App Store safe)" ON)
if(MGL_UNIVERSAL AND NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
  set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "" FORCE)
endif()

project(metalgldrv LANGUAGES C CXX OBJC OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)
include(GNUInstallDirs)

set(GEN_DIR ${CMAKE_BINARY_DIR}/gen)

# Driver version (git describe), regenerated on every build (cmake/version.cmake) so GL_VERSION
# and the macOS frontend name the commit the driver was built from.
add_custom_target(mgl_version
  COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_SOURCE_DIR} -DOUT=${GEN_DIR}/mgl_version.h -P ${CMAKE_SOURCE_DIR}/cmake/version.cmake
  BYPRODUCTS ${GEN_DIR}/mgl_version.h
  COMMENT "Driver version from git describe")
add_custom_command(
  OUTPUT ${GEN_DIR}/glapi.h ${GEN_DIR}/gl_stubs.cpp ${GEN_DIR}/gl_procs.cpp
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/gen_glapi.py ${CMAKE_SOURCE_DIR}/registry/gl.xml ${GEN_DIR}
  DEPENDS ${CMAKE_SOURCE_DIR}/scripts/gen_glapi.py ${CMAKE_SOURCE_DIR}/registry/gl.xml
  COMMENT "Generating GL API surface")

# --- MSL runtime libraries embedded into generated shaders ---
add_custom_command(
  OUTPUT ${GEN_DIR}/fp64_msl.inc
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/embed.py ${CMAKE_SOURCE_DIR}/src/glsl/runtime/fp64.metal ${GEN_DIR}/fp64_msl.inc
  DEPENDS ${CMAKE_SOURCE_DIR}/scripts/embed.py ${CMAKE_SOURCE_DIR}/src/glsl/runtime/fp64.metal
  COMMENT "Embedding fp64 MSL library")
add_custom_command(
  OUTPUT ${GEN_DIR}/multipass_msl.inc
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/embed.py ${CMAKE_SOURCE_DIR}/src/glsl/runtime/multipass.metal ${GEN_DIR}/multipass_msl.inc
  DEPENDS ${CMAKE_SOURCE_DIR}/scripts/embed.py ${CMAKE_SOURCE_DIR}/src/glsl/runtime/multipass.metal
  COMMENT "Embedding multi-pass MSL helpers")
add_custom_target(mgl_embedded DEPENDS ${GEN_DIR}/fp64_msl.inc ${GEN_DIR}/multipass_msl.inc)

# --- GLSL compiler (pure C++, no platform deps; also built for host tools) ---
file(GLOB_RECURSE GLSL_SOURCES CONFIGURE_DEPENDS src/glsl/*.cpp)
add_library(mglsl STATIC ${GLSL_SOURCES})
add_dependencies(mglsl mgl_embedded)
target_include_directories(mglsl PUBLIC src ${GEN_DIR})
set_target_properties(mglsl PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(mglsl PRIVATE -fvisibility=hidden -Wall -Wno-unused-parameter)

# --- driver ---
file(GLOB_RECURSE DRV_SOURCES CONFIGURE_DEPENDS
  src/gl/*.cpp src/gl/*.mm src/egl/*.cpp src/egl/*.mm src/mtl/*.mm src/mtl/*.cpp src/util/*.cpp src/util/*.mm)
# macOS system-GL replacement: CGL, NSOpenGL hooks, JVMTI agent, CrossOver glue.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
  file(GLOB MACOS_SOURCES CONFIGURE_DEPENDS src/macos/*.mm src/macos/*.cpp)
  list(APPEND DRV_SOURCES ${MACOS_SOURCES})
  # The NSOpenGL method replacements manage retain/release by hand.
  set_source_files_properties(${CMAKE_SOURCE_DIR}/src/macos/nsgl.mm PROPERTIES COMPILE_OPTIONS "-fno-objc-arc")
endif()
add_library(metalgldrv SHARED ${DRV_SOURCES} ${GEN_DIR}/gl_stubs.cpp ${GEN_DIR}/gl_procs.cpp)
add_dependencies(metalgldrv mgl_version)
target_include_directories(metalgldrv PRIVATE src ${GEN_DIR}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(metalgldrv PRIVATE mglsl "-framework Metal" "-framework QuartzCore" "-framework Foundation" "-framework CoreGraphics" "-framework ImageIO" "-framework MetalFX")
if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
  target_link_libraries(metalgldrv PRIVATE "-framework UIKit")
else()
  target_link_libraries(metalgldrv PRIVATE "-framework AppKit" "-framework CoreVideo")
endif()
target_compile_options(metalgldrv PRIVATE -fvisibility=hidden -fobjc-arc -Wall -Wno-unused-parameter -Wno-unused-function)
target_compile_definitions(metalgldrv PRIVATE METALGLDRV_PRIVATE_API=$<BOOL:${MGL_PRIVATE_API}>)
set_target_properties(metalgldrv PROPERTIES
  OUTPUT_NAME metalgldrv
  # >= the 1.0.0 that binaries linked against OpenGL.framework record (crossover-install.sh).
  MACHO_COMPATIBILITY_VERSION 1.0.0
  MACHO_CURRENT_VERSION 1.0.0
  INSTALL_NAME_DIR "@rpath"
  BUILD_WITH_INSTALL_NAME_DIR ON)

# --- install / packaging (docs/porting.md) ---
# cmake --install <build> --prefix <dir>: lib/libmetalgldrv.dylib, include/{GL,EGL,KHR,metalgldrv},
# lib/cmake/metalgldrv (find_package -> metalgldrv::metalgldrv) and lib/pkgconfig/metalgldrv.pc.
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
add_library(metalgldrv::metalgldrv ALIAS metalgldrv)  # the same name inside a parent project (add_subdirectory)
install(TARGETS metalgldrv EXPORT metalgldrvTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT metalgldrvTargets NAMESPACE metalgldrv:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/metalgldrv)
configure_package_config_file(cmake/metalgldrvConfig.cmake.in ${CMAKE_BINARY_DIR}/metalgldrvConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/metalgldrv)
install(FILES ${CMAKE_BINARY_DIR}/metalgldrvConfig.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/metalgldrv)
set(MGL_PC_VERSION 0.1.0)
configure_file(cmake/metalgldrv.pc.in ${CMAKE_BINARY_DIR}/metalgldrv.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/metalgldrv.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

option(MGL_BUILD_TESTS "Build host tests" ON)
if(MGL_BUILD_TESTS)
  add_subdirectory(tests)
  add_subdirectory(tools)
endif()
