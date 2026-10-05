cmake_minimum_required(VERSION 3.24)
project(metalgldrv LANGUAGES C CXX OBJC OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(GEN_DIR ${CMAKE_BINARY_DIR}/gen)
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
add_library(metalgldrv SHARED ${DRV_SOURCES} ${GEN_DIR}/gl_stubs.cpp ${GEN_DIR}/gl_procs.cpp)
target_include_directories(metalgldrv PRIVATE src ${GEN_DIR} include)
target_link_libraries(metalgldrv PRIVATE mglsl "-framework Metal" "-framework QuartzCore" "-framework Foundation" "-framework CoreGraphics" "-framework MetalFX")
if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
  target_link_libraries(metalgldrv PRIVATE "-framework UIKit")
else()
  target_link_libraries(metalgldrv PRIVATE "-framework AppKit")
endif()
target_compile_options(metalgldrv PRIVATE -fvisibility=hidden -fobjc-arc -Wall -Wno-unused-parameter -Wno-unused-function)
set_target_properties(metalgldrv PROPERTIES
  OUTPUT_NAME metalgldrv
  INSTALL_NAME_DIR "@rpath"
  BUILD_WITH_INSTALL_NAME_DIR ON)

option(MGL_BUILD_TESTS "Build host tests" ON)
if(MGL_BUILD_TESTS)
  add_subdirectory(tests)
  add_subdirectory(tools)
endif()
