add_executable(glslc glslc/glslc.mm)
target_link_libraries(glslc PRIVATE mglsl "-framework Metal" "-framework Foundation")
target_compile_options(glslc PRIVATE -fobjc-arc)
