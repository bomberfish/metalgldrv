#version 450 core
#extension GL_ARB_gpu_shader_int64 : require
layout(location = 0) in vec4 pos;
flat out int64_t big;
flat out u64vec3 v3;
void main() {
    big = -0x123456789ABl;
    v3 = u64vec3(1ul << 40, 0xFFFFFFFF00000000ul, 7ul);
    gl_Position = pos;
}
