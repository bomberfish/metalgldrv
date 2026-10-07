#version 450 core
#extension GL_ARB_gpu_shader_int64 : require
flat in int64_t big;
flat in u64vec3 v3;
out vec4 color;
void main() {
    bool ok = big == -0x123456789ABl && v3 == u64vec3(1ul << 40, 0xFFFFFFFF00000000ul, 7ul);
    color = ok ? vec4(0, 1, 0, 1) : vec4(1, 0, 0, 1);
}
