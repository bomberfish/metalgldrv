#version 450 core
#extension GL_ARB_gpu_shader_int64 : require
in vec2 ndc;
flat in int ok;
flat in int64_t pass64;
out vec4 color;
void main() {
    color = ok == 1 && pass64 == (-5l << 40) ? vec4(0, 1, 0, 1) : vec4(1, 0, 0, 1);
}
