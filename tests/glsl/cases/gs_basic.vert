#version 330 core
layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
out VS_OUT { vec3 normal; vec2 uv; } vs_out;
out float vDepth;
void main() {
    vs_out.normal = normal;
    vs_out.uv = pos.xy;
    vDepth = pos.z;
    gl_Position = vec4(pos, 1.0);
}
