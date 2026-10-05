#version 410 core
layout(location = 0) in vec3 pos;
out vec3 vPos;
out vec2 vUV;
void main() { vPos = pos; vUV = pos.xy * 0.5 + 0.5; gl_Position = vec4(pos, 1.0); }
