#version 410 core
layout(location = 0) in vec4 pos;
out vec2 vNdc;
void main() { vNdc = pos.xy; gl_Position = pos; }
