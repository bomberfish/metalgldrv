#version 330 core
layout(location = 0) in vec4 pos;
out float vx;
void main() { vx = pos.x; gl_Position = pos; }
