#version 450 core
in vec4 p;
out vec4 uv;
void main() { uv = p; gl_Position = p; }
