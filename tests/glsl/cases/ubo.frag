#version 430 core
in vec3 color;
out vec4 o;
void main() { o = vec4(color, 1.0); }
