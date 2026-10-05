#version 410 core
in vec2 teUV;
in vec4 teColor;
out vec4 color;
void main() { color = vec4(teUV, 0.0, 1.0) + teColor * float(gl_PrimitiveID + 1); }
