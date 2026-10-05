#version 330 core
out vec4 c;
void main() { vec3 a = vec3(1.0); vec2 b = a + vec2(1.0); c = vec4(b, 0.0, 1.0); }
