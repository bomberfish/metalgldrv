#version 330 core
in vec2 ndc;
in float gl_ClipDistance[2];
out vec4 color;
void main() {
    bool ok = abs(gl_ClipDistance[0] - (ndc.x + 10.0)) < 1e-3 && abs(gl_ClipDistance[1] - (ndc.y * 3.0 + 10.0)) < 1e-3;
    color = ok ? vec4(0, 1, 0, 1) : vec4(1, 0, 0, 1);
}
