#version 330 core
layout(location = 0) in vec4 pos;
out vec2 ndc;
void main() {
    ndc = pos.xy;
    gl_ClipDistance[0] = pos.x + 10.0;
    gl_ClipDistance[1] = pos.y * 3.0 + 10.0;
    gl_Position = pos;
}
