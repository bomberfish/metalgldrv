#version 330 core
layout(location = 0) in vec4 pos;
out vec2 ndc;
flat out int vid;
noperspective out vec2 lin;
void main() {
    ndc = pos.xy;
    lin = pos.xy;
    vid = gl_VertexID + 7;
    gl_Position = pos;
}
