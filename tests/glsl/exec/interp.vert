#version 400 core
layout(location = 0) in vec4 pos;
out vec2 ndc;
out Blk { vec3 v3; } blk;
void main() { ndc = pos.xy; blk.v3 = vec3(pos.xy, 1.0); gl_Position = pos; }
