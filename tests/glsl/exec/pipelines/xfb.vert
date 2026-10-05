#version 330 core
layout(location = 0) in vec4 pos;
out vec2 twice;
out float id;
void main() { twice = pos.xy * 2.0; id = float(gl_VertexID) + float(gl_InstanceID) * 100.0; gl_Position = pos; }
