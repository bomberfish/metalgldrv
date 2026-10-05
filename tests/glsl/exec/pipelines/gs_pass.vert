#version 330 core
layout(location = 0) in vec4 pos;
out VS { vec2 ndc; flat int vid; } vs;
void main() { vs.ndc = pos.xy; vs.vid = gl_VertexID; gl_Position = pos; }
