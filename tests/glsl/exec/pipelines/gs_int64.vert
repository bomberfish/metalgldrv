#version 450 core
#extension GL_ARB_gpu_shader_int64 : require
layout(location = 0) in vec4 pos;
out VS { vec2 ndc; flat int64_t big; flat u64vec2 v; } vs;
void main() { vs.ndc = pos.xy; vs.big = int64_t(gl_VertexID) - 0x100000000l; vs.v = u64vec2(1ul << 50, uint64_t(gl_VertexID)); gl_Position = pos; }
