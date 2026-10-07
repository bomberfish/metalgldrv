#version 450 core
#extension GL_ARB_gpu_shader_int64 : require
#extension GL_KHR_shader_subgroup_basic : require
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
in VS { vec2 ndc; flat int64_t big; flat u64vec2 v; } gin[];
out vec2 ndc;
flat out int ok;
flat out int64_t pass64;
void main() {
    bool good = gin[1].big == 1l - 0x100000000l && gin[2].v == u64vec2(1ul << 50, 2ul) && gl_SubgroupSize == 32u && gl_SubgroupInvocationID == 0u;
    for (int i = 0; i < 3; i++) {
        gl_Position = gl_in[i].gl_Position;
        ndc = gin[i].ndc;
        ok = good ? 1 : 0;
        pass64 = -5l << 40;
        EmitVertex();
    }
    EndPrimitive();
}
