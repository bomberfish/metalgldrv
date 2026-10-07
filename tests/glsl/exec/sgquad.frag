#version 450 core
#extension GL_KHR_shader_subgroup_quad : enable
out vec4 color;
// quad operations act on 2x2 pixel blocks: horizontal / vertical / diagonal neighbours
void main() {
    vec2 f = gl_FragCoord.xy;
    vec2 h = subgroupQuadSwapHorizontal(f), v = subgroupQuadSwapVertical(f), d = subgroupQuadSwapDiagonal(f);
    bool ok = h.y == f.y && abs(h.x - f.x) == 1.0 && v.x == f.x && abs(v.y - f.y) == 1.0 && abs(d.x - f.x) == 1.0 && abs(d.y - f.y) == 1.0;
    color = ok ? vec4(0, 1, 0, 1) : vec4(1, 0, 0, 1);
}
