#version 330 core
// gs_spill.geom with max_vertices = 200: Metal's 32 KB mesh size budget (200 vertices)
// spills most of the varyings, not only the ones past the 124-scalar limit.
layout(triangles) in;
layout(triangle_strip, max_vertices = 200) out;
in VS { vec2 ndc; flat int vid; } gin[];
out vec4 a[30];
noperspective out vec2 b;
flat out int c;
void main() {
    // a quad strip covering the input triangle's bounding box with distinct w per vertex
    vec2 lo = min(min(gin[0].ndc, gin[1].ndc), gin[2].ndc), hi = max(max(gin[0].ndc, gin[1].ndc), gin[2].ndc);
    vec2 p[4] = vec2[4](lo, vec2(hi.x, lo.y), vec2(lo.x, hi.y), hi);
    for (int i = 0; i < 4; i++) {
        float w = 1.0 + 0.75 * float(i);
        gl_Position = vec4(p[i] * w, 0.0, w);
        for (int k = 0; k < 30; k++) a[k] = vec4(p[i], float(k), float(i));
        b = p[i];
        c = 7;
        EmitVertex();
    }
    EndPrimitive();
}
