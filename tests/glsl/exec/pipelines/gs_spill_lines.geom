#version 330 core
// Spilled varyings on line strips (see gs_spill.geom).
layout(triangles) in;
layout(line_strip, max_vertices = 64) out;
in VS { vec2 ndc; flat int vid; } gin[];
out vec4 a[30];
noperspective out vec2 b;
flat out int c;
void main() {
    if (gin[0].vid != 0) return;
    for (int r = 0; r < 16; r++) {
        float y = -0.95 + 0.125 * float(r);
        for (int i = 0; i < 2; i++) {
            vec2 p = vec2(i == 0 ? -0.9 : 0.9, y);
            float w = 1.0 + 1.5 * float(i);
            gl_Position = vec4(p * w, 0.0, w);
            for (int k = 0; k < 30; k++) a[k] = vec4(p, float(k), float(i));
            b = p;
            c = 7;
            EmitVertex();
        }
        EndPrimitive();
    }
}
