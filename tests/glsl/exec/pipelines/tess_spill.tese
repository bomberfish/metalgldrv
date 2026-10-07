#version 410 core
// TES output past Metal's mesh output budget: the last varyings are spilled.
layout(triangles, equal_spacing, ccw) in;
in vec2 cNdc[];
out vec4 a[30];
noperspective out vec2 b;
flat out int c;
void main() {
    vec3 t = gl_TessCoord;
    vec2 p = t.x * cNdc[0] + t.y * cNdc[1] + t.z * cNdc[2];
    float w = 1.0 + t.x + 2.0 * t.y;
    for (int k = 0; k < 30; k++) a[k] = vec4(p, float(k), t.y);
    b = p;
    c = 7;
    gl_Position = vec4(p * w, 0.0, w);
}
