#version 410 core
layout(triangles, equal_spacing, ccw) in;
in vec2 cNdc[];
patch in float marker;
out vec2 ndc;
out vec3 bary;
flat out int ok;
void main() {
    vec3 t = gl_TessCoord;
    bary = t;
    ndc = t.x * cNdc[0] + t.y * cNdc[1] + t.z * cNdc[2];
    ok = (marker == 3.5 && gl_TessLevelOuter[2] == 5.0 && gl_TessLevelInner[0] == 4.0 && abs(t.x + t.y + t.z - 1.0) < 1e-5) ? 1 : 0;
    gl_Position = t.x * gl_in[0].gl_Position + t.y * gl_in[1].gl_Position + t.z * gl_in[2].gl_Position;
}
