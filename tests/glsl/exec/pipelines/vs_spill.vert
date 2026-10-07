#version 330 core
// 128 vertex output components read by the fragment shader: more than a classic Metal vertex
// function passes, so the program draws through a vertex mesh stage that spills the excess.
layout(location = 0) in vec4 pos;
out vec4 a[31];
noperspective out vec2 b;
flat out int c;
flat out int d;
void main() {
    for (int k = 0; k < 31; k++) a[k] = vec4(pos.xy, float(k), float(gl_VertexID));
    b = pos.xy;
    c = gl_VertexID;
    d = gl_VertexID;
    gl_Position = pos;
}
