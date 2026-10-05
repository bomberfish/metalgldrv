#version 410 core
layout(vertices = 3) out;
in vec2 vNdc[];
out vec2 cNdc[];
patch out float marker;
void main() {
    cNdc[gl_InvocationID] = vNdc[gl_InvocationID];
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    if (gl_InvocationID == 0) {
        gl_TessLevelOuter[0] = 3.0;
        gl_TessLevelOuter[1] = 4.0;
        gl_TessLevelOuter[2] = 5.0;
        gl_TessLevelInner[0] = 4.0;
        marker = float(gl_PatchVerticesIn) + 0.5;
    }
}
