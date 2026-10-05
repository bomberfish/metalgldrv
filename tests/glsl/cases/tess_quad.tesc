#version 410 core
layout(vertices = 4) out;
in vec3 vPos[];
in vec2 vUV[];
out vec3 tcPos[];
out vec2 tcUV[];
patch out vec4 patchColor;
uniform float level;
void main() {
    tcPos[gl_InvocationID] = vPos[gl_InvocationID];
    tcUV[gl_InvocationID] = vUV[gl_InvocationID];
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    barrier();
    if (gl_InvocationID == 0) {
        gl_TessLevelOuter[0] = level;
        gl_TessLevelOuter[1] = level + 1.0;
        gl_TessLevelOuter[2] = level;
        gl_TessLevelOuter[3] = level + 2.0;
        gl_TessLevelInner[0] = level;
        gl_TessLevelInner[1] = level;
        patchColor = vec4(tcUV[1], tcUV[2]);
    }
}
