#version 410 core
layout(quads, fractional_odd_spacing, ccw) in;
in vec3 tcPos[];
in vec2 tcUV[];
patch in vec4 patchColor;
out vec2 teUV;
out vec4 teColor;
void main() {
    vec2 uv = gl_TessCoord.xy;
    vec3 a = mix(tcPos[0], tcPos[1], uv.x);
    vec3 b = mix(tcPos[3], tcPos[2], uv.x);
    vec3 p = mix(a, b, uv.y);
    teUV = mix(mix(tcUV[0], tcUV[1], uv.x), mix(tcUV[3], tcUV[2], uv.x), uv.y);
    teColor = patchColor + vec4(gl_TessLevelOuter[0] * 0.01, float(gl_PrimitiveID), 0.0, 0.0);
    gl_Position = vec4(p, 1.0) + gl_in[0].gl_Position * 0.0;
}
