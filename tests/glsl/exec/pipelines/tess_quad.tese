#version 410 core
layout(quads, fractional_even_spacing, cw) in;
out vec2 uv;
void main() {
    uv = gl_TessCoord.xy;
    vec4 a = mix(gl_in[0].gl_Position, gl_in[1].gl_Position, uv.x);
    vec4 b = mix(gl_in[3].gl_Position, gl_in[2].gl_Position, uv.x);
    gl_Position = mix(a, b, uv.y);
}
