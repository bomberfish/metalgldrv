#version 330 core
// GL window coordinates: x = (ndc.x * 0.5 + 0.5) * 64, y = (ndc.y * 0.5 + 0.5) * 64 (origin bottom-left)
in vec2 ndc;
flat in int vid;
noperspective in vec2 lin;
out vec4 color;
void main() {
    vec2 expected = (ndc * 0.5 + 0.5) * 64.0;
    bool ok = all(lessThan(abs(gl_FragCoord.xy - expected), vec2(0.01)));
    ok = ok && fract(gl_FragCoord.x) == 0.5 && fract(gl_FragCoord.y) == 0.5;
    ok = ok && vid == 7;                       // flat: Metal provokes from the first vertex; GL last-vertex convention is emulated by the runtime
    ok = ok && all(lessThan(abs(lin - ndc), vec2(1e-4)));
    ok = ok && gl_FragCoord.z == 0.5;          // z = 0 -> depth range [0,1] midpoint
    color = ok ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
