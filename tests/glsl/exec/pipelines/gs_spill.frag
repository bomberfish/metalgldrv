#version 330 core
in vec4 a[30];
noperspective in vec2 b;
flat in int c;
out vec4 color;
void main() {
    vec2 ndc = gl_FragCoord.xy / 32.0 - 1.0;
    bool good = c == 7;
    good = good && all(lessThan(abs(a[29] - vec4(a[0].xy, 29.0, a[0].w)), vec4(1e-4)));  // spilled vs kept, perspective
    good = good && all(lessThan(abs(b - ndc), vec2(1e-3)));                            // spilled, noperspective
    good = good && abs(a[17].z - 17.0) < 1e-4;
    color = good ? vec4(0, 1, 0, 1) : vec4(1, 0, 0, 1);
}
