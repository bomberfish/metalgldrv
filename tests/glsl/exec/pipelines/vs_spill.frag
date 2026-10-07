#version 330 core
in vec4 a[31];
noperspective in vec2 b;
flat in int c;
flat in int d;
out vec4 color;
void main() {
    vec2 ndc = gl_FragCoord.xy / 32.0 - 1.0;
    bool good = c == d && c % 3 == 2;  // flat: GL's last-vertex convention, spilled == kept
    good = good && all(lessThan(abs(a[30] - vec4(a[0].xy, 30.0, a[0].w)), vec4(1e-4)));
    good = good && all(lessThan(abs(b - ndc), vec2(1e-3)));
    good = good && all(lessThan(abs(a[0].xy - ndc), vec2(1e-3)));
    color = good ? vec4(0, 1, 0, 1) : vec4(1, 0, 0, 1);
}
