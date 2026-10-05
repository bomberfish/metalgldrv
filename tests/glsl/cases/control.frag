#version 330 core
uniform int mode;
in vec2 uv;
out vec4 color;
void main() {
    vec4 c = vec4(0.0);
    switch (mode) {
        case 0:
            float t = uv.x;
            c = vec4(t);
            break;
        case 1:
        case 2: {
            int k = mode * 2;
            c.x = float(k);
        }
        // fallthrough
        default:
            c.y = 1.0;
    }
    int i = 0;
    do { i++; if (i == 3) continue; c.z += 0.1; } while (i < 10);
    for (int a = 0, b = 10; a < b; a++, b--) { if (a > 3) break; c.w += 1.0; }
    while (true) { c.x *= 0.5; if (c.x < 0.1) break; }
    c.x = mode > 1 ? c.x : (c.y, c.z);
    color = c;
}
