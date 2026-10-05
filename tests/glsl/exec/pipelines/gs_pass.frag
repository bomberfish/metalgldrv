#version 330 core
in vec2 ndc;
flat in int ok;
out vec4 color;
void main() {
    vec2 expected = (ndc * 0.5 + 0.5) * 64.0;
    bool good = ok == 1 && all(lessThan(abs(gl_FragCoord.xy - expected), vec2(0.01))) && gl_PrimitiveID == 7;
    color = good ? vec4(0, 1, 0, 1) : vec4(1, 0, 0, 1);
}
