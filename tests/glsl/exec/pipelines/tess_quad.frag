#version 410 core
in vec2 uv;
out vec4 color;
void main() {
    vec2 expected = uv * 64.0;
    bool good = all(lessThan(abs(gl_FragCoord.xy - expected), vec2(0.02)));
    color = good ? vec4(0, 1, 0, 1) : vec4(1, 0, 0, 1);
}
