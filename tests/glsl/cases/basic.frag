#version 330 core
in vec2 vUV;
in vec4 vColor;
uniform sampler2D uTex;
layout(location = 0) out vec4 fragColor;
void main() {
    vec4 c = texture(uTex, vUV) * vColor;
    if (c.a < 0.1) discard;
    fragColor = c;
}
