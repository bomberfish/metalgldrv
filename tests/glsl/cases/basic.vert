#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aUV;
in vec4 aColor;
uniform mat4 uMVP;
uniform float uScale[3];
out vec2 vUV;
out vec4 vColor;
void main() {
    vUV = aUV;
    vColor = aColor * uScale[gl_VertexID % 3];
    gl_Position = uMVP * vec4(aPos, 1.0);
}
