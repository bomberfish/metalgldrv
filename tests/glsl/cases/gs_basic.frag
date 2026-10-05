#version 330 core
in vec3 fNormal;
in vec2 fUV;
flat in int fLayerish;
out vec4 color;
void main() { color = vec4(fNormal * 0.5 + 0.5, float(fLayerish + gl_PrimitiveID)) + vec4(fUV, 0.0, 0.0); }
