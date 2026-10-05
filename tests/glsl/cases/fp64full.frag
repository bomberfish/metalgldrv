#version 450 core
flat in double vd;
flat in dvec3 vd3;
out vec4 c;
void main() { c = vec4(float(vd), vec3(vd3)); }
