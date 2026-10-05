#version 400 core
uniform double scale;
uniform dvec3 offset;
uniform dmat4 model;
in vec3 pos;
void main() {
    dvec3 p = dvec3(pos) * scale + offset;
    dvec4 w = model * dvec4(p, 1.0LF);
    double len = length(p);
    gl_Position = vec4(w) * float(len);
}
