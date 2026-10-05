#version 410 core
in VS_OUT {
    vec3 normal;
    flat int id;
    noperspective vec2 screen;
    vec4 extra[2];
} fs_in;
layout(location = 3) in vec4 anyName;
layout(location = 0, index = 0) out vec4 c0;
layout(location = 0, index = 1) out vec4 c1;
void main() {
    c0 = vec4(fs_in.normal, float(fs_in.id)) + fs_in.extra[1] + anyName;
    c1 = vec4(fs_in.screen, 0.0, 1.0);
}
