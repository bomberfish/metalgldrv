#version 410 core
in vec3 pos;
out VS_OUT {
    vec3 normal;
    flat int id;
    noperspective vec2 screen;
    vec4 extra[2];
} vs_out;
layout(location = 3) out vec4 locVarying;
void main() {
    vs_out.normal = pos;
    vs_out.id = gl_InstanceID;
    vs_out.screen = pos.xy;
    vs_out.extra[0] = vec4(1.0);
    vs_out.extra[1] = vec4(2.0);
    locVarying = vec4(pos, 1.0);
    gl_Position = vec4(pos, 1.0);
    gl_PointSize = 4.0;
}
