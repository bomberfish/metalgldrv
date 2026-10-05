#version 330 core
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;
flat out int corner;
void main() {
    vec4 c = gl_in[0].gl_Position;
    // lower-left quadrant of NDC: x,y in [-1, 0]
    vec2 o[4] = vec2[](vec2(-1, -1), vec2(0, -1), vec2(-1, 0), vec2(0, 0));
    for (int i = 0; i < 4; i++) {
        gl_Position = vec4(o[i] + c.xy * 0.0, 0.0, 1.0);
        corner = i;
        EmitVertex();
    }
}
