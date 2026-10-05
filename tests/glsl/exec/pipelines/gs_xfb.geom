#version 330 core
layout(triangles) in;
layout(triangle_strip, max_vertices = 4) out;
in float vx[];
out float val;
void main() {
    // strip of 4 vertices -> GL triangles (0,1,2) and (2,1,3)
    for (int i = 0; i < 4; i++) {
        val = float(gl_PrimitiveIDIn * 10 + i) + vx[0] * 0.0;
        gl_Position = vec4(0.0);
        EmitVertex();
    }
}
