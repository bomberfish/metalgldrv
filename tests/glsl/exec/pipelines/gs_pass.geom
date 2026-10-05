#version 330 core
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
in VS { vec2 ndc; flat int vid; } gin[];
out vec2 ndc;
flat out int ok;
void main() {
    bool good = gin[0].vid == 0 && gin[1].vid == 1 && gin[2].vid == 2 && gl_PrimitiveIDIn == 0;
    for (int i = 0; i < 3; i++) {
        gl_Position = gl_in[i].gl_Position;
        ndc = gin[i].ndc;
        ok = good ? 1 : 0;
        gl_PrimitiveID = 7;
        EmitVertex();
    }
    EndPrimitive();
}
