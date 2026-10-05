#version 330 core
layout(triangles) in;
layout(triangle_strip, max_vertices = 6) out;
in VS_OUT { vec3 normal; vec2 uv; } gs_in[];
in float vDepth[];
out vec3 fNormal;
out vec2 fUV;
flat out int fLayerish;
uniform float explode;
void main() {
    vec3 n = normalize(cross(gl_in[1].gl_Position.xyz - gl_in[0].gl_Position.xyz, gl_in[2].gl_Position.xyz - gl_in[0].gl_Position.xyz));
    for (int copy = 0; copy < 2; copy++) {
        for (int i = 0; i < 3; i++) {
            gl_Position = gl_in[i].gl_Position + vec4(n * explode * float(copy), 0.0);
            fNormal = gs_in[i].normal;
            fUV = gs_in[i].uv + vDepth[i];
            fLayerish = gl_PrimitiveIDIn + copy;
            gl_PrimitiveID = gl_PrimitiveIDIn * 2 + copy;
            EmitVertex();
        }
        EndPrimitive();
    }
}
