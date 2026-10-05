#version 420 core
in vec4 p;
out vec4 o;
struct P { vec2 a; int b; };
void main() {
    mat4 m4 = mat4(1.0);
    mat3 m3 = mat3(m4);
    mat2 m2 = mat2(p);
    mat4x3 m43 = mat4x3(m3);
    mat3 mixed = mat3(p.xy, p.z, p.w, p.x, vec3(1.0), 2.0);
    vec4 fromMat = vec4(m2);
    vec3 v3 = vec3(p);
    ivec2 iv = ivec2(p.xy);
    bvec3 bv = bvec3(p.xyz);
    uvec4 uv = uvec4(iv, 1, 2u);
    float arr[3] = { 1.0, 2.0, 3.0 };
    P pp = { vec2(1.0), 2 };
    vec2 sw = p.wx;
    float s = 1.0;
    vec3 ss = s.xxx;
    o = fromMat + vec4(v3, arr[2]) + vec4(mixed[1], m43[3].x) + vec4(pp.a, float(pp.b), float(bv.y)) + vec4(uv) + vec4(sw, ss.xy);
    o += vec4(m2[1], m3[2].xy) + vec4(mat2x3(p.xyz, p.yzw)[1], 1.0);
    gl_Position = m4 * p;
}
