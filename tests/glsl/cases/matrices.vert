#version 400 core
uniform mat4 m;
uniform mat3x2 m32;
in vec4 p;
out vec4 o;
void main() {
    mat4 inv = inverse(m);
    mat4 t = transpose(m);
    float d = determinant(mat3(m));
    mat3 op = outerProduct(p.xyz, p.yzw);
    mat4 cm = matrixCompMult(m, inv);
    mat4 acc = m;
    acc *= inv;
    acc += 1.0;
    acc = acc / 2.0;
    acc = -acc;
    vec4 v = p;
    v *= m;
    vec2 r = m32 * p.xyz;
    vec3 l = p.xy * m32;
    o = inv * t * cm * acc * v + vec4(d) + vec4(op[0], 1.0) + vec4(r, l.xy);
    gl_Position = m * p;
}
