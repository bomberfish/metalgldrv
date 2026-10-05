#version 450 core
uniform double scale;
uniform dvec3 offset;
uniform dmat4 model;
uniform dmat3x2 m32;
layout(std430, binding = 0) buffer D { double d; dvec3 v3; dmat2 m2; layout(row_major) dmat3 rm; double arr[]; } db;
layout(location = 0) in vec3 pos;
layout(location = 1) in dvec3 dpos;
layout(location = 3) in double dscalar;
flat out double vd;
flat out dvec3 vd3;
struct DS { double a; dvec2 b; };
dvec2 helper(inout double x, dvec2 y) { x += 1.0lf; return y * x; }
void main() {
    dvec3 p = dvec3(pos) * scale + offset + dpos * dscalar;
    dvec4 w = model * dvec4(p, 1.0LF);
    dmat4 inv = inverse(model);
    dmat3 t = transpose(dmat3(model));
    double det = determinant(t);
    dvec3 c = cross(p, dvec3(0.0lf, 1.0lf, 0.0lf));
    double len = length(p) + dot(p, c) + distance(p, c);
    dvec2 r2 = m32 * dvec3(1.0lf);
    dvec3 l3 = dvec2(1.0lf) * m32;
    double x = 2.0lf;
    dvec2 h = helper(x, r2);
    if (p.x < 0.0lf || all(lessThan(p, dvec3(1.0)))) p = -p;
    p = clamp(p, 0.0lf, 10.0lf) + mix(p, c, 0.5lf) + floor(p) + abs(p) + sqrt(p) + mod(p, 2.0lf) + min(p, c);
    p += dvec3(float(det));
    p++;
    db.d += len;
    db.v3.yz = h;
    db.m2[1] = r2 + l3.xy;
    db.rm[2][1] = x;
    db.arr[gl_VertexID] = double(int(len)) + double(uint(3)) + double(true);
    DS s = DS(1.0lf, dvec2(2.0lf));
    if (s == DS(x, h)) db.d = 0.0lf;
    uvec2 bits = unpackDouble2x32(packDouble2x32(uvec2(1u, 2u)));
    int e;
    double fr = frexp(len, e);
    vd = fr + double(bits.x) + double(e);
    vd3 = p;
    gl_Position = vec4(w) * float(len) + vec4(inv[0]) + vec4(float(db.arr[0]));
}
