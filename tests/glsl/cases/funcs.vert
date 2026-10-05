#version 400 core
struct S { vec3 a; float b[2]; };
void split(in vec4 v, out vec2 lo, inout vec2 hi) { lo = v.xy; hi += v.zw; }
float sum(float x[4]) { float s = 0.0; for (int i = 0; i < 4; i++) s += x[i]; return s; }
float[4] makeArr(float a) { return float[4](a, a * 2.0, a * 3.0, a * 4.0); }
S makeS(float f) { S s; s.a = vec3(f); s.b[0] = f; s.b[1] = -f; return s; }
float f(float x) { return x * 2.0; }
vec2 f(vec2 x) { return x * 3.0; }
int f(int x) { return x + 1; }
in vec4 p;
out vec4 o;
void main() {
    vec4 v = p;
    vec2 lo;
    split(v, lo, v.zw);
    split(p, o.xy, o.zw);
    float arr[4] = makeArr(1.0);
    S s = makeS(2.0);
    o.x += sum(arr) + s.b[1] + f(1.0) + float(f(2)) + f(vec2(1.0)).y;
    S t = S(vec3(1.0), float[2](1.0, 2.0));
    if (s == t) o.y = 1.0;
    if (arr == makeArr(1.0)) o.z = 2.0;
    gl_Position = v + vec4(lo, 0.0, 0.0);
}
