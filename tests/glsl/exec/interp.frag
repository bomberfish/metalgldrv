#version 400 core
in vec2 ndc;
in Blk { vec3 v3; } blk;
out vec4 color;
bool near(float a, float b) { return abs(a - b) < 1e-4; }
void main() {
    // ndc changes by 2/64 per pixel in x and y (GL window y up)
    float d = 2.0 / 64.0;
    bool ok = near(interpolateAtOffset(ndc, vec2(0.0)).x, ndc.x);
    ok = ok && near(interpolateAtOffset(ndc, vec2(0.5, 0.0)).x, ndc.x + 0.5 * d);
    ok = ok && near(interpolateAtOffset(ndc, vec2(0.0, -0.25)).y, ndc.y - 0.25 * d);
    ok = ok && near(interpolateAtCentroid(ndc.y), ndc.y);
    ok = ok && near(interpolateAtSample(blk.v3, 0).x, blk.v3.x);
    ok = ok && near(interpolateAtOffset(blk.v3.yx, vec2(0.25, 0.0)).y, ndc.x + 0.25 * d);
    color = ok ? vec4(0, 1, 0, 1) : vec4(1, 0, 0, 1);
}
