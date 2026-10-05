#version 430 core
layout(origin_upper_left, pixel_center_integer) in vec4 gl_FragCoord;
layout(depth_greater) out float gl_FragDepth;
out vec4 c;
void main() {
    c = gl_FragCoord + vec4(gl_PointCoord, float(gl_FrontFacing), 0.0);
    gl_FragDepth = gl_FragCoord.z + 0.01;
    c.x += dFdx(gl_FragCoord.x) + fwidth(c.y);
}
