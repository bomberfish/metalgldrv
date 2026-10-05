#version 330 core
flat in int corner;
out vec4 color;
void main() {
    // GL: flat values come from the last vertex of each strip triangle (vertices 2 and 3)
    bool good = corner == 2 || corner == 3;
    color = good ? vec4(0, 1, 0, 1) : vec4(1, 0, 0, 1);
}
