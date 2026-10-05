#version 430 core
struct Light { vec3 pos; float radius; vec4 color; };
layout(std140, binding = 1) uniform Scene {
    mat4 proj;
    layout(row_major) mat4 view;
    mat3 normalMat;
    vec3 eye;
    float time;
    Light lights[4];
    float weights[5];
    bvec2 flags;
    ivec3 counts;
} scene;
layout(std140) uniform Anon { vec2 offset2; mat2 rot; };
layout(location = 0) in vec3 pos;
out vec3 color;
void main() {
    vec3 c = vec3(0.0);
    for (int i = 0; i < 4; ++i) {
        float d = distance(scene.lights[i].pos, pos);
        c += scene.lights[i].color.rgb * max(0.0, 1.0 - d / scene.lights[i].radius) * scene.weights[i];
    }
    if (scene.flags.y) c *= 2.0;
    c += scene.normalMat * vec3(float(scene.counts.z));
    c.xy += rot * offset2 + scene.view[1].xy + scene.view[2][3];
    color = c;
    gl_Position = scene.proj * scene.view * vec4(pos + scene.eye * scene.time, 1.0);
}
