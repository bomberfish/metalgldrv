// EXT_mesh_shader: task + mesh pipelines, per-primitive outputs, culling, indirect draws.
#include "harness.h"

static GLuint meshProgram(const char* ts, const char* ms, const char* fs) {
    GLuint p = glCreateProgram();
    const char* srcs[3] = {ts, ms, fs};
    GLenum types[3] = {GL_TASK_SHADER_EXT, GL_MESH_SHADER_EXT, GL_FRAGMENT_SHADER};
    for (int i = 0; i < 3; i++) {
        if (!srcs[i]) continue;
        GLuint s = glCreateShader(types[i]);
        glShaderSource(s, 1, &srcs[i], nullptr);
        glCompileShader(s);
        GLint ok = 0;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char buf[4096] = {};
            glGetShaderInfoLog(s, sizeof buf, nullptr, buf);
            fprintf(stderr, "  shader compile failed: %s\n", buf);
        }
        glAttachShader(p, s);
        glDeleteShader(s);
    }
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char buf[4096] = {};
        glGetProgramInfoLog(p, sizeof buf, nullptr, buf);
        fprintf(stderr, "  link failed: %s\n", buf);
    }
    return p;
}

// Task workgroup i launches 2 mesh workgroups; mesh workgroup (i, j) draws a quad (two triangles) into
// cell (2i + j) of a 4x1 grid. The second triangle of cell 3 is culled; per-vertex colors are smooth, the
// blue channel is per primitive.
static const char* kTask = R"(#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 4) in;
struct P { uint base; uint cells[2]; };
taskPayloadSharedEXT P pl;
void main() {
    if (gl_LocalInvocationIndex == 0u) pl.base = gl_WorkGroupID.x * 2u;
    if (gl_LocalInvocationIndex < 2u) pl.cells[gl_LocalInvocationIndex] = gl_WorkGroupID.x * 2u + gl_LocalInvocationIndex;
    barrier();
    EmitMeshTasksEXT(2u, 1u, 1u);
}
)";
static const char* kMesh = R"(#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 4) in;
layout(triangles, max_vertices = 4, max_primitives = 2) out;
struct P { uint base; uint cells[2]; };
taskPayloadSharedEXT P pl;
layout(location = 0) out vec4 color[];
layout(location = 1) perprimitiveEXT out float blue[];
void main() {
    uint cell = pl.cells[gl_WorkGroupID.x];
    SetMeshOutputsEXT(4u, 2u);
    uint v = gl_LocalInvocationIndex;
    float x0 = -1.0 + 0.5 * float(cell), x1 = x0 + 0.5;
    vec2 pos = vec2((v & 1u) != 0u ? x1 : x0, (v & 2u) != 0u ? 1.0 : -1.0);
    gl_MeshVerticesEXT[v].gl_Position = vec4(pos, 0.0, 1.0);
    color[v] = vec4(1.0, 0.0, 0.0, 1.0);
    if (v < 2u) {
        gl_PrimitiveTriangleIndicesEXT[v] = v == 0u ? uvec3(0, 1, 2) : uvec3(1, 3, 2);
        blue[v] = float(pl.base + gl_WorkGroupID.x) / 4.0;
        gl_MeshPrimitivesEXT[v].gl_CullPrimitiveEXT = cell == 3u && v == 1u;
    }
}
)";
static const char* kFrag = R"(#version 460
#extension GL_EXT_mesh_shader : require
layout(location = 0) in vec4 color;
layout(location = 1) perprimitiveEXT in float blue;
out vec4 o;
void main() { o = vec4(color.rgb, 1.0) + vec4(0.0, 0.0, blue, 0.0); }
)";

TEST(mesh_task_payload_cull) {
    GLuint p = meshProgram(kTask, kMesh, kFrag);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    CHECK(ok);
    GLint ls[3] = {};
    glGetProgramiv(p, GL_MESH_WORK_GROUP_SIZE_EXT, ls);
    CHECK_EQ(ls[0], 4);
    GLint n = 0;
    glGetProgramiv(p, GL_MESH_VERTICES_OUT_EXT, &n);
    CHECK_EQ(n, 4);
    glGetProgramiv(p, GL_MESH_OUTPUT_TYPE_EXT, &n);
    CHECK_EQ(n, GL_TRIANGLES);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(p);
    glDrawMeshTasksEXT(2, 1, 1);
    CHECK_GL();
    // cells are 16 pixels wide; blue = cell / 4
    CHECK_PIXEL(4, 4, 255, 0, 0, 255);
    CHECK_PIXEL(20, 4, 255, 0, 64, 255);
    CHECK_PIXEL(36, 60, 255, 0, 128, 255);
    CHECK_PIXEL(52, 4, 255, 0, 191, 255);   // first triangle of cell 3 (lower left)
    CHECK_PIXEL(60, 60, 0, 0, 0, 255);      // culled triangle (upper right)
    // a regular draw with a mesh program is an error
    glDrawArrays(GL_TRIANGLES, 0, 3);
    CHECK_EQ(glGetError(), (GLenum)GL_INVALID_OPERATION);
    glDeleteProgram(p);
}

static const char* kMeshOnly = R"(#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 1) in;
layout(points, max_vertices = 1, max_primitives = 1) out;
layout(location = 0) out flat uint id[];
void main() {
    SetMeshOutputsEXT(1u, 1u);
    gl_MeshVerticesEXT[0].gl_Position = vec4(-0.9 + 0.5 * float(gl_DrawID) + 0.1 * float(gl_WorkGroupID.x), 0.03, 0.0, 1.0);
    gl_MeshVerticesEXT[0].gl_PointSize = 1.0;
    gl_PrimitivePointIndicesEXT[0] = 0u;
    id[0] = uint(gl_DrawID) + 1u;
}
)";
static const char* kFragId = R"(#version 460
layout(location = 0) flat in uint id;
out vec4 o;
void main() { o = vec4(float(id) / 4.0, 1.0, 0.0, 1.0); }
)";

TEST(mesh_multi_draw_indirect_count) {
    GLuint p = meshProgram(nullptr, kMeshOnly, kFragId);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    CHECK(ok);
    // three commands of {x, y, z}, stride 16; the count buffer limits the draw to two
    GLuint cmds[12] = {1, 1, 1, 0, 2, 1, 1, 0, 1, 1, 1, 0};
    GLuint count = 2;
    GLuint bufs[2];
    glGenBuffers(2, bufs);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, bufs[0]);
    glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof cmds, cmds, GL_STATIC_DRAW);
    glBindBuffer(GL_PARAMETER_BUFFER, bufs[1]);
    glBufferData(GL_PARAMETER_BUFFER, 4, &count, GL_STATIC_DRAW);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(p);
    glMultiDrawMeshTasksIndirectCountEXT(0, 0, 3, 16);
    CHECK_GL();
    // point x = -0.9 + 0.5 * drawID + 0.1 * workgroup -> pixel (x + 1) * 32
    CHECK_PIXEL(3, 32, 64, 255, 0, 255);    // draw 0
    CHECK_PIXEL(19, 32, 128, 255, 0, 255);  // draw 1, workgroup 0
    CHECK_PIXEL(22, 32, 128, 255, 0, 255);  // draw 1, workgroup 1
    CHECK_PIXEL(35, 32, 0, 0, 0, 255);      // draw 2 is past the count
    glMultiDrawMeshTasksIndirectEXT(0, 0, 16);
    CHECK_EQ(glGetError(), (GLenum)GL_INVALID_VALUE);
    glDeleteBuffers(2, bufs);
    glDeleteProgram(p);
}
