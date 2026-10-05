#version 120
varying vec2 texcoord;
varying vec4 glcolor;
varying vec3 normal;
uniform mat4 gbufferModelViewInverse;
attribute vec4 mc_Entity;
void main() {
    gl_Position = ftransform();
    texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    gl_TexCoord[1] = gl_MultiTexCoord1;
    glcolor = gl_Color;
    normal = normalize(gl_NormalMatrix * gl_Normal);
    gl_FrontColor = gl_Color * gl_LightSource[0].diffuse;
    gl_FogFragCoord = length((gl_ModelViewMatrix * gl_Vertex).xyz);
    if (mc_Entity.x == 10001.0) gl_Position.z -= 0.001;
}
