#version 120
uniform sampler2D texture;
uniform sampler2D lightmap;
uniform sampler2DShadow shadow;
varying vec2 texcoord;
varying vec4 glcolor;
varying vec3 normal;
void main() {
    vec4 color = texture2D(texture, texcoord) * glcolor * gl_Color;
    color *= texture2DLod(lightmap, gl_TexCoord[1].st, 0.0);
    color.rgb *= shadow2D(shadow, vec3(texcoord, 0.5)).r;
    float fog = clamp((gl_FogFragCoord - gl_Fog.start) * gl_Fog.scale, 0.0, 1.0);
    color.rgb = mix(color.rgb, gl_Fog.color.rgb, fog);
/* DRAWBUFFERS:01 */
    gl_FragData[0] = color;
    gl_FragData[1] = vec4(normal * 0.5 + 0.5, 1.0);
}
