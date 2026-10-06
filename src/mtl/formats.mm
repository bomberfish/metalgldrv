#import "mtl/formats.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace mgl {

#define SWZ(r, g, b, a) MTLTextureSwizzleChannelsMake(MTLTextureSwizzle##r, MTLTextureSwizzle##g, MTLTextureSwizzle##b, MTLTextureSwizzle##a)
#define IDENT SWZ(Red, Green, Blue, Alpha)
#define RGB1 SWZ(Red, Green, Blue, One)
#define RG01 SWZ(Red, Green, Zero, One)
#define R001 SWZ(Red, Zero, Zero, One)

static const GLenum UN = GL_UNSIGNED_NORMALIZED, SN = GL_SIGNED_NORMALIZED, FL = GL_FLOAT, UI = GL_UNSIGNED_INT, SI = GL_INT;

// clang-format off
static const FormatInfo kFormats[] = {
// internal                 mtl                                    base                   kind              comp        ch bpp bw bh srgb rend filt swizzle  order       r  g  b  a  d  s  type
{GL_R8,                     MTLPixelFormatR8Unorm,                 GL_RED,                FmtKind::Unorm,   Comp::U8,    1, 1, 1, 1, 0, 1, 1, R001, {0,1,2,3}, 8, 0, 0, 0, 0, 0, UN},
{GL_R8_SNORM,               MTLPixelFormatR8Snorm,                 GL_RED,                FmtKind::Snorm,   Comp::S8,    1, 1, 1, 1, 0, 1, 1, R001, {0,1,2,3}, 8, 0, 0, 0, 0, 0, SN},
{GL_R16,                    MTLPixelFormatR16Unorm,                GL_RED,                FmtKind::Unorm,   Comp::U16,   1, 2, 1, 1, 0, 1, 1, R001, {0,1,2,3},16, 0, 0, 0, 0, 0, UN},
{GL_R16_SNORM,              MTLPixelFormatR16Snorm,                GL_RED,                FmtKind::Snorm,   Comp::S16,   1, 2, 1, 1, 0, 1, 1, R001, {0,1,2,3},16, 0, 0, 0, 0, 0, SN},
{GL_RG8,                    MTLPixelFormatRG8Unorm,                GL_RG,                 FmtKind::Unorm,   Comp::U8,    2, 2, 1, 1, 0, 1, 1, RG01, {0,1,2,3}, 8, 8, 0, 0, 0, 0, UN},
{GL_RG8_SNORM,              MTLPixelFormatRG8Snorm,                GL_RG,                 FmtKind::Snorm,   Comp::S8,    2, 2, 1, 1, 0, 1, 1, RG01, {0,1,2,3}, 8, 8, 0, 0, 0, 0, SN},
{GL_RG16,                   MTLPixelFormatRG16Unorm,               GL_RG,                 FmtKind::Unorm,   Comp::U16,   2, 4, 1, 1, 0, 1, 1, RG01, {0,1,2,3},16,16, 0, 0, 0, 0, UN},
{GL_RG16_SNORM,             MTLPixelFormatRG16Snorm,               GL_RG,                 FmtKind::Snorm,   Comp::S16,   2, 4, 1, 1, 0, 1, 1, RG01, {0,1,2,3},16,16, 0, 0, 0, 0, SN},
{GL_R3_G3_B2,               MTLPixelFormatRGBA8Unorm,              GL_RGB,                FmtKind::Unorm,   Comp::U8,    4, 4, 1, 1, 0, 1, 1, RGB1, {0,1,2,3}, 8, 8, 8, 0, 0, 0, UN},
{GL_RGB4,                   MTLPixelFormatRGBA8Unorm,              GL_RGB,                FmtKind::Unorm,   Comp::U8,    4, 4, 1, 1, 0, 1, 1, RGB1, {0,1,2,3}, 8, 8, 8, 0, 0, 0, UN},
{GL_RGB5,                   MTLPixelFormatRGBA8Unorm,              GL_RGB,                FmtKind::Unorm,   Comp::U8,    4, 4, 1, 1, 0, 1, 1, RGB1, {0,1,2,3}, 8, 8, 8, 0, 0, 0, UN},
{GL_RGB565,                 MTLPixelFormatRGBA8Unorm,              GL_RGB,                FmtKind::Unorm,   Comp::U8,    4, 4, 1, 1, 0, 1, 1, RGB1, {0,1,2,3}, 8, 8, 8, 0, 0, 0, UN},
{GL_RGB8,                   MTLPixelFormatRGBA8Unorm,              GL_RGB,                FmtKind::Unorm,   Comp::U8,    4, 4, 1, 1, 0, 1, 1, RGB1, {0,1,2,3}, 8, 8, 8, 0, 0, 0, UN},
{GL_RGB8_SNORM,             MTLPixelFormatRGBA8Snorm,              GL_RGB,                FmtKind::Snorm,   Comp::S8,    4, 4, 1, 1, 0, 1, 1, RGB1, {0,1,2,3}, 8, 8, 8, 0, 0, 0, SN},
{GL_RGB10,                  MTLPixelFormatRGB10A2Unorm,            GL_RGB,                FmtKind::Unorm,   Comp::P1010102,4,4,1, 1, 0, 1, 1, RGB1, {0,1,2,3},10,10,10, 0, 0, 0, UN},
{GL_RGB12,                  MTLPixelFormatRGBA16Unorm,             GL_RGB,                FmtKind::Unorm,   Comp::U16,   4, 8, 1, 1, 0, 1, 1, RGB1, {0,1,2,3},16,16,16, 0, 0, 0, UN},
{GL_RGB16,                  MTLPixelFormatRGBA16Unorm,             GL_RGB,                FmtKind::Unorm,   Comp::U16,   4, 8, 1, 1, 0, 1, 1, RGB1, {0,1,2,3},16,16,16, 0, 0, 0, UN},
{GL_RGB16_SNORM,            MTLPixelFormatRGBA16Snorm,             GL_RGB,                FmtKind::Snorm,   Comp::S16,   4, 8, 1, 1, 0, 1, 1, RGB1, {0,1,2,3},16,16,16, 0, 0, 0, SN},
{GL_RGBA2,                  MTLPixelFormatRGBA8Unorm,              GL_RGBA,               FmtKind::Unorm,   Comp::U8,    4, 4, 1, 1, 0, 1, 1, IDENT,{0,1,2,3}, 8, 8, 8, 8, 0, 0, UN},
{GL_RGBA4,                  MTLPixelFormatRGBA8Unorm,              GL_RGBA,               FmtKind::Unorm,   Comp::U8,    4, 4, 1, 1, 0, 1, 1, IDENT,{0,1,2,3}, 8, 8, 8, 8, 0, 0, UN},
{GL_RGB5_A1,                MTLPixelFormatRGBA8Unorm,              GL_RGBA,               FmtKind::Unorm,   Comp::U8,    4, 4, 1, 1, 0, 1, 1, IDENT,{0,1,2,3}, 8, 8, 8, 8, 0, 0, UN},
{GL_RGBA8,                  MTLPixelFormatRGBA8Unorm,              GL_RGBA,               FmtKind::Unorm,   Comp::U8,    4, 4, 1, 1, 0, 1, 1, IDENT,{0,1,2,3}, 8, 8, 8, 8, 0, 0, UN},
{GL_RGBA8_SNORM,            MTLPixelFormatRGBA8Snorm,              GL_RGBA,               FmtKind::Snorm,   Comp::S8,    4, 4, 1, 1, 0, 1, 1, IDENT,{0,1,2,3}, 8, 8, 8, 8, 0, 0, SN},
{GL_RGB10_A2,               MTLPixelFormatRGB10A2Unorm,            GL_RGBA,               FmtKind::Unorm,   Comp::P1010102,4,4,1, 1, 0, 1, 1, IDENT,{0,1,2,3},10,10,10, 2, 0, 0, UN},
{GL_RGB10_A2UI,             MTLPixelFormatRGB10A2Uint,             GL_RGBA,               FmtKind::UInt,    Comp::P1010102,4,4,1, 1, 0, 1, 0, IDENT,{0,1,2,3},10,10,10, 2, 0, 0, UI},
{GL_RGBA12,                 MTLPixelFormatRGBA16Unorm,             GL_RGBA,               FmtKind::Unorm,   Comp::U16,   4, 8, 1, 1, 0, 1, 1, IDENT,{0,1,2,3},16,16,16,16, 0, 0, UN},
{GL_RGBA16,                 MTLPixelFormatRGBA16Unorm,             GL_RGBA,               FmtKind::Unorm,   Comp::U16,   4, 8, 1, 1, 0, 1, 1, IDENT,{0,1,2,3},16,16,16,16, 0, 0, UN},
{GL_RGBA16_SNORM,           MTLPixelFormatRGBA16Snorm,             GL_RGBA,               FmtKind::Snorm,   Comp::S16,   4, 8, 1, 1, 0, 1, 1, IDENT,{0,1,2,3},16,16,16,16, 0, 0, SN},
{GL_SRGB8,                  MTLPixelFormatRGBA8Unorm_sRGB,         GL_RGB,                FmtKind::Unorm,   Comp::U8,    4, 4, 1, 1, 1, 1, 1, RGB1, {0,1,2,3}, 8, 8, 8, 0, 0, 0, UN},
{GL_SRGB8_ALPHA8,           MTLPixelFormatRGBA8Unorm_sRGB,         GL_RGBA,               FmtKind::Unorm,   Comp::U8,    4, 4, 1, 1, 1, 1, 1, IDENT,{0,1,2,3}, 8, 8, 8, 8, 0, 0, UN},
{GL_R16F,                   MTLPixelFormatR16Float,                GL_RED,                FmtKind::Float,   Comp::F16,   1, 2, 1, 1, 0, 1, 1, R001, {0,1,2,3},16, 0, 0, 0, 0, 0, FL},
{GL_RG16F,                  MTLPixelFormatRG16Float,               GL_RG,                 FmtKind::Float,   Comp::F16,   2, 4, 1, 1, 0, 1, 1, RG01, {0,1,2,3},16,16, 0, 0, 0, 0, FL},
{GL_RGB16F,                 MTLPixelFormatRGBA16Float,             GL_RGB,                FmtKind::Float,   Comp::F16,   4, 8, 1, 1, 0, 1, 1, RGB1, {0,1,2,3},16,16,16, 0, 0, 0, FL},
{GL_RGBA16F,                MTLPixelFormatRGBA16Float,             GL_RGBA,               FmtKind::Float,   Comp::F16,   4, 8, 1, 1, 0, 1, 1, IDENT,{0,1,2,3},16,16,16,16, 0, 0, FL},
{GL_R32F,                   MTLPixelFormatR32Float,                GL_RED,                FmtKind::Float,   Comp::F32,   1, 4, 1, 1, 0, 1, 1, R001, {0,1,2,3},32, 0, 0, 0, 0, 0, FL},
{GL_RG32F,                  MTLPixelFormatRG32Float,               GL_RG,                 FmtKind::Float,   Comp::F32,   2, 8, 1, 1, 0, 1, 1, RG01, {0,1,2,3},32,32, 0, 0, 0, 0, FL},
{GL_RGB32F,                 MTLPixelFormatRGBA32Float,             GL_RGB,                FmtKind::Float,   Comp::F32,   4,16, 1, 1, 0, 1, 1, RGB1, {0,1,2,3},32,32,32, 0, 0, 0, FL},
{GL_RGBA32F,                MTLPixelFormatRGBA32Float,             GL_RGBA,               FmtKind::Float,   Comp::F32,   4,16, 1, 1, 0, 1, 1, IDENT,{0,1,2,3},32,32,32,32, 0, 0, FL},
{GL_R11F_G11F_B10F,         MTLPixelFormatRG11B10Float,            GL_RGB,                FmtKind::Float,   Comp::P111110F,3,4,1, 1, 0, 1, 1, RGB1, {0,1,2,3},11,11,10, 0, 0, 0, FL},
{GL_RGB9_E5,                MTLPixelFormatRGB9E5Float,             GL_RGB,                FmtKind::Float,   Comp::P999E5,3, 4, 1, 1, 0, 0, 1, RGB1, {0,1,2,3}, 9, 9, 9, 0, 0, 0, FL},
{GL_R8I,                    MTLPixelFormatR8Sint,                  GL_RED,                FmtKind::SInt,    Comp::S8,    1, 1, 1, 1, 0, 1, 0, R001, {0,1,2,3}, 8, 0, 0, 0, 0, 0, SI},
{GL_R8UI,                   MTLPixelFormatR8Uint,                  GL_RED,                FmtKind::UInt,    Comp::U8,    1, 1, 1, 1, 0, 1, 0, R001, {0,1,2,3}, 8, 0, 0, 0, 0, 0, UI},
{GL_R16I,                   MTLPixelFormatR16Sint,                 GL_RED,                FmtKind::SInt,    Comp::S16,   1, 2, 1, 1, 0, 1, 0, R001, {0,1,2,3},16, 0, 0, 0, 0, 0, SI},
{GL_R16UI,                  MTLPixelFormatR16Uint,                 GL_RED,                FmtKind::UInt,    Comp::U16,   1, 2, 1, 1, 0, 1, 0, R001, {0,1,2,3},16, 0, 0, 0, 0, 0, UI},
{GL_R32I,                   MTLPixelFormatR32Sint,                 GL_RED,                FmtKind::SInt,    Comp::S32,   1, 4, 1, 1, 0, 1, 0, R001, {0,1,2,3},32, 0, 0, 0, 0, 0, SI},
{GL_R32UI,                  MTLPixelFormatR32Uint,                 GL_RED,                FmtKind::UInt,    Comp::U32,   1, 4, 1, 1, 0, 1, 0, R001, {0,1,2,3},32, 0, 0, 0, 0, 0, UI},
{GL_RG8I,                   MTLPixelFormatRG8Sint,                 GL_RG,                 FmtKind::SInt,    Comp::S8,    2, 2, 1, 1, 0, 1, 0, RG01, {0,1,2,3}, 8, 8, 0, 0, 0, 0, SI},
{GL_RG8UI,                  MTLPixelFormatRG8Uint,                 GL_RG,                 FmtKind::UInt,    Comp::U8,    2, 2, 1, 1, 0, 1, 0, RG01, {0,1,2,3}, 8, 8, 0, 0, 0, 0, UI},
{GL_RG16I,                  MTLPixelFormatRG16Sint,                GL_RG,                 FmtKind::SInt,    Comp::S16,   2, 4, 1, 1, 0, 1, 0, RG01, {0,1,2,3},16,16, 0, 0, 0, 0, SI},
{GL_RG16UI,                 MTLPixelFormatRG16Uint,                GL_RG,                 FmtKind::UInt,    Comp::U16,   2, 4, 1, 1, 0, 1, 0, RG01, {0,1,2,3},16,16, 0, 0, 0, 0, UI},
{GL_RG32I,                  MTLPixelFormatRG32Sint,                GL_RG,                 FmtKind::SInt,    Comp::S32,   2, 8, 1, 1, 0, 1, 0, RG01, {0,1,2,3},32,32, 0, 0, 0, 0, SI},
{GL_RG32UI,                 MTLPixelFormatRG32Uint,                GL_RG,                 FmtKind::UInt,    Comp::U32,   2, 8, 1, 1, 0, 1, 0, RG01, {0,1,2,3},32,32, 0, 0, 0, 0, UI},
{GL_RGB8I,                  MTLPixelFormatRGBA8Sint,               GL_RGB,                FmtKind::SInt,    Comp::S8,    4, 4, 1, 1, 0, 1, 0, RGB1, {0,1,2,3}, 8, 8, 8, 0, 0, 0, SI},
{GL_RGB8UI,                 MTLPixelFormatRGBA8Uint,               GL_RGB,                FmtKind::UInt,    Comp::U8,    4, 4, 1, 1, 0, 1, 0, RGB1, {0,1,2,3}, 8, 8, 8, 0, 0, 0, UI},
{GL_RGB16I,                 MTLPixelFormatRGBA16Sint,              GL_RGB,                FmtKind::SInt,    Comp::S16,   4, 8, 1, 1, 0, 1, 0, RGB1, {0,1,2,3},16,16,16, 0, 0, 0, SI},
{GL_RGB16UI,                MTLPixelFormatRGBA16Uint,              GL_RGB,                FmtKind::UInt,    Comp::U16,   4, 8, 1, 1, 0, 1, 0, RGB1, {0,1,2,3},16,16,16, 0, 0, 0, UI},
{GL_RGB32I,                 MTLPixelFormatRGBA32Sint,              GL_RGB,                FmtKind::SInt,    Comp::S32,   4,16, 1, 1, 0, 1, 0, RGB1, {0,1,2,3},32,32,32, 0, 0, 0, SI},
{GL_RGB32UI,                MTLPixelFormatRGBA32Uint,              GL_RGB,                FmtKind::UInt,    Comp::U32,   4,16, 1, 1, 0, 1, 0, RGB1, {0,1,2,3},32,32,32, 0, 0, 0, UI},
{GL_RGBA8I,                 MTLPixelFormatRGBA8Sint,               GL_RGBA,               FmtKind::SInt,    Comp::S8,    4, 4, 1, 1, 0, 1, 0, IDENT,{0,1,2,3}, 8, 8, 8, 8, 0, 0, SI},
{GL_RGBA8UI,                MTLPixelFormatRGBA8Uint,               GL_RGBA,               FmtKind::UInt,    Comp::U8,    4, 4, 1, 1, 0, 1, 0, IDENT,{0,1,2,3}, 8, 8, 8, 8, 0, 0, UI},
{GL_RGBA16I,                MTLPixelFormatRGBA16Sint,              GL_RGBA,               FmtKind::SInt,    Comp::S16,   4, 8, 1, 1, 0, 1, 0, IDENT,{0,1,2,3},16,16,16,16, 0, 0, SI},
{GL_RGBA16UI,               MTLPixelFormatRGBA16Uint,              GL_RGBA,               FmtKind::UInt,    Comp::U16,   4, 8, 1, 1, 0, 1, 0, IDENT,{0,1,2,3},16,16,16,16, 0, 0, UI},
{GL_RGBA32I,                MTLPixelFormatRGBA32Sint,              GL_RGBA,               FmtKind::SInt,    Comp::S32,   4,16, 1, 1, 0, 1, 0, IDENT,{0,1,2,3},32,32,32,32, 0, 0, SI},
{GL_RGBA32UI,               MTLPixelFormatRGBA32Uint,              GL_RGBA,               FmtKind::UInt,    Comp::U32,   4,16, 1, 1, 0, 1, 0, IDENT,{0,1,2,3},32,32,32,32, 0, 0, UI},
// depth / stencil
{GL_DEPTH_COMPONENT16,      MTLPixelFormatDepth16Unorm,            GL_DEPTH_COMPONENT,    FmtKind::Depth,   Comp::D16,   1, 2, 1, 1, 0, 1, 1, R001, {0,1,2,3}, 0, 0, 0, 0,16, 0, UN},
{GL_DEPTH_COMPONENT24,      MTLPixelFormatDepth32Float,            GL_DEPTH_COMPONENT,    FmtKind::Depth,   Comp::D32F,  1, 4, 1, 1, 0, 1, 1, R001, {0,1,2,3}, 0, 0, 0, 0,24, 0, UN},
{GL_DEPTH_COMPONENT32,      MTLPixelFormatDepth32Float,            GL_DEPTH_COMPONENT,    FmtKind::Depth,   Comp::D32F,  1, 4, 1, 1, 0, 1, 1, R001, {0,1,2,3}, 0, 0, 0, 0,32, 0, UN},
{GL_DEPTH_COMPONENT32F,     MTLPixelFormatDepth32Float,            GL_DEPTH_COMPONENT,    FmtKind::Depth,   Comp::D32F,  1, 4, 1, 1, 0, 1, 1, R001, {0,1,2,3}, 0, 0, 0, 0,32, 0, FL},
{GL_DEPTH24_STENCIL8,       MTLPixelFormatDepth32Float_Stencil8,   GL_DEPTH_STENCIL,      FmtKind::DepthStencil, Comp::D32F_S8, 2, 8, 1, 1, 0, 1, 1, R001, {0,1,2,3}, 0, 0, 0, 0,24, 8, UN},
{GL_DEPTH32F_STENCIL8,      MTLPixelFormatDepth32Float_Stencil8,   GL_DEPTH_STENCIL,      FmtKind::DepthStencil, Comp::D32F_S8, 2, 8, 1, 1, 0, 1, 1, R001, {0,1,2,3}, 0, 0, 0, 0,32, 8, FL},
{GL_STENCIL_INDEX8,         MTLPixelFormatStencil8,                GL_STENCIL_INDEX,      FmtKind::Stencil, Comp::S8_ONLY,1,1, 1, 1, 0, 1, 0, R001, {0,1,2,3}, 0, 0, 0, 0, 0, 8, UI},
// legacy luminance / alpha / intensity (compatibility profile)
{GL_ALPHA8,                 MTLPixelFormatR8Unorm,                 GL_ALPHA,              FmtKind::Unorm,   Comp::U8,    1, 1, 1, 1, 0, 1, 1, SWZ(Zero, Zero, Zero, Red),   {3,0,1,2}, 0, 0, 0, 8, 0, 0, UN},
{GL_ALPHA16,                MTLPixelFormatR16Unorm,                GL_ALPHA,              FmtKind::Unorm,   Comp::U16,   1, 2, 1, 1, 0, 1, 1, SWZ(Zero, Zero, Zero, Red),   {3,0,1,2}, 0, 0, 0,16, 0, 0, UN},
{GL_LUMINANCE8,             MTLPixelFormatR8Unorm,                 GL_LUMINANCE,          FmtKind::Unorm,   Comp::U8,    1, 1, 1, 1, 0, 1, 1, SWZ(Red, Red, Red, One),      {0,1,2,3}, 8, 8, 8, 0, 0, 0, UN},
{GL_LUMINANCE16,            MTLPixelFormatR16Unorm,                GL_LUMINANCE,          FmtKind::Unorm,   Comp::U16,   1, 2, 1, 1, 0, 1, 1, SWZ(Red, Red, Red, One),      {0,1,2,3},16,16,16, 0, 0, 0, UN},
{GL_LUMINANCE8_ALPHA8,      MTLPixelFormatRG8Unorm,                GL_LUMINANCE_ALPHA,    FmtKind::Unorm,   Comp::U8,    2, 2, 1, 1, 0, 1, 1, SWZ(Red, Red, Red, Green),    {0,3,1,2}, 8, 8, 8, 8, 0, 0, UN},
{GL_LUMINANCE16_ALPHA16,    MTLPixelFormatRG16Unorm,               GL_LUMINANCE_ALPHA,    FmtKind::Unorm,   Comp::U16,   2, 4, 1, 1, 0, 1, 1, SWZ(Red, Red, Red, Green),    {0,3,1,2},16,16,16,16, 0, 0, UN},
{GL_INTENSITY8,             MTLPixelFormatR8Unorm,                 GL_INTENSITY,          FmtKind::Unorm,   Comp::U8,    1, 1, 1, 1, 0, 1, 1, SWZ(Red, Red, Red, Red),      {0,1,2,3}, 8, 8, 8, 8, 0, 0, UN},
{GL_INTENSITY16,            MTLPixelFormatR16Unorm,                GL_INTENSITY,          FmtKind::Unorm,   Comp::U16,   1, 2, 1, 1, 0, 1, 1, SWZ(Red, Red, Red, Red),      {0,1,2,3},16,16,16,16, 0, 0, UN},
// compressed
{GL_COMPRESSED_RGB_S3TC_DXT1_EXT,       MTLPixelFormatBC1_RGBA,         GL_RGB,  FmtKind::Unorm, Comp::Compressed, 4,  8, 4, 4, 0, 0, 1, RGB1, {0,1,2,3}, 0,0,0,0,0,0, UN},
{GL_COMPRESSED_RGBA_S3TC_DXT1_EXT,      MTLPixelFormatBC1_RGBA,         GL_RGBA, FmtKind::Unorm, Comp::Compressed, 4,  8, 4, 4, 0, 0, 1, IDENT,{0,1,2,3}, 0,0,0,0,0,0, UN},
{GL_COMPRESSED_RGBA_S3TC_DXT3_EXT,      MTLPixelFormatBC2_RGBA,         GL_RGBA, FmtKind::Unorm, Comp::Compressed, 4, 16, 4, 4, 0, 0, 1, IDENT,{0,1,2,3}, 0,0,0,0,0,0, UN},
{GL_COMPRESSED_RGBA_S3TC_DXT5_EXT,      MTLPixelFormatBC3_RGBA,         GL_RGBA, FmtKind::Unorm, Comp::Compressed, 4, 16, 4, 4, 0, 0, 1, IDENT,{0,1,2,3}, 0,0,0,0,0,0, UN},
{GL_COMPRESSED_SRGB_S3TC_DXT1_EXT,      MTLPixelFormatBC1_RGBA_sRGB,    GL_RGB,  FmtKind::Unorm, Comp::Compressed, 4,  8, 4, 4, 1, 0, 1, RGB1, {0,1,2,3}, 0,0,0,0,0,0, UN},
{GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT,MTLPixelFormatBC1_RGBA_sRGB,    GL_RGBA, FmtKind::Unorm, Comp::Compressed, 4,  8, 4, 4, 1, 0, 1, IDENT,{0,1,2,3}, 0,0,0,0,0,0, UN},
{GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT,MTLPixelFormatBC2_RGBA_sRGB,    GL_RGBA, FmtKind::Unorm, Comp::Compressed, 4, 16, 4, 4, 1, 0, 1, IDENT,{0,1,2,3}, 0,0,0,0,0,0, UN},
{GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT,MTLPixelFormatBC3_RGBA_sRGB,    GL_RGBA, FmtKind::Unorm, Comp::Compressed, 4, 16, 4, 4, 1, 0, 1, IDENT,{0,1,2,3}, 0,0,0,0,0,0, UN},
{GL_COMPRESSED_RED_RGTC1,               MTLPixelFormatBC4_RUnorm,       GL_RED,  FmtKind::Unorm, Comp::Compressed, 1,  8, 4, 4, 0, 0, 1, R001, {0,1,2,3}, 0,0,0,0,0,0, UN},
{GL_COMPRESSED_SIGNED_RED_RGTC1,        MTLPixelFormatBC4_RSnorm,       GL_RED,  FmtKind::Snorm, Comp::Compressed, 1,  8, 4, 4, 0, 0, 1, R001, {0,1,2,3}, 0,0,0,0,0,0, SN},
{GL_COMPRESSED_RG_RGTC2,                MTLPixelFormatBC5_RGUnorm,      GL_RG,   FmtKind::Unorm, Comp::Compressed, 2, 16, 4, 4, 0, 0, 1, RG01, {0,1,2,3}, 0,0,0,0,0,0, UN},
{GL_COMPRESSED_SIGNED_RG_RGTC2,         MTLPixelFormatBC5_RGSnorm,      GL_RG,   FmtKind::Snorm, Comp::Compressed, 2, 16, 4, 4, 0, 0, 1, RG01, {0,1,2,3}, 0,0,0,0,0,0, SN},
{GL_COMPRESSED_RGBA_BPTC_UNORM,         MTLPixelFormatBC7_RGBAUnorm,    GL_RGBA, FmtKind::Unorm, Comp::Compressed, 4, 16, 4, 4, 0, 0, 1, IDENT,{0,1,2,3}, 0,0,0,0,0,0, UN},
{GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM,   MTLPixelFormatBC7_RGBAUnorm_sRGB,GL_RGBA,FmtKind::Unorm, Comp::Compressed, 4, 16, 4, 4, 1, 0, 1, IDENT,{0,1,2,3}, 0,0,0,0,0,0, UN},
{GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT,   MTLPixelFormatBC6H_RGBFloat,    GL_RGB,  FmtKind::Float, Comp::Compressed, 3, 16, 4, 4, 0, 0, 1, RGB1, {0,1,2,3}, 0,0,0,0,0,0, FL},
{GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT, MTLPixelFormatBC6H_RGBUfloat,   GL_RGB,  FmtKind::Float, Comp::Compressed, 3, 16, 4, 4, 0, 0, 1, RGB1, {0,1,2,3}, 0,0,0,0,0,0, FL},
{GL_COMPRESSED_RGB8_ETC2,               MTLPixelFormatETC2_RGB8,        GL_RGB,  FmtKind::Unorm, Comp::Compressed, 3,  8, 4, 4, 0, 0, 1, RGB1, {0,1,2,3}, 0,0,0,0,0,0, UN},
{GL_COMPRESSED_SRGB8_ETC2,              MTLPixelFormatETC2_RGB8_sRGB,   GL_RGB,  FmtKind::Unorm, Comp::Compressed, 3,  8, 4, 4, 1, 0, 1, RGB1, {0,1,2,3}, 0,0,0,0,0,0, UN},
{GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2, MTLPixelFormatETC2_RGB8A1, GL_RGBA, FmtKind::Unorm, Comp::Compressed, 4,  8, 4, 4, 0, 0, 1, IDENT,{0,1,2,3}, 0,0,0,0,0,0, UN},
{GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2, MTLPixelFormatETC2_RGB8A1_sRGB, GL_RGBA, FmtKind::Unorm, Comp::Compressed, 4, 8, 4, 4, 1, 0, 1, IDENT,{0,1,2,3}, 0,0,0,0,0,0, UN},
{GL_COMPRESSED_RGBA8_ETC2_EAC,          MTLPixelFormatEAC_RGBA8,        GL_RGBA, FmtKind::Unorm, Comp::Compressed, 4, 16, 4, 4, 0, 0, 1, IDENT,{0,1,2,3}, 0,0,0,0,0,0, UN},
{GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC,   MTLPixelFormatEAC_RGBA8_sRGB,   GL_RGBA, FmtKind::Unorm, Comp::Compressed, 4, 16, 4, 4, 1, 0, 1, IDENT,{0,1,2,3}, 0,0,0,0,0,0, UN},
{GL_COMPRESSED_R11_EAC,                 MTLPixelFormatEAC_R11Unorm,     GL_RED,  FmtKind::Unorm, Comp::Compressed, 1,  8, 4, 4, 0, 0, 1, R001, {0,1,2,3}, 0,0,0,0,0,0, UN},
{GL_COMPRESSED_SIGNED_R11_EAC,          MTLPixelFormatEAC_R11Snorm,     GL_RED,  FmtKind::Snorm, Comp::Compressed, 1,  8, 4, 4, 0, 0, 1, R001, {0,1,2,3}, 0,0,0,0,0,0, SN},
{GL_COMPRESSED_RG11_EAC,                MTLPixelFormatEAC_RG11Unorm,    GL_RG,   FmtKind::Unorm, Comp::Compressed, 2, 16, 4, 4, 0, 0, 1, RG01, {0,1,2,3}, 0,0,0,0,0,0, UN},
{GL_COMPRESSED_SIGNED_RG11_EAC,         MTLPixelFormatEAC_RG11Snorm,    GL_RG,   FmtKind::Snorm, Comp::Compressed, 2, 16, 4, 4, 0, 0, 1, RG01, {0,1,2,3}, 0,0,0,0,0,0, SN},
// KHR_texture_compression_astc_ldr
#define ASTC(w, h) \
{GL_COMPRESSED_RGBA_ASTC_##w##x##h##_KHR,         MTLPixelFormatASTC_##w##x##h##_LDR,  GL_RGBA, FmtKind::Unorm, Comp::Compressed, 4, 16, w, h, 0, 0, 1, IDENT,{0,1,2,3}, 0,0,0,0,0,0, UN}, \
{GL_COMPRESSED_SRGB8_ALPHA8_ASTC_##w##x##h##_KHR, MTLPixelFormatASTC_##w##x##h##_sRGB, GL_RGBA, FmtKind::Unorm, Comp::Compressed, 4, 16, w, h, 1, 0, 1, IDENT,{0,1,2,3}, 0,0,0,0,0,0, UN},
ASTC(4, 4) ASTC(5, 4) ASTC(5, 5) ASTC(6, 5) ASTC(6, 6) ASTC(8, 5) ASTC(8, 6) ASTC(8, 8)
ASTC(10, 5) ASTC(10, 6) ASTC(10, 8) ASTC(10, 10) ASTC(12, 10) ASTC(12, 12)
#undef ASTC
};
// clang-format on

static const std::unordered_map<GLenum, const FormatInfo*>& formatMap() {
    static std::unordered_map<GLenum, const FormatInfo*> m = [] {
        std::unordered_map<GLenum, const FormatInfo*> r;
        for (auto& f : kFormats) r[f.internalFormat] = &f;
        return r;
    }();
    return m;
}

const FormatInfo* formatInfo(GLenum internalFormat) {
    auto& m = formatMap();
    auto it = m.find(internalFormat);
    return it == m.end() ? nullptr : it->second;
}

const FormatInfo* formatInfoForMTL(MTLPixelFormat f) {
    for (auto& e : kFormats)
        if (e.mtl == f && e.swizzle.red == MTLTextureSwizzleRed && e.swizzle.alpha == MTLTextureSwizzleAlpha) return &e;
    for (auto& e : kFormats)
        if (e.mtl == f) return &e;
    return nullptr;
}

bool isCompressedFormat(GLenum internalFormat) {
    auto f = formatInfo(internalFormat);
    return f && f->comp == Comp::Compressed;
}

bool isDepthOrStencil(const FormatInfo* f) {
    return f && (f->kind == FmtKind::Depth || f->kind == FmtKind::Stencil || f->kind == FmtKind::DepthStencil);
}

int viewClass(GLenum internalFormat) {
    switch (internalFormat) {
    case GL_RGBA32F: case GL_RGBA32UI: case GL_RGBA32I: return 128;
    case GL_RGB32F: case GL_RGB32UI: case GL_RGB32I: return 96;
    case GL_RGBA16F: case GL_RG32F: case GL_RGBA16UI: case GL_RG32UI: case GL_RGBA16I: case GL_RG32I: case GL_RGBA16:
    case GL_RGBA16_SNORM: return 64;
    case GL_RGB16: case GL_RGB16_SNORM: case GL_RGB16F: case GL_RGB16UI: case GL_RGB16I: return 48;
    case GL_RG16F: case GL_R11F_G11F_B10F: case GL_R32F: case GL_RGB10_A2UI: case GL_RGBA8UI: case GL_RG16UI: case GL_R32UI:
    case GL_RGBA8I: case GL_RG16I: case GL_R32I: case GL_RGB10_A2: case GL_RGBA8: case GL_RG16: case GL_RGBA8_SNORM:
    case GL_RG16_SNORM: case GL_SRGB8_ALPHA8: case GL_RGB9_E5: return 32;
    case GL_RGB8: case GL_RGB8_SNORM: case GL_SRGB8: case GL_RGB8UI: case GL_RGB8I: return 24;
    case GL_R16F: case GL_RG8UI: case GL_R16UI: case GL_RG8I: case GL_R16I: case GL_RG8: case GL_R16: case GL_RG8_SNORM:
    case GL_R16_SNORM: return 16;
    case GL_R8UI: case GL_R8I: case GL_R8: case GL_R8_SNORM: return 8;
    case GL_COMPRESSED_RED_RGTC1: case GL_COMPRESSED_SIGNED_RED_RGTC1: return 1001;
    case GL_COMPRESSED_RG_RGTC2: case GL_COMPRESSED_SIGNED_RG_RGTC2: return 1002;
    case GL_COMPRESSED_RGBA_BPTC_UNORM: case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM: return 1003;
    case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT: case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT: return 1004;
    case GL_COMPRESSED_RGB_S3TC_DXT1_EXT: case GL_COMPRESSED_SRGB_S3TC_DXT1_EXT: return 1005;
    case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT: case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT: return 1006;
    case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT: case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT: return 1007;
    case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT: case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT: return 1008;
    case GL_COMPRESSED_R11_EAC: case GL_COMPRESSED_SIGNED_R11_EAC: return 1009;
    case GL_COMPRESSED_RG11_EAC: case GL_COMPRESSED_SIGNED_RG11_EAC: return 1010;
    case GL_COMPRESSED_RGB8_ETC2: case GL_COMPRESSED_SRGB8_ETC2: return 1011;
    case GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2: case GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2: return 1012;
    case GL_COMPRESSED_RGBA8_ETC2_EAC: case GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC: return 1013;
    default:
        // ASTC: one class per block size, LDR and sRGB together.
        if (internalFormat >= GL_COMPRESSED_RGBA_ASTC_4x4_KHR && internalFormat <= GL_COMPRESSED_RGBA_ASTC_12x12_KHR)
            return 1100 + (int)(internalFormat - GL_COMPRESSED_RGBA_ASTC_4x4_KHR);
        if (internalFormat >= GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR && internalFormat <= GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR)
            return 1100 + (int)(internalFormat - GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR);
        return 0;
    }
}

bool copyImageCompatible(const FormatInfo* a, const FormatInfo* b) {
    if (a->internalFormat == b->internalFormat) return true;
    int ca = viewClass(a->internalFormat), cb = viewClass(b->internalFormat);
    if (!ca || !cb) return false;
    if (ca == cb) return true;
    bool ac = a->comp == Comp::Compressed, bc = b->comp == Comp::Compressed;
    // §18.3.3 / Table 18.4: an uncompressed format whose texel size equals the block size.
    if (ac != bc) {
        int texelBits = ac ? cb : ca;
        int blockBytes = ac ? a->bytesPerPixel : b->bytesPerPixel;
        return texelBits < 1000 && texelBits == blockBytes * 8;
    }
    return false;
}

const FormatInfo* resolveInternalFormat(GLenum internalFormat, GLenum format, GLenum type) {
    // Unsized / generic internal formats: choose based on client type.
    bool isFloat = type == GL_FLOAT;
    bool isHalf = type == GL_HALF_FLOAT;
    bool isShortN = type == GL_UNSIGNED_SHORT || type == GL_SHORT;
    switch (internalFormat) {
    case 1: case GL_RED: case GL_COMPRESSED_RED:
        return formatInfo(isFloat ? GL_R32F : isHalf ? GL_R16F : isShortN ? GL_R16 : GL_R8);
    case 2: case GL_RG: case GL_COMPRESSED_RG:
        return formatInfo(isFloat ? GL_RG32F : isHalf ? GL_RG16F : isShortN ? GL_RG16 : GL_RG8);
    case 3: case GL_RGB: case GL_COMPRESSED_RGB:
        if (type == GL_UNSIGNED_INT_10F_11F_11F_REV) return formatInfo(GL_R11F_G11F_B10F);
        if (type == GL_UNSIGNED_INT_5_9_9_9_REV) return formatInfo(GL_RGB9_E5);
        return formatInfo(isFloat ? GL_RGB32F : isHalf ? GL_RGB16F : isShortN ? GL_RGB16 : GL_RGB8);
    case 4: case GL_RGBA: case GL_COMPRESSED_RGBA:
        if (type == GL_UNSIGNED_INT_2_10_10_10_REV) return formatInfo(GL_RGB10_A2);
        return formatInfo(isFloat ? GL_RGBA32F : isHalf ? GL_RGBA16F : isShortN ? GL_RGBA16 : GL_RGBA8);
    case GL_BGRA: return formatInfo(GL_RGBA8);
    case GL_SRGB: case GL_COMPRESSED_SRGB: return formatInfo(GL_SRGB8);
    case GL_SRGB_ALPHA: case GL_COMPRESSED_SRGB_ALPHA: return formatInfo(GL_SRGB8_ALPHA8);
    case GL_DEPTH_COMPONENT:
        return formatInfo(isFloat ? GL_DEPTH_COMPONENT32F : type == GL_UNSIGNED_SHORT ? GL_DEPTH_COMPONENT16 : GL_DEPTH_COMPONENT24);
    case GL_DEPTH_STENCIL:
        return formatInfo(type == GL_FLOAT_32_UNSIGNED_INT_24_8_REV ? GL_DEPTH32F_STENCIL8 : GL_DEPTH24_STENCIL8);
    case GL_STENCIL_INDEX: return formatInfo(GL_STENCIL_INDEX8);
    case GL_ALPHA: case GL_ALPHA4: case GL_ALPHA12: case GL_COMPRESSED_ALPHA: return formatInfo(GL_ALPHA8);
    case GL_LUMINANCE: case GL_LUMINANCE4: case GL_LUMINANCE12: case GL_COMPRESSED_LUMINANCE: case GL_SLUMINANCE: case GL_SLUMINANCE8:
        return formatInfo(GL_LUMINANCE8);
    case GL_LUMINANCE_ALPHA: case GL_LUMINANCE4_ALPHA4: case GL_LUMINANCE6_ALPHA2: case GL_LUMINANCE12_ALPHA4:
    case GL_LUMINANCE12_ALPHA12: case GL_COMPRESSED_LUMINANCE_ALPHA: case GL_SLUMINANCE_ALPHA: case GL_SLUMINANCE8_ALPHA8:
        return formatInfo(GL_LUMINANCE8_ALPHA8);
    case GL_INTENSITY: case GL_INTENSITY4: case GL_INTENSITY12: case GL_COMPRESSED_INTENSITY: return formatInfo(GL_INTENSITY8);
    case GL_DEPTH_COMPONENT24: case GL_DEPTH_COMPONENT32: return formatInfo(internalFormat);
    default: return formatInfo(internalFormat);
    }
}

// ---------------------------------------------------------------------------
// client formats
// ---------------------------------------------------------------------------
static int formatComponents(GLenum format) {
    switch (format) {
    case GL_RED: case GL_GREEN: case GL_BLUE: case GL_ALPHA: case GL_LUMINANCE: case GL_INTENSITY:
    case GL_RED_INTEGER: case GL_GREEN_INTEGER: case GL_BLUE_INTEGER: case GL_ALPHA_INTEGER:
    case GL_DEPTH_COMPONENT: case GL_STENCIL_INDEX: case GL_COLOR_INDEX:
        return 1;
    case GL_RG: case GL_RG_INTEGER: case GL_LUMINANCE_ALPHA: case GL_DEPTH_STENCIL: return 2;
    case GL_RGB: case GL_BGR: case GL_RGB_INTEGER: case GL_BGR_INTEGER: return 3;
    case GL_RGBA: case GL_BGRA: case GL_RGBA_INTEGER: case GL_BGRA_INTEGER: return 4;
    default: return 0;
    }
}

static int typeSize(GLenum type) {
    switch (type) {
    case GL_UNSIGNED_BYTE: case GL_BYTE: return 1;
    case GL_UNSIGNED_SHORT: case GL_SHORT: case GL_HALF_FLOAT: return 2;
    case GL_UNSIGNED_INT: case GL_INT: case GL_FLOAT: return 4;
    default: return 0;
    }
}

static int packedSize(GLenum type, int& comps) {
    switch (type) {
    case GL_UNSIGNED_BYTE_3_3_2: case GL_UNSIGNED_BYTE_2_3_3_REV: comps = 3; return 1;
    case GL_UNSIGNED_SHORT_5_6_5: case GL_UNSIGNED_SHORT_5_6_5_REV: comps = 3; return 2;
    case GL_UNSIGNED_SHORT_4_4_4_4: case GL_UNSIGNED_SHORT_4_4_4_4_REV:
    case GL_UNSIGNED_SHORT_5_5_5_1: case GL_UNSIGNED_SHORT_1_5_5_5_REV: comps = 4; return 2;
    case GL_UNSIGNED_INT_8_8_8_8: case GL_UNSIGNED_INT_8_8_8_8_REV:
    case GL_UNSIGNED_INT_10_10_10_2: case GL_UNSIGNED_INT_2_10_10_10_REV: comps = 4; return 4;
    case GL_UNSIGNED_INT_10F_11F_11F_REV: case GL_UNSIGNED_INT_5_9_9_9_REV: comps = 3; return 4;
    case GL_UNSIGNED_INT_24_8: comps = 2; return 4;
    case GL_FLOAT_32_UNSIGNED_INT_24_8_REV: comps = 2; return 8;
    default: comps = 0; return 0;
    }
}

bool isIntegerFormat(GLenum format) {
    switch (format) {
    case GL_RED_INTEGER: case GL_GREEN_INTEGER: case GL_BLUE_INTEGER: case GL_ALPHA_INTEGER:
    case GL_RG_INTEGER: case GL_RGB_INTEGER: case GL_BGR_INTEGER: case GL_RGBA_INTEGER: case GL_BGRA_INTEGER:
        return true;
    default: return false;
    }
}

// GL 4.6 §8.4.4 / Tables 8.7–8.8: INVALID_ENUM for unknown format or type enums,
// INVALID_OPERATION for a known pair that cannot be combined.
GLenum pixelFormatTypeError(GLenum format, GLenum type) {
    int n = formatComponents(format);
    int pc;
    bool packed = packedSize(type, pc) != 0;
    if (!n || (!packed && !typeSize(type))) return GL_INVALID_ENUM;
    bool integer = isIntegerFormat(format);
    switch (type) {
    case GL_UNSIGNED_BYTE_3_3_2: case GL_UNSIGNED_BYTE_2_3_3_REV:
    case GL_UNSIGNED_SHORT_5_6_5: case GL_UNSIGNED_SHORT_5_6_5_REV:
        return format == GL_RGB || format == GL_RGB_INTEGER ? 0 : GL_INVALID_OPERATION;
    case GL_UNSIGNED_SHORT_4_4_4_4: case GL_UNSIGNED_SHORT_4_4_4_4_REV:
    case GL_UNSIGNED_SHORT_5_5_5_1: case GL_UNSIGNED_SHORT_1_5_5_5_REV:
    case GL_UNSIGNED_INT_8_8_8_8: case GL_UNSIGNED_INT_8_8_8_8_REV:
    case GL_UNSIGNED_INT_10_10_10_2: case GL_UNSIGNED_INT_2_10_10_10_REV:
        return format == GL_RGBA || format == GL_BGRA || format == GL_RGBA_INTEGER || format == GL_BGRA_INTEGER ? 0
                                                                                                             : GL_INVALID_OPERATION;
    case GL_UNSIGNED_INT_10F_11F_11F_REV: case GL_UNSIGNED_INT_5_9_9_9_REV:
        return format == GL_RGB ? 0 : GL_INVALID_OPERATION;
    case GL_UNSIGNED_INT_24_8: case GL_FLOAT_32_UNSIGNED_INT_24_8_REV:
        return format == GL_DEPTH_STENCIL ? 0 : GL_INVALID_OPERATION;
    default: break;
    }
    if (format == GL_DEPTH_STENCIL) return GL_INVALID_OPERATION;  // needs a packed depth/stencil type
    if (integer && (type == GL_FLOAT || type == GL_HALF_FLOAT)) return GL_INVALID_OPERATION;
    return 0;
}

GLenum texImageFormatError(const FormatInfo* f, GLenum format, GLenum type, GLenum target) {
    if (GLenum e = pixelFormatTypeError(format, type)) return e;
    bool depth = f->baseFormat == GL_DEPTH_COMPONENT, ds = f->baseFormat == GL_DEPTH_STENCIL,
         stencil = f->baseFormat == GL_STENCIL_INDEX;
    switch (format) {
    case GL_DEPTH_COMPONENT: case GL_DEPTH_STENCIL:
        if (!depth && !ds) return GL_INVALID_OPERATION;
        break;
    case GL_STENCIL_INDEX:
        if (!stencil) return GL_INVALID_OPERATION;
        break;
    default:
        if (depth || ds || stencil) return GL_INVALID_OPERATION;
        break;
    }
    bool intTex = f->kind == FmtKind::UInt || f->kind == FmtKind::SInt;
    if (intTex != isIntegerFormat(format)) return GL_INVALID_OPERATION;
    if ((target == GL_TEXTURE_3D || target == GL_PROXY_TEXTURE_3D) && (depth || ds || stencil)) return GL_INVALID_OPERATION;
    return 0;
}

bool isValidFormatType(GLenum format, GLenum type) {
    int n = formatComponents(format);
    if (!n) return false;
    int pc;
    if (packedSize(type, pc)) {
        if (type == GL_UNSIGNED_INT_24_8 || type == GL_FLOAT_32_UNSIGNED_INT_24_8_REV) return format == GL_DEPTH_STENCIL;
        if (pc == 4) return n == 4;
        return n == 3 && format != GL_BGR && format != GL_BGR_INTEGER ? true : n == 3;
    }
    if (format == GL_DEPTH_STENCIL) return false;
    return typeSize(type) != 0;
}

int clientPixelSize(GLenum format, GLenum type) {
    int pc;
    int ps = packedSize(type, pc);
    if (ps) return ps;
    return formatComponents(format) * typeSize(type);
}

ClientLayout clientLayout(GLenum format, GLenum type, int width, int height, const PixelStore& ps) {
    ClientLayout l{};
    l.pixelSize = clientPixelSize(format, type);
    int rowPixels = ps.rowLength > 0 ? ps.rowLength : width;
    size_t rowBytes = (size_t)rowPixels * l.pixelSize;
    int pc;
    int elemSize = packedSize(type, pc) ? (int)l.pixelSize : std::max(1, typeSize(type));
    size_t align = ps.alignment;
    if (elemSize < (int)align) rowBytes = (rowBytes + align - 1) / align * align;
    l.rowStride = rowBytes;
    int imgRows = ps.imageHeight > 0 ? ps.imageHeight : height;
    l.imageStride = rowBytes * imgRows;
    l.skipBytes = (size_t)ps.skipImages * l.imageStride + (size_t)ps.skipRows * rowBytes + (size_t)ps.skipPixels * l.pixelSize;
    return l;
}

size_t clientImageSize(GLenum format, GLenum type, int width, int height, int depth, const PixelStore& ps) {
    if (width <= 0 || height <= 0 || depth <= 0) return 0;
    ClientLayout l = clientLayout(format, type, width, height, ps);
    return l.skipBytes + (size_t)(depth - 1) * l.imageStride + (size_t)(height - 1) * l.rowStride + (size_t)width * l.pixelSize;
}

size_t storageRowBytes(const FormatInfo* f, int width) {
    if (f->comp == Comp::Compressed) return (size_t)((width + f->blockW - 1) / f->blockW) * f->bytesPerPixel;
    return (size_t)width * f->bytesPerPixel;
}

size_t storageImageBytes(const FormatInfo* f, int width, int height) {
    int rows = f->comp == Comp::Compressed ? (height + f->blockH - 1) / f->blockH : height;
    return storageRowBytes(f, width) * rows;
}

CompressedLayout compressedLayout(const FormatInfo* f, int width, int height, int dims, const PixelStore& ps) {
    CompressedLayout l;
    l.rowBytes = storageRowBytes(f, width);
    l.rowStride = l.rowBytes;
    l.rows = (height + f->blockH - 1) / f->blockH;
    int rowsPerImage = l.rows;
    l.skipBytes = 0;
    int bs = ps.compressedBlockSize;
    if (bs && ps.compressedBlockWidth) {
        int bw = ps.compressedBlockWidth;
        if (ps.rowLength) l.rowStride = (size_t)bs * ((ps.rowLength + bw - 1) / bw);
        l.skipBytes += (size_t)ps.skipPixels / bw * bs;
    }
    if (dims > 1 && bs && ps.compressedBlockHeight) {
        int bh = ps.compressedBlockHeight;
        l.skipBytes += (size_t)ps.skipRows / bh * l.rowStride;
        if (ps.imageHeight) rowsPerImage = (ps.imageHeight + bh - 1) / bh;
    }
    l.imageStride = l.rowStride * rowsPerImage;
    if (dims > 2 && bs && ps.compressedBlockDepth) l.skipBytes += (size_t)ps.skipImages / ps.compressedBlockDepth * l.imageStride;
    return l;
}

bool directUploadCompatible(const FormatInfo* f, GLenum format, GLenum type) {
    switch (f->mtl) {
    case MTLPixelFormatRGBA8Unorm: case MTLPixelFormatRGBA8Unorm_sRGB:
        return f->baseFormat == GL_RGBA && format == GL_RGBA && (type == GL_UNSIGNED_BYTE || type == GL_UNSIGNED_INT_8_8_8_8_REV);
    case MTLPixelFormatRGBA8Snorm: return f->baseFormat == GL_RGBA && format == GL_RGBA && type == GL_BYTE;
    case MTLPixelFormatRGBA8Uint: return f->baseFormat == GL_RGBA && format == GL_RGBA_INTEGER && type == GL_UNSIGNED_BYTE;
    case MTLPixelFormatRGBA8Sint: return f->baseFormat == GL_RGBA && format == GL_RGBA_INTEGER && type == GL_BYTE;
    case MTLPixelFormatR8Unorm:
        return (f->baseFormat == GL_RED && format == GL_RED && type == GL_UNSIGNED_BYTE) ||
               (f->baseFormat == GL_ALPHA && format == GL_ALPHA && type == GL_UNSIGNED_BYTE) ||
               (f->baseFormat == GL_LUMINANCE && format == GL_LUMINANCE && type == GL_UNSIGNED_BYTE) ||
               (f->baseFormat == GL_INTENSITY && (format == GL_LUMINANCE || format == GL_RED) && type == GL_UNSIGNED_BYTE);
    case MTLPixelFormatRG8Unorm:
        return (f->baseFormat == GL_RG && format == GL_RG && type == GL_UNSIGNED_BYTE) ||
               (f->baseFormat == GL_LUMINANCE_ALPHA && format == GL_LUMINANCE_ALPHA && type == GL_UNSIGNED_BYTE);
    case MTLPixelFormatR16Unorm: return f->baseFormat == GL_RED && format == GL_RED && type == GL_UNSIGNED_SHORT;
    case MTLPixelFormatRG16Unorm: return f->baseFormat == GL_RG && format == GL_RG && type == GL_UNSIGNED_SHORT;
    case MTLPixelFormatRGBA16Unorm: return f->baseFormat == GL_RGBA && format == GL_RGBA && type == GL_UNSIGNED_SHORT;
    case MTLPixelFormatR16Float: return format == GL_RED && type == GL_HALF_FLOAT;
    case MTLPixelFormatRG16Float: return format == GL_RG && type == GL_HALF_FLOAT;
    case MTLPixelFormatRGBA16Float: return f->baseFormat == GL_RGBA && format == GL_RGBA && type == GL_HALF_FLOAT;
    case MTLPixelFormatR32Float: return format == GL_RED && type == GL_FLOAT;
    case MTLPixelFormatRG32Float: return format == GL_RG && type == GL_FLOAT;
    case MTLPixelFormatRGBA32Float: return f->baseFormat == GL_RGBA && format == GL_RGBA && type == GL_FLOAT;
    case MTLPixelFormatR32Uint: return format == GL_RED_INTEGER && type == GL_UNSIGNED_INT;
    case MTLPixelFormatR32Sint: return format == GL_RED_INTEGER && type == GL_INT;
    case MTLPixelFormatRG32Uint: return format == GL_RG_INTEGER && type == GL_UNSIGNED_INT;
    case MTLPixelFormatRG32Sint: return format == GL_RG_INTEGER && type == GL_INT;
    case MTLPixelFormatRGBA32Uint: return f->baseFormat == GL_RGBA && format == GL_RGBA_INTEGER && type == GL_UNSIGNED_INT;
    case MTLPixelFormatRGBA32Sint: return f->baseFormat == GL_RGBA && format == GL_RGBA_INTEGER && type == GL_INT;
    case MTLPixelFormatR16Uint: return format == GL_RED_INTEGER && type == GL_UNSIGNED_SHORT;
    case MTLPixelFormatR16Sint: return format == GL_RED_INTEGER && type == GL_SHORT;
    case MTLPixelFormatR8Uint: return format == GL_RED_INTEGER && type == GL_UNSIGNED_BYTE;
    case MTLPixelFormatR8Sint: return format == GL_RED_INTEGER && type == GL_BYTE;
    case MTLPixelFormatRGB10A2Unorm: return f->baseFormat == GL_RGBA && format == GL_RGBA && type == GL_UNSIGNED_INT_2_10_10_10_REV;
    case MTLPixelFormatRG11B10Float: return format == GL_RGB && type == GL_UNSIGNED_INT_10F_11F_11F_REV;
    case MTLPixelFormatRGB9E5Float: return format == GL_RGB && type == GL_UNSIGNED_INT_5_9_9_9_REV;
    case MTLPixelFormatDepth16Unorm: return format == GL_DEPTH_COMPONENT && type == GL_UNSIGNED_SHORT;
    case MTLPixelFormatDepth32Float: return format == GL_DEPTH_COMPONENT && type == GL_FLOAT;
    case MTLPixelFormatStencil8: return format == GL_STENCIL_INDEX && type == GL_UNSIGNED_BYTE;
    default: return false;
    }
}

// ---------------------------------------------------------------------------
// generic conversion
// ---------------------------------------------------------------------------
namespace {

struct Px {
    float f[4] = {0, 0, 0, 1};
    int64_t i[4] = {0, 0, 0, 1};
    bool isInt = false;
    float depth = 0;
    uint32_t stencil = 0;
};

inline float halfToFloat(uint16_t h) {
    __fp16 v;
    memcpy(&v, &h, 2);
    return (float)v;
}
inline uint16_t floatToHalf(float f) {
    __fp16 v = (__fp16)f;
    uint16_t h;
    memcpy(&h, &v, 2);
    return h;
}

inline float unorm(uint64_t v, int bits) { return (float)((double)v / (double)((1ull << bits) - 1)); }
inline float snorm(int64_t v, int bits) { return std::max(-1.0f, (float)((double)v / (double)((1ll << (bits - 1)) - 1))); }
inline uint64_t toUnorm(float f, int bits) {
    if (!(f > 0)) f = 0;
    if (f > 1) f = 1;
    return (uint64_t)llround((double)f * (double)((1ull << bits) - 1));
}
inline int64_t toSnorm(float f, int bits) {
    if (!(f >= -1)) f = -1;
    if (f > 1) f = 1;
    return (int64_t)llround((double)f * (double)((1ll << (bits - 1)) - 1));
}

float unpackF11(uint32_t v) {  // 6e5
    uint32_t e = (v >> 6) & 0x1f, m = v & 0x3f;
    if (e == 0) return ldexpf((float)m, -14 - 6);
    if (e == 31) return m ? NAN : INFINITY;
    return ldexpf(1.0f + m / 64.0f, (int)e - 15);
}
float unpackF10(uint32_t v) {  // 5e5
    uint32_t e = (v >> 5) & 0x1f, m = v & 0x1f;
    if (e == 0) return ldexpf((float)m, -14 - 5);
    if (e == 31) return m ? NAN : INFINITY;
    return ldexpf(1.0f + m / 32.0f, (int)e - 15);
}
uint32_t packSmallFloat(float f, int mbits) {
    if (std::isnan(f)) return (31u << mbits) | 1u;
    if (!(f > 0)) return 0;
    if (std::isinf(f)) return 31u << mbits;
    __fp16 h = (__fp16)f;
    uint16_t b;
    memcpy(&b, &h, 2);
    uint32_t e = (b >> 10) & 0x1f, m = b & 0x3ff;
    if (e == 31) return (31u << mbits);
    return (e << mbits) | (m >> (10 - mbits));
}
void unpack999E5(uint32_t v, float* out) {
    int e = (int)(v >> 27) - 15 - 9;
    for (int c = 0; c < 3; c++) out[c] = ldexpf((float)((v >> (9 * c)) & 0x1ff), e);
}
uint32_t pack999E5(const float* in) {
    const float maxv = (511.0f / 512.0f) * 65536.0f;
    float r = std::clamp(in[0], 0.0f, maxv), g = std::clamp(in[1], 0.0f, maxv), b = std::clamp(in[2], 0.0f, maxv);
    if (std::isnan(r)) r = 0;
    if (std::isnan(g)) g = 0;
    if (std::isnan(b)) b = 0;
    float mx = std::max(r, std::max(g, b));
    int ep = std::max(-16, (int)floorf(log2f(std::max(mx, 1e-30f)))) + 1 + 15;
    int maxm = (int)floorf(mx / ldexpf(1.0f, ep - 15 - 9) + 0.5f);
    if (maxm == 512) ep++;
    float scale = ldexpf(1.0f, ep - 15 - 9);
    uint32_t rm = (uint32_t)floorf(r / scale + 0.5f), gm = (uint32_t)floorf(g / scale + 0.5f), bm = (uint32_t)floorf(b / scale + 0.5f);
    return (uint32_t)ep << 27 | (bm & 0x1ff) << 18 | (gm & 0x1ff) << 9 | (rm & 0x1ff);
}

template <typename T>
T rd(const uint8_t* p, bool swap) {
    T v;
    memcpy(&v, p, sizeof(T));
    if (swap) {
        if constexpr (sizeof(T) == 2) {
            uint16_t u;
            memcpy(&u, &v, 2);
            u = __builtin_bswap16(u);
            memcpy(&v, &u, 2);
        } else if constexpr (sizeof(T) == 4) {
            uint32_t u;
            memcpy(&u, &v, 4);
            u = __builtin_bswap32(u);
            memcpy(&v, &u, 4);
        }
    }
    return v;
}

// Maps client format components to RGBA indices. -1 = unused.
void clientMapping(GLenum format, int map[4], bool& isInt, bool& lum) {
    isInt = false;
    lum = false;
    for (int i = 0; i < 4; i++) map[i] = -1;
    switch (format) {
    case GL_RED_INTEGER: isInt = true; [[fallthrough]];
    case GL_RED: map[0] = 0; break;
    case GL_GREEN_INTEGER: isInt = true; [[fallthrough]];
    case GL_GREEN: map[0] = 1; break;
    case GL_BLUE_INTEGER: isInt = true; [[fallthrough]];
    case GL_BLUE: map[0] = 2; break;
    case GL_ALPHA_INTEGER: isInt = true; [[fallthrough]];
    case GL_ALPHA: map[0] = 3; break;
    case GL_RG_INTEGER: isInt = true; [[fallthrough]];
    case GL_RG: map[0] = 0; map[1] = 1; break;
    case GL_RGB_INTEGER: isInt = true; [[fallthrough]];
    case GL_RGB: map[0] = 0; map[1] = 1; map[2] = 2; break;
    case GL_BGR_INTEGER: isInt = true; [[fallthrough]];
    case GL_BGR: map[0] = 2; map[1] = 1; map[2] = 0; break;
    case GL_RGBA_INTEGER: isInt = true; [[fallthrough]];
    case GL_RGBA: map[0] = 0; map[1] = 1; map[2] = 2; map[3] = 3; break;
    case GL_BGRA_INTEGER: isInt = true; [[fallthrough]];
    case GL_BGRA: map[0] = 2; map[1] = 1; map[2] = 0; map[3] = 3; break;
    case GL_LUMINANCE: lum = true; map[0] = 0; break;
    case GL_INTENSITY: lum = true; map[0] = 0; break;
    case GL_LUMINANCE_ALPHA: lum = true; map[0] = 0; map[1] = 3; break;
    default: break;
    }
}

void decodeClient(GLenum format, GLenum type, const uint8_t* p, bool swap, Px& px) {
    if (format == GL_DEPTH_COMPONENT) {
        switch (type) {
        case GL_UNSIGNED_BYTE: px.depth = unorm(p[0], 8); break;
        case GL_BYTE: px.depth = std::max(0.0f, snorm((int8_t)p[0], 8)); break;
        case GL_UNSIGNED_SHORT: px.depth = unorm(rd<uint16_t>(p, swap), 16); break;
        case GL_SHORT: px.depth = std::max(0.0f, snorm(rd<int16_t>(p, swap), 16)); break;
        case GL_UNSIGNED_INT: px.depth = unorm(rd<uint32_t>(p, swap), 32); break;
        case GL_INT: px.depth = std::max(0.0f, snorm(rd<int32_t>(p, swap), 32)); break;
        case GL_HALF_FLOAT: px.depth = halfToFloat(rd<uint16_t>(p, swap)); break;
        case GL_FLOAT: px.depth = rd<float>(p, swap); break;
        }
        return;
    }
    if (format == GL_STENCIL_INDEX) {
        switch (type) {
        case GL_UNSIGNED_BYTE: case GL_BYTE: px.stencil = p[0]; break;
        case GL_UNSIGNED_SHORT: case GL_SHORT: px.stencil = rd<uint16_t>(p, swap); break;
        case GL_UNSIGNED_INT: case GL_INT: px.stencil = rd<uint32_t>(p, swap); break;
        case GL_FLOAT: px.stencil = (uint32_t)rd<float>(p, swap); break;
        }
        return;
    }
    if (format == GL_DEPTH_STENCIL) {
        if (type == GL_UNSIGNED_INT_24_8) {
            uint32_t v = rd<uint32_t>(p, swap);
            px.depth = unorm(v >> 8, 24);
            px.stencil = v & 0xff;
        } else {
            px.depth = rd<float>(p, swap);
            px.stencil = rd<uint32_t>(p + 4, swap) & 0xff;
        }
        return;
    }
    int map[4];
    bool isInt, lum;
    clientMapping(format, map, isInt, lum);
    px.isInt = isInt;
    float c[4] = {0, 0, 0, 1};
    int64_t ci[4] = {0, 0, 0, 1};
    auto put = [&](int comp, float fv, int64_t iv) {
        int dst = map[comp];
        if (dst < 0) return;
        c[dst] = fv;
        ci[dst] = iv;
    };
    switch (type) {
    case GL_UNSIGNED_BYTE: for (int k = 0; k < 4 && map[k] >= 0; k++) put(k, unorm(p[k], 8), p[k]); break;
    case GL_BYTE: for (int k = 0; k < 4 && map[k] >= 0; k++) put(k, snorm((int8_t)p[k], 8), (int8_t)p[k]); break;
    case GL_UNSIGNED_SHORT:
        for (int k = 0; k < 4 && map[k] >= 0; k++) { uint16_t v = rd<uint16_t>(p + 2 * k, swap); put(k, unorm(v, 16), v); }
        break;
    case GL_SHORT:
        for (int k = 0; k < 4 && map[k] >= 0; k++) { int16_t v = rd<int16_t>(p + 2 * k, swap); put(k, snorm(v, 16), v); }
        break;
    case GL_UNSIGNED_INT:
        for (int k = 0; k < 4 && map[k] >= 0; k++) { uint32_t v = rd<uint32_t>(p + 4 * k, swap); put(k, unorm(v, 32), v); }
        break;
    case GL_INT:
        for (int k = 0; k < 4 && map[k] >= 0; k++) { int32_t v = rd<int32_t>(p + 4 * k, swap); put(k, snorm(v, 32), v); }
        break;
    case GL_HALF_FLOAT:
        for (int k = 0; k < 4 && map[k] >= 0; k++) { float v = halfToFloat(rd<uint16_t>(p + 2 * k, swap)); put(k, v, (int64_t)v); }
        break;
    case GL_FLOAT:
        for (int k = 0; k < 4 && map[k] >= 0; k++) { float v = rd<float>(p + 4 * k, swap); put(k, v, (int64_t)v); }
        break;
    default: {
        // packed: extract components in "first component first" order
        uint32_t v = 0;
        int sz = 0, nc = 0;
        sz = packedSize(type, nc);
        if (sz == 1) v = p[0];
        else if (sz == 2) v = rd<uint16_t>(p, swap);
        else v = rd<uint32_t>(p, swap);
        uint32_t comps[4] = {0, 0, 0, 0};
        int bits[4] = {0, 0, 0, 0};
        auto ext = [&](int idx, int shift, int nb) { comps[idx] = (v >> shift) & ((1u << nb) - 1); bits[idx] = nb; };
        switch (type) {
        case GL_UNSIGNED_BYTE_3_3_2: ext(0, 5, 3); ext(1, 2, 3); ext(2, 0, 2); break;
        case GL_UNSIGNED_BYTE_2_3_3_REV: ext(0, 0, 3); ext(1, 3, 3); ext(2, 6, 2); break;
        case GL_UNSIGNED_SHORT_5_6_5: ext(0, 11, 5); ext(1, 5, 6); ext(2, 0, 5); break;
        case GL_UNSIGNED_SHORT_5_6_5_REV: ext(0, 0, 5); ext(1, 5, 6); ext(2, 11, 5); break;
        case GL_UNSIGNED_SHORT_4_4_4_4: ext(0, 12, 4); ext(1, 8, 4); ext(2, 4, 4); ext(3, 0, 4); break;
        case GL_UNSIGNED_SHORT_4_4_4_4_REV: ext(0, 0, 4); ext(1, 4, 4); ext(2, 8, 4); ext(3, 12, 4); break;
        case GL_UNSIGNED_SHORT_5_5_5_1: ext(0, 11, 5); ext(1, 6, 5); ext(2, 1, 5); ext(3, 0, 1); break;
        case GL_UNSIGNED_SHORT_1_5_5_5_REV: ext(0, 0, 5); ext(1, 5, 5); ext(2, 10, 5); ext(3, 15, 1); break;
        case GL_UNSIGNED_INT_8_8_8_8: ext(0, 24, 8); ext(1, 16, 8); ext(2, 8, 8); ext(3, 0, 8); break;
        case GL_UNSIGNED_INT_8_8_8_8_REV: ext(0, 0, 8); ext(1, 8, 8); ext(2, 16, 8); ext(3, 24, 8); break;
        case GL_UNSIGNED_INT_10_10_10_2: ext(0, 22, 10); ext(1, 12, 10); ext(2, 2, 10); ext(3, 0, 2); break;
        case GL_UNSIGNED_INT_2_10_10_10_REV: ext(0, 0, 10); ext(1, 10, 10); ext(2, 20, 10); ext(3, 30, 2); break;
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
            c[0] = unpackF11(v & 0x7ff); c[1] = unpackF11((v >> 11) & 0x7ff); c[2] = unpackF10((v >> 22) & 0x3ff); c[3] = 1;
            for (int k = 0; k < 4; k++) px.f[k] = c[k];
            return;
        case GL_UNSIGNED_INT_5_9_9_9_REV:
            unpack999E5(v, c); c[3] = 1;
            for (int k = 0; k < 4; k++) px.f[k] = c[k];
            return;
        }
        for (int k = 0; k < nc; k++) put(k, unorm(comps[k], bits[k]), comps[k]);
        break;
    }
    }
    if (lum) {
        // luminance replicated into RGB when converting to RGB storage
        if (format == GL_INTENSITY) { c[1] = c[2] = c[3] = c[0]; ci[1] = ci[2] = ci[3] = ci[0]; }
        else { c[1] = c[2] = c[0]; ci[1] = ci[2] = ci[0]; }
    }
    for (int k = 0; k < 4; k++) { px.f[k] = c[k]; px.i[k] = ci[k]; }
}

void encodeStorage(const FormatInfo* f, const Px& px, uint8_t* dst, GLenum srcFormat) {
    // Map RGBA -> stored channels. For luminance/alpha formats the "order"
    // array says which RGBA component each stored channel comes from.
    float c[4];
    int64_t ci[4];
    for (int k = 0; k < 4; k++) {
        int src = f->order[k];
        c[k] = px.f[src];
        ci[k] = px.isInt ? px.i[src] : (int64_t)px.f[src];
    }
    // Luminance/intensity storage from luminance client data: channel0 = L(=R)
    if ((f->baseFormat == GL_LUMINANCE || f->baseFormat == GL_INTENSITY || f->baseFormat == GL_LUMINANCE_ALPHA)) {
        c[0] = px.f[0];
        if (f->baseFormat == GL_LUMINANCE_ALPHA) c[1] = px.f[3];
    }
    int n = f->channels;
    switch (f->comp) {
    case Comp::U8:
        for (int k = 0; k < n; k++) dst[k] = (uint8_t)(f->kind == FmtKind::UInt ? std::clamp<int64_t>(ci[k], 0, 255) : toUnorm(c[k], 8));
        break;
    case Comp::S8:
        for (int k = 0; k < n; k++) dst[k] = (uint8_t)(int8_t)(f->kind == FmtKind::SInt ? std::clamp<int64_t>(ci[k], -128, 127) : toSnorm(c[k], 8));
        break;
    case Comp::U16:
        for (int k = 0; k < n; k++) {
            uint16_t v = (uint16_t)(f->kind == FmtKind::UInt ? std::clamp<int64_t>(ci[k], 0, 65535) : toUnorm(c[k], 16));
            memcpy(dst + 2 * k, &v, 2);
        }
        break;
    case Comp::S16:
        for (int k = 0; k < n; k++) {
            int16_t v = (int16_t)(f->kind == FmtKind::SInt ? std::clamp<int64_t>(ci[k], -32768, 32767) : toSnorm(c[k], 16));
            memcpy(dst + 2 * k, &v, 2);
        }
        break;
    case Comp::U32:
        for (int k = 0; k < n; k++) { uint32_t v = (uint32_t)std::clamp<int64_t>(ci[k], 0, 0xffffffffll); memcpy(dst + 4 * k, &v, 4); }
        break;
    case Comp::S32:
        for (int k = 0; k < n; k++) { int32_t v = (int32_t)std::clamp<int64_t>(ci[k], INT32_MIN, INT32_MAX); memcpy(dst + 4 * k, &v, 4); }
        break;
    case Comp::F16:
        for (int k = 0; k < n; k++) { uint16_t v = floatToHalf(c[k]); memcpy(dst + 2 * k, &v, 2); }
        break;
    case Comp::F32:
        for (int k = 0; k < n; k++) memcpy(dst + 4 * k, &c[k], 4);
        break;
    case Comp::P1010102: {
        uint32_t v;
        if (f->kind == FmtKind::UInt)
            v = (uint32_t)std::clamp<int64_t>(ci[0], 0, 1023) | (uint32_t)std::clamp<int64_t>(ci[1], 0, 1023) << 10 |
                (uint32_t)std::clamp<int64_t>(ci[2], 0, 1023) << 20 | (uint32_t)std::clamp<int64_t>(ci[3], 0, 3) << 30;
        else
            v = (uint32_t)toUnorm(c[0], 10) | (uint32_t)toUnorm(c[1], 10) << 10 | (uint32_t)toUnorm(c[2], 10) << 20 |
                (uint32_t)toUnorm(c[3], 2) << 30;
        memcpy(dst, &v, 4);
        break;
    }
    case Comp::P111110F: {
        uint32_t v = packSmallFloat(c[0], 6) | packSmallFloat(c[1], 6) << 11 | packSmallFloat(c[2], 5) << 22;
        memcpy(dst, &v, 4);
        break;
    }
    case Comp::P999E5: {
        uint32_t v = pack999E5(c);
        memcpy(dst, &v, 4);
        break;
    }
    case Comp::D16: { uint16_t v = (uint16_t)toUnorm(px.depth, 16); memcpy(dst, &v, 2); break; }
    case Comp::D32F: { float d = std::clamp(px.depth, 0.0f, 1.0f); if (f->glType == GL_FLOAT) d = px.depth; memcpy(dst, &d, 4); break; }
    case Comp::S8_ONLY: dst[0] = (uint8_t)px.stencil; break;
    case Comp::D32F_S8: {
        float d = f->glType == GL_FLOAT ? px.depth : std::clamp(px.depth, 0.0f, 1.0f);
        memcpy(dst, &d, 4);
        uint32_t s = px.stencil & 0xff;
        memcpy(dst + 4, &s, 4);
        break;
    }
    default: break;
    }
}

void decodeStorage(const FormatInfo* f, const uint8_t* src, Px& px) {
    float c[4] = {0, 0, 0, 1};
    int64_t ci[4] = {0, 0, 0, 1};
    int n = f->channels;
    bool isInt = f->kind == FmtKind::UInt || f->kind == FmtKind::SInt;
    switch (f->comp) {
    case Comp::U8: for (int k = 0; k < n; k++) { c[k] = unorm(src[k], 8); ci[k] = src[k]; } break;
    case Comp::S8: for (int k = 0; k < n; k++) { c[k] = snorm((int8_t)src[k], 8); ci[k] = (int8_t)src[k]; } break;
    case Comp::U16: for (int k = 0; k < n; k++) { uint16_t v; memcpy(&v, src + 2 * k, 2); c[k] = unorm(v, 16); ci[k] = v; } break;
    case Comp::S16: for (int k = 0; k < n; k++) { int16_t v; memcpy(&v, src + 2 * k, 2); c[k] = snorm(v, 16); ci[k] = v; } break;
    case Comp::U32: for (int k = 0; k < n; k++) { uint32_t v; memcpy(&v, src + 4 * k, 4); ci[k] = v; c[k] = (float)v; } break;
    case Comp::S32: for (int k = 0; k < n; k++) { int32_t v; memcpy(&v, src + 4 * k, 4); ci[k] = v; c[k] = (float)v; } break;
    case Comp::F16: for (int k = 0; k < n; k++) { uint16_t v; memcpy(&v, src + 2 * k, 2); c[k] = halfToFloat(v); } break;
    case Comp::F32: for (int k = 0; k < n; k++) memcpy(&c[k], src + 4 * k, 4); break;
    case Comp::P1010102: {
        uint32_t v; memcpy(&v, src, 4);
        uint32_t r = v & 1023, g = (v >> 10) & 1023, b = (v >> 20) & 1023, a = v >> 30;
        ci[0] = r; ci[1] = g; ci[2] = b; ci[3] = a;
        c[0] = unorm(r, 10); c[1] = unorm(g, 10); c[2] = unorm(b, 10); c[3] = unorm(a, 2);
        break;
    }
    case Comp::P111110F: {
        uint32_t v; memcpy(&v, src, 4);
        c[0] = unpackF11(v & 0x7ff); c[1] = unpackF11((v >> 11) & 0x7ff); c[2] = unpackF10(v >> 22);
        break;
    }
    case Comp::P999E5: { uint32_t v; memcpy(&v, src, 4); unpack999E5(v, c); break; }
    case Comp::D16: { uint16_t v; memcpy(&v, src, 2); px.depth = unorm(v, 16); return; }
    case Comp::D32F: memcpy(&px.depth, src, 4); return;
    case Comp::S8_ONLY: px.stencil = src[0]; return;
    case Comp::D32F_S8: memcpy(&px.depth, src, 4); px.stencil = src[4]; return;
    default: return;
    }
    // stored channel k -> RGBA component order[k]
    float rgba[4] = {0, 0, 0, 1};
    int64_t irgba[4] = {0, 0, 0, 1};
    for (int k = 0; k < n; k++) { rgba[f->order[k]] = c[k]; irgba[f->order[k]] = ci[k]; }
    if (f->baseFormat == GL_LUMINANCE || f->baseFormat == GL_INTENSITY) {
        rgba[1] = rgba[2] = rgba[0];
        if (f->baseFormat == GL_INTENSITY) rgba[3] = rgba[0];
    } else if (f->baseFormat == GL_LUMINANCE_ALPHA) {
        rgba[0] = c[0]; rgba[1] = rgba[2] = c[0]; rgba[3] = c[1];
    } else if (f->baseFormat == GL_RGB || f->baseFormat == GL_RG || f->baseFormat == GL_RED) {
        rgba[3] = 1; irgba[3] = 1;
        if (f->baseFormat != GL_RGB) { rgba[2] = 0; irgba[2] = 0; }
        if (f->baseFormat == GL_RED) { rgba[1] = 0; irgba[1] = 0; }
    }
    px.isInt = isInt;
    for (int k = 0; k < 4; k++) { px.f[k] = rgba[k]; px.i[k] = irgba[k]; }
}

template <typename T>
void wr(uint8_t* p, T v, bool swap) {
    if (swap) {
        if constexpr (sizeof(T) == 2) { uint16_t u; memcpy(&u, &v, 2); u = __builtin_bswap16(u); memcpy(p, &u, 2); return; }
        if constexpr (sizeof(T) == 4) { uint32_t u; memcpy(&u, &v, 4); u = __builtin_bswap32(u); memcpy(p, &u, 4); return; }
    }
    memcpy(p, &v, sizeof(T));
}

void encodeClient(GLenum format, GLenum type, const Px& px, uint8_t* p, bool swap, bool clampColor) {
    if (format == GL_DEPTH_COMPONENT) {
        float d = px.depth;
        switch (type) {
        case GL_UNSIGNED_BYTE: p[0] = (uint8_t)toUnorm(d, 8); break;
        case GL_BYTE: p[0] = (uint8_t)(int8_t)toSnorm(d, 8); break;
        case GL_UNSIGNED_SHORT: wr<uint16_t>(p, (uint16_t)toUnorm(d, 16), swap); break;
        case GL_SHORT: wr<int16_t>(p, (int16_t)toSnorm(d, 16), swap); break;
        case GL_UNSIGNED_INT: wr<uint32_t>(p, (uint32_t)toUnorm(d, 32), swap); break;
        case GL_INT: wr<int32_t>(p, (int32_t)toSnorm(d, 32), swap); break;
        case GL_HALF_FLOAT: wr<uint16_t>(p, floatToHalf(d), swap); break;
        case GL_FLOAT: wr<float>(p, d, swap); break;
        }
        return;
    }
    if (format == GL_STENCIL_INDEX) {
        switch (type) {
        case GL_UNSIGNED_BYTE: case GL_BYTE: p[0] = (uint8_t)px.stencil; break;
        case GL_UNSIGNED_SHORT: case GL_SHORT: wr<uint16_t>(p, (uint16_t)px.stencil, swap); break;
        case GL_UNSIGNED_INT: case GL_INT: wr<uint32_t>(p, px.stencil, swap); break;
        case GL_FLOAT: wr<float>(p, (float)px.stencil, swap); break;
        }
        return;
    }
    if (format == GL_DEPTH_STENCIL) {
        if (type == GL_UNSIGNED_INT_24_8) wr<uint32_t>(p, (uint32_t)toUnorm(px.depth, 24) << 8 | (px.stencil & 0xff), swap);
        else { wr<float>(p, px.depth, swap); wr<uint32_t>(p + 4, px.stencil & 0xff, swap); }
        return;
    }
    int map[4];
    bool isInt, lum;
    clientMapping(format, map, isInt, lum);
    float c[4];
    int64_t ci[4];
    for (int k = 0; k < 4; k++) {
        c[k] = px.f[k];
        if (clampColor) c[k] = std::clamp(c[k], 0.0f, 1.0f);
        ci[k] = px.isInt ? px.i[k] : (int64_t)px.f[k];
    }
    auto comp = [&](int k) { return map[k] >= 0 ? c[map[k]] : 0.0f; };
    auto icomp = [&](int k) { return map[k] >= 0 ? ci[map[k]] : 0; };
    // Integer values are clamped to the destination type's range (GL 4.6 §8.4.4.2 / §18.2.8).
    auto ic = [&](int k, int64_t lo, int64_t hi) { return std::clamp<int64_t>(icomp(k), lo, hi); };
    int nc = 0;
    while (nc < 4 && map[nc] >= 0) nc++;
    switch (type) {
    case GL_UNSIGNED_BYTE: for (int k = 0; k < nc; k++) p[k] = (uint8_t)(isInt ? ic(k, 0, 255) : toUnorm(comp(k), 8)); break;
    case GL_BYTE: for (int k = 0; k < nc; k++) p[k] = (uint8_t)(int8_t)(isInt ? ic(k, -128, 127) : toSnorm(comp(k), 8)); break;
    case GL_UNSIGNED_SHORT: for (int k = 0; k < nc; k++) wr<uint16_t>(p + 2 * k, (uint16_t)(isInt ? ic(k, 0, 65535) : toUnorm(comp(k), 16)), swap); break;
    case GL_SHORT: for (int k = 0; k < nc; k++) wr<int16_t>(p + 2 * k, (int16_t)(isInt ? ic(k, -32768, 32767) : toSnorm(comp(k), 16)), swap); break;
    case GL_UNSIGNED_INT: for (int k = 0; k < nc; k++) wr<uint32_t>(p + 4 * k, (uint32_t)(isInt ? ic(k, 0, 4294967295LL) : toUnorm(comp(k), 32)), swap); break;
    case GL_INT: for (int k = 0; k < nc; k++) wr<int32_t>(p + 4 * k, (int32_t)(isInt ? ic(k, INT32_MIN, INT32_MAX) : toSnorm(comp(k), 32)), swap); break;
    case GL_HALF_FLOAT: for (int k = 0; k < nc; k++) wr<uint16_t>(p + 2 * k, floatToHalf(comp(k)), swap); break;
    case GL_FLOAT: for (int k = 0; k < nc; k++) wr<float>(p + 4 * k, comp(k), swap); break;
    default: {
        auto u = [&](int k, int bits) -> uint32_t { return isInt ? (uint32_t)ic(k, 0, (1LL << bits) - 1) : (uint32_t)toUnorm(comp(k), bits); };
        uint32_t v = 0;
        int sz = 4;
        switch (type) {
        case GL_UNSIGNED_BYTE_3_3_2: v = u(0, 3) << 5 | u(1, 3) << 2 | u(2, 2); sz = 1; break;
        case GL_UNSIGNED_BYTE_2_3_3_REV: v = u(0, 3) | u(1, 3) << 3 | u(2, 2) << 6; sz = 1; break;
        case GL_UNSIGNED_SHORT_5_6_5: v = u(0, 5) << 11 | u(1, 6) << 5 | u(2, 5); sz = 2; break;
        case GL_UNSIGNED_SHORT_5_6_5_REV: v = u(0, 5) | u(1, 6) << 5 | u(2, 5) << 11; sz = 2; break;
        case GL_UNSIGNED_SHORT_4_4_4_4: v = u(0, 4) << 12 | u(1, 4) << 8 | u(2, 4) << 4 | u(3, 4); sz = 2; break;
        case GL_UNSIGNED_SHORT_4_4_4_4_REV: v = u(0, 4) | u(1, 4) << 4 | u(2, 4) << 8 | u(3, 4) << 12; sz = 2; break;
        case GL_UNSIGNED_SHORT_5_5_5_1: v = u(0, 5) << 11 | u(1, 5) << 6 | u(2, 5) << 1 | u(3, 1); sz = 2; break;
        case GL_UNSIGNED_SHORT_1_5_5_5_REV: v = u(0, 5) | u(1, 5) << 5 | u(2, 5) << 10 | u(3, 1) << 15; sz = 2; break;
        case GL_UNSIGNED_INT_8_8_8_8: v = u(0, 8) << 24 | u(1, 8) << 16 | u(2, 8) << 8 | u(3, 8); break;
        case GL_UNSIGNED_INT_8_8_8_8_REV: v = u(0, 8) | u(1, 8) << 8 | u(2, 8) << 16 | u(3, 8) << 24; break;
        case GL_UNSIGNED_INT_10_10_10_2: v = u(0, 10) << 22 | u(1, 10) << 12 | u(2, 10) << 2 | u(3, 2); break;
        case GL_UNSIGNED_INT_2_10_10_10_REV: v = u(0, 10) | u(1, 10) << 10 | u(2, 10) << 20 | u(3, 2) << 30; break;
        case GL_UNSIGNED_INT_10F_11F_11F_REV: v = packSmallFloat(px.f[0], 6) | packSmallFloat(px.f[1], 6) << 11 | packSmallFloat(px.f[2], 5) << 22; break;
        case GL_UNSIGNED_INT_5_9_9_9_REV: v = pack999E5(px.f); break;
        }
        if (sz == 1) p[0] = (uint8_t)v;
        else if (sz == 2) wr<uint16_t>(p, (uint16_t)v, swap);
        else wr<uint32_t>(p, v, swap);
        break;
    }
    }
}

}  // namespace

void convertToStorage(const FormatInfo* f, GLenum format, GLenum type, const void* src, size_t srcRowStride,
                      size_t srcImageStride, void* dst, size_t dstRowStride, size_t dstImageStride, int width,
                      int height, int depth, bool swapBytes) {
    const uint8_t* s = (const uint8_t*)src;
    uint8_t* d = (uint8_t*)dst;
    if (directUploadCompatible(f, format, type) && !swapBytes) {
        size_t row = (size_t)width * f->bytesPerPixel;
        for (int z = 0; z < depth; z++)
            for (int y = 0; y < height; y++) memcpy(d + z * dstImageStride + y * dstRowStride, s + z * srcImageStride + y * srcRowStride, row);
        return;
    }
    int ps = clientPixelSize(format, type);
    for (int z = 0; z < depth; z++)
        for (int y = 0; y < height; y++) {
            const uint8_t* sr = s + z * srcImageStride + y * srcRowStride;
            uint8_t* dr = d + z * dstImageStride + y * dstRowStride;
            for (int x = 0; x < width; x++) {
                Px px;
                decodeClient(format, type, sr + x * ps, swapBytes, px);
                // Writing integer client data into normalized storage (or vice versa) is an
                // error caught earlier; float -> int keeps truncation.
                if (!px.isInt && (f->kind == FmtKind::UInt || f->kind == FmtKind::SInt))
                    for (int k = 0; k < 4; k++) px.i[k] = (int64_t)px.f[k];
                encodeStorage(f, px, dr + x * f->bytesPerPixel, format);
            }
        }
}

void convertFromStorage(const FormatInfo* f, GLenum format, GLenum type, const void* src, size_t srcRowStride,
                        void* dst, size_t dstRowStride, int width, int height, bool swapBytes, bool depthPlane) {
    const uint8_t* s = (const uint8_t*)src;
    uint8_t* d = (uint8_t*)dst;
    if (directUploadCompatible(f, format, type) && !swapBytes && f->baseFormat != GL_LUMINANCE &&
        f->baseFormat != GL_LUMINANCE_ALPHA && f->baseFormat != GL_INTENSITY && f->baseFormat != GL_ALPHA) {
        size_t row = (size_t)width * f->bytesPerPixel;
        for (int y = 0; y < height; y++) memcpy(d + y * dstRowStride, s + y * srcRowStride, row);
        return;
    }
    int ps = clientPixelSize(format, type);
    bool clampColor = f->kind == FmtKind::Unorm && type != GL_FLOAT && type != GL_HALF_FLOAT;
    (void)clampColor;
    for (int y = 0; y < height; y++) {
        const uint8_t* sr = s + y * srcRowStride;
        uint8_t* dr = d + y * dstRowStride;
        for (int x = 0; x < width; x++) {
            Px px;
            decodeStorage(f, sr + x * f->bytesPerPixel, px);
            encodeClient(format, type, px, dr + x * ps, swapBytes, false);
        }
    }
}

}  // namespace mgl
