// GL <-> Metal format mapping and CPU pixel transfer conversion.
#pragma once
#import <Metal/Metal.h>
#include <cstddef>
#include <cstdint>
#include "glapi.h"

namespace mgl {

enum class FmtKind : uint8_t { Unorm, Snorm, Float, UInt, SInt, Depth, Stencil, DepthStencil };

// How a Metal storage format is laid out in memory, used by the converter.
enum class Comp : uint8_t {
    U8, S8, U16, S16, U32, S32, F16, F32,   // per-channel arrays
    P565, P5551, P4444, P1010102, P111110F, P999E5, P24S8_F32S8,  // packed
    D16, D32F, S8_ONLY, D32F_S8,
    Compressed,
};

struct FormatInfo {
    GLenum internalFormat;    // sized GL internal format
    MTLPixelFormat mtl;       // storage format
    GLenum baseFormat;        // GL_RGBA, GL_RGB, GL_RG, GL_RED, GL_DEPTH_COMPONENT, ...
    FmtKind kind;
    Comp comp;                // memory layout of `mtl`
    uint8_t channels;         // channels stored in `mtl` (memory order for per-channel arrays)
    uint8_t bytesPerPixel;    // or bytes per block for compressed
    uint8_t blockW, blockH;   // compressed block size (1 for uncompressed)
    bool srgb;
    bool renderable;
    bool filterable;
    // Swizzle applied by texture views so sampling returns GL semantics
    // (e.g. GL_RGB stored as RGBA -> alpha = ONE; luminance -> RRR1).
    MTLTextureSwizzleChannels swizzle;
    // Channel order of memory for per-channel arrays: index into RGBA. E.g. BGRA8 = {2,1,0,3}.
    uint8_t order[4];
    // GL sizes for queries (GL_TEXTURE_RED_SIZE etc).
    uint8_t rBits, gBits, bBits, aBits, dBits, sBits;
    GLenum glType;            // GL_TEXTURE_RED_TYPE etc component type (GL_UNSIGNED_NORMALIZED...)
};

// Lookup by sized or unsized internal format (unsized formats are resolved
// with the client format/type like GL does). Returns nullptr if unsupported.
const FormatInfo* formatInfo(GLenum internalFormat);
const FormatInfo* resolveInternalFormat(GLenum internalFormat, GLenum format, GLenum type);
const FormatInfo* formatInfoForMTL(MTLPixelFormat f);
bool isCompressedFormat(GLenum internalFormat);
bool isDepthOrStencil(const FormatInfo* f);
// GL 4.6 Table 8.27 view class: texel bits for uncompressed classes, 1000+n for
// compressed classes, 0 if the format belongs to no class.
int viewClass(GLenum internalFormat);
// Apple GPU lossless compression canonicalizes RGB9E5 encodings and RG11B10F NaNs,
// but GL requires bit-exact uploads, readbacks and copies: such textures opt out.
inline bool losslessCompressionSafe(MTLPixelFormat f) {
    return f != MTLPixelFormatRGB9E5Float && f != MTLPixelFormatRG11B10Float;
}
// glCopyImageSubData format compatibility (§18.3.3).
bool copyImageCompatible(const FormatInfo* a, const FormatInfo* b);

// --- client pixel layout ---------------------------------------------------
struct PixelStore {
    int alignment = 4, rowLength = 0, imageHeight = 0, skipPixels = 0, skipRows = 0, skipImages = 0;
    bool swapBytes = false, lsbFirst = false;
    // compressed block params (GL 4.2)
    int compressedBlockWidth = 0, compressedBlockHeight = 0, compressedBlockDepth = 0, compressedBlockSize = 0;
};

// Bytes per pixel of client data (format,type), 0 if invalid combination.
int clientPixelSize(GLenum format, GLenum type);
bool isValidFormatType(GLenum format, GLenum type);
// 0, GL_INVALID_ENUM (unknown enums) or GL_INVALID_OPERATION (incompatible pair).
GLenum pixelFormatTypeError(GLenum format, GLenum type);
// Also checks the client format against the texture's internal format (integer-ness,
// depth/stencil) and the target, as glTexImage*/glTexSubImage* do.
GLenum texImageFormatError(const FormatInfo* f, GLenum format, GLenum type, GLenum target);
// Computes client image addressing per GL unpack/pack rules.
struct ClientLayout {
    size_t pixelSize, rowStride, imageStride, skipBytes;
};
ClientLayout clientLayout(GLenum format, GLenum type, int width, int height, const PixelStore& ps);
size_t clientImageSize(GLenum format, GLenum type, int width, int height, int depth, const PixelStore& ps);

// True if client data can be memcpy'd row-wise into the Metal format.
bool directUploadCompatible(const FormatInfo* f, GLenum format, GLenum type);

// Converts client pixels -> Metal storage layout (tightly packed rows of
// width * bytesPerPixel). `src` points at the first pixel (skip applied).
void convertToStorage(const FormatInfo* f, GLenum format, GLenum type, const void* src, size_t srcRowStride,
                      size_t srcImageStride, void* dst, size_t dstRowStride, size_t dstImageStride, int width,
                      int height, int depth, bool swapBytes);

// Converts Metal storage -> client pixels (glReadPixels / glGetTexImage).
void convertFromStorage(const FormatInfo* f, GLenum format, GLenum type, const void* src, size_t srcRowStride,
                        void* dst, size_t dstRowStride, int width, int height, bool swapBytes,
                        bool depthPlane = true);

// Size in bytes of one tightly packed image in Metal storage.
size_t storageRowBytes(const FormatInfo* f, int width);
size_t storageImageBytes(const FormatInfo* f, int width, int height);

}  // namespace mgl
