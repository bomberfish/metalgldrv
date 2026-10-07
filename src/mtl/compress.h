// CPU block compression for glTex(Sub)Image* with a compressed internal format and
// uncompressed client data (GL requires the image to really be stored compressed).
#pragma once
#include <cstddef>
#include <cstdint>
#include "mtl/formats.h"

namespace mgl {

// True if `f` is a compressed format with a CPU encoder.
bool cpuCompressible(const FormatInfo* f);

// Compresses a w x h image of RGBA float texels (tightly packed, 4 floats per texel;
// values already in the format's encoding, e.g. sRGB-encoded for sRGB formats) into
// ceil(w/bw) x ceil(h/bh) blocks written tightly packed to `out`.
void compressImage(const FormatInfo* f, const float* rgba, int w, int h, uint8_t* out);

}  // namespace mgl
