// Compatibility-profile state (fixed function). Filled in by the compat milestone.
#import "gl/context.h"

namespace mgl {}
using namespace mgl;

bool mglCompatCap(Context* ctx, GLenum cap, bool v) { return false; }
bool mglCompatIsEnabled(Context* ctx, GLenum cap, bool* out) { return false; }
bool mglCompatGet(Context* ctx, GLenum pname, double* out, int* n, bool* isFloat) { return false; }
