// Queries, sync objects, conditional rendering.
#import "gl/context.h"
#include "util/log.h"

using namespace mgl;
#define API extern "C" __attribute__((visibility("default")))

namespace mgl {
bool SyncObj::signaled() const { return Device::get()->isComplete(serial); }
}

bool mglCheckDrawConditional(Context* ctx) { return true; }
