// fsbench FSB_CAPTURE: Metal GPU capture of the process's default device (Apple's OpenGL and metalgldrv both use it).
#import <Metal/Metal.h>
#import <objc/runtime.h>

bool mtlCaptureStart(const char* path) {
    MTLCaptureDescriptor* d = [MTLCaptureDescriptor new];
    d.captureObject = MTLCreateSystemDefaultDevice();
    d.destination = MTLCaptureDestinationGPUTraceDocument;
    d.outputURL = [NSURL fileURLWithPath:@(path)];
    NSError* err = nil;
    if (![[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:d error:&err]) {
        fprintf(stderr, "capture: %s\n", err.localizedDescription.UTF8String);
        return false;
    }
    return true;
}

void mtlCaptureStop() { [[MTLCaptureManager sharedCaptureManager] stopCapture]; }

// FSB_LOGTEX=1: logs the descriptors of the textures the GL driver creates (swizzles the default device's
// -newTextureWithDescriptor:), e.g. whether Apple's OpenGL keeps Metal's GPU-optimized (compressed) layout.
static IMP g_origNewTexture;
static id loggedNewTexture(id self, SEL _cmd, MTLTextureDescriptor* d) {
    fprintf(stderr, "texture %lux%lu fmt %lu type %lu levels %lu usage 0x%lx storage %lu gpuOptimized %d compression %ld\n",
            (unsigned long)d.width, (unsigned long)d.height, (unsigned long)d.pixelFormat, (unsigned long)d.textureType,
            (unsigned long)d.mipmapLevelCount, (unsigned long)d.usage, (unsigned long)d.storageMode, (int)d.allowGPUOptimizedContents,
            (long)d.compressionType);
    return ((id (*)(id, SEL, MTLTextureDescriptor*))g_origNewTexture)(self, _cmd, d);
}
void mtlLogTextures() {
    Class c = [MTLCreateSystemDefaultDevice() class];
    Method m = class_getInstanceMethod(c, @selector(newTextureWithDescriptor:));
    if (m) g_origNewTexture = method_setImplementation(m, (IMP)loggedNewTexture);
}
