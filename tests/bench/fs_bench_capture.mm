// fsbench FSB_CAPTURE: Metal GPU capture of the process's default device (Apple's OpenGL and metalgldrv both use it).
#import <Metal/Metal.h>

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
