// iOS host app for the metalgldrv test suite. Runs tests on a background
// thread after launch, prints results to stdout (captured by devicectl
// --console) and exits.
#import <UIKit/UIKit.h>
#import <QuartzCore/CAMetalLayer.h>
#include "harness.h"

@interface MGLMetalView : UIView
@end
@implementation MGLMetalView
+ (Class)layerClass { return [CAMetalLayer class]; }
@end

CAMetalLayer* g_runnerLayer = nil;  // available to window-surface tests

static void startTests() {
    NSArray<NSString*>* args = NSProcessInfo.processInfo.arguments;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        std::vector<std::string> storage;
        std::vector<char*> argv;
        for (NSString* a in args) storage.push_back(a.UTF8String);
        for (auto& s : storage) argv.push_back((char*)s.c_str());
        setvbuf(stdout, nullptr, _IOLBF, 0);
        int rc = mglRunTests((int)argv.size(), argv.data());
        printf("MGL_TESTS_DONE rc=%d\n", rc);
        fflush(stdout);
        fflush(stderr);
        exit(rc);
    });
}

@interface MGLSceneDelegate : UIResponder <UIWindowSceneDelegate>
@property(strong, nonatomic) UIWindow* window;
@end

@implementation MGLSceneDelegate
- (void)scene:(UIScene*)scene willConnectToSession:(UISceneSession*)session options:(UISceneConnectionOptions*)options {
    UIWindowScene* ws = (UIWindowScene*)scene;
    self.window = [[UIWindow alloc] initWithWindowScene:ws];
    UIViewController* vc = [UIViewController new];
    MGLMetalView* v = [[MGLMetalView alloc] initWithFrame:ws.coordinateSpace.bounds];
    v.contentScaleFactor = ws.traitCollection.displayScale;
    vc.view = v;
    self.window.rootViewController = vc;
    [self.window makeKeyAndVisible];
    g_runnerLayer = (CAMetalLayer*)v.layer;
    startTests();
}
@end

@interface MGLAppDelegate : UIResponder <UIApplicationDelegate>
@end

@implementation MGLAppDelegate
- (BOOL)application:(UIApplication*)app didFinishLaunchingWithOptions:(NSDictionary*)opts {
    return YES;
}
- (UISceneConfiguration*)application:(UIApplication*)app configurationForConnectingSceneSession:(UISceneSession*)session
                             options:(UISceneConnectionOptions*)options {
    UISceneConfiguration* c = [[UISceneConfiguration alloc] initWithName:@"Default" sessionRole:session.role];
    c.delegateClass = [MGLSceneDelegate class];
    return c;
}
@end

int main(int argc, char* argv[]) {
    @autoreleasepool {
        return UIApplicationMain(argc, argv, nil, NSStringFromClass([MGLAppDelegate class]));
    }
}
