// Compile a .metal file the way the engine does (runtime, safe math); print errors.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
int main(int argc, char **argv) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSString *src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:nil];
        MTLCompileOptions *o = [MTLCompileOptions new];
        o.mathMode = MTLMathModeSafe;
        NSError *e = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:o error:&e];
        if (!lib) { fprintf(stderr, "%s\n", e.localizedDescription.UTF8String); return 1; }
        for (NSString *n in lib.functionNames) printf("%s\n", n.UTF8String);
        return 0;
    }
}
