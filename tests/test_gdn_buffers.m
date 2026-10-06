// Fail each chunk-preprocessing allocation and verify atomic capacity replacement.
#include "../clef_metal.m"

@interface FailBufferDevice : NSObject
@property(nonatomic, strong) id<MTLDevice> inner;
@property(nonatomic) int calls, fail_at;
@end
@implementation FailBufferDevice
- (id)forwardingTargetForSelector:(SEL)selector { return self.inner; }
- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options {
    if (++self.calls == self.fail_at) return nil;
    return [self.inner newBufferWithLength:length options:options];
}
@end

static void require(bool ok, const char *why) {
    if (!ok) { fprintf(stderr, "gdn buffers: %s\n", why); exit(1); }
}

int main(void) {
    @autoreleasepool {
        clef_gpu *g = calloc(1, sizeof(*g)); require(g != NULL, "host allocation");
        FailBufferDevice *dev = [FailBufferDevice new];
        dev.inner = MTLCreateSystemDefaultDevice(); require(dev.inner != nil, "no GPU");
        g->dev = (id<MTLDevice>)dev;
        clef_config c = {0}; c.Hv = 48;
        char err[256];
        require(ensure_gdn_capacity(g, &c, 0, err, sizeof(err)) && !g->gdn_cap && !dev.calls, "empty capacity allocates");
        require(ensure_gdn_capacity(g, &c, 31, err, sizeof(err)) && g->gdn_cap == 32, "initial capacity");
        int calls = dev.calls;
        require(ensure_gdn_capacity(g, &c, 32, err, sizeof(err)) && dev.calls == calls, "existing capacity reallocates");
        for (int failure = 1; failure <= 5; failure++) {
            const int cap = g->gdn_cap;
            NSArray *old = @[g->gdn_w, g->gdn_u, g->gdn_ke, g->gdn_a, g->gdn_e];
            dev.calls = 0; dev.fail_at = failure;
            require(!ensure_gdn_capacity(g, &c, cap + 1, err, sizeof(err)), "allocation failure accepted");
            require(strstr(err, "previous buffers kept") != NULL && g->gdn_cap == cap, "failure contract");
            NSArray *after = @[g->gdn_w, g->gdn_u, g->gdn_ke, g->gdn_a, g->gdn_e];
            for (int i = 0; i < 5; i++) require(old[i] == after[i], "partial buffer replacement");
            calls = dev.calls;
            require(ensure_gdn_capacity(g, &c, cap, err, sizeof(err)) && calls == dev.calls, "old capacity unusable");
            dev.calls = dev.fail_at = 0;
            require(ensure_gdn_capacity(g, &c, cap + 1, err, sizeof(err)) && g->gdn_cap == cap + 32, "growth recovery");
        }
        calls = dev.calls;
        require(!ensure_gdn_capacity(g, &c, INT_MAX, err, sizeof(err)) && dev.calls == calls, "rounded length overflow");
        require(!gdn_chunked(&c, 4095) && gdn_chunked(&c, 4096) && gdn_chunked(&c, 8192), "27B dispatch boundary");
        c.Hv = 32; require(!gdn_chunked(&c, 16347), "unqualified Flash dispatch");
        clef_gpu_close(g);
        puts("gdn buffers: 5 allocation failures, recovery, overflow and dispatch PASS");
    }
    return 0;
}
