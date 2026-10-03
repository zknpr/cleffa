/* Exercise real forwards, substituting only a completed command buffer's error
 * property. A hardware timeout is neither deterministic nor needed for this test. */
#include "../clef_metal.m"

@interface ErrorBuffer : NSObject
@property(nonatomic, strong) id<MTLCommandBuffer> inner;
@property(nonatomic) BOOL fail;
@end
@implementation ErrorBuffer
- (id)forwardingTargetForSelector:(SEL)selector { return self.inner; }
- (NSError *)error {
    if (self.fail) return [NSError errorWithDomain:@"clef-test" code:1
                                        userInfo:@{NSLocalizedDescriptionKey: @"injected execution failure"}];
    return self.inner.error;
}
@end

@interface ErrorQueue : NSObject
@property(nonatomic, strong) id<MTLCommandQueue> inner;
@property(nonatomic) int count, failAt;
@end
@implementation ErrorQueue
- (id<MTLCommandBuffer>)commandBuffer {
    ErrorBuffer *b = [ErrorBuffer new];
    b.inner = [self.inner commandBuffer];
    b.fail = ++self.count == self.failAt;
    return (id<MTLCommandBuffer>)b;
}
@end

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: test-metal-errors MODEL.gguf\n"); return 2; }
    @autoreleasepool {
        char err[512];
        clef_engine *e = clef_open(argv[1], err, sizeof(err));
        if (!e) { fprintf(stderr, "%s\n", err); return 1; }
        const char text[] = "{\"model\":\"m\",\"state\":\"test\",\"questions\":{\"q\":{\"type\":\"noul\"}}}";
        jarena *a = jarena_new();
        jval *req = json_parse(a, text, sizeof(text) - 1, err, sizeof(err));
        clef_record rec;
        clef_encode_opts opts = CLEF_ENCODE_DEFAULTS;
        if (!req || !clef_encode_request(e->tok, req, opts, &rec, err, sizeof(err))) {
            fprintf(stderr, "%s\n", err); return 1;
        }
        ErrorQueue *queue = [ErrorQueue new];
        queue.inner = e->gpu->queue;
        e->gpu->queue = (id<MTLCommandQueue>)queue;
        e->gpu->profile = true;
        float ***probs = NULL;
        if (!clef_run(e, &rec, 1, &probs, err, sizeof(err))) {
            fprintf(stderr, "clean forward: %s\n", err); return 1;
        }
        float clean[2];
        memcpy(clean, probs[0][0], sizeof(clean));
        clef_free_probs(&rec, 1, probs);
        const int points[] = { 1, 3, queue.count / 2, queue.count };
        int failures = 0;
        for (size_t i = 0; i < sizeof(points) / sizeof(points[0]); i++) {
            queue.count = 0;
            queue.failAt = points[i];
            probs = NULL;
            bool ok = clef_run(e, &rec, 1, &probs, err, sizeof(err));
            bool rejected = !ok && !probs && strstr(err, "injected execution failure") && queue.count == points[i];
            printf("execution error at buffer %d: %s\n", points[i], rejected ? "PASS" : "FAIL");
            failures += !rejected;
            clef_free_probs(&rec, 1, probs);

            queue.count = queue.failAt = 0;
            probs = NULL;
            ok = clef_run(e, &rec, 1, &probs, err, sizeof(err));
            bool recovered = ok && !memcmp(clean, probs[0][0], sizeof(clean));
            printf("subsequent forward: %s\n", recovered ? "PASS" : "FAIL");
            failures += !recovered;
            clef_free_probs(&rec, 1, probs);
        }
        clef_record_free(&rec);
        jarena_free(a);
        clef_close(e);
        return failures ? 1 : 0;
    }
}
