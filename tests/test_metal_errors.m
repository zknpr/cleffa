/* Exercise real forwards, substituting only a completed command buffer's error
 * property. A hardware timeout is neither deterministic nor needed for this test. */
#include "../clef_metal.m"

@interface ErrorBuffer : NSObject
@property(nonatomic, strong) id<MTLCommandBuffer> inner;
@property(nonatomic) BOOL fail;
@property(nonatomic) BOOL nilEncoder;
@end
@implementation ErrorBuffer
- (id)forwardingTargetForSelector:(SEL)selector { return self.inner; }
- (id<MTLComputeCommandEncoder>)computeCommandEncoder {
    return self.nilEncoder ? nil : [self.inner computeCommandEncoder];
}
- (NSError *)error {
    if (self.fail) return [NSError errorWithDomain:@"clef-test" code:1
                                        userInfo:@{NSLocalizedDescriptionKey: @"injected execution failure"}];
    return self.inner.error;
}
@end

@interface ErrorQueue : NSObject
@property(nonatomic, strong) id<MTLCommandQueue> inner;
@property(nonatomic) int count, failAt, nilAt, nilEncoderAt;
@end
@implementation ErrorQueue
- (id<MTLCommandBuffer>)commandBuffer {
    self.count++;
    if (self.count == self.nilAt) return nil;
    ErrorBuffer *b = [ErrorBuffer new];
    b.inner = [self.inner commandBuffer];
    b.fail = self.count == self.failAt;
    b.nilEncoder = self.count == self.nilEncoderAt;
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
        if (!clef_keep_warm(e, err, sizeof(err))) {
            fprintf(stderr, "keep-warm before first forward: %s\n", err); return 1;
        }
        queue.count = 0;
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
        // Idle passes may read every allocation, including padding, but must not
        // write activation data that the CPU head still owns until the next forward.
        id<MTLBuffer> buffers[] = { e->gpu->ids, e->gpu->pos, e->gpu->seq_start, e->gpu->seq_bounds,
            e->gpu->x, e->gpu->xn, e->gpu->P, e->gpu->Q, e->gpu->K, e->gpu->V, e->gpu->G,
            e->gpu->attn_blk,
            e->gpu->A, e->gpu->Xc, e->gpu->beta, e->gpu->gate, e->gpu->O, e->gpu->hfin,
            e->gpu->nh32, e->gpu->nh16, e->gpu->mem, e->gpu->mem16, e->gpu->mn16,
            e->gpu->mn32, e->gpu->ovf, e->gpu->inv_freq };
        NSMutableArray<id<MTLBuffer>> *checked = [NSMutableArray new];
        for (size_t i = 0; i < sizeof(buffers) / sizeof(buffers[0]); i++)
            if (buffers[i]) [checked addObject:buffers[i]];
        for (int i = 0; i < e->gpu->n_kv; i++) if (e->gpu->kv[i]) [checked addObject:e->gpu->kv[i]];
        NSMutableArray<NSData *> *snapshots = [NSMutableArray new];
        for (id<MTLBuffer> b in checked) [snapshots addObject:[NSData dataWithBytes:b.contents length:b.length]];
        bool unchanged = clef_keep_warm(e, err, sizeof(err));
        for (NSUInteger i = 0; i < checked.count; i++)
            unchanged = unchanged && !memcmp(checked[i].contents, snapshots[i].bytes, snapshots[i].length);
        printf("keep-warm preserves all allocated activation bytes: %s\n", unchanged ? "PASS" : "FAIL");
        failures += !unchanged;
        snapshots = nil;
        for (int kind = 0; kind < 3; kind++) {
            queue.count = 0;
            queue.nilAt = kind == 0; queue.nilEncoderAt = kind == 1; queue.failAt = kind == 2;
            bool ok = clef_keep_warm(e, err, sizeof(err));
            bool rejected = !ok && strstr(err, kind == 2 ? "injected execution failure" : "cannot create a command buffer");
            printf("keep-warm error %d propagated: %s\n", kind, rejected ? "PASS" : "FAIL");
            failures += !rejected;
            queue.count = queue.nilAt = queue.nilEncoderAt = queue.failAt = 0;
            ok = clef_keep_warm(e, err, sizeof(err));
            printf("keep-warm subsequent call: %s\n", ok ? "PASS" : "FAIL");
            failures += !ok;
        }
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
