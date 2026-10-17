// A failed arena allocation must retain the previous state, including warm buffers.
#include "../clef_metal_omni.m"
#include <assert.h>
#include <unistd.h>

@interface OmniFakeBuffer : NSObject
@property NSMutableData *storage;
@end
@implementation OmniFakeBuffer
- (void *)contents { return self.storage.mutableBytes; }
- (NSUInteger)length { return self.storage.length; }
@end

@interface OmniFakeDevice : NSObject
@property NSUInteger maxBufferLength;
@property int failAt;
@property int calls;
@property bool backed;
@end
@implementation OmniFakeDevice
- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)n options:(MTLResourceOptions)options {
    self.calls++;
    if (self.calls == self.failAt) return nil;
    if (self.backed) {
        OmniFakeBuffer *b = [OmniFakeBuffer new];
        b.storage = [NSMutableData dataWithLength:n];
        return (id<MTLBuffer>)b;
    }
    // Reservation tests inspect ownership only; no shader or buffer contents are used.
    return (id<MTLBuffer>)[NSObject new];
}
@end

int main(void) { @autoreleasepool {
    OmniFakeDevice *dev = [OmniFakeDevice new];
    dev.maxBufferLength = (NSUInteger)1 << 34;
    clef_config c = { .H = 2048, .ffn = 768, .n_expert = 128, .n_expert_used = 8 };
    char err[256];
    clef_gpu *g = calloc(1, sizeof(*g));
    g->dev = (id<MTLDevice>)dev;
    assert(clef_omni_reserve(g, &c, 295, err, sizeof(err)));
    int allocations = dev.calls;
    assert(allocations > 0);
    clef_omni_gpu *previous = g->omni;
    assert(clef_omni_reserve(g, &c, 65536, err, sizeof(err)) && g->omni == previous);
    assert(dev.calls == allocations);  // context length does not grow the expert arena
    assert(!clef_omni_reserve(g, &c, -1, err, sizeof(err)) && g->omni == previous);
    clef_config bad = c; bad.H = INT_MAX;
    assert(!clef_omni_reserve(g, &bad, 1, err, sizeof(err)) && g->omni == previous);
    clef_omni_free(g); assert(g->omni == NULL);
    for (int fail = 1; fail <= allocations; fail++) {
        dev.calls = 0; dev.failAt = fail;
        assert(!clef_omni_reserve(g, &c, 295, err, sizeof(err)) && !g->omni);
        dev.failAt = 0;
        assert(clef_omni_reserve(g, &c, 1, err, sizeof(err)));
        previous = g->omni;
        dev.failAt = dev.calls+1;
        assert(clef_omni_reserve(g, &c, 64000, err, sizeof(err)) && g->omni == previous);
        clef_omni_free(g);
    }
    dev.failAt = 0;
    assert(clef_omni_reserve(g, &c, 295, err, sizeof(err)));
    dev.backed = true;
    assert(clef_omni_trace_reserve(g, 295, 2, 3, true, err, sizeof(err)));
    id<MTLBuffer> old_trace = g->omni->trace;
    assert(old_trace.length == 6*sizeof(omni_route_row));
    dev.failAt = dev.calls+1;
    assert(!clef_omni_trace_reserve(g, 295, 2, 4, true, err, sizeof(err)));
    assert(g->omni->trace == old_trace && g->omni->trace_rows == 3);
    assert(!clef_omni_trace_reserve(g, 295, 2, -1, true, err, sizeof(err)));
    omni_route_row *rows = old_trace.contents;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 128; j++) rows[i].logits[j] = 128-j;
        for (int j = 0; j < 8; j++) { rows[i].ids[j] = j; rows[i].weights[j] = 0.125f; }
    }
    char path[] = "/tmp/cleffa-route-test-XXXXXX";
    int fd = mkstemp(path); assert(fd >= 0); close(fd);
    clef_engine e = {.gpu = g, .routes_valid = true, .capture_routes = true};
    assert(clef_dump_routes(&e, path, err, sizeof(err)));
    FILE *f = fopen(path, "r"); assert(f);
    char line[4096]; int lines = 0;
    while (fgets(line, sizeof(line), f)) {
        char prefix[80];
        snprintf(prefix, sizeof(prefix), "{\"layer\":%d,\"token\":%d,", lines/3, 292+lines%3);
        assert(!strncmp(line, prefix, strlen(prefix)) && strstr(line, "\"margin\":1}"));
        lines++;
    }
    assert(lines == 6); fclose(f); unlink(path);
    rows[0].logits[0] = NAN;
    assert(!clef_dump_routes(&e, path, err, sizeof(err)) && access(path, F_OK));
    rows[0].logits[0] = 128;
    // A failed host pass must invalidate even a previously complete GPU capture.
    assert(!clef_run(&e, NULL, 2, NULL, err, sizeof(err)));
    assert(!clef_dump_routes(&e, path, err, sizeof(err)) && access(path, F_OK));
    assert(clef_omni_trace_reserve(g, 295, 2, 3, false, err, sizeof(err)));
    assert(!g->omni->trace && !g->omni->trace_rows);
    clef_omni_free(g);
    g->dev = nil; free(g);
    printf("Omni scratch: %d allocation failures/recovery and context-independent arena passed\n", allocations);
} return 0; }
