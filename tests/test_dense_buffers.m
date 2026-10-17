// Dense head temporaries reuse completed decoder storage. Growth still commits atomically.
#include "../clef_metal.m"

@interface DenseFailDevice : NSObject
@property(nonatomic, strong) id<MTLDevice> inner;
@property(nonatomic) int calls, fail_at;
@end
@implementation DenseFailDevice
- (id)forwardingTargetForSelector:(SEL)selector { return self.inner; }
- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)n options:(MTLResourceOptions)options {
    if (++self.calls == self.fail_at) return nil;
    return [self.inner newBufferWithLength:n options:options];
}
@end

static void require(bool ok, const char *why) {
    if (!ok) { fprintf(stderr, "dense buffers: %s\n", why); exit(1); }
}
static NSArray *buffers(clef_gpu *g) {
    NSMutableArray *all = [NSMutableArray arrayWithArray:@[g->ids, g->pos, g->seq_start,
        g->seq_bounds, g->x, g->xn, g->P, g->Q, g->K, g->V, g->G, g->A, g->Xc,
        g->beta, g->gate, g->O, g->hfin, g->nh32, g->nh16, g->mem, g->mem16,
        g->mn16, g->mn32, g->ovf, g->attn_blk, g->img_row]];
    for (int i = 0; i < g->n_kv; i++) [all addObject:g->kv[i]];
    return all;
}
static void check_layout(clef_gpu *g, const clef_config *c, bool small_projection) {
    const bool omni = c->model.backbone == CLEF_OMNI;
    require((g->nh32 == g->x) == !omni, "hidden normalization storage reuse");
    require((g->hfin == g->P) == (!omni && !small_projection), "final normalization storage reuse");
    require(g->hfin != g->nh32, "normalization input aliases its output");
    require(g->hfin.length >= (size_t)g->cap * c->H * 4, "final normalization capacity");
    require(g->nh32.length >= (size_t)g->cap * c->H * 4, "hidden normalization capacity");
}
int main(void) {
    int failures_checked = 0;
    for (int omni = 0; omni < 2; omni++) for (int f32 = 0; f32 < 2; f32++)
        for (int small = 0; small < 2; small++) @autoreleasepool {
        clef_gpu *g = calloc(1, sizeof(*g)); require(g != NULL, "host allocation");
        DenseFailDevice *dev = [DenseFailDevice new]; dev.inner = MTLCreateSystemDefaultDevice();
        require(dev.inner != nil, "no GPU"); g->dev = (id<MTLDevice>)dev;
        g->head_f32 = f32; g->n_kv = 2;
        clef_config c = {0}; c.model.backbone = omni ? CLEF_OMNI : CLEF_QWEN35;
        c.H = small ? 1024 : 64; c.W = 32; c.ffn = 64;
        c.nh = c.nkv = c.Hk = c.Hv = 1; c.hd = 32; c.dk = c.dv = 16;
        char err[256];
        require(ensure_capacity(g, &c, 1, 1, err, sizeof(err)), err);
        check_layout(g, &c, small);
        const int allocations = dev.calls;
        NSArray *old = buffers(g);
        const int cap = g->cap;
        for (int i = 1; i <= allocations; i++) @autoreleasepool {
            dev.calls = 0; dev.fail_at = i;
            require(!ensure_capacity(g, &c, cap + 1, 1, err, sizeof(err)), "growth failure accepted");
            require(strstr(err, "previous buffers kept") != NULL, "missing allocation error");
            require(g->cap == cap && [old isEqualToArray:buffers(g)], "failed growth replaced live buffers");
            check_layout(g, &c, small);
            failures_checked++;
        }
        dev.calls = dev.fail_at = 0;
        require(ensure_capacity(g, &c, cap + 1, 1, err, sizeof(err)), err);
        require(g->cap == 2 * cap && g->x != old[4], "growth did not replace storage");
        check_layout(g, &c, small);
        require(ensure_capacity(g, &c, 1, 1, err, sizeof(err)), "smaller request failed");
        check_layout(g, &c, small);
        clef_gpu_close(g);
    }
    printf("dense buffers: 8 layouts, %d allocation failures, growth and shrink reuse PASS\n", failures_checked);
    return 0;
}
