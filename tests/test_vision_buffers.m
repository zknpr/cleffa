// Vision capacity replacement is atomic, including the compensation scratch buffer.
// ASan also exercises keep-warm with every activation and all 16 KV slots populated.
#include "../clef_metal.m"

@interface VisionFailDevice : NSObject
@property(nonatomic, strong) id<MTLDevice> inner;
@property(nonatomic) int calls, fail_at;
@end
@implementation VisionFailDevice
- (id)forwardingTargetForSelector:(SEL)selector { return self.inner; }
- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)n options:(MTLResourceOptions)options {
    if (++self.calls == self.fail_at) return nil;
    return [self.inner newBufferWithLength:n options:options];
}
@end

static void require(bool ok, const char *why) {
    if (!ok) { fprintf(stderr, "vision buffers: %s\n", why); exit(1); }
}
static NSArray *vision_buffers(clef_gpu *g) {
    return @[g->vpatch, g->vpatch16, g->vposidx, g->vposw, g->vx, g->vxn, g->vqkv,
             g->vq, g->vk, g->vv, g->va, g->vff, g->vff16, g->vm, g->vm16, g->vsplit];
}
int main(void) {
    @autoreleasepool {
        clef_gpu *g = calloc(1, sizeof(*g)); require(g != NULL, "host allocation");
        VisionFailDevice *dev = [VisionFailDevice new]; dev.inner = MTLCreateSystemDefaultDevice();
        require(dev.inner != nil, "no GPU"); g->dev = (id<MTLDevice>)dev;
        g->vis_f32 = g->vis_comp = true;
        clef_config c = {0}; c.v_E = 1152; c.v_ff = 4304; c.v_in = 1536; c.v_merge = 2;
        char err[256];
        require(ensure_vision_capacity(g, &c, 4, 4, err, sizeof(err)), "initial capacity");
        require(g->vsplit.length == (size_t)g->vcap * c.v_in * 4, "compensation scratch size");
        NSArray *old = vision_buffers(g); const int cap = g->vcap, total = g->vcap_total;
        for (int fail = 1; fail <= (int)old.count; fail++) {
            dev.calls = 0; dev.fail_at = fail;
            require(!ensure_vision_capacity(g, &c, cap + 1, total + 1, err, sizeof(err)), "allocation failure accepted");
            require(strstr(err, "previous buffers kept") && g->vcap == cap && g->vcap_total == total, "failure contract");
            NSArray *after = vision_buffers(g);
            for (NSUInteger i = 0; i < old.count; i++) require(old[i] == after[i], "partial buffer replacement");
            int calls = dev.calls;
            require(ensure_vision_capacity(g, &c, cap, total, err, sizeof(err)) && calls == dev.calls, "old capacity unusable");
        }
        dev.calls = dev.fail_at = 0;
        require(ensure_vision_capacity(g, &c, cap + 1, total + 1, err, sizeof(err)), "growth recovery");
        require(g->vcap == 2 * cap && g->vcap_total == 2 * total, "growth dimensions");
        // Fill every other activation slot; vision buffers above are already populated.
        id<MTLBuffer> tiny = buf(g, 16);
        g->weights = g->ids = g->pos = g->seq_start = g->seq_bounds = tiny;
        g->x = g->xn = g->P = g->Q = g->K = g->V = g->G = g->A = g->Xc = g->beta = g->gate = g->O = tiny;
        g->hfin = g->nh32 = g->nh16 = g->mem = g->mem16 = g->mn16 = g->mn32 = g->ovf = g->attn_blk = tiny;
        g->gdn_w = g->gdn_u = g->gdn_ke = g->gdn_a = g->gdn_e = g->img_row = g->feat = tiny;
        g->n_kv = 16; for (int i = 0; i < g->n_kv; i++) g->kv[i] = tiny;
        g->keep = buf(g, 16); g->queue = [dev.inner newCommandQueue];
        NSError *error = nil;
        NSString *src = [[NSString alloc] initWithBytes:clef_metal_src length:clef_metal_src_len encoding:NSUTF8StringEncoding];
        MTLCompileOptions *opts = [MTLCompileOptions new]; opts.mathMode = MTLMathModeSafe;
        id<MTLLibrary> lib = [dev.inner newLibraryWithSource:src options:opts error:&error];
        require(lib != nil, error.localizedDescription.UTF8String);
        g->ps = [NSMutableDictionary new]; require(pipeline(g, lib, "keepalive", err, sizeof(err)), err);
        require(clef_gpu_keepalive(g, err, sizeof(err)), err);
        clef_gpu_prefix *px = clef_gpu_prefix_new(); require(px != NULL, "prefix allocation");
        require(prefix_vision_reserve(g, px, 64, err, sizeof(err)), "initial cached features");
        id<MTLBuffer> saved = px->vision;
        dev.calls = 0; dev.fail_at = 1;
        require(!prefix_vision_reserve(g, px, 128, err, sizeof(err)) && px->vision == saved,
                "failed feature growth replaced the live allocation");
        dev.calls = dev.fail_at = 0;
        require(prefix_vision_reserve(g, px, 128, err, sizeof(err)), "cached feature growth recovery");
        require(clef_gpu_prefix_bytes(px) == 128, "cached features missing from budget");
        px->cap = 1;
        require(clef_gpu_prefix_keepalive(g, px, err, sizeof(err)), err);
        require(prefix_vision_reserve(g, px, 0, err, sizeof(err)) && !px->vision,
                "text replacement retained image features");
        clef_gpu_prefix_free(px);
        clef_gpu_close(g);
        puts("vision buffers: 16 allocation failures/recovery, cache feature growth/budget and maximum keep-warm buffer list PASS");
    }
    return 0;
}
