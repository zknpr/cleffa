// Reservation is atomic: every failed growth preserves the live audio arena.
#include "../clef_metal_audio.m"
#include <assert.h>

@interface AudioFakeBuffer : NSObject
@property NSMutableData *data;
@end
@implementation AudioFakeBuffer
- (void *)contents { return self.data.mutableBytes; }
@end
@interface AudioFakeDevice : NSObject
@property NSUInteger maxBufferLength;
@property int calls;
@property int failAt;
@end
@implementation AudioFakeDevice
- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)n options:(MTLResourceOptions)options {
    self.calls++;
    if(self.calls==self.failAt)return nil;
    // Only the position buffer is initialized during reservation; avoid huge
    // fake allocations while checking ownership of every other arena buffer.
    AudioFakeBuffer *b=[AudioFakeBuffer new];
    if(n==AUDIO_ROWS*AUDIO_H*4)b.data=[NSMutableData dataWithLength:n];
    return (id<MTLBuffer>)b;
}
@end
int main(void) { @autoreleasepool {
    AudioFakeDevice *dev=[AudioFakeDevice new];dev.maxBufferLength=(NSUInteger)1<<34;
    clef_gpu *g=calloc(1,sizeof(*g));assert(g);g->dev=(id<MTLDevice>)dev;
    clef_audio_config c={.layers=32,.H=1280,.ffn=5120,.heads=20,.hd=64,.mel_bins=128,.conv_channels=480,
        .output=2048,.chunk_frames=100,.window_frames=800,.eps=1e-5f};
    char err[256];
    assert(clef_audio_gpu_reserve(g,&c,1,err,sizeof(err)));
    int allocations=dev.calls;assert(allocations==17);
    clef_audio_gpu *saved=g->audio;
    assert(clef_audio_gpu_reserve(g,&c,32,err,sizeof(err)) && g->audio==saved && dev.calls==allocations);
    for(int fail=1;fail<=allocations;fail++) {
        dev.calls=0;dev.failAt=fail;
        assert(!clef_audio_gpu_reserve(g,&c,33,err,sizeof(err)) && g->audio==saved && g->audio->cap==32);
    }
    dev.failAt=0;
    assert(clef_audio_gpu_reserve(g,&c,33,err,sizeof(err)) && g->audio!=saved && g->audio->cap==64);
    saved=g->audio;
    int sizes[]={-1,0,65537,INT_MAX};
    for(unsigned i=0;i<sizeof(sizes)/sizeof(sizes[0]);i++)
        assert(!clef_audio_gpu_reserve(g,&c,sizes[i],err,sizeof(err)) && g->audio==saved);
    c.H=INT_MAX;assert(!clef_audio_gpu_reserve(g,&c,2,err,sizeof(err)) && g->audio==saved);c.H=1280;
    dev.maxBufferLength=1024;
    assert(!clef_audio_gpu_reserve(g,&c,65,err,sizeof(err)) && g->audio==saved);
    clef_audio_gpu_free(g);assert(!g->audio);clef_audio_gpu_free(g);g->dev=nil;free(g);
    puts("audio arena: 17 failed growth points, recovery and bounds passed");
}return 0;}
