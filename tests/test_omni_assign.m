// Stable integer bucketing must match a CPU oracle, including unused padding.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { E = 128, TOP = 8, GUARD = 16 };
typedef struct { int rows, H, F, padded; } moe_args;
static const int poison = 0x5a5ac3c3;
static void require(bool ok, const char *why) {
    if (!ok) { fprintf(stderr, "Omni assignment: %s\n", why); exit(1); }
}
static id<MTLBuffer> buffer(id<MTLDevice> dev, int words) {
    id<MTLBuffer> b = [dev newBufferWithLength:(NSUInteger)words*4 options:MTLResourceStorageModeShared];
    require(b != nil, "allocation");
    for (int i=0;i<words;i++) ((int *)b.contents)[i]=poison;
    return b;
}
static int expert_id(int i, int pattern) {
    switch (pattern) {
        case 0: return (i*37+(i/8)*13)%E;
        case 1: return i%8;
        case 2: return 120+i%8;
        case 3: return 56+i%8;
        case 4: return (i/32)%E;
        case 5: return 127; // More concentrated than the router permits.
        default: return ((uint32_t)i*2654435761u >> 16)%E;
    }
}
static void run_case(id<MTLDevice> dev, id<MTLCommandQueue> queue,
                     NSDictionary *ps, int rows, int pattern, int shift) {
    int live=rows*TOP, padded=live+(live<E?live:E)*31;
    int count[5]={live,E,E+1,padded,live};
    int starts[5]={GUARD+shift,GUARD,GUARD,GUARD,GUARD};
    id<MTLBuffer> buffers[5]; int *expected[5]; int words[5];
    for (int j=0;j<5;j++) {
        words[j]=starts[j]+count[j]+GUARD;
        buffers[j]=buffer(dev,words[j]);
        expected[j]=malloc((size_t)words[j]*4); require(expected[j]!=NULL,"CPU allocation");
        memcpy(expected[j],buffers[j].contents,(size_t)words[j]*4);
    }
    int *ids=expected[0]+starts[0], *counts=expected[1]+starts[1];
    int *offsets=expected[2]+starts[2], *rowmap=expected[3]+starts[3];
    int *slots=expected[4]+starts[4], cursor[E]={0};
    memset(counts,0,E*4);
    for (int i=0;i<live;i++) counts[ids[i]=expert_id(i,pattern)]++;
    offsets[0]=0;
    for (int e=0;e<E;e++) offsets[e+1]=offsets[e]+((counts[e]+31)/32)*32;
    require(offsets[E]<=padded,"CPU padding bound");
    for (int i=0;i<padded;i++) rowmap[i]=-1;
    for (int i=0;i<live;i++) {
        int e=ids[i], row=offsets[e]+cursor[e]++;
        rowmap[row]=i/8; slots[i]=row;
    }
    memcpy(buffers[0].contents,expected[0],(size_t)words[0]*4);
    moe_args a={rows,32,16,padded};
    id<MTLCommandBuffer> cb=[queue commandBuffer];
    id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    require(cb && enc,"command allocation");
    NSArray *stages=@[@"omni_assign_count",@"omni_assign_offsets",@"omni_assign_scatter"];
    for (NSString *name in stages) {
        [enc setComputePipelineState:ps[name]];
        [enc setBytes:&a length:sizeof(a) atIndex:0];
        for (int j=0;j<5;j++) [enc setBuffer:buffers[j] offset:(NSUInteger)starts[j]*4 atIndex:j+1];
        bool scan=[name isEqualToString:@"omni_assign_count"] || [name isEqualToString:@"omni_assign_scatter"];
        [enc dispatchThreadgroups:MTLSizeMake(scan?E:1,1,1) threadsPerThreadgroup:MTLSizeMake(scan?32:1,1,1)];
    }
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    require(cb.status==MTLCommandBufferStatusCompleted && !cb.error,cb.error.localizedDescription.UTF8String);
    for (int j=0;j<5;j++) {
        const int *got=buffers[j].contents;
        for (int i=0;i<words[j];i++) if (got[i]!=expected[j][i]) {
            fprintf(stderr,"rows=%d pattern=%d shift=%d buffer=%d word=%d got=%d expected=%d\n",
                    rows,pattern,shift,j,i,got[i],expected[j][i]); exit(1);
        }
    }
    for (int j=0;j<5;j++) free(expected[j]);
}
int main(int argc, char **argv) { @autoreleasepool {
    require(argc==1,"no arguments expected");
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); require(dev!=nil,"Metal device");
    NSError *error=nil;
    NSString *src=[NSString stringWithContentsOfFile:@"metal/clef.metal" encoding:NSUTF8StringEncoding error:&error];
    require(src!=nil,error.localizedDescription.UTF8String);
    MTLCompileOptions *opt=[MTLCompileOptions new]; opt.mathMode=MTLMathModeSafe;
    id<MTLLibrary> lib=[dev newLibraryWithSource:src options:opt error:&error]; require(lib!=nil,error.localizedDescription.UTF8String);
    NSMutableDictionary *ps=[NSMutableDictionary new];
    for (NSString *name in @[@"omni_assign_count",@"omni_assign_offsets",@"omni_assign_scatter"]) {
        id<MTLFunction> fn=[lib newFunctionWithName:name]; require(fn!=nil,"missing kernel");
        id<MTLComputePipelineState> pipeline=[dev newComputePipelineStateWithFunction:fn error:&error];
        require(pipeline!=nil,error.localizedDescription.UTF8String);
        require(pipeline.threadExecutionWidth==32,"requires a 32-lane SIMD group"); ps[name]=pipeline;
    }
    id<MTLCommandQueue> queue=[dev newCommandQueue]; require(queue!=nil,"command queue");
    const int sizes[]={1,15,16,17,31,32,33,127,128,129,255,256,257,1023,1024};
    int cases=0;
    for (size_t s=0;s<sizeof(sizes)/sizeof(sizes[0]);s++) for (int pattern=0;pattern<7;pattern++) for (int shift=0;shift<=37;shift+=37) {
        run_case(dev,queue,ps,sizes[s],pattern,shift); cases++;
    }
    printf("Omni assignment: %d cases exact against CPU; counts, offsets, row map, slots, input and guards passed\n",cases);
} return 0; }
