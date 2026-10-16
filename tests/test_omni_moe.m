// Model-free routed expert oracle. Inputs/weights are fixed binary fractions.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { E = 128, TOP = 8, H = 32, F = 16, CHUNK = 1024, PAD = CHUNK*8+128*31 };
typedef struct { int rows, H, F, padded; } moe_args;
typedef struct { int T, N, K; } gemm_args;
static void require(bool ok, const char *why) { if (!ok) { fprintf(stderr, "Omni MoE: %s\n", why); exit(1); } }
static id<MTLBuffer> buffer(id<MTLDevice> dev, size_t bytes) {
    id<MTLBuffer> b = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    require(b != nil, "allocation"); memset(b.contents, 0xff, bytes); return b;
}
static float sample(int i) { return ((i*19+7)%31-15)/64.f; }
static uint16_t bf16(float f) { uint32_t bits; memcpy(&bits, &f, 4); require((bits&65535)==0, "test sample not exact BF16"); return bits>>16; }

static float *run(id<MTLDevice> dev, id<MTLCommandQueue> queue, NSDictionary *ps, int total, int mode, int chunk) {
    id<MTLBuffer> x = buffer(dev, (size_t)total*H*4), residual = buffer(dev, ((size_t)total*H+16)*4);
    id<MTLBuffer> scores = buffer(dev, (size_t)total*E*4), ids = buffer(dev, (size_t)total*TOP*4), weights = buffer(dev, (size_t)total*TOP*4);
    id<MTLBuffer> low = buffer(dev, (size_t)total*E*4);
    memset(low.contents, 0, total*E*4);
    id<MTLBuffer> status = buffer(dev, 4), counts = buffer(dev, E*4), offsets = buffer(dev, (E+1)*4);
    id<MTLBuffer> rowmap = buffer(dev, PAD*4), slots = buffer(dev, CHUNK*TOP*4);
    id<MTLBuffer> gathered = buffer(dev, PAD*H*4), up = buffer(dev, PAD*2*F*4), act = buffer(dev, PAD*F*4), down = buffer(dev, PAD*H*4);
    id<MTLBuffer> uw = buffer(dev, E*2*F*H*2), dw = buffer(dev, E*H*F*2);
    float *xx=x.contents, *yy=residual.contents, *lg=scores.contents;
    uint16_t *u=uw.contents, *d=dw.contents;
    *(uint32_t *)status.contents=0;
    for (int i=0;i<total*H;i++) xx[i]=sample(i), yy[i]=sample(i+2);
    for (int i=0;i<E*2*F*H;i++) u[i]=bf16(sample(i));
    for (int i=0;i<E*H*F;i++) d[i]=bf16(sample(i+5));
    for (int t=0;t<total;t++) for(int e=0;e<E;e++) {
        lg[t*E+e] = mode == 1 ? 0.f : mode == 2 || mode == 6 ? -(float)e/32.f : (float)((e*37+t*13)%E)/64.f;
        if (mode == 7) lg[t*E+e] = (float)e/32.f;
        if (mode == 8) lg[t*E+e] = e >= 56 && e < 64 ? 0.f : -16.f;
    }
    if (mode == 2) lg[8] = nextafterf(lg[7], INFINITY);
    if (mode == 6) { lg[8] = lg[7]; ((float *)low.contents)[8] = 1e-9f; }
    if (mode >= 3 && mode <= 5) lg[11] = mode == 3 ? NAN : mode == 4 ? INFINITY : -INFINITY;
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    require(cb && enc, "command allocation");
    for (int start=0;start<total;start+=chunk) {
        int n=total-start<chunk?total-start:chunk;
        int live=n*TOP, experts=live<E?live:E, padded=live+experts*31;
        moe_args a={n,H,F,padded};
        [enc setComputePipelineState:ps[@"omni_route"]];
        [enc setBuffer:scores offset:(NSUInteger)start*E*4 atIndex:0];
        [enc setBuffer:ids offset:(NSUInteger)start*TOP*4 atIndex:1];
        [enc setBuffer:weights offset:(NSUInteger)start*TOP*4 atIndex:2];
        [enc setBuffer:status offset:0 atIndex:3];
        [enc setBuffer:low offset:(NSUInteger)start*E*4 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [enc setComputePipelineState:ps[@"omni_assign_count"]];
        [enc setBytes:&a length:sizeof(a) atIndex:0];
        [enc setBuffer:ids offset:(NSUInteger)start*TOP*4 atIndex:1];
        [enc setBuffer:counts offset:0 atIndex:2]; [enc setBuffer:offsets offset:0 atIndex:3];
        [enc setBuffer:rowmap offset:0 atIndex:4]; [enc setBuffer:slots offset:0 atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(E,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [enc setComputePipelineState:ps[@"omni_assign_offsets"]];
        [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [enc setComputePipelineState:ps[@"omni_assign_scatter"]];
        [enc dispatchThreadgroups:MTLSizeMake(E,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [enc setComputePipelineState:ps[@"omni_gather"]];
        [enc setBytes:&a length:sizeof(a) atIndex:0];
        [enc setBuffer:x offset:(NSUInteger)start*H*4 atIndex:1]; [enc setBuffer:rowmap offset:0 atIndex:2];
        [enc setBuffer:gathered offset:0 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(H,padded,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        gemm_args ga={n,2*F,H};
        [enc setComputePipelineState:ps[@"omni_expert_gemm"]];
        [enc setBytes:&ga length:sizeof(ga) atIndex:0];
        [enc setBuffer:gathered offset:0 atIndex:1]; [enc setBuffer:uw offset:0 atIndex:2]; [enc setBuffer:up offset:0 atIndex:3];
        [enc setBuffer:counts offset:0 atIndex:4]; [enc setBuffer:offsets offset:0 atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((2*F+127)/128,(padded+31)/32,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        [enc setComputePipelineState:ps[@"omni_swiglu"]];
        [enc setBytes:&a length:sizeof(a) atIndex:0];
        [enc setBuffer:up offset:0 atIndex:1]; [enc setBuffer:act offset:0 atIndex:2]; [enc setBuffer:rowmap offset:0 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(F,padded,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        ga=(gemm_args){n,H,F};
        [enc setComputePipelineState:ps[@"omni_expert_gemm"]];
        [enc setBytes:&ga length:sizeof(ga) atIndex:0];
        [enc setBuffer:act offset:0 atIndex:1]; [enc setBuffer:dw offset:0 atIndex:2]; [enc setBuffer:down offset:0 atIndex:3];
        [enc setBuffer:counts offset:0 atIndex:4]; [enc setBuffer:offsets offset:0 atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((H+127)/128,(padded+31)/32,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        [enc setComputePipelineState:ps[@"omni_combine"]];
        [enc setBytes:&a length:sizeof(a) atIndex:0]; [enc setBuffer:down offset:0 atIndex:1];
        [enc setBuffer:ids offset:(NSUInteger)start*TOP*4 atIndex:2]; [enc setBuffer:weights offset:(NSUInteger)start*TOP*4 atIndex:3];
        [enc setBuffer:slots offset:0 atIndex:4]; [enc setBuffer:residual offset:(NSUInteger)start*H*4 atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    }
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted]; require(!cb.error, cb.error.localizedDescription.UTF8String);
    if (mode >= 3 && mode <= 5) { require(*(uint32_t *)status.contents != 0,"nonfinite router not reported"); return NULL; }
    require(*(uint32_t *)status.contents==0,"unexpected route error");
    double worst=0;
    for (int t=0;t<total;t++) {
        int order[E]; for(int e=0;e<E;e++)order[e]=e;
        double score[E]; for(int e=0;e<E;e++)score[e]=(double)lg[t*E+e]+((float *)low.contents)[t*E+e];
        for(int i=1;i<E;i++) { int j=i, e=order[i]; while(j>0 && score[order[j-1]]<score[e]) { order[j]=order[j-1];j--; } order[j]=e; }
        double den=0, prob[TOP]; for(int j=0;j<TOP;j++) { prob[j]=exp((double)lg[t*E+order[j]]-lg[t*E+order[0]]);den+=prob[j]; }
        double out[H]={0};
        for(int j=0;j<TOP;j++) {
            require(((int *)ids.contents)[t*TOP+j]==order[j],"wrong selected expert or tie order");
            double weight=((float *)weights.contents)[t*TOP+j];
            require(fabs(weight-prob[j]/den)<2e-7,"router probability mismatch");
            int e=order[j]; double intermediate[F];
            for(int f=0;f<F;f++) { double gate=0,upv=0; for(int h=0;h<H;h++) {gate+=xx[t*H+h]*sample((e*2*F+f)*H+h);upv+=xx[t*H+h]*sample((e*2*F+F+f)*H+h);} intermediate[f]=gate/(1+exp(-gate))*upv; }
            for(int h=0;h<H;h++) for(int f=0;f<F;f++) out[h]+=intermediate[f]*sample((e*H+h)*F+f+5)*weight;
        }
        for(int h=0;h<H;h++) { double err=fabs(yy[t*H+h]-(sample(t*H+h+2)+out[h]));if(err>worst)worst=err;require(isfinite(yy[t*H+h])&&err<2e-6,"expert FP64 mismatch or poisoned padding"); }
    }
    for(int i=0;i<16;i++)require(isnan(yy[total*H+i]),"output guard overwritten");
    printf("Omni MoE rows=%d chunk=%d mode=%d max FP64 error %.3g\n",total,chunk,mode,worst);
    float *copy = malloc((size_t)total*H*4);
    require(copy != NULL, "result allocation");
    memcpy(copy,yy,(size_t)total*H*4);
    return copy;
}

int main(int argc, char **argv) { @autoreleasepool {
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice();require(dev!=nil,"no Metal device");
    NSError *error=nil;NSString *src=[NSString stringWithContentsOfFile:@"metal/clef.metal" encoding:NSUTF8StringEncoding error:&error];
    MTLCompileOptions *opt=[MTLCompileOptions new];opt.mathMode=MTLMathModeSafe;
    id<MTLLibrary> lib=[dev newLibraryWithSource:src options:opt error:&error];require(lib!=nil,error.localizedDescription.UTF8String);
    NSMutableDictionary *ps=[NSMutableDictionary new];
    for(NSString *name in @[@"omni_route",@"omni_assign_count",@"omni_assign_offsets",@"omni_assign_scatter",@"omni_gather",@"omni_expert_gemm",@"omni_swiglu",@"omni_combine"]) {
        id<MTLFunction> fn=[lib newFunctionWithName:name];require(fn!=nil,"missing Omni MoE kernel");
        ps[name]=[dev newComputePipelineStateWithFunction:fn error:&error];require(ps[name]!=nil,error.localizedDescription.UTF8String);
    }
    if(argc==2 && !strcmp(argv[1],"--compile-only"))return 0;
    id<MTLCommandQueue> queue=[dev newCommandQueue];
    for (int mode = 0; mode < 3; mode++) {
        for (int chunk = 256; chunk <= CHUNK; chunk *= 2) {
            float *a = run(dev,queue,ps,chunk-1,mode,chunk), *b = run(dev,queue,ps,chunk,mode,chunk);
            float *c = run(dev,queue,ps,chunk+1,mode,chunk);
            require(!memcmp(a,b,(chunk-1)*H*4) && !memcmp(b,c,chunk*H*4), "chunk edge or expert occupancy changed prior rows");
            free(a); free(b); free(c);
        }
        float *small = run(dev,queue,ps,CHUNK+1,mode,256);
        float *large = run(dev,queue,ps,CHUNK+1,mode,CHUNK);
        require(!memcmp(small,large,(CHUNK+1)*H*4), "changing expert chunk size changed output bytes");
        free(small); free(large);
    }
    // Exercise the live-expert padding bound when fewer than 128 assignments
    // exist, and tile edges where an expert's row count changes its GEMM mask.
    int short_rows[] = {1,15,16,17,31,32,33,127,128,129};
    for (size_t i=0;i<sizeof(short_rows)/sizeof(short_rows[0]);i++) free(run(dev,queue,ps,short_rows[i],0,CHUNK));
    // Empty experts repeat their offset. Concentrate routes at the final eight
    // experts or in the middle to cover leading, internal and trailing empty runs.
    int sparse_rows[] = {1,31,32,33,CHUNK-1,CHUNK,CHUNK+1};
    for (int mode=7;mode<=8;mode++) for (size_t i=0;i<sizeof(sparse_rows)/sizeof(sparse_rows[0]);i++) {
        int n=sparse_rows[i];
        float *small=run(dev,queue,ps,n,mode,256), *large=run(dev,queue,ps,n,mode,CHUNK);
        require(!memcmp(small,large,(size_t)n*H*4), "sparse expert dispatch changed chunk output bytes");
        free(small); free(large);
    }
    for (int mode = 3; mode <= 5; mode++) for (int n = CHUNK-1; n <= CHUNK+1; n++) run(dev,queue,ps,n,mode,CHUNK);
    free(run(dev,queue,ps,CHUNK+1,6,CHUNK));
    puts("Omni MoE: routing, expert products, ties, chunk edges, nonfinite status and poison passed");
} return 0; }
