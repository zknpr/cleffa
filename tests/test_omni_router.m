// Compensated FP32 router products against independently accumulated CPU FP64.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void require(bool ok, const char *why) { if (!ok) { fprintf(stderr,"Omni router: %s\n",why); exit(1); } }
static uint32_t state = 97;
static float sample(void) { state=state*1664525u+1013904223u; return ((int)(state>>16)-32768)*.001234567f; }
static float decode(uint16_t b) { uint32_t u=(uint32_t)b<<16;float f;memcpy(&f,&u,4);return f; }
static id<MTLBuffer> alloc(id<MTLDevice> dev,size_t n) {
    id<MTLBuffer>b=[dev newBufferWithLength:n options:MTLResourceStorageModeShared]; require(b!=nil,"allocation");
    memset(b.contents,0xff,n);return b;
}
int main(void) { @autoreleasepool {
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice();require(dev!=nil,"device");
    NSError *err=nil;MTLCompileOptions *opts=[MTLCompileOptions new];opts.mathMode=MTLMathModeSafe;
    NSString *src=[NSString stringWithContentsOfFile:@"metal/clef.metal" encoding:NSUTF8StringEncoding error:&err];
    id<MTLLibrary>lib=[dev newLibraryWithSource:src options:opts error:&err];require(lib!=nil,err.localizedDescription.UTF8String);
    id<MTLFunction>fn=[lib newFunctionWithName:@"omni_router"];require(fn!=nil,"missing compensated router kernel");
    id<MTLComputePipelineState>ps=[dev newComputePipelineStateWithFunction:fn error:&err];require(ps!=nil,err.localizedDescription.UTF8String);
    id<MTLCommandQueue>queue=[dev newCommandQueue];
    const int K=2048,N=128,T=33;
    id<MTLBuffer>x=alloc(dev,T*K*4),w=alloc(dev,N*K*2),hi=alloc(dev,(T*N+16)*4),lo=alloc(dev,(T*N+16)*4);
    float *xx=x.contents,*hh=hi.contents,*ll=lo.contents;uint16_t*ww=w.contents;
    for(int i=0;i<T*K;i++)xx[i]=sample();
    for(int i=0;i<N*K;i++){float f=sample()*.01f;uint32_t bits;memcpy(&bits,&f,4);ww[i]=bits>>16;}
    float previous[33*128];
    for(int rows=31;rows<=33;rows++) {
        struct {int T,N,K;} a={rows,N,K};
        id<MTLCommandBuffer>cb=[queue commandBuffer];id<MTLComputeCommandEncoder>enc=[cb computeCommandEncoder];require(cb&&enc,"commands");
        [enc setComputePipelineState:ps];[enc setBytes:&a length:sizeof(a) atIndex:0];
        [enc setBuffer:x offset:0 atIndex:1];[enc setBuffer:w offset:0 atIndex:2];
        [enc setBuffer:hi offset:0 atIndex:3];[enc setBuffer:lo offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(rows,N,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [enc endEncoding];[cb commit];[cb waitUntilCompleted];require(!cb.error,cb.error.localizedDescription.UTF8String);
        double worst=0;
        for(int t=0;t<rows;t++)for(int e=0;e<N;e++) {
            double want=0;for(int k=0;k<K;k++)want+=(double)xx[t*K+k]*decode(ww[e*K+k]);
            double error=fabs(((double)hh[t*N+e]+ll[t*N+e])-want);if(error>worst)worst=error;
            require(isfinite(hh[t*N+e])&&isfinite(ll[t*N+e])&&error<1e-9,"FP64 product mismatch");
        }
        if(rows>31)require(!memcmp(previous,hh,(rows-1)*N*4),"neighbor rows changed projection");
        memcpy(previous,hh,rows*N*4);
        for(int i=0;i<16;i++)require(isnan(hh[T*N+i])&&isnan(ll[T*N+i]),"output guard overwritten");
        printf("Omni router rows=%d compensated FP64 max error %.3g\n",rows,worst);
    }
}return 0;}
