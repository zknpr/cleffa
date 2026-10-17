// Extra mergers normalize each four-patch group without consuming its pending bias.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void require(bool ok,const char *why) { if(!ok){fprintf(stderr,"Omni DeepStack: %s\n",why);exit(1);} }
static id<MTLBuffer> buffer(id<MTLDevice>d,size_t n) {
    id<MTLBuffer>b=[d newBufferWithLength:n options:MTLResourceStorageModeShared];require(b!=nil,"allocation");
    memset(b.contents,0xff,n);return b;
}
int main(void) { @autoreleasepool {
    id<MTLDevice>d=MTLCreateSystemDefaultDevice();require(d!=nil,"device");
    NSError *err=nil;MTLCompileOptions *opts=[MTLCompileOptions new];opts.mathMode=MTLMathModeSafe;
    NSString *src=[NSString stringWithContentsOfFile:@"metal/clef.metal" encoding:NSUTF8StringEncoding error:&err];
    id<MTLLibrary>lib=[d newLibraryWithSource:src options:opts error:&err];require(lib!=nil,err.localizedDescription.UTF8String);
    id<MTLFunction>fn=[lib newFunctionWithName:@"omni_vision_merger_norm"];require(fn!=nil,"missing grouped merger norm");
    id<MTLComputePipelineState>norm=[d newComputePipelineStateWithFunction:fn error:&err];require(norm!=nil,err.localizedDescription.UTF8String);
    fn=[lib newFunctionWithName:@"omni_vision_inject"];require(fn!=nil,"missing visual injection");
    id<MTLComputePipelineState>inject=[d newComputePipelineStateWithFunction:fn error:&err];require(inject!=nil,err.localizedDescription.UTF8String);
    fn=[lib newFunctionWithName:@"omni_vision_residual"];require(fn!=nil,"missing bias-before-residual kernel");
    id<MTLComputePipelineState>residual=[d newComputePipelineStateWithFunction:fn error:&err];require(residual!=nil,err.localizedDescription.UTF8String);
    id<MTLCommandQueue>queue=[d newCommandQueue];
    const int E=1152,M=4*E,R=3,H=64,T=7;
    {
        id<MTLBuffer>branch=buffer(d,16),old=buffer(d,16),bias=buffer(d,16);
        float values[]={-1e8f,1e8f,1.f,-1.f},state[]={1e8f,-1e8f,.125f,-.125f},biases[]={1,-1,.25f,-.25f};
        memcpy(branch.contents,values,16);memcpy(old.contents,state,16);memcpy(bias.contents,biases,16);
        struct{int H,rows;}a={4,1};
        id<MTLCommandBuffer>cb=[queue commandBuffer];id<MTLComputeCommandEncoder>enc=[cb computeCommandEncoder];
        [enc setComputePipelineState:residual];[enc setBytes:&a length:sizeof(a) atIndex:0];
        [enc setBuffer:branch offset:0 atIndex:1];[enc setBuffer:bias offset:0 atIndex:2];[enc setBuffer:old offset:0 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(4,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [enc endEncoding];[cb commit];[cb waitUntilCompleted];require(!cb.error,cb.error.localizedDescription.UTF8String);
        for(int i=0;i<4;i++)require(((float *)old.contents)[i]==state[i]+(values[i]+biases[i]),"projection bias must round before residual addition");
        require(((float *)old.contents)[0]==0,"old residual-then-bias order survived");
    }
    id<MTLBuffer>x=buffer(d,R*M*4),bias=buffer(d,E*4),w=buffer(d,M*4),b=buffer(d,M*4),out=buffer(d,(R*M+16)*4);
    float *xx=x.contents,*bb=bias.contents,*ww=w.contents,*bn=b.contents,*yy=out.contents;
    for(int i=0;i<R*M;i++)xx[i]=(i%41-20)*.07f+(i/E)*3.f;
    for(int i=0;i<E;i++)bb[i]=(i%17-8)*.01f;
    for(int i=0;i<M;i++){ww[i]=.5f+(i%7)*.125f;bn[i]=(i%13-6)*.03f;}
    NSData *saved=[NSData dataWithBytes:xx length:R*M*4];
    float clean[R*M];
    for(int pending=0;pending<2;pending++)for(int pass=0;pass<2;pass++) {
        if(pass)memset(yy,0xff,(R*M+16)*4);
        struct {int E,rows;float eps;int pending;} a={E,R,1e-6f,pending};
        id<MTLCommandBuffer>cb=[queue commandBuffer];id<MTLComputeCommandEncoder>enc=[cb computeCommandEncoder];
        [enc setComputePipelineState:norm];[enc setBytes:&a length:sizeof(a) atIndex:0];
        [enc setBuffer:x offset:0 atIndex:1];[enc setBuffer:bias offset:0 atIndex:2];
        [enc setBuffer:w offset:0 atIndex:3];[enc setBuffer:b offset:0 atIndex:4];[enc setBuffer:out offset:0 atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(R,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];[cb commit];[cb waitUntilCompleted];require(!cb.error,cb.error.localizedDescription.UTF8String);
        require(!memcmp(saved.bytes,xx,R*M*4),"merger consumed pending bias in the tower stream");
        for(int r=0;r<R;r++) {
            double mean=0,var=0;for(int i=0;i<M;i++)mean+=(float)(xx[r*M+i]+(pending?bb[i%E]:0));mean/=M;
            for(int i=0;i<M;i++){double z=(float)(xx[r*M+i]+(pending?bb[i%E]:0))-mean;var+=z*z;}var/=M;
            for(int i=0;i<M;i++) {
                double want=((float)(xx[r*M+i]+(pending?bb[i%E]:0))-mean)/sqrt(var+1e-6f)*ww[i]+bn[i];
                require(isfinite(yy[r*M+i])&&fabs(yy[r*M+i]-want)<3e-6,"grouped norm disagrees with FP64");
            }
        }
        if(pass)require(!memcmp(clean,yy,sizeof(clean)),"poison changed norm output");else memcpy(clean,yy,sizeof(clean));
        for(int i=0;i<16;i++)require(isnan(yy[R*M+i]),"norm guard overwritten");
    }
    id<MTLBuffer>hidden=buffer(d,(T*H+16)*4),features=buffer(d,R*H*4),map=buffer(d,T*4);
    int indices[]={-1,2,-1,0,-1,1,-1};memcpy(map.contents,indices,sizeof(indices));
    float original[T*H],previous[T*H];for(int i=0;i<T*H;i++)original[i]=(i%11-5)*.25f;
    for(int level=0;level<3;level++) {
        float *hh=hidden.contents,*ff=features.contents;memcpy(hh,original,sizeof(original));
        for(int i=0;i<R*H;i++)ff[i]=(level+1)*.125f+(i%7)*.0625f;
        struct{int T,H;}a={T,H};
        id<MTLCommandBuffer>cb=[queue commandBuffer];id<MTLComputeCommandEncoder>enc=[cb computeCommandEncoder];
        [enc setComputePipelineState:inject];[enc setBytes:&a length:sizeof(a) atIndex:0];
        [enc setBuffer:hidden offset:0 atIndex:1];[enc setBuffer:features offset:0 atIndex:2];[enc setBuffer:map offset:0 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(H,T,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        [enc endEncoding];[cb commit];[cb waitUntilCompleted];require(!cb.error,cb.error.localizedDescription.UTF8String);
        for(int t=0;t<T;t++)for(int i=0;i<H;i++) {
            float want=original[t*H+i]+(indices[t]<0?0:ff[indices[t]*H+i]);
            require(hh[t*H+i]==want,"injection changed a text row or used the wrong feature row");
        }
        if(level)require(memcmp(previous,hh,sizeof(previous)),"different extras gave identical output");
        memcpy(previous,hh,sizeof(previous));
        for(int i=0;i<16;i++)require(isnan(hh[T*H+i]),"injection guard overwritten");
    }
    puts("Omni DeepStack: grouped FP64 norm, pending bias ownership, poison, mapped injection and guards PASS");
}return 0;}
