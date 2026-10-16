// Independent FP64 windowed attention, packing invariance and invalid-value reporting.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void require(bool ok,const char *why) {
    if(!ok){fprintf(stderr,"Omni audio: %s\n",why);exit(1);}
}
static id<MTLBuffer> buffer(id<MTLDevice> dev,size_t bytes) {
    id<MTLBuffer> b=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    require(b!=nil,"allocation");
    for(size_t i=0;i<bytes/4;i++)((float *)b.contents)[i]=NAN;
    return b;
}
static float sample(int t,int d,int seed){return sinf((t*71+d*13+seed)*.013f)*.7f;}
static void finish(id<MTLCommandBuffer> cb,id<MTLComputeCommandEncoder> enc) {
    require(cb && enc,"command allocation");
    [enc endEncoding];[cb commit];[cb waitUntilCompleted];
    require(cb.error==nil,cb.error.localizedDescription.UTF8String);
}
static float *attention(id<MTLDevice> dev,id<MTLCommandQueue> queue,id<MTLComputePipelineState> ps,
                        int count,int before,int mode) {
    const int heads=3,H=heads*64,T=before+count+5;
    id<MTLBuffer> q=buffer(dev,T*H*4),k=buffer(dev,T*H*4),v=buffer(dev,T*H*4);
    id<MTLBuffer> starts=buffer(dev,T*4),ends=buffer(dev,T*4),out=buffer(dev,(T*H+32)*4),status=buffer(dev,4);
    *(uint32_t *)status.contents=0;
    for(int t=0;t<T;t++) {
        int base=t<before?0:t<before+count?before:before+count;
        int stop=t<before?before:t<before+count?before+count:T;
        ((int *)starts.contents)[t]=base;((int *)ends.contents)[t]=stop;
        for(int d=0;d<H;d++) {
            ((float *)q.contents)[t*H+d]=mode==1?0:sample(t-base,d,1);
            ((float *)k.contents)[t*H+d]=mode==1?0:sample(t-base,d,7);
            ((float *)v.contents)[t*H+d]=mode==1?.1234567f+d*.0001f:sample(t-base,d,13);
        }
    }
    if(mode==2)((float *)q.contents)[before*H]=NAN;
    if(mode==3)((float *)k.contents)[before*H]=INFINITY;
    if(mode==4)((float *)v.contents)[before*H]=-INFINITY;
    if(mode==5)((int *)ends.contents)[before]=T+1;
    struct{int T,H,heads;float scale;} args={T,H,heads,.125f};
    id<MTLCommandBuffer> cb=[queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    [enc setComputePipelineState:ps];[enc setBytes:&args length:sizeof(args) atIndex:0];
    id<MTLBuffer> buffers[]={q,k,v,starts,ends,out,status};
    for(int i=0;i<7;i++)[enc setBuffer:buffers[i] offset:0 atIndex:i+1];
    [enc dispatchThreadgroups:MTLSizeMake(T,heads,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    finish(cb,enc);
    if(mode>=2){require(*(uint32_t *)status.contents!=0,"invalid attention was hidden");return NULL;}
    require(*(uint32_t *)status.contents==0,"unexpected nonfinite status");
    double worst=0;
    for(int t=before;t<before+count;t++)for(int h=0;h<heads;h++) {
        double scores[104],max=-INFINITY,sum=0;
        for(int s=0;s<count;s++) {
            double dot=0;
            for(int d=0;d<64;d++)dot+=(double)((float *)q.contents)[t*H+h*64+d]*((float *)k.contents)[(before+s)*H+h*64+d];
            scores[s]=dot*.125;if(scores[s]>max)max=scores[s];
        }
        for(int s=0;s<count;s++){scores[s]=exp(scores[s]-max);sum+=scores[s];}
        for(int d=0;d<64;d++) {
            double value=0;
            for(int s=0;s<count;s++)value+=scores[s]/sum*((float *)v.contents)[(before+s)*H+h*64+d];
            float got=((float *)out.contents)[t*H+h*64+d];
            require(isfinite(got),"nonfinite output");
            double delta=fabs(got-value);if(delta>worst)worst=delta;
        }
    }
    require(worst<(mode==1?6e-8:2e-6),"FP64 attention disagreement");
    for(int i=T*H;i<T*H+32;i++)require(isnan(((float *)out.contents)[i]),"output guard overwritten");
    float *copy=malloc(count*H*4);require(copy!=NULL,"copy allocation");
    memcpy(copy,(float *)out.contents+before*H,count*H*4);
    printf("audio attention %d mode%d: max %.9g\n",count,mode,worst);
    return copy;
}

static void norm_bias(id<MTLDevice> dev,id<MTLCommandQueue> queue,NSDictionary *ps) {
    const int H=1280,T=3;
    id<MTLBuffer> x=buffer(dev,T*H*4),w=buffer(dev,H*4),bias=buffer(dev,H*4);
    id<MTLBuffer> out=buffer(dev,(T*H+32)*4),residual=buffer(dev,T*H*4),status=buffer(dev,4);
    for(int d=0;d<H;d++){((float *)w.contents)[d]=1+sample(0,d,3);((float *)bias.contents)[d]=sample(0,d,5);}
    for(int t=0;t<T;t++)for(int d=0;d<H;d++)((float *)x.contents)[t*H+d]=sample(t,d,8)+t*3;
    *(uint32_t *)status.contents=0;
    struct{int H;float eps;} args={H,1e-5f};
    id<MTLCommandBuffer> cb=[queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    [enc setComputePipelineState:ps[@"audio_norm"]];[enc setBytes:&args length:sizeof(args) atIndex:0];
    id<MTLBuffer> buffers[]={x,w,bias,out,status};
    for(int i=0;i<5;i++)[enc setBuffer:buffers[i] offset:0 atIndex:i+1];
    [enc dispatchThreadgroups:MTLSizeMake(T,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];finish(cb,enc);
    for(int t=0;t<T;t++) {
        double mean=0,var=0;for(int d=0;d<H;d++)mean+=((float *)x.contents)[t*H+d];mean/=H;
        for(int d=0;d<H;d++){double delta=((float *)x.contents)[t*H+d]-mean;var+=delta*delta;}var/=H;
        for(int d=0;d<H;d++) {
            double ref=(((float *)x.contents)[t*H+d]-mean)/sqrt(var+args.eps)*((float *)w.contents)[d]+((float *)bias.contents)[d];
            require(fabs(((float *)out.contents)[t*H+d]-ref)<4e-6,"FP64 layer norm");
        }
    }
    for(int d=T*H;d<T*H+32;d++)require(isnan(((float *)out.contents)[d]),"norm output guard");
    for(int mode=0;mode<3;mode++) {
        for(int i=0;i<T*H;i++){((float *)out.contents)[i]=sample(i/H,i%H,9)*10;((float *)residual.contents)[i]=.25f;}
        struct{int H,rows,mode;} a={H,T,mode};
        cb=[queue commandBuffer];enc=[cb computeCommandEncoder];
        [enc setComputePipelineState:ps[@"audio_bias"]];[enc setBytes:&a length:sizeof(a) atIndex:0];
        [enc setBuffer:out offset:0 atIndex:1];[enc setBuffer:bias offset:0 atIndex:2];
        [enc setBuffer:residual offset:0 atIndex:3];[enc setBuffer:status offset:0 atIndex:4];
        [enc dispatchThreads:MTLSizeMake(T*H,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];finish(cb,enc);
        for(int i=0;i<T*H;i++) {
            float b=sample(i/H,i%H,9)*10+((float *)bias.contents)[i%H];
            double ref=mode==1?.5*b*(1+erf(b/sqrt(2.))):mode==2?(float)(b+.25f):b;
            float got=((float *)(mode==2?residual:out).contents)[i];
            require(isfinite(got) && fabs(got-ref)<1e-6,"FP64 bias/GELU/residual");
        }
    }
    require(!*(uint32_t *)status.contents,"unexpected norm/bias status");
}

static void convolution_layout(id<MTLDevice> dev,id<MTLCommandQueue> queue,NSDictionary *ps) {
    // Deliberately odd widths and multiple channels reveal NHWC/channel-first mixups.
    for(int width=1;width<=7;width+=2) {
        const int chunks=3,height=5,channels=4,oh=3,ow=(width+1)/2,K=channels*9;
        id<MTLBuffer> x=buffer(dev,chunks*height*width*channels*4);
        id<MTLBuffer> out=buffer(dev,(chunks*oh*ow*K+32)*4);
        for(int i=0;i<chunks*height*width*channels;i++)((float *)x.contents)[i]=i+1;
        struct{int chunks,height,width,channels;} a={chunks,height,width,channels};
        id<MTLCommandBuffer> cb=[queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
        [enc setComputePipelineState:ps[@"audio_im2col"]];[enc setBytes:&a length:sizeof(a) atIndex:0];
        [enc setBuffer:x offset:0 atIndex:1];[enc setBuffer:out offset:0 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(chunks*oh*ow*K,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];finish(cb,enc);
        for(int c=0;c<chunks;c++)for(int y=0;y<oh;y++)for(int t=0;t<ow;t++)
        for(int ch=0;ch<channels;ch++)for(int dy=0;dy<3;dy++)for(int dt=0;dt<3;dt++) {
            int sy=y*2-1+dy,st=t*2-1+dt;
            float ref=sy<0 || sy>=height || st<0 || st>=width?0:((float *)x.contents)[((c*height+sy)*width+st)*channels+ch];
            int index=((c*oh+y)*ow+t)*K+ch*9+dy*3+dt;
            require(((float *)out.contents)[index]==ref,"im2col index/padding");
        }
        for(int i=chunks*oh*ow*K;i<chunks*oh*ow*K+32;i++)require(isnan(((float *)out.contents)[i]),"im2col guard");
        cb=[queue commandBuffer];enc=[cb computeCommandEncoder];
        [enc setComputePipelineState:ps[@"audio_conv_pack"]];[enc setBytes:&a length:sizeof(a) atIndex:0];
        [enc setBuffer:x offset:0 atIndex:1];[enc setBuffer:out offset:0 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(chunks*height*width*channels,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];finish(cb,enc);
        for(int c=0;c<chunks;c++)for(int t=0;t<width;t++)for(int ch=0;ch<channels;ch++)for(int f=0;f<height;f++)
            require(((float *)out.contents)[((c*width+t)*channels+ch)*height+f]==
                    ((float *)x.contents)[((c*height+f)*width+t)*channels+ch],"conv projection packing");
    }
}

int main(int argc,char **argv) {
    @autoreleasepool {
        id<MTLDevice> dev=MTLCreateSystemDefaultDevice();require(dev!=nil,"no Metal device");
        NSError *error=nil;
        NSString *src=[NSString stringWithContentsOfFile:@"metal/clef.metal" encoding:NSUTF8StringEncoding error:&error];
        MTLCompileOptions *options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
        id<MTLLibrary> lib=[dev newLibraryWithSource:src options:options error:&error];
        require(lib!=nil,error.localizedDescription.UTF8String);
        NSMutableDictionary *pipelines=[NSMutableDictionary new];
        for(NSString *name in @[@"audio_attention",@"audio_norm",@"audio_bias",@"audio_im2col",@"audio_conv_pack",@"audio_gather"]) {
            id<MTLFunction> fn=[lib newFunctionWithName:name];require(fn!=nil,name.UTF8String);
            id<MTLComputePipelineState> p=[dev newComputePipelineStateWithFunction:fn error:&error];
            require(p!=nil,error.localizedDescription.UTF8String);require(p.threadExecutionWidth==32,"SIMD width");
            pipelines[name]=p;
        }
        if(argc==2 && !strcmp(argv[1],"--compile-only"))return 0;
        id<MTLCommandQueue> queue=[dev newCommandQueue];require(queue!=nil,"queue");
        norm_bias(dev,queue,pipelines);convolution_layout(dev,queue,pipelines);
        int sizes[]={1,7,8,13,31,32,33,103,104};
        for(int mode=0;mode<2;mode++)for(unsigned i=0;i<sizeof(sizes)/sizeof(sizes[0]);i++) {
            float *single=attention(dev,queue,pipelines[@"audio_attention"],sizes[i],0,mode);
            float *packed=attention(dev,queue,pipelines[@"audio_attention"],sizes[i],3,mode);
            require(!memcmp(single,packed,sizes[i]*3*64*4),"packing changed bytes");free(single);free(packed);
        }
        for(int mode=2;mode<=5;mode++)attention(dev,queue,pipelines[@"audio_attention"],33,3,mode);
        puts("Omni audio kernels passed");
    }
    return 0;
}
