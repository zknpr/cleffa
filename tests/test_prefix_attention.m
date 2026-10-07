// Exact cached-versus-full FP32 attention, reusing the production fixture and float64 oracle.
#define main original_attention_main
#include "../bench/attention_bench.m"
#undef main

typedef struct { int nh, nkv, hd, T; float scale; int kvT, qoff; } prefix_args;

static id<MTLComputePipelineState> named_pipeline(id<MTLDevice> dev, const char *path, NSString *name) {
    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:@(path) encoding:NSUTF8StringEncoding error:&error];
    require(source != nil, error.localizedDescription.UTF8String);
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.mathMode = MTLMathModeSafe;
    id<MTLLibrary> lib = [dev newLibraryWithSource:source options:options error:&error];
    require(lib != nil, error.localizedDescription.UTF8String);
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    require(fn != nil, "missing named kernel");
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&error];
    require(ps != nil, error.localizedDescription.UTF8String);
    require(ps.threadExecutionWidth == 32, "cached attention requires 32-lane SIMDgroups");
    return ps;
}

typedef struct { int nh, nkv, hd, n_rot, row; float eps; } prep_args;
typedef struct { int nh, nkv, hd, n_rot, row; float eps; int kvT, koff; } prep_px_args;

static void prep_run(id<MTLCommandQueue> queue, id<MTLComputePipelineState> ps,
                     prep_px_args a, NSArray<id<MTLBuffer>> *b, int T, int L, bool cached) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    require(cb != nil && enc != nil, "prep command creation failed");
    [enc setComputePipelineState:ps];
    [enc setBytes:&a length:cached ? sizeof(a) : sizeof(prep_args) atIndex:0];
    for (int i=0;i<9;i++) {
        const NSUInteger offset = i==0 ? (NSUInteger)L*a.row*4 : i==3 ? (NSUInteger)L*4 : 0;
        [enc setBuffer:b[i] offset:offset atIndex:i+1];
    }
    [enc setBytes:&T length:sizeof(T) atIndex:10];
    [enc dispatchThreadgroups:MTLSizeMake(T,a.nh+a.nkv,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    require(cb.error == nil, cb.error.localizedDescription.UTF8String);
}

static void prep_checks(id<MTLDevice> dev, id<MTLCommandQueue> queue, const char *old, const char *candidate) {
    id<MTLComputePipelineState> plain = named_pipeline(dev,old,@"attn_prep");
    id<MTLComputePipelineState> cached = named_pipeline(dev,candidate,@"attn_prep_prefix");
    const int lengths[] = {129,161,1025,2049};
    const uint32_t guard = 0x7fc12345;
    int checks=0;
    full_fp32_values=true;
    for (int nh=16;nh<=24;nh+=8) for (int ti=0;ti<4;ti++) @autoreleasepool {
        const int T=lengths[ti], hd=256, nkv=4, row=2*(nh+nkv)*hd;
        const size_t nq=(size_t)nh*T*hd, nk=(size_t)nkv*T*hd;
        NSArray<id<MTLBuffer>> *full=@[buffer(dev,(size_t)T*row*4),buffer(dev,hd*4),buffer(dev,hd*4),
            buffer(dev,T*4),buffer(dev,32*4),buffer(dev,nq*4),buffer(dev,nk*4),buffer(dev,nk*4),buffer(dev,nq*4)];
        for (size_t i=0;i<(size_t)T*row;i++) ((float *)full[0].contents)[i]=value((uint32_t)i+31)*3;
        for (int d=0;d<hd;d++) {
            ((float *)full[1].contents)[d]=1+value(d+71)*.2f;
            ((float *)full[2].contents)[d]=1+value(d+97)*.2f;
        }
        for (int t=0;t<T;t++) ((int *)full[3].contents)[t]=t+13000;
        for (int d=0;d<32;d++) ((float *)full[4].contents)[d]=powf(1000000.0f,-(float)d/32);
        prep_px_args a={nh,nkv,hd,64,row,1e-6f,T,0};
        prep_run(queue,plain,a,full,T,0,false);
        const int caps[]={T,(T+1023)/1024*1024,(T+1023)/1024*1024+1024};
        for (int L=32;L<=96;L+=32) for (int ci=0;ci<3;ci++) @autoreleasepool {
            const int n=T-L, cap=caps[ci];
            const size_t cnq=(size_t)nh*n*hd, cnk=(size_t)nkv*cap*hd;
            NSArray<id<MTLBuffer>> *b=@[full[0],full[1],full[2],full[3],full[4],
                buffer(dev,(cnq+16)*4),buffer(dev,(cnk+16)*4),buffer(dev,(cnk+16)*4),buffer(dev,(cnq+16)*4)];
            for (int i=5;i<9;i++) for (size_t j=0;j<b[i].length/4;j++) ((uint32_t *)b[i].contents)[j]=guard;
            a.kvT=cap; a.koff=L;
            prep_run(queue,cached,a,b,n,L,true);
            for (int h=0;h<nh;h++)
                require(!memcmp((float *)b[5].contents+(size_t)h*n*hd,
                                (float *)full[5].contents+((size_t)h*T+L)*hd,(size_t)n*hd*4),"cached prep Q changed");
            require(!memcmp(b[8].contents,(float *)full[8].contents+(size_t)L*nh*hd,cnq*4),"cached prep G changed");
            for (int i=6;i<8;i++) {
                const uint32_t *got=b[i].contents;
                for (int h=0;h<nkv;h++) {
                    require(!memcmp(got+((size_t)h*cap+L)*hd,
                                    (float *)full[i].contents+((size_t)h*T+L)*hd,(size_t)n*hd*4),"cached prep K/V changed");
                    for (int t=0;t<cap;t++) if (t<L || t>=T)
                        for (int d=0;d<hd;d++) require(got[((size_t)h*cap+t)*hd+d]==guard,"cached prep overwrote prefix or capacity slack");
                }
            }
            for (int i=5;i<9;i++) {
                const size_t count=(i==6||i==7)?cnk:cnq;
                const uint32_t *got=b[i].contents;
                for (size_t j=count;j<count+16;j++) require(got[j]==guard,"cached prep output overrun");
            }
            checks++;
        }
    }
    printf("PASS %d cached prep layouts: exact Q/K/V/G and untouched prefix/slack/guards\n",checks);fflush(stdout);
}

static NSArray<id<MTLBuffer>> *prefix_inputs(id<MTLDevice> dev, attn_args a,
                                           NSArray<id<MTLBuffer>> *full, int L, int cap) {
    const int T = a.T-L;
    size_t nq = (size_t)a.nh*T*a.hd, nk = (size_t)a.nkv*cap*a.hd;
    NSArray<id<MTLBuffer>> *b = @[
        buffer(dev,(nq+128*a.hd)*4), buffer(dev,(nk+128*a.hd)*4), buffer(dev,(nk+128*a.hd)*4),
        buffer(dev,nq*4), buffer(dev,T*4), buffer(dev,(nq+16)*2), buffer(dev,T*4)];
    float *q=b[0].contents, *k=b[1].contents, *v=b[2].contents;
    for (size_t i=0;i<nk+128*a.hd;i++) k[i]=v[i]=NAN;
    for (int h=0;h<a.nh;h++)
        memcpy(q+(size_t)h*T*a.hd,(float *)full[0].contents+((size_t)h*a.T+L)*a.hd,(size_t)T*a.hd*4);
    for (int h=0;h<a.nkv;h++) {
        memcpy(k+(size_t)h*cap*a.hd,(float *)full[1].contents+(size_t)h*a.T*a.hd,(size_t)a.T*a.hd*4);
        memcpy(v+(size_t)h*cap*a.hd,(float *)full[2].contents+(size_t)h*a.T*a.hd,(size_t)a.T*a.hd*4);
        const int padding=h==a.nkv-1?64:MIN(64,cap-a.T);
        memset(k+((size_t)h*cap+a.T)*a.hd,0,(size_t)padding*a.hd*4);
        memset(v+((size_t)h*cap+a.T)*a.hd,0,(size_t)padding*a.hd*4);
    }
    memcpy(b[3].contents,(float *)full[3].contents+(size_t)L*a.nh*a.hd,nq*4);
    for (size_t i=nq+8*a.hd;i<nq+128*a.hd;i++) q[i]=NAN;
    uint16_t *output=b[5].contents;
    for (size_t i=0;i<nq+16;i++) output[i]=0x7fc1;
    return b;
}

static void prefix_run(id<MTLCommandQueue> queue,id<MTLComputePipelineState> ps,prefix_args a,
                       NSArray<id<MTLBuffer>> *b,int f16) {
    id<MTLCommandBuffer> cb=[queue commandBuffer];
    id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    require(cb!=nil&&enc!=nil,"cached command creation failed");
    act_args ac={f16,65504.0f};
    [enc setComputePipelineState:ps];
    [enc setBytes:&a length:sizeof(a) atIndex:0];
    for(int i=0;i<6;i++) [enc setBuffer:b[i] offset:0 atIndex:i+1];
    [enc setBytes:&ac length:sizeof(ac) atIndex:7];
    [enc setBuffer:b[6] offset:0 atIndex:8];
    [enc setThreadgroupMemoryLength:4*(8*64+64)*4 atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((a.T+31)/32,a.nh,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    [enc endEncoding];[cb commit];[cb waitUntilCompleted];
    require(cb.error==nil,cb.error.localizedDescription.UTF8String);
}

int main(int argc,char **argv) {
    require(argc==3,"usage: prefix-attention-checks QUALIFIED.metal CANDIDATE.metal");
    @autoreleasepool {
        id<MTLDevice> dev=MTLCreateSystemDefaultDevice();
        require(dev!=nil,"no Metal device");
        id<MTLCommandQueue> queue=[dev newCommandQueue];
        require(queue!=nil,"no command queue");
        id<MTLComputePipelineState> fa=pipeline(dev,argv[1],original);
        const attn_variant prefetch={4,64};
        id<MTLComputePipelineState> pf=pipeline(dev,argv[1],prefetch);
        prep_checks(dev,queue,argv[1],argv[2]);
        id<MTLComputePipelineState> cached=named_pipeline(dev,argv[2],@"attention_prefix_64");
        const int lengths[]={129,159,160,161,255,256,257,1023,1024,1025,2047,2048,2049,8072,16347};
        int checks=0;
        for(int mode=0;mode<2;mode++) {
            full_fp32_values=mode!=0;
            for(int nh=16;nh<=24;nh+=8) for(int f16=0;f16<=1;f16++)
            for(size_t shape=0;shape<sizeof(lengths)/sizeof(lengths[0]);shape++) @autoreleasepool {
                const int T=lengths[shape];
                attn_args a={nh,4,256,T,1.0f/16};
                NSArray<id<MTLBuffer>> *full=inputs(dev,a,&T,1,64);
                const attn_variant variant=T>=1024?prefetch:original;
                run(queue,T>=1024?pf:fa,a,full,f16,variant);
                printf("T=%d nh=%d f16=%d values=%s",T,nh,f16,mode?"fp32":"grid");
                check(a,full,f16,false);
                int cuts[]={32,64,96,(T-1)/32*32};
                int caps[]={T,(T+1023)/1024*1024,(T+1023)/1024*1024+1024};
                for(int ci=0;ci<4;ci++) for(int bi=0;bi<3;bi++) @autoreleasepool {
                    const int L=cuts[ci],cap=caps[bi],remaining=T-L;
                    prefix_args pa={nh,4,256,remaining,1.0f/16,cap,L};
                    NSArray<id<MTLBuffer>> *b=prefix_inputs(dev,a,full,L,cap);
                    prefix_run(queue,cached,pa,b,f16);
                    const size_t count=(size_t)remaining*nh*256;
                    const uint16_t *output=b[5].contents;
                    require(!memcmp(output,(uint16_t *)full[5].contents+(size_t)L*nh*256,count*2),
                            "cached attention changed full-pass bits");
                    for(size_t i=0;i<count;i++) require(isfinite(decode(output[i],f16)),"nonfinite cached output");
                    for(size_t i=count;i<count+16;i++) require(output[i]==0x7fc1,"cached output overrun");
                    for(int t=0;t<remaining;t++) require(((int *)b[6].contents)[t]==0,"unexpected cached overflow flag");
                    checks++;
                }
                puts(" cache_bits=identical guards=OK");fflush(stdout);
            }
        }
        printf("PASS %d cached layouts: exact bits, float64 baseline and poisoned guards\n",checks);
    }
    return 0;
}
