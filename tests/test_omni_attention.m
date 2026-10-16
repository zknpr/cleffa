// FP64 normalization/rotary/causal oracle, packing invariance and poisoned tails.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int nh, nkv, T; float eps; } prep_args;
typedef struct { int nh, nkv, hd, T; float scale; } attention_args;
static void require(bool ok, const char *why) {
    if (!ok) { fprintf(stderr, "Omni attention: %s\n", why); exit(1); }
}
static id<MTLBuffer> buffer(id<MTLDevice> dev, size_t bytes) {
    id<MTLBuffer> b = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    require(b != nil, "allocation");
    for (size_t i = 0; i < bytes/4; i++) ((float *)b.contents)[i] = NAN;
    return b;
}
static float sample(int i) { return sinf((float)i*.37f)*.7f; }

static float *run(id<MTLDevice> dev, id<MTLCommandQueue> queue,
                  id<MTLComputePipelineState> prep, id<MTLComputePipelineState> attention,
                  int length, bool packed, int mode) {
    const bool sharp = mode == 1;
    const int nh = 8, nkv = 1, hd = 128, before = packed ? 7 : 0, after = packed ? 3 : 0;
    const int T = before+length+after, row = (nh+2*nkv)*hd;
    id<MTLBuffer> qkv = buffer(dev, (size_t)T*row*4), norm = buffer(dev, 256*4);
    id<MTLBuffer> pos = buffer(dev, (size_t)T*3*4), freq = buffer(dev, 64*4), starts = buffer(dev, (size_t)T*4);
    id<MTLBuffer> q = buffer(dev, ((size_t)T*nh+32)*hd*4);
    id<MTLBuffer> k = buffer(dev, ((size_t)T*nkv+32)*hd*4), v = buffer(dev, ((size_t)T*nkv+32)*hd*4);
    id<MTLBuffer> out = buffer(dev, ((size_t)T*nh*hd+32)*4);
    id<MTLBuffer> status = buffer(dev, 4);
    *(uint32_t *)status.contents = 0;
    float *source = qkv.contents, *norms = norm.contents, *iv = freq.contents;
    int32_t *positions = pos.contents, *ss = starts.contents;
    for (int i = 0; i < 256; i++) norms[i] = (sharp ? 8.f : 1.f)+sample(i+3)*.2f;
    for (int i = 0; i < 64; i++) iv[i] = 1.f/powf(1e6f, 2.f*i/128.f);
    for (int t = 0; t < T; t++) {
        int start = t < before ? 0 : t < before+length ? before : before+length;
        int local = t-start;
        ss[t] = start;
        positions[3*t] = local+5; positions[3*t+1] = 2*local+1; positions[3*t+2] = 3*local+2;
        for (int i = 0; i < row; i++) source[(size_t)t*row+i] = sample(local*row+i);
        if (mode == 2 || mode == 6) memset(source+(size_t)t*row, 0, (nh+nkv)*hd*4); // exactly flat scores
        if (mode == 6) for (int d=0; d<hd; d++) source[(size_t)t*row+(nh+nkv)*hd+d] = .1234567f+(float)d*.0001f;
    }
    if (mode >= 3 && mode <= 5) source[(size_t)(before+length-1)*row] = mode == 3 ? NAN : mode == 4 ? INFINITY : -INFINITY;
    prep_args pa = {nh, nkv, T, 1e-6f};
    attention_args aa = {nh, nkv, hd, T, 1.f/sqrtf(128.f)};
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    require(cb && enc, "command allocation");
    [enc setComputePipelineState:prep];
    [enc setBytes:&pa length:sizeof(pa) atIndex:0];
    [enc setBuffer:qkv offset:0 atIndex:1];
    [enc setBuffer:norm offset:0 atIndex:2];
    [enc setBuffer:norm offset:128*4 atIndex:3];
    [enc setBuffer:pos offset:0 atIndex:4];
    [enc setBuffer:freq offset:0 atIndex:5];
    [enc setBuffer:q offset:0 atIndex:6];
    [enc setBuffer:k offset:0 atIndex:7];
    [enc setBuffer:v offset:0 atIndex:8];
    [enc setBuffer:status offset:0 atIndex:9];
    [enc dispatchThreadgroups:MTLSizeMake(T, nh+nkv, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc setComputePipelineState:attention];
    [enc setBytes:&aa length:sizeof(aa) atIndex:0];
    [enc setBuffer:q offset:0 atIndex:1];
    [enc setBuffer:k offset:0 atIndex:2];
    [enc setBuffer:v offset:0 atIndex:3];
    [enc setBuffer:starts offset:0 atIndex:4];
    [enc setBuffer:out offset:0 atIndex:5];
    [enc setThreadgroupMemoryLength:nh*(8*32+64)*4 atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((T+7)/8, nkv, 1) threadsPerThreadgroup:MTLSizeMake(32*nh, 1, 1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    require(cb.error == nil, cb.error.localizedDescription.UTF8String);
    if (mode >= 3 && mode <= 5) {
        // Softmax max/masking can hide invalid live Q. Preparation must report
        // it explicitly before any masked reduction can turn it into zero.
        require(!isfinite(((float *)q.contents)[(size_t)(before+length-1)*hd]), "nonfinite Q was hidden");
        require(*(uint32_t *)status.contents != 0, "nonfinite attention was not reported");
        return NULL;
    }
    require(*(uint32_t *)status.contents == 0, "unexpected nonfinite attention status");
    if (mode == 6) {
        double worst=0;
        for (int t=before; t<before+length; t++) for (int h=0; h<nh; h++) for (int d=0; d<hd; d++) {
            float expected=source[(size_t)t*row+(nh+nkv)*hd+d];
            double delta=fabs(((float *)out.contents)[((size_t)t*nh+h)*hd+d]-expected);
            if(delta>worst) worst=delta;
        }
        printf("long flat %d: max %.9g\n",length,worst);
        require(worst < 6e-8,"long attention accumulation lost constant values");
        float *copy=malloc((size_t)length*nh*hd*4); require(copy!=NULL,"long copy");
        memcpy(copy,(float *)out.contents+(size_t)before*nh*hd,(size_t)length*nh*hd*4);
        return copy;
    }
    // Independent FP64 RMS/RoPE, including the four temporal-only leftover pairs.
    double *cq = calloc((size_t)T*nh*hd, sizeof(double)), *ck = calloc((size_t)T*nkv*hd, sizeof(double));
    require(cq && ck, "CPU oracle allocation");
    for (int t = 0; t < T; t++) for (int h = 0; h < nh+nkv; h++) {
        double x[128], sum = 0;
        for (int d = 0; d < 128; d++) { x[d] = source[(size_t)t*row+h*hd+d]; sum += x[d]*x[d]; }
        for (int d = 0; d < 128; d++) x[d] *= norms[(h < nh ? 0 : 128)+d]/sqrt(sum/128+pa.eps);
        double *dst = (h < nh ? cq+(size_t)h*T*hd : ck+(size_t)(h-nh)*T*hd)+t*hd;
        float *gpu = (h < nh ? (float *)q.contents+(size_t)h*T*hd : (float *)k.contents+(size_t)(h-nh)*T*hd)+t*hd;
        for (int d = 0; d < 64; d++) {
            int axis = d < 60 ? d%3 : 0;
            // The product feeding GPU sin/cos is float, just as the HF rotary op.
            double angle = (float)(positions[3*t+axis]*iv[d]);
            dst[d] = x[d]*cos(angle)-x[d+64]*sin(angle);
            dst[d+64] = x[d+64]*cos(angle)+x[d]*sin(angle);
            require(isfinite(gpu[d]) && fabs(gpu[d]-dst[d]) < (sharp ? 1e-5 : 2e-6), "RMS/RoPE low-half mismatch");
            require(isfinite(gpu[d+64]) && fabs(gpu[d+64]-dst[d+64]) < (sharp ? 1e-5 : 2e-6), "RMS/RoPE high-half mismatch");
        }
    }
    float *got = out.contents;
    double worst = 0;
    for (int t = before; t < before+length; t++) for (int h = 0; h < nh; h++) {
        double scores[256], mx = -INFINITY, sum = 0, result[128] = {0};
        for (int s = before; s <= t; s++) {
            double dot = 0;
            for (int d = 0; d < hd; d++) dot += cq[((size_t)h*T+t)*hd+d]*ck[(size_t)s*hd+d];
            scores[s-before] = dot*aa.scale;
            if (dot*aa.scale > mx) mx = dot*aa.scale;
        }
        for (int s = before; s <= t; s++) {
            double prob = exp(scores[s-before]-mx); sum += prob;
            for (int d = 0; d < hd; d++) result[d] += prob*source[(size_t)s*row+(nh+nkv)*hd+d];
        }
        for (int d = 0; d < hd; d++) {
            float y = got[((size_t)t*nh+h)*hd+d];
            double err = fabs(y-result[d]/sum);
            if (err > worst) worst = err;
            require(isfinite(y) && err < (sharp ? 3e-4 : 3e-6), "attention FP64 mismatch or poisoned output");
        }
    }
    for (int d = 0; d < 32; d++) require(isnan(got[(size_t)T*nh*hd+d]), "output guard overwritten");
    printf("length %d packed=%d mode=%d: FP64 max %.3g\n", length, packed, mode, worst);
    float *copy = malloc((size_t)length*nh*hd*4);
    require(copy != NULL, "output allocation");
    memcpy(copy, got+(size_t)before*nh*hd, (size_t)length*nh*hd*4);
    free(cq); free(ck);
    return copy;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        require(dev != nil, "no Metal device");
        NSError *error = nil;
        NSString *source = [NSString stringWithContentsOfFile:@"metal/clef.metal" encoding:NSUTF8StringEncoding error:&error];
        MTLCompileOptions *opts = [MTLCompileOptions new]; opts.mathMode = MTLMathModeSafe;
        id<MTLLibrary> lib = [dev newLibraryWithSource:source options:opts error:&error];
        require(lib != nil, error.localizedDescription.UTF8String);
        id<MTLFunction> pfn = [lib newFunctionWithName:@"omni_attn_prep"], afn = [lib newFunctionWithName:@"omni_attention"];
        require(pfn && afn, "missing Omni attention kernels");
        id<MTLComputePipelineState> prep = [dev newComputePipelineStateWithFunction:pfn error:&error];
        id<MTLComputePipelineState> attn = [dev newComputePipelineStateWithFunction:afn error:&error];
        require(prep && attn, error.localizedDescription.UTF8String);
        if (argc == 2 && !strcmp(argv[1], "--compile-only")) return 0;
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        int lengths[] = {1,31,32,33,127,128,129};
        for (int mode = 0; mode < 3; mode++) for (size_t i = 0; i < sizeof(lengths)/sizeof(lengths[0]); i++) {
            float *single = run(dev, queue, prep, attn, lengths[i], false, mode);
            float *packed = run(dev, queue, prep, attn, lengths[i], true, mode);
            require(!memcmp(single, packed, (size_t)lengths[i]*8*128*4), "packing changed output bytes");
            free(single); free(packed);
        }
        for (int mode = 3; mode <= 5; mode++) {
            run(dev, queue, prep, attn, 1, false, mode);
            run(dev, queue, prep, attn, 33, true, mode);
        }
        float *long_single=run(dev,queue,prep,attn,8193,false,6);
        float *long_packed=run(dev,queue,prep,attn,8193,true,6);
        require(!memcmp(long_single,long_packed,(size_t)8193*8*128*4),"long packing changed bytes");
        free(long_single);free(long_packed);
        puts("Omni attention: FP64, packed equality, poison and output guards passed");
    }
    return 0;
}
