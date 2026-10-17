// Native Omni audio tower. Small convolution batches bound im2col scratch;
// chunk padding and transformer windows are determined by the whole record.
#include "clef_metal_internal.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { AUDIO_CHUNKS=8, AUDIO_WIDTH=100, AUDIO_ROWS=13, AUDIO_H=1280, AUDIO_FF=5120, AUDIO_C=480 };
typedef struct { int chunks,height,width,channels; } audio_conv_args;
typedef struct { int H,rows,mode; } audio_bias_args;
typedef struct { int H;float eps; } audio_norm_args;
typedef struct { int T,H,heads;float scale; } audio_attention_args;
typedef struct { int rows,H,padded_rows; } audio_gather_args;

struct clef_audio_gpu {
    int cap;
    id<MTLBuffer> input,col,conv_a,conv_b,projected,indices,position;
    id<MTLBuffer> x,xn,q,k,v,attn,ff,output,windows,status;
};

static void audio_free_state(clef_audio_gpu *s) {
    if(!s)return;
    s->input=s->col=s->conv_a=s->conv_b=s->projected=s->indices=s->position=nil;
    s->x=s->xn=s->q=s->k=s->v=s->attn=s->ff=s->output=s->windows=s->status=nil;
    free(s);
}
void clef_audio_gpu_free(clef_gpu *g) { audio_free_state(g->audio);g->audio=NULL; }

static id<MTLBuffer> audio_allocate(clef_gpu *g,size_t count) {
    if(!count || count>SIZE_MAX/4 || count*4>g->dev.maxBufferLength)return nil;
    return [g->dev newBufferWithLength:count*4 options:MTLResourceStorageModeShared];
}

bool clef_audio_gpu_reserve(clef_gpu *g,const clef_audio_config *c,int tokens,char *err,size_t errlen) {
    if(tokens<=0 || tokens>65536 || c->H!=AUDIO_H || c->ffn!=AUDIO_FF || c->hd!=64 || c->heads!=20 ||
       c->layers!=32 || c->mel_bins!=128 || c->conv_channels!=AUDIO_C || c->output!=2048 ||
       c->chunk_frames!=100 || c->window_frames!=800 || c->eps!=1e-5f) {
        snprintf(err,errlen,"metal: invalid audio scratch dimensions");return false;
    }
    if(g->audio && g->audio->cap>=tokens)return true;
    clef_audio_gpu *s=calloc(1,sizeof(*s));
    if(!s){snprintf(err,errlen,"metal: cannot allocate audio state");return false;}
    s->cap=(tokens+31)/32*32;
    s->input=audio_allocate(g,AUDIO_CHUNKS*128*AUDIO_WIDTH);
    s->col=audio_allocate(g,AUDIO_CHUNKS*32*25*AUDIO_C*9);
    s->conv_a=audio_allocate(g,AUDIO_CHUNKS*64*50*AUDIO_C);
    s->conv_b=audio_allocate(g,AUDIO_CHUNKS*32*25*AUDIO_C);
    s->projected=audio_allocate(g,AUDIO_CHUNKS*AUDIO_ROWS*AUDIO_H);
    s->indices=audio_allocate(g,s->cap>AUDIO_CHUNKS*AUDIO_ROWS?s->cap:AUDIO_CHUNKS*AUDIO_ROWS);
    s->position=audio_allocate(g,AUDIO_ROWS*AUDIO_H);
    s->x=audio_allocate(g,(size_t)s->cap*AUDIO_H);
    s->xn=audio_allocate(g,(size_t)s->cap*AUDIO_H);
    s->q=audio_allocate(g,(size_t)s->cap*AUDIO_H);
    s->k=audio_allocate(g,(size_t)s->cap*AUDIO_H);
    s->v=audio_allocate(g,(size_t)s->cap*AUDIO_H);
    s->attn=audio_allocate(g,(size_t)s->cap*AUDIO_H);
    s->ff=audio_allocate(g,(size_t)s->cap*AUDIO_FF);
    s->output=audio_allocate(g,(size_t)s->cap*2048);
    s->windows=audio_allocate(g,(size_t)s->cap*2);
    s->status=audio_allocate(g,1);
    if(!s->input || !s->col || !s->conv_a || !s->conv_b || !s->projected || !s->indices || !s->position ||
       !s->x || !s->xn || !s->q || !s->k || !s->v || !s->attn || !s->ff || !s->output || !s->windows || !s->status) {
        audio_free_state(s);snprintf(err,errlen,"metal: cannot allocate audio arena (previous buffers kept)");return false;
    }
    // The pinned model uses FP32 exp, multiply, sin and cos for its nonpersistent
    // position buffer. This is numerical tower state, not a token position map.
    float increment=(float)(log(10000.0)/(AUDIO_H/2-1));
    float *position=s->position.contents;
    for(int d=0;d<AUDIO_H/2;d++) {
        float frequency=expf(-increment*d);
        for(int t=0;t<AUDIO_ROWS;t++) {
            float angle=t*frequency;
            position[t*AUDIO_H+d]=sinf(angle);position[t*AUDIO_H+d+AUDIO_H/2]=cosf(angle);
        }
    }
    audio_free_state(g->audio);g->audio=s;return true;
}

void clef_audio_gpu_keepalive(clef_gpu *g,id<MTLComputeCommandEncoder> enc) {
    clef_audio_gpu *s=g->audio;if(!s)return;
    id<MTLBuffer> buffers[]={s->input,s->col,s->conv_a,s->conv_b,s->projected,s->indices,s->position,
        s->x,s->xn,s->q,s->k,s->v,s->attn,s->ff,s->output,s->windows,s->status};
    for(size_t i=0;i<sizeof(buffers)/sizeof(buffers[0]);i++) {
        [enc setBuffer:buffers[i] offset:0 atIndex:0];
        [enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
    }
}

static void audio_threads(id<MTLComputeCommandEncoder> enc,NSUInteger count) {
    [enc dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
}
static bool audio_command(clef_gpu *g,id<MTLCommandBuffer> __strong *cb,
                           id<MTLComputeCommandEncoder> __strong *enc,char *err,size_t errlen) {
    *cb=++g->cb_count==g->nil_cb_at?nil:[g->queue commandBuffer];
    *enc=*cb?[*cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial]:nil;
    if(!*cb || !*enc){g->cb_failed=true;snprintf(err,errlen,"metal: cannot create audio command buffer");return false;}
    return true;
}
static bool audio_finish(clef_gpu *g,id<MTLCommandBuffer> cb,id<MTLComputeCommandEncoder> enc,char *err,size_t errlen) {
    [enc endEncoding];[cb commit];[cb waitUntilCompleted];
    if(cb.status!=MTLCommandBufferStatusCompleted || cb.error) {
        snprintf(err,errlen,"metal: audio command failed: %s",cb.error?cb.error.localizedDescription.UTF8String:"incomplete");return false;
    }
    if(*(uint32_t *)g->audio->status.contents){snprintf(err,errlen,"metal: nonfinite audio tower values");return false;}
    return true;
}
static void audio_bias(clef_gpu *g,id<MTLComputeCommandEncoder> enc,id<MTLBuffer> x,const gguf_tensor *bias,
                        id<MTLBuffer> residual,int rows,int H,int mode) {
    audio_bias_args a={H,rows,mode};
    [enc setComputePipelineState:g->ps[@"audio_bias"]];[enc setBytes:&a length:sizeof(a) atIndex:0];
    [enc setBuffer:x offset:0 atIndex:1];[enc setBuffer:g->weights offset:bias->offset atIndex:2];
    [enc setBuffer:residual?residual:x offset:0 atIndex:3];[enc setBuffer:g->audio->status offset:0 atIndex:4];
    audio_threads(enc,(NSUInteger)rows*H);
}
static void audio_norm(clef_gpu *g,id<MTLComputeCommandEncoder> enc,const gguf_tensor *weight,const gguf_tensor *bias,int T) {
    audio_norm_args a={AUDIO_H,1e-5f};
    [enc setComputePipelineState:g->ps[@"audio_norm"]];[enc setBytes:&a length:sizeof(a) atIndex:0];
    [enc setBuffer:g->audio->x offset:0 atIndex:1];[enc setBuffer:g->weights offset:weight->offset atIndex:2];
    [enc setBuffer:g->weights offset:bias->offset atIndex:3];[enc setBuffer:g->audio->xn offset:0 atIndex:4];
    [enc setBuffer:g->audio->status offset:0 atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(T,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
}
static void audio_linear(clef_gpu *g,id<MTLComputeCommandEncoder> enc,id<MTLBuffer> input,const gguf_tensor *weight,
                          const gguf_tensor *bias,id<MTLBuffer> output,int T,int N,int K,int mode) {
    clef_metal_gemm(g,enc,X_F32,false,input,0,weight->offset,N,K,output,0,T);
    if(bias)audio_bias(g,enc,output,bias,mode==2?g->audio->x:nil,T,N,mode);
}

bool clef_audio_gpu_run(clef_gpu *g,const clef_engine *e,const clef_audio_features *clips,int n,
                        id<MTLBuffer> __strong *output,clef_audio_capture capture,void *context,char *err,size_t errlen) {
    if(!output){snprintf(err,errlen,"metal: missing audio output");return false;}
    *output=nil;
    clef_audio_layout layout={0};bool ok=false;
    if(!clef_audio_layout_make(clips,n,&layout,err,errlen))return false;
    if(!clef_audio_gpu_reserve(g,&e->cfg.audio,layout.tokens,err,errlen))goto done;
    {
        clef_audio_gpu *s=g->audio;const clef_audio_weights *w=&e->w.audio;
        const int T=layout.tokens,width=layout.max_chunk_frames,padded_rows=(width+7)/8;
        *(uint32_t *)s->status.contents=0;
        if(g->debug_poison) {
            id<MTLBuffer> buffers[]={s->input,s->col,s->conv_a,s->conv_b,s->projected,s->x,s->xn,s->q,s->k,s->v,s->attn,s->ff,s->output};
            for(size_t b=0;b<sizeof(buffers)/sizeof(buffers[0]);b++) {
                float *p=buffers[b].contents;for(NSUInteger j=0;j<buffers[b].length/4;j++)p[j]=NAN;
            }
        }
        memcpy(s->windows.contents,layout.window_start,(size_t)T*4);
        memcpy((int32_t *)s->windows.contents+s->cap,layout.window_end,(size_t)T*4);
        for(int first=0;first<layout.n_chunks;first+=AUDIO_CHUNKS) {
            int count=layout.n_chunks-first;if(count>AUDIO_CHUNKS)count=AUDIO_CHUNKS;
            float *input=s->input.contents;int *indices=s->indices.contents,valid=0;
            memset(input,0,(size_t)count*128*width*4);
            for(int c=0;c<count;c++) {
                const clef_audio_chunk *chunk=layout.chunks+first+c;
                const clef_audio_features *clip=clips+chunk->clip;
                for(int f=0;f<128;f++)for(int t=0;t<chunk->frames;t++) {
                    float value=clip->mel[(size_t)f*clip->frames+chunk->frame+t];
                    if(!isfinite(value)){snprintf(err,errlen,"metal: nonfinite audio features");goto done;}
                    input[((size_t)c*128+f)*width+t]=value;
                }
                for(int t=0;t<(chunk->frames+7)/8;t++)indices[valid++]=c*padded_rows+t;
            }
            id<MTLCommandBuffer> cb=nil;id<MTLComputeCommandEncoder> enc=nil;
            if(!audio_command(g,&cb,&enc,err,errlen))goto done;
            id<MTLBuffer> source=s->input;
            int height=128,time=width,channels=1;
            for(int layer=0;layer<3;layer++) {
                audio_conv_args a={count,height,time,channels};
                int rows=count*((height+1)/2)*((time+1)/2);
                [enc setComputePipelineState:g->ps[@"audio_im2col"]];[enc setBytes:&a length:sizeof(a) atIndex:0];
                [enc setBuffer:source offset:0 atIndex:1];[enc setBuffer:s->col offset:0 atIndex:2];
                audio_threads(enc,(NSUInteger)rows*channels*9);
                id<MTLBuffer> dest=layer==1?s->conv_b:s->conv_a;
                audio_linear(g,enc,s->col,w->conv_w[layer],w->conv_b[layer],dest,rows,AUDIO_C,channels*9,1);
                if(capture) {
                    char name[32];snprintf(name,sizeof(name),"conv2d%d.gelu",layer+1);
                    if(!audio_finish(g,cb,enc,err,errlen) ||
                       !capture(context,name,first,dest.contents,rows,AUDIO_C,err,errlen) ||
                       !audio_command(g,&cb,&enc,err,errlen))goto done;
                }
                source=dest;height=(height+1)/2;time=(time+1)/2;channels=AUDIO_C;
            }
            audio_conv_args pack={count,height,time,channels};
            [enc setComputePipelineState:g->ps[@"audio_conv_pack"]];[enc setBytes:&pack length:sizeof(pack) atIndex:0];
            [enc setBuffer:source offset:0 atIndex:1];[enc setBuffer:s->col offset:0 atIndex:2];
            audio_threads(enc,(NSUInteger)count*height*time*channels);
            audio_linear(g,enc,s->col,w->conv_out,NULL,s->projected,count*time,AUDIO_H,channels*height,0);
            audio_gather_args gather={valid,AUDIO_H,padded_rows};
            [enc setComputePipelineState:g->ps[@"audio_gather"]];[enc setBytes:&gather length:sizeof(gather) atIndex:0];
            [enc setBuffer:s->projected offset:0 atIndex:1];[enc setBuffer:s->indices offset:0 atIndex:2];
            [enc setBuffer:s->position offset:0 atIndex:3];
            [enc setBuffer:s->x offset:(NSUInteger)layout.chunks[first].first_token*AUDIO_H*4 atIndex:4];
            [enc setBuffer:s->status offset:0 atIndex:5];audio_threads(enc,(NSUInteger)valid*AUDIO_H);
            if(!audio_finish(g,cb,enc,err,errlen))goto done;
        }
        if(capture && (!capture(context,"position",0,s->position.contents,AUDIO_ROWS,AUDIO_H,err,errlen) ||
                       !capture(context,"conv",0,s->x.contents,T,AUDIO_H,err,errlen)))goto done;
        for(int layer=0;layer<e->cfg.audio.layers;layer++) {
            const clef_alayer_w *lw=w->layer+layer;
            id<MTLCommandBuffer> cb=nil;id<MTLComputeCommandEncoder> enc=nil;
            if(!audio_command(g,&cb,&enc,err,errlen))goto done;
            audio_norm(g,enc,lw->ln1_w,lw->ln1_b,T);
            audio_linear(g,enc,s->xn,lw->q_w,lw->q_b,s->q,T,AUDIO_H,AUDIO_H,0);
            audio_linear(g,enc,s->xn,lw->k_w,lw->k_b,s->k,T,AUDIO_H,AUDIO_H,0);
            audio_linear(g,enc,s->xn,lw->v_w,lw->v_b,s->v,T,AUDIO_H,AUDIO_H,0);
            audio_attention_args a={T,AUDIO_H,20,.125f};
            [enc setComputePipelineState:g->ps[@"audio_attention"]];[enc setBytes:&a length:sizeof(a) atIndex:0];
            [enc setBuffer:s->q offset:0 atIndex:1];[enc setBuffer:s->k offset:0 atIndex:2];[enc setBuffer:s->v offset:0 atIndex:3];
            [enc setBuffer:s->windows offset:0 atIndex:4];[enc setBuffer:s->windows offset:(NSUInteger)s->cap*4 atIndex:5];
            [enc setBuffer:s->attn offset:0 atIndex:6];[enc setBuffer:s->status offset:0 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(T,20,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            audio_linear(g,enc,s->attn,lw->out_w,lw->out_b,s->xn,T,AUDIO_H,AUDIO_H,2);
            audio_norm(g,enc,lw->ln2_w,lw->ln2_b,T);
            audio_linear(g,enc,s->xn,lw->up_w,lw->up_b,s->ff,T,AUDIO_FF,AUDIO_H,1);
            audio_linear(g,enc,s->ff,lw->down_w,lw->down_b,s->xn,T,AUDIO_H,AUDIO_FF,2);
            if(!audio_finish(g,cb,enc,err,errlen))goto done;
            if(capture) {
                char name[32];snprintf(name,sizeof(name),"layer.%02d",layer);
                if(!capture(context,name,0,s->x.contents,T,AUDIO_H,err,errlen))goto done;
            }
        }
        {
            id<MTLCommandBuffer> cb=nil;id<MTLComputeCommandEncoder> enc=nil;
            if(!audio_command(g,&cb,&enc,err,errlen))goto done;
            audio_norm(g,enc,w->ln_w,w->ln_b,T);
            audio_linear(g,enc,s->xn,w->proj1_w,w->proj1_b,s->x,T,AUDIO_H,AUDIO_H,1);
            audio_linear(g,enc,s->x,w->proj2_w,w->proj2_b,s->output,T,2048,AUDIO_H,0);
            if(!audio_finish(g,cb,enc,err,errlen))goto done;
        }
        if(capture && !capture(context,"output",0,s->output.contents,T,2048,err,errlen))goto done;
        *output=s->output;ok=true;
    }
done:
    clef_audio_layout_free(&layout);return ok;
}

bool clef_audio_gpu_record(clef_gpu *g,const clef_engine *e,const clef_record *record,int base,
                           int total,char *err,size_t errlen) {
    if(!record->n_audio)return true;
    if(e->cfg.model.backbone!=CLEF_OMNI || !(e->cfg.model.capabilities&CLEF_CAP_AUDIO) ||
       record->n_audio<0 || !record->audio || base<0 || total<base || record->ids.len>(size_t)(total-base)) {
        snprintf(err,errlen,"metal: invalid audio record");return false;
    }
    clef_audio_features *features=calloc((size_t)record->n_audio,sizeof(*features));
    if(!features){snprintf(err,errlen,"metal: cannot allocate audio descriptors");return false;}
    int64_t rows=0;int previous=-1;
    for(int i=0;i<record->n_audio;i++) {
        const clef_audio_ref *a=record->audio+i;features[i]=a->feat;
        if(!a->token_rows || a->feat.n_tokens<=0 || (rows+=a->feat.n_tokens)>65536)goto invalid;
        for(int t=0;t<a->feat.n_tokens;t++) {
            int row=a->token_rows[t];
            if(row<=previous || row<0 || (size_t)row>=record->ids.len || record->ids.ids[row]!=e->audio.token_id)goto invalid;
            previous=row;
        }
    }
    if(rows!=record->n_audio_tokens)goto invalid;
    {
        id<MTLBuffer> output=nil;
        bool ok=clef_audio_gpu_run(g,e,features,record->n_audio,&output,NULL,NULL,err,errlen);
        free(features);features=NULL;if(!ok)return false;
        int32_t *indices=g->audio->indices.contents;int index=0;
        for(int i=0;i<record->n_audio;i++)for(int t=0;t<record->audio[i].feat.n_tokens;t++)
            indices[index++]=base+record->audio[i].token_rows[t];
        id<MTLCommandBuffer> cb=nil;id<MTLComputeCommandEncoder> enc=nil;
        if(!audio_command(g,&cb,&enc,err,errlen))return false;
        struct{int rows,H;}a={(int)rows,2048};
        [enc setComputePipelineState:g->ps[@"audio_scatter"]];[enc setBytes:&a length:sizeof(a) atIndex:0];
        [enc setBuffer:output offset:0 atIndex:1];[enc setBuffer:g->audio->indices offset:0 atIndex:2];
        [enc setBuffer:g->x offset:0 atIndex:3];audio_threads(enc,(NSUInteger)rows*2048);
        return audio_finish(g,cb,enc,err,errlen);
    }
invalid:
    free(features);snprintf(err,errlen,"metal: invalid audio token placement");return false;
}
