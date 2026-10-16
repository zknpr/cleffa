#ifndef CLEF_METAL_INTERNAL_H
#define CLEF_METAL_INTERNAL_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "clef_engine.h"
#include "clef_audio.h"

typedef struct clef_omni_gpu clef_omni_gpu;
typedef struct clef_audio_gpu clef_audio_gpu;

struct clef_gpu {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLBuffer> weights;            // the whole mmap'd GGUF, no copy
    NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *ps;
    id<MTLBuffer> inv_freq;
    id<MTLBuffer> keep;               // keep-warm sink (clef_gpu_keepalive)
    int cap;                          // token capacity of the activation buffers
    int gdn_cap;                      // longest record capacity, rounded to 32 tokens
    id<MTLBuffer> gdn_w, gdn_u, gdn_ke, gdn_a, gdn_e;
    id<MTLBuffer> ids, pos, seq_start, seq_bounds;
    id<MTLBuffer> x, xn, P, Q, K, V, G, A, Xc, beta, gate, O, hfin, nh32, nh16, mem, mem16, mn16, mn32;
    id<MTLBuffer> ovf;                // FP16 overflow flags, int per token, set at each record's first token
    id<MTLBuffer> attn_blk;           // attention_tu query blocks: 8 ints each, at most one block per token
    id<MTLBuffer> kv[16];
    int n_kv;
    // vision tower (encode_vision): per-pass patch uploads sized by the pass's patches, activations
    // by its largest image; feat holds every image's features, img_row maps tokens to them
    id<MTLBuffer> vis_inv_freq;
    int vcap, vcap_total;             // patch capacity of the per-image activations and of the uploads
    id<MTLBuffer> vpatch, vpatch16, vposidx, vposw;
    id<MTLBuffer> vx, vxn, vqkv, vq, vk, vv, va, vff, vff16, vm, vm16, vsplit;
    id<MTLBuffer> img_row, feat;
    id<MTLBuffer> deepstack[3];
    int act_f16;                      // FP16 GEMM inputs per producer class (ACT_*), else BF16
    float f16_lim;                    // magnitude past which an FP16 operand counts as overflow
    bool head_f32;                    // head inputs unrounded, head GEMMs on f32 activations
    int scan_lpc;                     // lanes per value column in gdn_scan (CLEF_SCAN_LPC)
    bool attn_ref;                    // CLEF_ATTN_REF=1: the simple reference attention kernel
    bool attn_tu;                     // attention_tu in FP16 passes; false = FP32 kernels everywhere (CLEF_ATTN_TU=0)
    bool vis_mpp;                     // CLEF_VIS_MPP=0 keeps simdgroup vision attention
    bool vis_comp;                    // CLEF_VIS_COMP=0 keeps direct FP32 vision GEMMs
    bool vis_group;                   // combine equal-grid GEMMs within one record
    bool vis_f32;                     // retain FP32 vision operands; non-residual GEMMs may compensate (vis_comp)
    bool debug_poison;                // CLEF_DEBUG_POISON=1 (tests): fill every activation buffer with NaN before each forward
    bool flash_gemm;                  // Flash matrix shapes qualify for selected short 64-row tiles
    bool profile;                     // CLEF_PROFILE=1: GPU time per kernel category (serializes!)
    double prof_ms[16];
    bool stage_time;                  // CLEF_STAGE_TIME=1: host wall time per forward stage (does not serialize)
    bool cb_failed;                   // a command buffer or encoder could not be created this pass
    int nil_cb_at, cb_count;          // test hook CLEF_DEBUG_NIL_CMDBUF: creation #nil_cb_at fails
    int ck_fail_at, ck_count;         // test hook CLEF_DEBUG_PREFIX_CKPT_FAIL: checkpoint allocation #ck_fail_at fails
    clef_omni_gpu *omni;              // family-specific scratch, owned by this backend
    clef_audio_gpu *audio;
};

enum { ACT_NORM = 1, ACT_ATTN = 2, ACT_GDN = 4, ACT_MLP = 8, ACT_VIS = 16 };
enum { X_BF16, X_F16, X_F32 };

void clef_metal_gemm(clef_gpu *g, id<MTLComputeCommandEncoder> enc, int xt, bool acc,
                     id<MTLBuffer> X, NSUInteger xoff, NSUInteger w_off, int N, int K,
                     id<MTLBuffer> Y, NSUInteger yoff, int T);
bool clef_omni_reserve(clef_gpu *g, const clef_config *c, int T, char *err, size_t errlen);
void clef_omni_free(clef_gpu *g);
void clef_omni_begin(clef_gpu *g);
bool clef_omni_trace_reserve(clef_gpu *g, int T, int layers, int rows, bool enabled, char *err, size_t errlen);
void clef_omni_keepalive(clef_gpu *g, id<MTLComputeCommandEncoder> enc);
bool clef_omni_complete(clef_gpu *g, char *err, size_t errlen);
bool clef_omni_encode_attention(clef_gpu *g, const clef_engine *e, id<MTLComputeCommandEncoder> enc,
                               int layer, int T, char *err, size_t errlen);
bool clef_omni_encode_layer(clef_gpu *g, const clef_engine *e, id<MTLComputeCommandEncoder> enc,
                            int layer, int T, int n_seq, bool bf16_only, char *err, size_t errlen);

// Optional stage consumer is used by offline tower verification. It sees only
// completed buffers and must report I/O/allocation errors through its return.
typedef bool (*clef_audio_capture)(void *context, const char *stage, int first, const float *values,
                                    int rows, int columns, char *err, size_t errlen);
bool clef_audio_gpu_reserve(clef_gpu *g, const clef_audio_config *c, int tokens, char *err, size_t errlen);
void clef_audio_gpu_free(clef_gpu *g);
void clef_audio_gpu_keepalive(clef_gpu *g, id<MTLComputeCommandEncoder> enc);
bool clef_audio_gpu_run(clef_gpu *g, const clef_engine *e, const clef_audio_features *clips, int n,
                        id<MTLBuffer> __strong *output, clef_audio_capture capture, void *context,
                        char *err, size_t errlen);
bool clef_audio_gpu_record(clef_gpu *g, const clef_engine *e, const clef_record *record, int base,
                           int total, char *err, size_t errlen);

#endif
