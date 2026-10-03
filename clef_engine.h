#ifndef CLEF_ENGINE_H
#define CLEF_ENGINE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "clef_gguf.h"
#include "clef_record.h"
#include "clef_tok.h"

#define CLEF_MAX_LAYERS 128

typedef struct {
    int n_layer, H, ffn, nh, nkv, hd, n_rot;
    float eps, rope_theta;
    int layer_full[CLEF_MAX_LAYERS];         /* 1: gated full attention, 0: Gated DeltaNet */
    int ssm_kernel, Hk, Hv, dk, dv;
    int vocab;
    /* joint head */
    int W, routing_layers, head_layers, head_heads, head_ff;
} clef_config;

typedef struct {
    const gguf_tensor *attn_norm, *ffn_norm, *ffn_gate_up, *ffn_down;
    const gguf_tensor *attn_qkv, *attn_output, *attn_q_norm, *attn_k_norm;
    const gguf_tensor *ssm_in, *ssm_out, *ssm_conv1d, *ssm_dt, *ssm_a, *ssm_norm;
} clef_layer_w;

typedef struct {
    const gguf_tensor *token_embd, *output_norm, *output;
    clef_layer_w layer[CLEF_MAX_LAYERS];
} clef_weights;

typedef struct clef_head clef_head;
typedef struct clef_gpu clef_gpu;

typedef struct {
    gguf_file gguf;
    clef_config cfg;
    clef_weights w;
    clef_tokenizer *tok;
    clef_head *head;
    clef_gpu *gpu;
} clef_engine;

/* GPU outputs the head consumes, for one packed batch (row-major, token-major). */
typedef struct {
    const float *nh;       /* hidden_norm(last_hidden_state)  [T][H] */
    const float *kv[16];   /* per head attention module: memory K|V projections without bias [T][2W] */
    int n_kv;              /* routing_layers + head_layers */
} clef_head_inputs;

clef_engine *clef_open(const char *path, char *err, size_t errlen);
void clef_close(clef_engine *e);

/* Runs a batch of encoded records. probs[r][q] gets n_opt floats (caller frees with
 * clef_free_probs). Returns false with err set on failure. */
bool clef_run(clef_engine *e, const clef_record *recs, int n, float ****probs, char *err, size_t errlen);
void clef_free_probs(const clef_record *recs, int n, float ***probs);

/* ---- internal: head (CPU) ---- */
clef_head *clef_head_load(const gguf_file *f, const clef_config *cfg, char *err, size_t errlen);
void clef_head_free(clef_head *h);
/* logits for one record; `base` is the record's first token row in the batch. */
bool clef_head_run(const clef_head *h, const clef_config *cfg, const gguf_tensor *lm_head,
                   const clef_head_inputs *in, int base, const clef_record *rec, float **logits);

/* ---- internal: GPU ---- */
clef_gpu *clef_gpu_open(const clef_engine *e, char *err, size_t errlen);
void clef_gpu_close(clef_gpu *g);
/* Backbone + head-side GPU work for a packed batch. On success *in points into GPU
 * buffers that stay valid until the next call. dump_layers (optional, [n_layer+2][R][H]: embedding, every
 * layer, final norm; R = dump_rows last token rows, or all T when dump_rows <= 0).
 * bf16_only forces BF16 GEMM activations. overflow ([n_seq]; required unless bf16_only or FP16 is
 * off) reports the records whose FP16 activations left FP16's range or were NaN: their outputs
 * are invalid and must be recomputed with bf16_only (clef_run_ex does). The other records'
 * outputs are unaffected (act16 keeps the discarded values finite). */
bool clef_gpu_forward(clef_gpu *g, const clef_engine *e, const int32_t *ids, const int32_t *pos,
                      const int32_t *seq_start, const int32_t *seq_bounds, int n_seq, int T, bool bf16_only,
                      bool *overflow, clef_head_inputs *in, float *dump_layers, int dump_rows, char *err, size_t errlen);

#endif
