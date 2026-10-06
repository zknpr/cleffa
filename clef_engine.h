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
typedef struct clef_gpu_prefix clef_gpu_prefix;
typedef struct clef_prefix clef_prefix;

typedef struct {
    gguf_file gguf;
    clef_config cfg;
    clef_weights w;
    clef_tokenizer *tok;
    clef_head *head;
    clef_gpu *gpu;
    uint64_t instance_id;  /* cache ownership; unique for each successful or attempted open */
} clef_engine;

/* GPU outputs the head consumes, for one packed batch (row-major, token-major). */
typedef struct {
    const float *nh;       /* hidden_norm(last_hidden_state)  [T][H] */
    int nh_skip;           /* a record's first nh_skip tokens have no nh row (prefix cache): token t is row base + t - nh_skip */
    const float *kv[16];   /* per head attention module: memory K|V projections without bias [T][2W] */
    int n_kv;              /* routing_layers + head_layers */
} clef_head_inputs;

clef_engine *clef_open(const char *path, char *err, size_t errlen);
void clef_close(clef_engine *e);

/* Runs a batch of encoded records. probs[r][q] gets n_opt floats (caller frees with
 * clef_free_probs). Returns false with err set on failure. */
bool clef_run(clef_engine *e, const clef_record *recs, int n, float ****probs, char *err, size_t errlen);
void clef_free_probs(const clef_record *recs, int n, float ***probs);

/* Prefix cache. An entry keeps the backbone's state for the leading tokens of the last request
 * served through it, up to where that request's schema begins. A request that starts with some
 * of the same tokens resumes from the last checkpoint inside them and computes the rest: the
 * entry holds the recurrent state at its snapshot, every 2,048 tokens, and where a request last
 * left its tokens (clef.c). Logits are bitwise those of clef_run_ex for the
 * same record: the pass resumes on the tile and block boundaries an uncached pass has, and an
 * entry is used only by records in the same DeltaNet class (clef_gpu_prefix_class). A record
 * that shares no checkpoint with the entry is computed whole and replaces it. *reused gets the
 * tokens taken from the entry. An entry binds to its first engine; using another engine returns an error,
 * including after closing and reopening a model. Scope an entry to one tenant: whether a request hits it shows in its latency. */
clef_prefix *clef_prefix_new(void);
void clef_prefix_free(clef_prefix *p);
size_t clef_prefix_bytes(const clef_prefix *p);   /* GPU memory the entry holds */
bool clef_prefix_usable(const clef_prefix *p);    /* holds state a matching record can resume from */
/* GPU memory the entry would hold after clef_run_prefix on this record (an upper bound; 0 when the
 * pass would take the plain path), for a caller that enforces a budget before anything is allocated. */
size_t clef_prefix_estimate(const clef_engine *e, const clef_prefix *p, const clef_record *rec);
bool clef_prefix_keep_warm(clef_engine *e, const clef_prefix *p, char *err, size_t errlen);
/* Template reuse. Every request starts with the same template tokens, so an entry pinned to
 * their first 32 reuses only public state; no request key is needed. The entry buffers also
 * retain suffix rows, which are overwritten before use on the next request. Reuse is limited
 * to measured tile/padding ranges up to 2,048 tokens; other records take the plain path
 * without invalidating the entry. Bitwise the plain result within an attention mode,
 * as with clef_run_prefix. Give it an entry of its own, not one clef_run_prefix also uses. */
bool clef_run_template(clef_engine *e, clef_prefix *p, const clef_record *rec, float ****out, bool raw,
                       int *reused, char *err, size_t errlen);
bool clef_run_prefix(clef_engine *e, clef_prefix *p, const clef_record *rec, float ****out, bool raw,
                     int *reused, char *err, size_t errlen);

/* Submits one-thread GPU passes over the weights and activation buffers. Called about twice a
 * second while idle, it keeps the next request from starting late (see clef_gpu_keepalive).
 * Never changes a result. Not thread-safe against clef_run. */
bool clef_keep_warm(clef_engine *e, char *err, size_t errlen);

/* ---- internal: head (CPU) ---- */
clef_head *clef_head_load(const gguf_file *f, const clef_config *cfg, char *err, size_t errlen);
void clef_head_free(clef_head *h);
/* logits for one record; `base` is the record's first token row in the batch. */
bool clef_head_run(const clef_head *h, const clef_config *cfg, const gguf_tensor *lm_head,
                   const clef_head_inputs *in, int base, const clef_record *rec, float **logits);

/* ---- internal: GPU ---- */
clef_gpu *clef_gpu_open(const clef_engine *e, char *err, size_t errlen);
void clef_gpu_close(clef_gpu *g);
bool clef_gpu_keepalive(clef_gpu *g, char *err, size_t errlen);
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
/* An entry keeps the DeltaNet state and conv tail of up to this many rows of its record, one
 * per slot. Which row a slot holds is the caller's bookkeeping. */
#define CLEF_PREFIX_CKPT 12
/* The checkpoints one prefix pass uses: the slot it resumes from at row L (ignored when L is 0)
 * and the rows whose state it stores, ascending multiples of 32 with L < row < T, each with the
 * slot it goes to. The slots are distinct and none is the one being read. */
typedef struct { int load, n, row[CLEF_PREFIX_CKPT], slot[CLEF_PREFIX_CKPT]; } clef_prefix_plan;
/* One record of T tokens (ids) whose first L are in the entry px; L is a multiple of 32, L < T.
 * Computes tokens L..T-1 with FP16 activations, leaves their rows in the entry and stores the
 * recurrent state of the plan's rows in its slots. On failure or overflow the entry is unusable
 * until a pass from L = 0 succeeds. in->nh has rows for tokens L.. only (nh_skip). */
clef_gpu_prefix *clef_gpu_prefix_new(void);
void clef_gpu_prefix_free(clef_gpu_prefix *px);
size_t clef_gpu_prefix_bytes(const clef_gpu_prefix *px);
size_t clef_gpu_prefix_estimate(const clef_gpu *g, const clef_config *c, const clef_gpu_prefix *px, int rows, int new_ckpts);
bool clef_gpu_prefix_supported(const clef_gpu *g);
bool clef_gpu_prefix_keepalive(clef_gpu *g, const clef_gpu_prefix *px, char *err, size_t errlen);
int clef_gpu_prefix_class(const clef_engine *e, int length);
bool clef_gpu_forward_prefix(clef_gpu *g, const clef_engine *e, clef_gpu_prefix *px, const int32_t *ids, int T,
                             int L, const clef_prefix_plan *plan, bool *overflow, clef_head_inputs *in, char *err, size_t errlen);

#endif
