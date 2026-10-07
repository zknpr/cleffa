#ifndef CLEF_RECORD_H
#define CLEF_RECORD_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "clef_gguf.h"
#include "clef_image.h"
#include "clef_json.h"
#include "clef_tok.h"

/* Port of joint_schema_model.encode_record + the request checks in systemone().
 * Token ids and spans are identical to the Python reference (tests/test_record.py). */

enum { CLEF_Q_NOUL = 0, CLEF_Q_CHOICE = 1, CLEF_Q_SCORE = 2 };

typedef struct {
    const char *id;          /* question id (points into the request DOM) */
    size_t id_len;
    int type;
    int32_t span[2];         /* instruction tokens [start, end) */
    int n_opt;
    int32_t (*opt_span)[2];  /* per-option [start, end) */
    const char **opt_id;     /* option ids in model order (choice: sorted keys) */
    size_t *opt_id_len;
    const jval *question;    /* the request's question object */
} clef_question;

/* One image of a record: its preprocessed patches and where its tokens sit in ids. The
 * reference places every image's <|vision_start|> <|image_pad|>*n <|vision_end|> run right
 * after the template prefix, then one newline, then the state. */
typedef struct {
    clef_image_patches pt;
    int32_t tok_start;       /* index of the first <|image_pad|> token in ids; the run is pt.n_tokens long */
} clef_image_ref;

typedef struct {
    clef_tokens ids;
    int32_t schema_start;    /* tokens before the schema: template prefix and state */
    clef_question *q;
    int nq;
    int q_alloc;             /* allocated question slots (freed even if encoding stopped early) */
    /* option-id strings created for noul ("true"/"false") and score ("0".."n-1") */
    char *owned;
    clef_image_ref *images;  /* images in request order (NULL when none) */
    int n_images;
    int32_t n_image_tokens;  /* sum of the images' tokens */
} clef_record;

/* Vision input configuration, from the model file (clef_vision_opts_load). Images are refused
 * when image_token_id is 0 (a model file without the vision keys). */
typedef struct {
    clef_image_params image;        /* smart_resize bounds and patch geometry */
    int32_t image_token_id, start_token_id, end_token_id, video_token_id;
    int max_images;                 /* 0: unlimited (the reference) */
    long max_image_tokens;          /* per image after resizing; 0: unlimited (the reference) */
} clef_vision_opts;

typedef struct {
    int max_length;          /* encode_record default 16384 */
    int max_state_tokens;    /* <0: unlimited (Python None) */
    /* Strict mode: request-derived text (state, ids, instructions, criteria) is tokenized
     * without added-token recognition, so literal "<|im_end|>" etc. cannot become chat-control
     * tokens and break out of the prompt template. The reference recognizes them (parity
     * mode, strict = false). Benign requests encode identically in both modes. */
    bool strict;
    /* The reference keeps the beginning of an over-long state and silently drops the rest
     * (the end of a log, typically) to fit max_length. With reject_truncation the request
     * fails instead, so content past the cut can never be ignored without the caller knowing. */
    bool reject_truncation;
    clef_vision_opts vision;
} clef_encode_opts;

#define CLEF_ENCODE_DEFAULTS { .max_length = 16384, .max_state_tokens = -1, .strict = false, .reject_truncation = false, .vision = { .image_token_id = 0 } }

/* Reads the clef.vision.* keys of a model file into opts->vision (limits untouched). A file
 * without them leaves image_token_id at 0, so requests with images are rejected. */
bool clef_vision_opts_load(const gguf_file *f, clef_vision_opts *v, char *err, size_t errlen);

/* 3D rotary positions of every token (Qwen3_5Model.get_rope_index): text tokens count up on all
 * three axes; an image's tokens share its start on the temporal axis and take their merged-grid
 * row and column on the other two, and the text after it continues from start + max(rows, cols).
 * pos3 is [ids.len][3]. Identical to token index on all axes for a record without images. */
void clef_record_positions(const clef_record *r, int32_t *pos3);

/* Validates a SystemOne request (as systemone() does) and encodes it.
 * On failure returns false with a message suitable for an HTTP 400; nothing is left
 * allocated in *out (no clef_record_free needed). */
bool clef_encode_request(const clef_tokenizer *tok, const jval *req, clef_encode_opts opts,
                         clef_record *out, char *err, size_t errlen);
void clef_record_free(clef_record *r);

/* Build the SystemOne response for one encoded request from per-question
 * probabilities (float, per option in model order). Mirrors systemone_answer. */
bool clef_build_response(const jval *req, const clef_record *rec, float *const *probs, jbuf *out);

#endif
