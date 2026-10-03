#ifndef CLEF_RECORD_H
#define CLEF_RECORD_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

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

typedef struct {
    clef_tokens ids;
    clef_question *q;
    int nq;
    int q_alloc;             /* allocated question slots (freed even if encoding stopped early) */
    /* option-id strings created for noul ("true"/"false") and score ("0".."n-1") */
    char *owned;
} clef_record;

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
} clef_encode_opts;

#define CLEF_ENCODE_DEFAULTS { .max_length = 16384, .max_state_tokens = -1, .strict = false, .reject_truncation = false }

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
