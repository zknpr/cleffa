#ifndef CLEF_TOK_H
#define CLEF_TOK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "clef_gguf.h"

/* Byte-level BPE tokenizer matching the HF `tokenizers` pipeline in Clef's
 * tokenizer.json:  added-token split -> NFC -> Qwen split regex -> ByteLevel -> BPE.
 * Equivalent to tokenizer(text, add_special_tokens=False).input_ids. */

typedef struct {
    int32_t *ids;
    size_t len, cap;
} clef_tokens;

typedef struct clef_tokenizer clef_tokenizer;

clef_tokenizer *clef_tok_load(const gguf_file *f, char *err, size_t errlen);
void clef_tok_free(clef_tokenizer *t);

/* Appends the tokens of `text` (UTF-8, `len` bytes). Returns false on OOM.
 * Invalid UTF-8 is handled the way HF does after Python decoding cannot occur
 * (requests arrive as JSON, so strings are valid UTF-8); stray bytes are encoded
 * byte-level and never dropped. */
bool clef_tok_encode(const clef_tokenizer *t, const char *text, size_t len, clef_tokens *out);

/* Same, but stops once at least max_tokens tokens were appended; the first max_tokens
 * appended tokens are exactly those clef_tok_encode would produce (may append a few more). */
bool clef_tok_encode_max(const clef_tokenizer *t, const char *text, size_t len, size_t max_tokens, clef_tokens *out);

/* split_special: do not recognize added tokens in `text` (strict mode for untrusted content);
 * equals HF `tokenizers` with the added-token list removed. */
bool clef_tok_encode_ex(const clef_tokenizer *t, const char *text, size_t len, size_t max_tokens,
                        bool split_special, clef_tokens *out);

int32_t clef_tok_pad_id(const clef_tokenizer *t);
int32_t clef_tok_vocab_size(const clef_tokenizer *t);

void clef_tokens_free(clef_tokens *v);
bool clef_tokens_push(clef_tokens *v, int32_t id);

/* Exposed for tests: NFC-normalize UTF-8. Caller frees *out. */
bool clef_nfc(const char *s, size_t len, char **out, size_t *out_len);

#endif
