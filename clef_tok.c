#include "clef_tok.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "clef_unicode.inc"

/* ---- containers ------------------------------------------------------------ */

bool clef_tokens_push(clef_tokens *v, int32_t id) {
    if (v->len == v->cap) {
        size_t cap = v->cap ? v->cap * 2 : 64;
        int32_t *p = realloc(v->ids, cap * sizeof(*p));
        if (!p) return false;
        v->ids = p;
        v->cap = cap;
    }
    v->ids[v->len++] = id;
    return true;
}

void clef_tokens_free(clef_tokens *v) {
    free(v->ids);
    memset(v, 0, sizeof(*v));
}

typedef struct { uint32_t *cp; size_t len, cap; } cpvec;

static bool cp_push(cpvec *v, uint32_t c) {
    if (v->len == v->cap) {
        size_t cap = v->cap ? v->cap * 2 : 64;
        uint32_t *p = realloc(v->cp, cap * sizeof(*p));
        if (!p) return false;
        v->cp = p;
        v->cap = cap;
    }
    v->cp[v->len++] = c;
    return true;
}

/* ---- hash tables ------------------------------------------------------------- */

static uint64_t fnv1a(const char *s, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= (uint8_t)s[i]; h *= 1099511628211ull; }
    return h;
}

static uint64_t mix64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull;
    return x ^ (x >> 33);
}

static size_t pow2_at_least(size_t n) {
    size_t c = 16;
    while (c < n * 2) c <<= 1;
    return c;
}

/* token string -> id; strings point into the GGUF mapping */
typedef struct { const char *s; uint32_t len; int32_t id; } str_slot;
typedef struct { str_slot *slots; size_t mask; } str_table;

static bool str_table_init(str_table *t, size_t n) {
    size_t cap = pow2_at_least(n);
    t->slots = calloc(cap, sizeof(*t->slots));
    t->mask = cap - 1;
    return t->slots != NULL;
}

static void str_table_put(str_table *t, const char *s, uint32_t len, int32_t id) {
    for (size_t i = fnv1a(s, len) & t->mask;; i = (i + 1) & t->mask) {
        str_slot *e = &t->slots[i];
        if (!e->s) { e->s = s; e->len = len; e->id = id; return; }
        if (e->len == len && !memcmp(e->s, s, len)) return;  /* first id wins */
    }
}

static int32_t str_table_get(const str_table *t, const char *s, size_t len) {
    for (size_t i = fnv1a(s, len) & t->mask;; i = (i + 1) & t->mask) {
        const str_slot *e = &t->slots[i];
        if (!e->s) return -1;
        if (e->len == len && !memcmp(e->s, s, len)) return e->id;
    }
}

/* (left id, right id) -> (rank, merged id) */
typedef struct { uint64_t key; int32_t rank; int32_t id; } pair_slot;
typedef struct { pair_slot *slots; size_t mask; } pair_table;

#define PAIR_EMPTY UINT64_MAX

static bool pair_table_init(pair_table *t, size_t n) {
    size_t cap = pow2_at_least(n);
    t->slots = malloc(cap * sizeof(*t->slots));
    if (!t->slots) return false;
    for (size_t i = 0; i < cap; i++) t->slots[i].key = PAIR_EMPTY;
    t->mask = cap - 1;
    return true;
}

static void pair_table_put(pair_table *t, int32_t a, int32_t b, int32_t rank, int32_t id) {
    uint64_t key = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    for (size_t i = mix64(key) & t->mask;; i = (i + 1) & t->mask) {
        pair_slot *e = &t->slots[i];
        if (e->key == PAIR_EMPTY) { e->key = key; e->rank = rank; e->id = id; return; }
        if (e->key == key) return;  /* HF keeps the first (lowest-rank) occurrence */
    }
}

static const pair_slot *pair_table_get(const pair_table *t, int32_t a, int32_t b) {
    uint64_t key = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    for (size_t i = mix64(key) & t->mask;; i = (i + 1) & t->mask) {
        const pair_slot *e = &t->slots[i];
        if (e->key == PAIR_EMPTY) return NULL;
        if (e->key == key) return e;
    }
}

/* ---- tokenizer state ------------------------------------------------------- */

#define MAX_ADDED 64

struct clef_tokenizer {
    int32_t n_vocab;
    gguf_str *tokens;
    str_table token_to_id;
    pair_table merges;
    int32_t byte_id[256];
    struct { const char *s; uint32_t len; int32_t id; } added[MAX_ADDED];
    int n_added;
    int32_t pad_id;
};

int32_t clef_tok_pad_id(const clef_tokenizer *t) { return t->pad_id; }
int32_t clef_tok_vocab_size(const clef_tokenizer *t) { return t->n_vocab; }

static uint32_t gpt2_byte_to_cp(uint8_t b) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174) return b;
    uint32_t n = 0;
    for (uint32_t x = 0; x < b; x++) {
        if (!((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || x >= 174)) n++;
    }
    return 256 + n;
}

static size_t utf8_put(char *p, uint32_t cp) {
    if (cp <= 0x7f) { p[0] = (char)cp; return 1; }
    if (cp <= 0x7ff) { p[0] = (char)(0xc0 | (cp >> 6)); p[1] = (char)(0x80 | (cp & 0x3f)); return 2; }
    if (cp <= 0xffff) {
        p[0] = (char)(0xe0 | (cp >> 12)); p[1] = (char)(0x80 | ((cp >> 6) & 0x3f));
        p[2] = (char)(0x80 | (cp & 0x3f)); return 3;
    }
    p[0] = (char)(0xf0 | (cp >> 18)); p[1] = (char)(0x80 | ((cp >> 12) & 0x3f));
    p[2] = (char)(0x80 | ((cp >> 6) & 0x3f)); p[3] = (char)(0x80 | (cp & 0x3f)); return 4;
}

static bool set_err(char *err, size_t errlen, const char *msg) {
    snprintf(err, errlen, "%s", msg);
    return false;
}

clef_tokenizer *clef_tok_load(const gguf_file *f, char *err, size_t errlen) {
    clef_tokenizer *t = calloc(1, sizeof(*t));
    if (!t) { set_err(err, errlen, "tokenizer: out of memory"); return NULL; }

    gguf_str model, pre;
    if (!gguf_get_str(f, "tokenizer.ggml.model", &model) || model.len != 4 || memcmp(model.ptr, "gpt2", 4)) {
        set_err(err, errlen, "tokenizer: expected byte-level BPE (gpt2)"); goto fail;
    }
    if (!gguf_get_str(f, "tokenizer.ggml.pre", &pre) || pre.len != 6 || memcmp(pre.ptr, "qwen35", 6)) {
        set_err(err, errlen, "tokenizer: unsupported pre-tokenizer"); goto fail;
    }
    /* The split regex is hardcoded below; refuse a GGUF whose recorded definition differs. */
    gguf_str pre_def, norm_def;
    static const char expect_regex[] =
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?\\\\p{L}+|\\\\p{N}| ?[^\\\\s\\\\p{L}\\\\p{N}]+[\\\\r\\\\n]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+";
    if (!gguf_get_str(f, "clef.tokenizer.pre_tokenizer", &pre_def) ||
        !memmem(pre_def.ptr, pre_def.len, expect_regex, sizeof(expect_regex) - 1)) {
        set_err(err, errlen, "tokenizer: split regex differs from the implemented one"); goto fail;
    }
    if (!gguf_get_str(f, "clef.tokenizer.normalizer", &norm_def) ||
        !memmem(norm_def.ptr, norm_def.len, "\"type\": \"NFC\"", 13)) {
        set_err(err, errlen, "tokenizer: normalizer is not NFC"); goto fail;
    }

    uint32_t et; uint64_t n, pos;
    if (!gguf_get_array(f, "tokenizer.ggml.tokens", &et, &n, &pos) || et != GGUF_T_STRING || n == 0 || n > INT32_MAX) {
        set_err(err, errlen, "tokenizer: token list missing"); goto fail;
    }
    t->n_vocab = (int32_t)n;
    t->tokens = calloc(n, sizeof(*t->tokens));
    if (!t->tokens || !gguf_read_strings(f, pos, n, t->tokens) || !str_table_init(&t->token_to_id, n)) {
        set_err(err, errlen, "tokenizer: bad token list"); goto fail;
    }
    for (uint64_t i = 0; i < n; i++) {
        if (t->tokens[i].len > UINT32_MAX) { set_err(err, errlen, "tokenizer: token too long"); goto fail; }
        if (t->tokens[i].len) str_table_put(&t->token_to_id, t->tokens[i].ptr, (uint32_t)t->tokens[i].len, (int32_t)i);
    }

    /* Added tokens: CONTROL (3) and USER_DEFINED (4) types are matched literally in raw text. */
    uint64_t tn, tpos;
    if (!gguf_get_array(f, "tokenizer.ggml.token_type", &et, &tn, &tpos) || (et != GGUF_T_I32 && et != GGUF_T_U32) || tn != n) {
        set_err(err, errlen, "tokenizer: token types missing"); goto fail;
    }
    for (uint64_t i = 0; i < n; i++) {
        int32_t ty;
        memcpy(&ty, f->base + tpos + 4 * i, 4);
        if (ty == 3 || ty == 4) {
            if (t->n_added == MAX_ADDED) { set_err(err, errlen, "tokenizer: too many added tokens"); goto fail; }
            /* clef_tok_encode finds added tokens by scanning for '<' */
            if (t->tokens[i].len && t->tokens[i].ptr[0] != '<') { set_err(err, errlen, "tokenizer: added token does not start with '<'"); goto fail; }
            if (t->tokens[i].len == 0) { set_err(err, errlen, "tokenizer: empty added token"); goto fail; }
            t->added[t->n_added].s = t->tokens[i].ptr;
            t->added[t->n_added].len = (uint32_t)t->tokens[i].len;
            t->added[t->n_added].id = (int32_t)i;
            t->n_added++;
        }
    }

    for (int b = 0; b < 256; b++) {
        char buf[4];
        size_t l = utf8_put(buf, gpt2_byte_to_cp((uint8_t)b));
        t->byte_id[b] = str_table_get(&t->token_to_id, buf, l);
        if (t->byte_id[b] < 0) { set_err(err, errlen, "tokenizer: byte token missing"); goto fail; }
    }

    if (!gguf_get_array(f, "tokenizer.ggml.merges", &et, &n, &pos) || et != GGUF_T_STRING || n > INT32_MAX) {
        set_err(err, errlen, "tokenizer: merges missing"); goto fail;
    }
    gguf_str *merges = calloc(n ? n : 1, sizeof(*merges));
    if (!merges || !gguf_read_strings(f, pos, n, merges) || !pair_table_init(&t->merges, n)) {
        free(merges); set_err(err, errlen, "tokenizer: bad merges"); goto fail;
    }
    char *joined = NULL;
    size_t joined_cap = 0;
    for (uint64_t r = 0; r < n; r++) {
        const char *m = merges[r].ptr;
        const char *sp = memchr(m, ' ', merges[r].len);
        if (!sp) { free(merges); free(joined); set_err(err, errlen, "tokenizer: malformed merge"); goto fail; }
        size_t la = (size_t)(sp - m), lb = merges[r].len - la - 1;
        int32_t a = str_table_get(&t->token_to_id, m, la);
        int32_t b = str_table_get(&t->token_to_id, sp + 1, lb);
        if (la + lb > joined_cap) {
            joined_cap = (la + lb) * 2;
            char *p = realloc(joined, joined_cap);
            if (!p) { free(merges); free(joined); set_err(err, errlen, "tokenizer: out of memory"); goto fail; }
            joined = p;
        }
        memcpy(joined, m, la);
        memcpy(joined + la, sp + 1, lb);
        int32_t id = str_table_get(&t->token_to_id, joined, la + lb);
        if (a < 0 || b < 0 || id < 0) { free(merges); free(joined); set_err(err, errlen, "tokenizer: merge refers to unknown token"); goto fail; }
        pair_table_put(&t->merges, a, b, (int32_t)r, id);
    }
    free(joined);
    free(merges);

    uint32_t pad;
    if (!gguf_get_u32(f, "tokenizer.ggml.padding_token_id", &pad) || pad >= (uint32_t)t->n_vocab) {
        set_err(err, errlen, "tokenizer: pad token missing"); goto fail;
    }
    t->pad_id = (int32_t)pad;
    return t;

fail:
    clef_tok_free(t);
    return NULL;
}

void clef_tok_free(clef_tokenizer *t) {
    if (!t) return;
    free(t->tokens);
    free(t->token_to_id.slots);
    free(t->merges.slots);
    free(t);
}

/* ---- Unicode helpers --------------------------------------------------------- */

static bool in_ranges(const clef_urange *r, size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < r[mid].lo) hi = mid;
        else if (cp > r[mid].hi) lo = mid + 1;
        else return true;
    }
    return false;
}

#define NELEM(a) (sizeof(a) / sizeof((a)[0]))
static bool is_letter(uint32_t cp) { return cp < 0x80 ? ((cp | 0x20) >= 'a' && (cp | 0x20) <= 'z') : in_ranges(clef_letter_ranges, NELEM(clef_letter_ranges), cp); }
static bool is_number(uint32_t cp) { return cp < 0x80 ? (cp >= '0' && cp <= '9') : in_ranges(clef_number_ranges, NELEM(clef_number_ranges), cp); }
static bool is_space(uint32_t cp) { return in_ranges(clef_space_ranges, NELEM(clef_space_ranges), cp); }

static uint8_t ccc_of(uint32_t cp) {
    if (cp < 0x300) return 0;
    size_t lo = 0, hi = NELEM(clef_ccc_ranges);
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < clef_ccc_ranges[mid].lo) hi = mid;
        else if (cp > clef_ccc_ranges[mid].hi) lo = mid + 1;
        else return clef_ccc_ranges[mid].ccc;
    }
    return 0;
}

static const clef_decomp_entry *decomp_of(uint32_t cp) {
    size_t lo = 0, hi = NELEM(clef_decomp);
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < clef_decomp[mid].cp) hi = mid;
        else if (cp > clef_decomp[mid].cp) lo = mid + 1;
        else return &clef_decomp[mid];
    }
    return NULL;
}

enum { S_BASE = 0xAC00, L_BASE = 0x1100, V_BASE = 0x1161, T_BASE = 0x11A7,
       L_COUNT = 19, V_COUNT = 21, T_COUNT = 28, N_COUNT = V_COUNT * T_COUNT, S_COUNT = L_COUNT * N_COUNT };

static uint32_t compose_pair(uint32_t a, uint32_t b) {
    if (a >= L_BASE && a < L_BASE + L_COUNT && b >= V_BASE && b < V_BASE + V_COUNT) {
        return S_BASE + ((a - L_BASE) * V_COUNT + (b - V_BASE)) * T_COUNT;
    }
    if (a >= S_BASE && a < S_BASE + S_COUNT && (a - S_BASE) % T_COUNT == 0 &&
        b > T_BASE && b < T_BASE + T_COUNT) {
        return a + (b - T_BASE);
    }
    size_t lo = 0, hi = NELEM(clef_comp);
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        const clef_comp_entry *e = &clef_comp[mid];
        if (a < e->a || (a == e->a && b < e->b)) hi = mid;
        else if (a > e->a || b > e->b) lo = mid + 1;
        else return e->cp;
    }
    return 0;
}

static bool decompose_into(uint32_t cp, cpvec *out, int depth) {
    if (cp >= S_BASE && cp < S_BASE + S_COUNT) {
        uint32_t s = cp - S_BASE;
        if (!cp_push(out, L_BASE + s / N_COUNT) || !cp_push(out, V_BASE + (s % N_COUNT) / T_COUNT)) return false;
        if (s % T_COUNT) return cp_push(out, T_BASE + s % T_COUNT);
        return true;
    }
    const clef_decomp_entry *d = depth < 8 ? decomp_of(cp) : NULL;
    if (!d) return cp_push(out, cp);
    if (!decompose_into(d->a, out, depth + 1)) return false;
    return d->b ? decompose_into(d->b, out, depth + 1) : true;
}

/* Decode one code point; invalid sequences decode to U+FFFD and consume one byte. */
static uint32_t utf8_next(const uint8_t *s, size_t len, size_t *pos) {
    size_t i = *pos;
    uint8_t c = s[i];
    if (c < 0x80) { *pos = i + 1; return c; }
    int n = (c & 0xe0) == 0xc0 ? 2 : (c & 0xf0) == 0xe0 ? 3 : (c & 0xf8) == 0xf0 ? 4 : 0;
    if (n == 0 || i + (size_t)n > len) { *pos = i + 1; return 0xFFFD; }
    uint32_t cp = c & (0x7f >> n);
    for (int k = 1; k < n; k++) {
        if ((s[i + k] & 0xc0) != 0x80) { *pos = i + 1; return 0xFFFD; }
        cp = (cp << 6) | (s[i + k] & 0x3f);
    }
    static const uint32_t min_cp[5] = {0, 0, 0x80, 0x800, 0x10000};
    if (cp < min_cp[n] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) { *pos = i + 1; return 0xFFFD; }
    *pos = i + (size_t)n;
    return cp;
}


static bool canonical_order(uint32_t *cp, size_t n) {
    uint32_t *tmp = NULL;
    size_t tmp_cap = 0;
    for (size_t i = 0; i < n;) {
        if (!ccc_of(cp[i])) { i++; continue; }
        size_t j = i;
        while (j < n && ccc_of(cp[j])) j++;
        const size_t len = j - i;
        if (len > 1) {
            if (len > tmp_cap) {
                uint32_t *t = realloc(tmp, len * sizeof(*t));
                if (!t) { free(tmp); return false; }
                tmp = t;
                tmp_cap = len;
            }
            size_t start[257] = {0};
            for (size_t k = i; k < j; k++) start[ccc_of(cp[k]) + 1]++;
            for (int c = 1; c <= 256; c++) start[c] += start[c - 1];
            for (size_t k = i; k < j; k++) tmp[start[ccc_of(cp[k])]++] = cp[k];
            memcpy(cp + i, tmp, len * sizeof(*cp));
        }
        i = j;
    }
    free(tmp);
    return true;
}

bool clef_nfc(const char *s, size_t len, char **out, size_t *out_len) {
    cpvec cps = {0};
    for (size_t p = 0; p < len;) {
        if (!decompose_into(utf8_next((const uint8_t *)s, len, &p), &cps, 0)) { free(cps.cp); return false; }
    }
    /* Canonical ordering: stable sort of each maximal run of non-starters by ccc.
     * A counting sort keeps this linear; an insertion sort made a long run of
     * reverse-ordered combining marks quadratic (160 KB: 4.7 s). */
    if (!canonical_order(cps.cp, cps.len)) { free(cps.cp); return false; }
    /* Canonical composition (UAX #15 reference algorithm). */
    size_t comp_len = cps.len;
    if (cps.len > 1) {
        size_t starter = 0, w = 1;
        int last_cc = ccc_of(cps.cp[0]);
        if (last_cc) last_cc = 256;
        for (size_t r = 1; r < cps.len; r++) {
            uint32_t ch = cps.cp[r];
            int cc = ccc_of(ch);
            uint32_t composite = compose_pair(cps.cp[starter], ch);
            if (composite && (last_cc < cc || last_cc == 0)) {
                cps.cp[starter] = composite;
                continue;
            }
            if (cc == 0) starter = w;
            last_cc = cc;
            cps.cp[w++] = ch;
        }
        comp_len = w;
    }
    char *buf = malloc(comp_len * 4 + 1);
    if (!buf) { free(cps.cp); return false; }
    size_t o = 0;
    for (size_t i = 0; i < comp_len; i++) o += utf8_put(buf + o, cps.cp[i]);
    buf[o] = '\0';
    free(cps.cp);
    *out = buf;
    *out_len = o;
    return true;
}

/* ---- BPE -------------------------------------------------------------------- */

typedef struct { int32_t rank; int32_t pos; int32_t id; } merge_item;

static bool heap_less(const merge_item *a, const merge_item *b) {
    return a->rank < b->rank || (a->rank == b->rank && a->pos < b->pos);
}

static void heap_push(merge_item *h, size_t *n, merge_item it) {
    size_t i = (*n)++;
    h[i] = it;
    while (i > 0) {
        size_t p = (i - 1) / 2;
        if (!heap_less(&h[i], &h[p])) break;
        merge_item tmp = h[i]; h[i] = h[p]; h[p] = tmp;
        i = p;
    }
}

static merge_item heap_pop(merge_item *h, size_t *n) {
    merge_item top = h[0];
    h[0] = h[--(*n)];
    size_t i = 0;
    for (;;) {
        size_t l = 2 * i + 1, r = l + 1, m = i;
        if (l < *n && heap_less(&h[l], &h[m])) m = l;
        if (r < *n && heap_less(&h[r], &h[m])) m = r;
        if (m == i) break;
        merge_item tmp = h[i]; h[i] = h[m]; h[m] = tmp;
        i = m;
    }
    return top;
}

/* Mirrors tokenizers' Word::merge_all: a min-heap of (rank, position), with stale
 * entries skipped. O(n log n), so pathological pieces cannot stall the engine. */
static bool bpe_piece(const clef_tokenizer *t, const uint8_t *s, size_t n, clef_tokens *out) {
    if (n == 1) return clef_tokens_push(out, t->byte_id[s[0]]);
    int32_t *id = malloc(n * sizeof(*id));
    int32_t *prev = malloc(n * sizeof(*prev));
    int32_t *next = malloc(n * sizeof(*next));
    uint8_t *alive = malloc(n);
    /* every merge pushes at most two new pairs, plus n-1 initial pairs */
    merge_item *heap = malloc((3 * n + 1) * sizeof(*heap));
    bool ok = id && prev && next && alive && heap;
    size_t hn = 0;
    if (ok) {
        for (size_t i = 0; i < n; i++) {
            id[i] = t->byte_id[s[i]];
            prev[i] = (int32_t)i - 1;
            next[i] = i + 1 < n ? (int32_t)i + 1 : -1;
            alive[i] = 1;
        }
        for (size_t i = 0; i + 1 < n; i++) {
            const pair_slot *m = pair_table_get(&t->merges, id[i], id[i + 1]);
            if (m) heap_push(heap, &hn, (merge_item){ m->rank, (int32_t)i, m->id });
        }
        while (hn) {
            merge_item top = heap_pop(heap, &hn);
            int32_t p = top.pos;
            if (!alive[p] || next[p] < 0) continue;
            int32_t q = next[p];
            const pair_slot *m = pair_table_get(&t->merges, id[p], id[q]);
            if (!m || m->id != top.id) continue;
            id[p] = top.id;
            alive[q] = 0;
            next[p] = next[q];
            if (next[q] >= 0) prev[next[q]] = p;
            if (prev[p] >= 0) {
                const pair_slot *ml = pair_table_get(&t->merges, id[prev[p]], id[p]);
                if (ml) heap_push(heap, &hn, (merge_item){ ml->rank, prev[p], ml->id });
            }
            if (next[p] >= 0) {
                const pair_slot *mr = pair_table_get(&t->merges, id[p], id[next[p]]);
                if (mr) heap_push(heap, &hn, (merge_item){ mr->rank, p, mr->id });
            }
        }
        for (int32_t i = 0; i >= 0 && ok; i = next[i]) ok = clef_tokens_push(out, id[i]);
    }
    free(id); free(prev); free(next); free(alive); free(heap);
    return ok;
}

/* ---- pre-tokenizer ---------------------------------------------------------- */

typedef struct { uint32_t cp; size_t next; bool valid, letter, number, space; } cinfo;

static cinfo char_at(const uint8_t *s, size_t len, size_t pos) {
    cinfo c = {0};
    if (pos >= len) return c;
    c.valid = true;
    size_t p = pos;
    c.cp = utf8_next(s, len, &p);
    c.next = p;
    c.letter = is_letter(c.cp);
    c.number = is_number(c.cp);
    c.space = is_space(c.cp);
    return c;
}

static uint32_t fold_ascii(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return cp + 32;
    if (cp == 0x017F) return 's';  /* LATIN SMALL LETTER LONG S case-folds to 's' */
    return cp;
}

/* The split regex, as an ordered alternation evaluated at each position:
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N} |
 *    ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
 * Combining marks (\p{M}) are neither letters nor numbers here, so they end letter
 * runs and belong to the punctuation alternative. Adapted from ds4's qwen35 splitter
 * (https://github.com/antirez/ds4, MIT; see THIRD_PARTY_NOTICES.md), which implements the
 * later [\p{L}\p{M}] variant of this regex. */
static bool pretokenize(const clef_tokenizer *t, const uint8_t *s, size_t len, size_t limit, clef_tokens *out) {
    size_t pos = 0;
    while (pos < len && out->len < limit) {
        const size_t start = pos;
        cinfo cur = char_at(s, len, pos);

        if (cur.cp == '\'' && cur.next < len) {
            cinfo n1 = char_at(s, len, cur.next);
            uint32_t c1 = fold_ascii(n1.cp);
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                pos = n1.next;
                if (!bpe_piece(t, s + start, pos - start, out)) return false;
                continue;
            }
            if (n1.next < len) {
                cinfo n2 = char_at(s, len, n1.next);
                uint32_t c2 = fold_ascii(n2.cp);
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                    pos = n2.next;
                    if (!bpe_piece(t, s + start, pos - start, out)) return false;
                    continue;
                }
            }
        }

        /* [^\r\n\p{L}\p{N}]?\p{L}+ */
        {
            size_t run = SIZE_MAX;
            if (cur.letter) {
                run = cur.next;
            } else if (cur.cp != '\r' && cur.cp != '\n' && !cur.number) {
                cinfo n1 = char_at(s, len, cur.next);
                if (n1.valid && n1.letter) run = n1.next;
            }
            if (run != SIZE_MAX) {
                pos = run;
                while (pos < len) {
                    cinfo c = char_at(s, len, pos);
                    if (!c.letter) break;
                    pos = c.next;
                }
                if (!bpe_piece(t, s + start, pos - start, out)) return false;
                continue;
            }
        }

        if (cur.number) {
            pos = cur.next;
            if (!bpe_piece(t, s + start, pos - start, out)) return false;
            continue;
        }

        /*  ?[^\s\p{L}\p{N}]+[\r\n]* */
        {
            cinfo punct = cur;
            size_t ppos = pos;
            if (cur.cp == ' ') { ppos = cur.next; punct = char_at(s, len, ppos); }
            if (punct.valid && !punct.space && !punct.letter && !punct.number) {
                pos = ppos;
                while (pos < len) {
                    cinfo c = char_at(s, len, pos);
                    if (c.space || c.letter || c.number) break;
                    pos = c.next;
                }
                while (pos < len && (s[pos] == '\r' || s[pos] == '\n')) pos++;
                if (!bpe_piece(t, s + start, pos - start, out)) return false;
                continue;
            }
        }

        if (cur.space) {
            /* \s*[\r\n]+  |  \s+(?!\S)  |  \s+ */
            size_t p = pos, last_nl_end = 0, last_ws_start = pos;
            int nspace = 0;
            while (p < len) {
                cinfo c = char_at(s, len, p);
                if (!c.space) break;
                last_ws_start = p;
                if (c.cp == '\r' || c.cp == '\n') last_nl_end = c.next;
                p = c.next;
                nspace++;
            }
            if (last_nl_end) pos = last_nl_end;
            else if (nspace > 1 && p < len) pos = last_ws_start;
            else pos = p;
            if (!bpe_piece(t, s + start, pos - start, out)) return false;
            continue;
        }

        /* Unreachable for valid input (every code point is in one class above);
         * consume one character so malformed input cannot loop. */
        pos = cur.next > pos ? cur.next : pos + 1;
        if (!bpe_piece(t, s + start, pos - start, out)) return false;
    }
    return true;
}

static bool encode_segment(const clef_tokenizer *t, const char *s, size_t n, size_t limit, clef_tokens *out) {
    if (!n) return true;
    bool ascii = true;
    for (size_t i = 0; i < n && ascii; i++) ascii = (uint8_t)s[i] < 0x80;
    if (ascii) return pretokenize(t, (const uint8_t *)s, n, limit, out);
    char *norm;
    size_t norm_len;
    if (!clef_nfc(s, n, &norm, &norm_len)) return false;
    bool ok = pretokenize(t, (const uint8_t *)norm, norm_len, limit, out);
    free(norm);
    return ok;
}

/* Pre-tokenized pieces are encoded independently and in order, and each piece boundary
 * depends only on bytes up to the piece's end (+1 lookahead), so stopping after the piece
 * that reaches the limit yields exactly the first `limit` tokens of the full tokenization.
 * Lets callers that truncate (the state) skip tokenizing megabytes they will discard. */
bool clef_tok_encode_ex(const clef_tokenizer *t, const char *text, size_t len, size_t max_tokens,
                        bool split_special, clef_tokens *out) {
    const size_t limit = max_tokens > SIZE_MAX - out->len ? SIZE_MAX : out->len + max_tokens;
    /* strict: added tokens (<|im_end|>, <think>, ...) are not recognized; their text is
     * normalized and BPE-encoded like any other text, so content cannot emit control tokens */
    if (split_special) return encode_segment(t, text, len, limit, out);
    /* Split on added tokens (leftmost, then longest), as tokenizers' AddedVocabulary does. */
    size_t seg = 0, pos = 0;
    while (pos < len && out->len < limit) {
        const char *lt = memchr(text + pos, '<', len - pos);
        if (!lt) break;
        pos = (size_t)(lt - text);
        int best = -1;
        for (int i = 0; i < t->n_added; i++) {
            if (t->added[i].len <= len - pos && !memcmp(text + pos, t->added[i].s, t->added[i].len) &&
                (best < 0 || t->added[i].len > t->added[best].len)) {
                best = i;
            }
        }
        if (best < 0) { pos++; continue; }
        if (!encode_segment(t, text + seg, pos - seg, limit, out)) return false;
        if (out->len >= limit) return true;
        if (!clef_tokens_push(out, t->added[best].id)) return false;
        pos += t->added[best].len;
        seg = pos;
    }
    if (out->len >= limit) return true;
    return encode_segment(t, text + seg, len - seg, limit, out);
}

bool clef_tok_encode_max(const clef_tokenizer *t, const char *text, size_t len, size_t max_tokens, clef_tokens *out) {
    return clef_tok_encode_ex(t, text, len, max_tokens, false, out);
}

bool clef_tok_encode(const clef_tokenizer *t, const char *text, size_t len, clef_tokens *out) {
    return clef_tok_encode_ex(t, text, len, SIZE_MAX, false, out);
}
