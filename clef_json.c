#include "clef_json.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define JSON_MAX_DEPTH 512  /* Python allows ~1000 (recursion limit); bounded for server thread stacks */
/* Python 3.12 refuses int <-> str conversion above 4300 digits (sys.int_info). */
#define PY_INT_MAX_DIGITS 4300

/* ---- arena ------------------------------------------------------------------ */

typedef struct jblock { struct jblock *next; size_t used, cap; char data[]; } jblock;
struct jarena { jblock *head; };

jarena *jarena_new(void) { return calloc(1, sizeof(jarena)); }

void jarena_free(jarena *a) {
    if (!a) return;
    for (jblock *b = a->head; b;) { jblock *n = b->next; free(b); b = n; }
    free(a);
}

static void *aalloc(jarena *a, size_t n) {
    n = (n + 15) & ~(size_t)15;
    if (!a->head || a->head->cap - a->head->used < n) {
        size_t cap = n > 65536 ? n : 65536;
        jblock *b = malloc(sizeof(jblock) + cap);
        if (!b) return NULL;
        b->next = a->head; b->used = 0; b->cap = cap;
        a->head = b;
    }
    void *p = a->head->data + a->head->used;
    a->head->used += n;
    return p;
}

/* ---- parser ----------------------------------------------------------------- */

typedef struct {
    jarena *a;
    const char *s;
    size_t len, pos;
    char *err;
    size_t errlen;
    bool failed;
} parser;

static jval *perr(parser *p, const char *msg) {
    if (!p->failed) snprintf(p->err, p->errlen, "json: %s at byte %zu", msg, p->pos);
    p->failed = true;
    return NULL;
}

static void skip_ws(parser *p) {
    while (p->pos < p->len) {
        char c = p->s[p->pos];
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') break;
        p->pos++;
    }
}

static jval *newval(parser *p, jtype t) {
    jval *v = aalloc(p->a, sizeof(*v));
    if (!v) return perr(p, "out of memory");
    memset(v, 0, sizeof(*v));
    v->type = t;
    return v;
}

static bool lit(parser *p, const char *z) {
    size_t n = strlen(z);
    if (p->len - p->pos >= n && !memcmp(p->s + p->pos, z, n)) { p->pos += n; return true; }
    return false;
}

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static bool read_hex4(parser *p, uint32_t *out) {
    if (p->len - p->pos < 4) return false;
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        int h = hexval(p->s[p->pos + i]);
        if (h < 0) return false;
        v = (v << 4) | (uint32_t)h;
    }
    p->pos += 4;
    *out = v;
    return true;
}

static size_t put_utf8(char *o, uint32_t cp) {
    if (cp < 0x80) { o[0] = (char)cp; return 1; }
    if (cp < 0x800) { o[0] = (char)(0xc0 | (cp >> 6)); o[1] = (char)(0x80 | (cp & 0x3f)); return 2; }
    if (cp < 0x10000) {
        o[0] = (char)(0xe0 | (cp >> 12)); o[1] = (char)(0x80 | ((cp >> 6) & 0x3f));
        o[2] = (char)(0x80 | (cp & 0x3f)); return 3;
    }
    o[0] = (char)(0xf0 | (cp >> 18)); o[1] = (char)(0x80 | ((cp >> 12) & 0x3f));
    o[2] = (char)(0x80 | ((cp >> 6) & 0x3f)); o[3] = (char)(0x80 | (cp & 0x3f)); return 4;
}

/* Validates one raw UTF-8 sequence starting at s[i]; returns its length or 0. */
static size_t utf8_valid_len(const unsigned char *s, size_t avail) {
    unsigned char c = s[0];
    size_t n; uint32_t cp;
    if (c < 0x80) return 1;
    if ((c & 0xe0) == 0xc0) { n = 2; cp = c & 0x1f; }
    else if ((c & 0xf0) == 0xe0) { n = 3; cp = c & 0x0f; }
    else if ((c & 0xf8) == 0xf0) { n = 4; cp = c & 0x07; }
    else return 0;
    if (n > avail) return 0;
    for (size_t k = 1; k < n; k++) {
        if ((s[k] & 0xc0) != 0x80) return 0;
        cp = (cp << 6) | (s[k] & 0x3f);
    }
    static const uint32_t min_cp[5] = {0, 0, 0x80, 0x800, 0x10000};
    if (cp < min_cp[n] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return 0;
    return n;
}

static bool parse_string_raw(parser *p, const char **out, size_t *out_len) {
    /* p->pos is at the opening quote */
    p->pos++;
    size_t start = p->pos, decoded = 0;
    bool has_escape = false;
    for (size_t i = start;; ) {
        if (i >= p->len) { p->pos = i; perr(p, "unterminated string"); return false; }
        unsigned char c = (unsigned char)p->s[i];
        if (c == '"') break;
        if (c < 0x20) { p->pos = i; perr(p, "control character in string"); return false; }
        if (c == '\\') { has_escape = true; i += 2; decoded += 4; continue; }
        size_t n = utf8_valid_len((const unsigned char *)p->s + i, p->len - i);
        if (!n) { p->pos = i; perr(p, "invalid UTF-8"); return false; }
        i += n; decoded += n;
    }
    if (!has_escape) {
        const char *q = memchr(p->s + start, '"', p->len - start);
        *out = p->s + start;
        *out_len = (size_t)(q - (p->s + start));
        p->pos = (size_t)(q - p->s) + 1;
        return true;
    }
    char *buf = aalloc(p->a, decoded + 1);
    if (!buf) { perr(p, "out of memory"); return false; }
    size_t o = 0;
    while (p->s[p->pos] != '"') {
        char c = p->s[p->pos];
        if (c != '\\') { buf[o++] = c; p->pos++; continue; }
        p->pos++;
        if (p->pos >= p->len) { perr(p, "bad escape"); return false; }
        char e = p->s[p->pos++];
        switch (e) {
        case '"': buf[o++] = '"'; break;
        case '\\': buf[o++] = '\\'; break;
        case '/': buf[o++] = '/'; break;
        case 'b': buf[o++] = '\b'; break;
        case 'f': buf[o++] = '\f'; break;
        case 'n': buf[o++] = '\n'; break;
        case 'r': buf[o++] = '\r'; break;
        case 't': buf[o++] = '\t'; break;
        case 'u': {
            uint32_t cp;
            if (!read_hex4(p, &cp)) { perr(p, "bad \\u escape"); return false; }
            if (cp >= 0xD800 && cp <= 0xDBFF && p->len - p->pos >= 6 &&
                p->s[p->pos] == '\\' && p->s[p->pos + 1] == 'u') {
                size_t save = p->pos;
                p->pos += 2;
                uint32_t lo;
                if (read_hex4(p, &lo) && lo >= 0xDC00 && lo <= 0xDFFF) {
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                } else {
                    p->pos = save;
                }
            }
            /* json.loads accepts lone surrogates, but the reference tokenizer cannot
             * encode them (UnicodeEncodeError), so the request fails there too. */
            if (cp >= 0xD800 && cp <= 0xDFFF) { perr(p, "lone surrogate in string"); return false; }
            o += put_utf8(buf + o, cp);
            break;
        }
        default: perr(p, "bad escape"); return false;
        }
    }
    p->pos++;
    buf[o] = '\0';
    *out = buf;
    *out_len = o;
    return true;
}


/* Duplicate-key lookup. A linear scan per key made an object with n distinct keys cost
 * O(n^2) (80k keys: 3.8 s), a cheap way for one request to stall the engine; above a
 * few keys an open-addressed index of member positions keeps parsing linear. */
#define KEY_INDEX_MIN 16

typedef struct { uint32_t *slot; size_t mask; } key_index;   /* slot: member index + 1, 0 = empty */

static uint64_t key_hash(const char *k, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= (uint8_t)k[i]; h *= 1099511628211ull; }
    return h;
}

static size_t find_member(const jval *v, const key_index *ix, const char *k, size_t kl) {
    if (!ix->slot) {
        for (size_t i = 0; i < v->n; i++) {
            if (v->members[i].klen == kl && !memcmp(v->members[i].key, k, kl)) return i;
        }
        return v->n;
    }
    for (size_t s = key_hash(k, kl) & ix->mask;; s = (s + 1) & ix->mask) {
        uint32_t e = ix->slot[s];
        if (!e) return v->n;
        const jmember *m = &v->members[e - 1];
        if (m->klen == kl && !memcmp(m->key, k, kl)) return e - 1;
    }
}

static void index_put(key_index *ix, const jval *v, size_t i) {
    const jmember *m = &v->members[i];
    size_t s = key_hash(m->key, m->klen) & ix->mask;
    while (ix->slot[s]) s = (s + 1) & ix->mask;
    ix->slot[s] = (uint32_t)(i + 1);
}

/* Called after appending member v->n - 1: build or grow the index (load <= 1/2). */
static bool index_member(jarena *a, const jval *v, key_index *ix) {
    if (v->n < KEY_INDEX_MIN) return true;
    if (v->n > UINT32_MAX - 1) return false;
    if (ix->slot && v->n * 2 <= ix->mask + 1) { index_put(ix, v, v->n - 1); return true; }
    size_t cap = 64;
    while (cap < v->n * 4) cap *= 2;
    uint32_t *slot = aalloc(a, cap * sizeof(*slot));
    if (!slot) return false;
    memset(slot, 0, cap * sizeof(*slot));
    ix->slot = slot;
    ix->mask = cap - 1;
    for (size_t i = 0; i < v->n; i++) index_put(ix, v, i);
    return true;
}

static jval *parse_value(parser *p, int depth);

static jval *parse_number(parser *p) {
    size_t start = p->pos;
    bool neg = false;
    if (p->s[p->pos] == '-') { neg = true; p->pos++; }
    if (lit(p, "Infinity")) {
        jval *v = newval(p, J_FLOAT);
        if (v) v->f = neg ? -INFINITY : INFINITY;
        return v;
    }
    size_t int_start = p->pos;
    if (p->pos < p->len && p->s[p->pos] == '0') p->pos++;
    else if (p->pos < p->len && p->s[p->pos] >= '1' && p->s[p->pos] <= '9') {
        while (p->pos < p->len && p->s[p->pos] >= '0' && p->s[p->pos] <= '9') p->pos++;
    } else return perr(p, "bad number");
    size_t int_end = p->pos;
    bool is_float = false;
    if (p->pos < p->len && p->s[p->pos] == '.' && p->pos + 1 < p->len &&
        p->s[p->pos + 1] >= '0' && p->s[p->pos + 1] <= '9') {
        is_float = true;
        p->pos++;
        while (p->pos < p->len && p->s[p->pos] >= '0' && p->s[p->pos] <= '9') p->pos++;
    }
    if (p->pos < p->len && (p->s[p->pos] == 'e' || p->s[p->pos] == 'E')) {
        size_t save = p->pos;
        p->pos++;
        if (p->pos < p->len && (p->s[p->pos] == '+' || p->s[p->pos] == '-')) p->pos++;
        if (p->pos < p->len && p->s[p->pos] >= '0' && p->s[p->pos] <= '9') {
            is_float = true;
            while (p->pos < p->len && p->s[p->pos] >= '0' && p->s[p->pos] <= '9') p->pos++;
        } else {
            p->pos = save;  /* Python's scanner stops before a dangling exponent */
        }
    }
    if (!is_float) {
        size_t nd = int_end - int_start;
        if (nd > PY_INT_MAX_DIGITS) return perr(p, "integer exceeds Python's 4300-digit limit");
        jval *v = newval(p, J_INT);
        if (!v) return NULL;
        bool zero = nd == 1 && p->s[int_start] == '0';
        v->s = zero ? p->s + int_start : p->s + start;  /* int('-0') == 0 */
        v->len = zero ? 1 : int_end - start;
        return v;
    }
    size_t n = p->pos - start;
    char stackbuf[128];
    char *tmp = n < sizeof(stackbuf) ? stackbuf : malloc(n + 1);
    if (!tmp) return perr(p, "out of memory");
    memcpy(tmp, p->s + start, n);
    tmp[n] = '\0';
    jval *v = newval(p, J_FLOAT);
    if (v) v->f = strtod(tmp, NULL);  /* correctly rounded; overflow -> inf like float() */
    if (tmp != stackbuf) free(tmp);
    return v;
}

static jval *parse_value(parser *p, int depth) {
    if (depth > JSON_MAX_DEPTH) return perr(p, "nesting too deep");
    skip_ws(p);
    if (p->pos >= p->len) return perr(p, "unexpected end of input");
    char c = p->s[p->pos];
    if (c == '{') {
        p->pos++;
        jval *v = newval(p, J_OBJECT);
        if (!v) return NULL;
        size_t cap = 0;
        key_index idx = {0};
        skip_ws(p);
        if (p->pos < p->len && p->s[p->pos] == '}') { p->pos++; return v; }
        for (;;) {
            skip_ws(p);
            if (p->pos >= p->len || p->s[p->pos] != '"') return perr(p, "expected object key");
            const char *k; size_t kl;
            if (!parse_string_raw(p, &k, &kl)) return NULL;
            skip_ws(p);
            if (p->pos >= p->len || p->s[p->pos] != ':') return perr(p, "expected ':'");
            p->pos++;
            jval *val = parse_value(p, depth + 1);
            if (!val) return NULL;
            size_t i = find_member(v, &idx, k, kl);
            if (i < v->n) {
                v->members[i].val = val;
            } else {
                if (v->n == cap) {
                    size_t nc = cap ? cap * 2 : 8;
                    jmember *m = aalloc(p->a, nc * sizeof(*m));
                    if (!m) return perr(p, "out of memory");
                    if (v->n) memcpy(m, v->members, v->n * sizeof(*m));
                    v->members = m;
                    cap = nc;
                }
                v->members[v->n++] = (jmember){ k, kl, val };
                if (!index_member(p->a, v, &idx)) return perr(p, "out of memory");
            }
            skip_ws(p);
            if (p->pos < p->len && p->s[p->pos] == ',') { p->pos++; continue; }
            if (p->pos < p->len && p->s[p->pos] == '}') { p->pos++; return v; }
            return perr(p, "expected ',' or '}'");
        }
    }
    if (c == '[') {
        p->pos++;
        jval *v = newval(p, J_ARRAY);
        if (!v) return NULL;
        size_t cap = 0;
        skip_ws(p);
        if (p->pos < p->len && p->s[p->pos] == ']') { p->pos++; return v; }
        for (;;) {
            jval *item = parse_value(p, depth + 1);
            if (!item) return NULL;
            if (v->n == cap) {
                size_t nc = cap ? cap * 2 : 8;
                jval **it = aalloc(p->a, nc * sizeof(*it));
                if (!it) return perr(p, "out of memory");
                if (v->n) memcpy(it, v->items, v->n * sizeof(*it));
                v->items = it;
                cap = nc;
            }
            v->items[v->n++] = item;
            skip_ws(p);
            if (p->pos < p->len && p->s[p->pos] == ',') { p->pos++; continue; }
            if (p->pos < p->len && p->s[p->pos] == ']') { p->pos++; return v; }
            return perr(p, "expected ',' or ']'");
        }
    }
    if (c == '"') {
        jval *v = newval(p, J_STRING);
        if (!v) return NULL;
        return parse_string_raw(p, &v->s, &v->len) ? v : NULL;
    }
    if (lit(p, "null")) return newval(p, J_NULL);
    if (lit(p, "true")) return newval(p, J_TRUE);
    if (lit(p, "false")) return newval(p, J_FALSE);
    if (lit(p, "NaN")) { jval *v = newval(p, J_FLOAT); if (v) v->f = NAN; return v; }
    if (lit(p, "Infinity")) { jval *v = newval(p, J_FLOAT); if (v) v->f = INFINITY; return v; }
    if (c == '-' || (c >= '0' && c <= '9')) return parse_number(p);
    return perr(p, "unexpected character");
}

jval *json_parse(jarena *a, const char *text, size_t len, char *err, size_t errlen) {
    parser p = { a, text, len, 0, err, errlen, false };
    jval *v = parse_value(&p, 0);
    if (!v) return NULL;
    skip_ws(&p);
    if (p.pos != p.len) return perr(&p, "trailing data");
    return v;
}

jval *json_get(const jval *obj, const char *key) {
    if (!obj || obj->type != J_OBJECT) return NULL;
    size_t n = strlen(key);
    for (size_t i = 0; i < obj->n; i++) {
        if (obj->members[i].klen == n && !memcmp(obj->members[i].key, key, n)) return obj->members[i].val;
    }
    return NULL;
}

bool json_str_eq(const jval *v, const char *z) {
    size_t n = strlen(z);
    return v && v->type == J_STRING && v->len == n && !memcmp(v->s, z, n);
}

/* ---- writer ------------------------------------------------------------------ */

void jbuf_put(jbuf *b, const char *s, size_t n) {
    if (b->oom) return;
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 256;
        while (cap < b->len + n + 1) cap *= 2;
        char *p = realloc(b->p, cap);
        if (!p) { b->oom = true; return; }
        b->p = p;
        b->cap = cap;
    }
    memcpy(b->p + b->len, s, n);
    b->len += n;
    b->p[b->len] = '\0';
}

void jbuf_puts(jbuf *b, const char *z) { jbuf_put(b, z, strlen(z)); }

void jbuf_free(jbuf *b) { free(b->p); memset(b, 0, sizeof(*b)); }

void json_put_string(jbuf *b, const char *s, size_t len) {
    jbuf_put(b, "\"", 1);
    size_t run = 0;
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];
        const char *esc = NULL;
        char ubuf[8];
        switch (c) {
        case '"': esc = "\\\""; break;
        case '\\': esc = "\\\\"; break;
        case '\n': esc = "\\n"; break;
        case '\r': esc = "\\r"; break;
        case '\t': esc = "\\t"; break;
        case '\b': esc = "\\b"; break;
        case '\f': esc = "\\f"; break;
        default:
            if (c < 0x20) { snprintf(ubuf, sizeof(ubuf), "\\u%04x", c); esc = ubuf; }
        }
        if (esc) {
            jbuf_put(b, s + i - run, run);
            run = 0;
            jbuf_puts(b, esc);
        } else {
            run++;
        }
    }
    jbuf_put(b, s + len - run, run);
    jbuf_put(b, "\"", 1);
}

static bool roundtrips(const char *digits_e, double x) {
    return strtod(digits_e, NULL) == x;
}

/* Shortest round-trip digits, nearest among equally short: the contract of
 * Python's repr (David Gay's dtoa mode 0). For each precision the correctly
 * rounded candidate is tried first; if it does not round-trip, its decimal
 * neighbours are tried, which covers the asymmetric rounding interval at
 * powers of two. Verified against Python by tests/test_json.py. */
static int shortest_digits(double x, char *digits, int *decpt) {
    char buf[64];
    for (int prec = 1; prec <= 17; prec++) {
        snprintf(buf, sizeof(buf), "%.*e", prec - 1, x);
        bool ok = roundtrips(buf, x);
        char cand[3][64];
        int ncand = 0;
        if (ok) {
            memcpy(cand[ncand++], buf, sizeof(buf));
        } else if (prec < 17) {
            /* neighbours: +/- one unit in the last digit */
            for (int dir = -1; dir <= 1; dir += 2) {
                char *e = strchr(buf, 'e');
                int exp10 = atoi(e + 1);
                char mant[32];
                int m = 0;
                for (char *q = buf; q < e; q++) if (*q >= '0' && *q <= '9') mant[m++] = *q;
                mant[m] = '\0';
                /* add dir to the integer formed by mant */
                int i = m - 1;
                if (dir > 0) {
                    while (i >= 0 && mant[i] == '9') { mant[i] = '0'; i--; }
                    if (i < 0) { memmove(mant + 1, mant, (size_t)m + 1); mant[0] = '1'; mant[m] = '\0'; exp10++; }
                    else mant[i]++;
                } else {
                    while (i >= 0 && mant[i] == '0') { mant[i] = '9'; i--; }
                    if (i < 0) continue;
                    mant[i]--;
                    if (mant[0] == '0') continue;  /* would lose a digit: covered by prec-1 */
                }
                char c2[64];
                snprintf(c2, sizeof(c2), "%c.%se%d", mant[0], mant + 1, exp10);
                if (roundtrips(c2, x)) memcpy(cand[ncand++], c2, sizeof(c2));
            }
        }
        if (!ncand) continue;
        /* The correctly rounded candidate is the closest p-digit decimal to x, so when it
         * falls outside the rounding interval only the neighbour on the other side of x
         * can fall inside: at most one candidate exists here. */
        const char *best = cand[0];
        const char *e = strchr(best, 'e');
        int n = 0;
        for (const char *q = best; q < e; q++) if (*q >= '0' && *q <= '9') digits[n++] = *q;
        while (n > 1 && digits[n - 1] == '0') n--;
        digits[n] = '\0';
        *decpt = atoi(e + 1) + 1;
        return n;
    }
    return 0;
}

void json_put_float(jbuf *b, double x) {
    if (isnan(x)) { jbuf_puts(b, "NaN"); return; }
    if (isinf(x)) { jbuf_puts(b, x > 0 ? "Infinity" : "-Infinity"); return; }
    if (x == 0) { jbuf_puts(b, signbit(x) ? "-0.0" : "0.0"); return; }
    char out[64];
    size_t o = 0;
    if (x < 0) { out[o++] = '-'; x = -x; }
    char digits[32];
    int decpt;
    int n = shortest_digits(x, digits, &decpt);
    if (decpt > -4 && decpt <= 16) {
        if (decpt <= 0) {
            out[o++] = '0'; out[o++] = '.';
            for (int i = 0; i < -decpt; i++) out[o++] = '0';
            memcpy(out + o, digits, (size_t)n); o += (size_t)n;
        } else if (n <= decpt) {
            memcpy(out + o, digits, (size_t)n); o += (size_t)n;
            for (int i = n; i < decpt; i++) out[o++] = '0';
            out[o++] = '.'; out[o++] = '0';
        } else {
            memcpy(out + o, digits, (size_t)decpt); o += (size_t)decpt;
            out[o++] = '.';
            memcpy(out + o, digits + decpt, (size_t)(n - decpt)); o += (size_t)(n - decpt);
        }
    } else {
        out[o++] = digits[0];
        if (n > 1) { out[o++] = '.'; memcpy(out + o, digits + 1, (size_t)(n - 1)); o += (size_t)(n - 1); }
        int e = decpt - 1;
        o += (size_t)snprintf(out + o, sizeof(out) - o, "e%c%02d", e < 0 ? '-' : '+', e < 0 ? -e : e);
    }
    jbuf_put(b, out, o);
}

static int member_cmp(const void *a, const void *b) {
    const jmember *x = a, *y = b;
    size_t n = x->klen < y->klen ? x->klen : y->klen;
    int c = memcmp(x->key, y->key, n);  /* UTF-8 byte order == code point order */
    if (c) return c;
    return (x->klen > y->klen) - (x->klen < y->klen);
}

void json_dump(jbuf *b, const jval *v, bool sort_keys) {
    switch (v->type) {
    case J_NULL: jbuf_puts(b, "null"); return;
    case J_TRUE: jbuf_puts(b, "true"); return;
    case J_FALSE: jbuf_puts(b, "false"); return;
    case J_INT: jbuf_put(b, v->s, v->len); return;
    case J_FLOAT: json_put_float(b, v->f); return;
    case J_STRING: json_put_string(b, v->s, v->len); return;
    case J_ARRAY:
        jbuf_put(b, "[", 1);
        for (size_t i = 0; i < v->n; i++) {
            if (i) jbuf_put(b, ",", 1);
            json_dump(b, v->items[i], sort_keys);
        }
        jbuf_put(b, "]", 1);
        return;
    case J_OBJECT: {
        jmember *m = v->members;
        jmember *sorted = NULL;
        if (sort_keys && v->n > 1) {
            sorted = malloc(v->n * sizeof(*sorted));
            if (!sorted) { b->oom = true; return; }
            memcpy(sorted, v->members, v->n * sizeof(*sorted));
            qsort(sorted, v->n, sizeof(*sorted), member_cmp);
            m = sorted;
        }
        jbuf_put(b, "{", 1);
        for (size_t i = 0; i < v->n; i++) {
            if (i) jbuf_put(b, ",", 1);
            json_put_string(b, m[i].key, m[i].klen);
            jbuf_put(b, ":", 1);
            json_dump(b, m[i].val, sort_keys);
        }
        jbuf_put(b, "}", 1);
        free(sorted);
        return;
    }
    }
}

double py_round(double x, int ndigits) {
    /* Python rounds the exact binary value half-to-even in decimal, then converts the
     * decimal string back to the nearest double. printf("%.*f") is correctly rounded
     * on the exact value (round-half-even) in Apple's libc, so this is the same op. */
    if (!isfinite(x)) return x;
    char buf[512];
    snprintf(buf, sizeof(buf), "%.*f", ndigits, x);
    return strtod(buf, NULL);
}
