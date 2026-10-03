#ifndef CLEF_JSON_H
#define CLEF_JSON_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* JSON DOM with Python `json` semantics, because the reference model's prompt is
 * built from json.loads + json.dumps(sort_keys=True, ensure_ascii=False,
 * separators=(",", ":")) and the engine must reproduce those bytes exactly:
 *   - integers keep their exact digits (Python ints are arbitrary precision);
 *   - floats are parsed with correct rounding and printed as Python repr();
 *   - NaN / Infinity / -Infinity are accepted, as json.loads does;
 *   - duplicate object keys: last value wins, first position is kept (dict update).
 * Inputs are untrusted: depth and size are bounded and every failure is an error. */

typedef enum { J_NULL, J_FALSE, J_TRUE, J_INT, J_FLOAT, J_STRING, J_ARRAY, J_OBJECT } jtype;

typedef struct jval jval;
typedef struct { const char *key; size_t klen; jval *val; } jmember;

struct jval {
    jtype type;
    const char *s;     /* J_STRING: decoded UTF-8; J_INT: canonical decimal text */
    size_t len;
    double f;          /* J_FLOAT */
    jval **items;      /* J_ARRAY */
    jmember *members;  /* J_OBJECT, in first-insertion order */
    size_t n;
};

typedef struct jarena jarena;

jarena *jarena_new(void);
void jarena_free(jarena *a);

/* Parse `len` bytes. On failure returns NULL and writes a message into err. */
jval *json_parse(jarena *a, const char *text, size_t len, char *err, size_t errlen);

jval *json_get(const jval *obj, const char *key);
bool json_str_eq(const jval *v, const char *z);

typedef struct { char *p; size_t len, cap; bool oom; } jbuf;

void jbuf_put(jbuf *b, const char *s, size_t n);
void jbuf_puts(jbuf *b, const char *z);
void jbuf_free(jbuf *b);

/* json.dumps(v, ensure_ascii=False, separators=(",", ":"), sort_keys=sort_keys) */
void json_dump(jbuf *b, const jval *v, bool sort_keys);
/* Python repr(float) */
void json_put_float(jbuf *b, double x);
/* A Python str serialized by json.dumps (quoted, escaped). */
void json_put_string(jbuf *b, const char *s, size_t len);

/* Python round(x, ndigits) for a finite double. */
double py_round(double x, int ndigits);

#endif
