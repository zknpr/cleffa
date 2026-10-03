#include "clef_gguf.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* The GGUF is an untrusted input (it can come from anywhere), so every read is
 * bounds-checked against the mapping and every count is checked before it is
 * used to size an allocation or a loop. */

typedef struct {
    const gguf_file *f;
    uint64_t pos;
    bool ok;
} cursor;

static bool need(cursor *c, uint64_t n) {
    if (!c->ok || n > c->f->size || c->pos > c->f->size - n) {
        c->ok = false;
        return false;
    }
    return true;
}

static uint32_t rd_u32(cursor *c) {
    uint32_t v = 0;
    if (need(c, 4)) { memcpy(&v, c->f->base + c->pos, 4); c->pos += 4; }
    return v;
}

static uint64_t rd_u64(cursor *c) {
    uint64_t v = 0;
    if (need(c, 8)) { memcpy(&v, c->f->base + c->pos, 8); c->pos += 8; }
    return v;
}

static gguf_str rd_str(cursor *c) {
    gguf_str s = {0};
    uint64_t n = rd_u64(c);
    if (need(c, n)) { s.ptr = (const char *)c->f->base + c->pos; s.len = n; c->pos += n; }
    return s;
}

static uint64_t scalar_size(uint32_t t) {
    switch (t) {
    case GGUF_T_U8: case GGUF_T_I8: case GGUF_T_BOOL: return 1;
    case GGUF_T_U16: case GGUF_T_I16: return 2;
    case GGUF_T_U32: case GGUF_T_I32: case GGUF_T_F32: return 4;
    case GGUF_T_U64: case GGUF_T_I64: case GGUF_T_F64: return 8;
    default: return 0;
    }
}

static void skip_value(cursor *c, uint32_t t, int depth) {
    if (depth > 2) { c->ok = false; return; }
    if (t == GGUF_T_STRING) { (void)rd_str(c); return; }
    if (t == GGUF_T_ARRAY) {
        uint32_t et = rd_u32(c);
        uint64_t n = rd_u64(c);
        uint64_t es = scalar_size(et);
        if (es) {
            if (n > c->f->size / es) { c->ok = false; return; }
            if (need(c, n * es)) c->pos += n * es;
            return;
        }
        for (uint64_t i = 0; i < n && c->ok; i++) skip_value(c, et, depth + 1);
        return;
    }
    uint64_t s = scalar_size(t);
    if (!s) { c->ok = false; return; }
    if (need(c, s)) c->pos += s;
}

static uint64_t type_nbytes(uint32_t type, uint64_t nelem, bool *ok) {
    uint64_t es;
    switch (type) {
    case GGML_F32: es = 4; break;
    case GGML_F16: case GGML_BF16: es = 2; break;
    default: *ok = false; return 0;
    }
    if (nelem > UINT64_MAX / es) { *ok = false; return 0; }
    return nelem * es;
}

static bool fail(char *err, size_t errlen, const char *msg) {
    snprintf(err, errlen, "%s", msg);
    return false;
}

bool gguf_open(gguf_file *f, const char *path, char *err, size_t errlen) {
    memset(f, 0, sizeof(*f));
    f->fd = open(path, O_RDONLY);
    if (f->fd < 0) { snprintf(err, errlen, "open %s failed", path); return false; }
    struct stat st;
    if (fstat(f->fd, &st) != 0 || st.st_size < 24) { gguf_close(f); return fail(err, errlen, "gguf: file too small"); }
    f->size = (uint64_t)st.st_size;
    void *m = mmap(NULL, (size_t)f->size, PROT_READ, MAP_SHARED, f->fd, 0);
    if (m == MAP_FAILED) { gguf_close(f); return fail(err, errlen, "gguf: mmap failed"); }
    f->base = m;

    cursor c = { f, 0, true };
    if (rd_u32(&c) != 0x46554747u) { gguf_close(f); return fail(err, errlen, "gguf: bad magic"); }
    if (rd_u32(&c) != 3) { gguf_close(f); return fail(err, errlen, "gguf: unsupported version"); }
    f->n_tensors = rd_u64(&c);
    f->n_kv = rd_u64(&c);
    /* Each KV and tensor record is at least 8 bytes on disk; reject counts the file cannot hold. */
    if (!c.ok || f->n_kv > f->size / 8 || f->n_tensors > f->size / 8) {
        gguf_close(f); return fail(err, errlen, "gguf: bad header counts");
    }
    f->kv = calloc(f->n_kv ? f->n_kv : 1, sizeof(*f->kv));
    f->tensors = calloc(f->n_tensors ? f->n_tensors : 1, sizeof(*f->tensors));
    if (!f->kv || !f->tensors) { gguf_close(f); return fail(err, errlen, "gguf: out of memory"); }

    uint32_t alignment = 32;
    for (uint64_t i = 0; i < f->n_kv && c.ok; i++) {
        gguf_kv *kv = &f->kv[i];
        kv->key = rd_str(&c);
        kv->type = rd_u32(&c);
        kv->value_pos = c.pos;
        if (kv->type == GGUF_T_U32 && kv->key.len == 17 && !memcmp(kv->key.ptr, "general.alignment", 17)) {
            cursor a = c;
            alignment = rd_u32(&a);
        }
        skip_value(&c, kv->type, 0);
    }
    if (!c.ok) { gguf_close(f); return fail(err, errlen, "gguf: truncated metadata"); }
    if (alignment == 0 || (alignment & (alignment - 1))) { gguf_close(f); return fail(err, errlen, "gguf: bad alignment"); }

    for (uint64_t i = 0; i < f->n_tensors && c.ok; i++) {
        gguf_tensor *t = &f->tensors[i];
        t->name = rd_str(&c);
        t->n_dims = rd_u32(&c);
        if (t->n_dims == 0 || t->n_dims > 4) { c.ok = false; break; }
        uint64_t nelem = 1;
        for (uint32_t d = 0; d < 4; d++) t->ne[d] = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) {
            t->ne[d] = rd_u64(&c);
            if (t->ne[d] == 0 || nelem > UINT64_MAX / t->ne[d]) { c.ok = false; break; }
            nelem *= t->ne[d];
        }
        t->type = rd_u32(&c);
        t->offset = rd_u64(&c);
        bool ok = true;
        t->nbytes = type_nbytes(t->type, nelem, &ok);
        if (!ok) { c.ok = false; break; }
    }
    if (!c.ok) { gguf_close(f); return fail(err, errlen, "gguf: bad tensor table"); }

    f->data_offset = (c.pos + alignment - 1) & ~(uint64_t)(alignment - 1);
    for (uint64_t i = 0; i < f->n_tensors; i++) {
        gguf_tensor *t = &f->tensors[i];
        if (t->offset > f->size || f->data_offset > f->size - t->offset ||
            t->nbytes > f->size - f->data_offset - t->offset) {
            gguf_close(f); return fail(err, errlen, "gguf: tensor data out of bounds");
        }
        t->offset += f->data_offset;
        t->data = f->base + t->offset;
    }
    return true;
}

void gguf_close(gguf_file *f) {
    if (f->base) munmap((void *)f->base, (size_t)f->size);
    if (f->fd > 0) close(f->fd);
    free(f->kv);
    free(f->tensors);
    memset(f, 0, sizeof(*f));
}

static bool str_eq(gguf_str s, const char *z) {
    size_t n = strlen(z);
    return s.len == n && !memcmp(s.ptr, z, n);
}

const gguf_kv *gguf_find_kv(const gguf_file *f, const char *key) {
    for (uint64_t i = 0; i < f->n_kv; i++) if (str_eq(f->kv[i].key, key)) return &f->kv[i];
    return NULL;
}

const gguf_tensor *gguf_find_tensor(const gguf_file *f, const char *name) {
    for (uint64_t i = 0; i < f->n_tensors; i++) if (str_eq(f->tensors[i].name, name)) return &f->tensors[i];
    return NULL;
}

bool gguf_get_u32(const gguf_file *f, const char *key, uint32_t *out) {
    const gguf_kv *kv = gguf_find_kv(f, key);
    if (!kv || kv->type != GGUF_T_U32) return false;
    cursor c = { f, kv->value_pos, true };
    *out = rd_u32(&c);
    return c.ok;
}

bool gguf_get_f32(const gguf_file *f, const char *key, float *out) {
    const gguf_kv *kv = gguf_find_kv(f, key);
    if (!kv || kv->type != GGUF_T_F32) return false;
    cursor c = { f, kv->value_pos, true };
    uint32_t bits = rd_u32(&c);
    memcpy(out, &bits, 4);
    return c.ok;
}

bool gguf_get_str(const gguf_file *f, const char *key, gguf_str *out) {
    const gguf_kv *kv = gguf_find_kv(f, key);
    if (!kv || kv->type != GGUF_T_STRING) return false;
    cursor c = { f, kv->value_pos, true };
    *out = rd_str(&c);
    return c.ok;
}

bool gguf_get_array(const gguf_file *f, const char *key, uint32_t *elem_type, uint64_t *count, uint64_t *pos) {
    const gguf_kv *kv = gguf_find_kv(f, key);
    if (!kv || kv->type != GGUF_T_ARRAY) return false;
    cursor c = { f, kv->value_pos, true };
    *elem_type = rd_u32(&c);
    *count = rd_u64(&c);
    *pos = c.pos;
    return c.ok;
}

bool gguf_read_strings(const gguf_file *f, uint64_t pos, uint64_t count, gguf_str *out) {
    cursor c = { f, pos, true };
    for (uint64_t i = 0; i < count && c.ok; i++) out[i] = rd_str(&c);
    return c.ok;
}

bool gguf_read_i32_array(const gguf_file *f, const char *key, int32_t *out, uint64_t max, uint64_t *count) {
    uint32_t et; uint64_t n, pos;
    if (!gguf_get_array(f, key, &et, &n, &pos)) return false;
    if ((et != GGUF_T_I32 && et != GGUF_T_U32) || n > max) return false;
    cursor c = { f, pos, true };
    for (uint64_t i = 0; i < n && c.ok; i++) out[i] = (int32_t)rd_u32(&c);
    *count = n;
    return c.ok;
}
