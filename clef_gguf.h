#ifndef CLEF_GGUF_H
#define CLEF_GGUF_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Read-only GGUF v3 view over an mmap'd file. Strings point into the mapping
 * (not NUL-terminated); every accessor validates bounds against the file size. */

enum {
    GGUF_T_U8 = 0, GGUF_T_I8 = 1, GGUF_T_U16 = 2, GGUF_T_I16 = 3,
    GGUF_T_U32 = 4, GGUF_T_I32 = 5, GGUF_T_F32 = 6, GGUF_T_BOOL = 7,
    GGUF_T_STRING = 8, GGUF_T_ARRAY = 9, GGUF_T_U64 = 10, GGUF_T_I64 = 11,
    GGUF_T_F64 = 12,
};

enum { GGML_F32 = 0, GGML_F16 = 1, GGML_BF16 = 30 };

typedef struct {
    const char *ptr;
    uint64_t len;
} gguf_str;

typedef struct {
    gguf_str key;
    uint32_t type;
    uint64_t value_pos;  /* file offset of the value payload */
} gguf_kv;

typedef struct {
    gguf_str name;
    uint32_t n_dims;
    uint64_t ne[4];      /* ne[0] is the fastest-varying (row length) */
    uint32_t type;
    uint64_t offset;     /* absolute file offset */
    uint64_t nbytes;
    const void *data;
} gguf_tensor;

typedef struct {
    int fd;
    const uint8_t *base;
    uint64_t size;
    uint64_t n_kv, n_tensors;
    gguf_kv *kv;
    gguf_tensor *tensors;
    uint64_t data_offset;
} gguf_file;

/* Returns false and fills err (NUL-terminated) on any malformed input. */
bool gguf_open(gguf_file *f, const char *path, char *err, size_t errlen);
void gguf_close(gguf_file *f);

const gguf_kv *gguf_find_kv(const gguf_file *f, const char *key);
const gguf_tensor *gguf_find_tensor(const gguf_file *f, const char *name);

bool gguf_get_u32(const gguf_file *f, const char *key, uint32_t *out);
bool gguf_get_f32(const gguf_file *f, const char *key, float *out);
bool gguf_get_str(const gguf_file *f, const char *key, gguf_str *out);

/* Arrays: element type and count; element access walks from the payload. */
bool gguf_get_array(const gguf_file *f, const char *key, uint32_t *elem_type, uint64_t *count, uint64_t *pos);
/* Read `count` consecutive strings starting at pos (array payload). */
bool gguf_read_strings(const gguf_file *f, uint64_t pos, uint64_t count, gguf_str *out);
bool gguf_read_i32_array(const gguf_file *f, const char *key, int32_t *out, uint64_t max, uint64_t *count);

#endif
