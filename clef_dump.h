#ifndef CLEF_DUMP_H
#define CLEF_DUMP_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>

/* Diagnostic layers can be written as they finish. Keeping a full 64k-token
 * trace in anonymous memory otherwise retains another 24.4 GiB until inference
 * ends. Offsets allow a BF16 overflow retry to replace the first pass. */
typedef struct { float *memory; FILE *file; size_t elements; } clef_dump;

static inline bool clef_dump_write(clef_dump *dump, size_t offset, const float *values,
                                   size_t count, char *err, size_t errlen) {
    if (!dump) return true;
    if (offset > dump->elements || count > dump->elements - offset ||
        dump->elements > INT64_MAX / sizeof(float) ||
        ((dump->memory != NULL) == (dump->file != NULL))) {
        snprintf(err, errlen, "invalid residual dump bounds or destination");
        return false;
    }
    if (dump->memory) {
        memcpy(dump->memory + offset, values, count * sizeof(float));
        return true;
    }
    if (fseeko(dump->file, (off_t)(offset * sizeof(float)), SEEK_SET) ||
        fwrite(values, sizeof(float), count, dump->file) != count || fflush(dump->file)) {
        snprintf(err, errlen, "cannot write residual dump (incomplete dump)");
        return false;
    }
    return true;
}

#endif
