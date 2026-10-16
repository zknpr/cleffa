#ifndef CLEF_MODEL_H
#define CLEF_MODEL_H

#include "clef_gguf.h"

typedef enum { CLEF_QWEN35 = 0, CLEF_OMNI = 1 } clef_backbone;
enum { CLEF_CAP_TEXT = 1, CLEF_CAP_VISION = 2, CLEF_CAP_AUDIO = 4 };
typedef struct {
    clef_backbone backbone;
    uint32_t capabilities;
    int context_default, context_length;   /* length 0: no new limit imposed on legacy files */
} clef_model_info;

bool clef_model_info_load(const gguf_file *f, clef_model_info *out, char *err, size_t errlen);

#endif
