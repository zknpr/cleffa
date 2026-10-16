#ifndef CLEF_RECORD_OMNI_H
#define CLEF_RECORD_OMNI_H

#include "clef_record.h"

bool clef_omni_append_visual_tokens(const clef_tokenizer *tok, clef_record *record,
                                     const clef_encode_opts *opts, clef_tokens *prefix,
                                     char *err, size_t errlen);
bool clef_omni_encode_audio(const jval *audio, int tracks, const clef_encode_opts *opts, size_t *reserve,
                            clef_record *record, char *err, size_t errlen);
bool clef_omni_encode_record_audio(const jval *request, const clef_encode_opts *opts, size_t *reserve,
                                   clef_record *record, char *err, size_t errlen);
bool clef_omni_positions(clef_record *record, char *err, size_t errlen);

#endif
