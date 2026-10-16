#ifndef CLEF_AUDIO_H
#define CLEF_AUDIO_H

#include "clef_json.h"
#include "clef_gguf.h"

typedef struct {
    int layers, H, ffn, heads, hd, mel_bins, conv_channels, output;
    int chunk_frames, window_frames, conv_chunks, max_positions;
    float eps;
} clef_audio_config;
bool clef_audio_config_load(const gguf_file *f, clef_audio_config *out, char *err, size_t errlen);

typedef struct { float *data; size_t count; } clef_audio_samples;
typedef struct { float *mel; size_t frames, padded_frames; int n_tokens; } clef_audio_features;
typedef struct { int clip, frame, frames, first_token; } clef_audio_chunk;
typedef struct {
    clef_audio_chunk *chunks;
    int n_chunks, max_chunk_frames, tokens;
    int32_t *token_position, *window_start, *window_end;
} clef_audio_layout;

/* Owned, finite mono samples at 16000 Hz. WAV accepts little-endian PCM16 or
 * IEEE float32 only. A zero sample cap removes that soft limit. Output must be
 * empty; validation and allocation failure leave it empty. No paths or codecs. */
bool clef_audio_decode(const void *bytes, size_t n, size_t max_samples,
                       clef_audio_samples *out, char *err, size_t errlen);
bool clef_audio_from_json(const jval *value, size_t max_samples,
                          clef_audio_samples *out, char *err, size_t errlen);
void clef_audio_samples_free(clef_audio_samples *samples);

/* clips are one record: right-pad to its longest clip before centered STFT.
 * Output is [128][padded_frames]. Every output must be empty; failure leaves
 * all outputs empty. The caller enforces item/sample/context limits first. */
bool clef_audio_mel_batch(const clef_audio_samples *clips, int n, clef_audio_features *out,
                          char *err, size_t errlen);
void clef_audio_features_free(clef_audio_features *features);
bool clef_audio_layout_make(const clef_audio_features *clips, int n, clef_audio_layout *out,
                            char *err, size_t errlen);
void clef_audio_layout_free(clef_audio_layout *layout);

#endif
