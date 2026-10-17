#include "clef_audio.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(float)==4 && FLT_RADIX==2,"audio requires IEEE float32");

static bool audio_error(char *err,size_t n,const char *message) {
    if (n) snprintf(err,n,"audio: %s",message);
    return false;
}
static uint16_t read16(const uint8_t *p) { return (uint16_t)p[0]|((uint16_t)p[1]<<8); }
static uint32_t read32(const uint8_t *p) {
    return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);
}
static bool sample_count(size_t count,size_t limit) {
    return count && count<=SIZE_MAX/sizeof(float) && (!limit || count<=limit);
}

void clef_audio_samples_free(clef_audio_samples *samples) {
    if (samples) { free(samples->data); *samples=(clef_audio_samples){0}; }
}

bool clef_audio_decode(const void *bytes,size_t n,size_t max_samples,
                       clef_audio_samples *out,char *err,size_t errlen) {
    if (!out || out->data || out->count) return audio_error(err,errlen,"output must be empty");
    const uint8_t *p=bytes;
    if (!p || n<12 || memcmp(p,"RIFF",4) || memcmp(p+8,"WAVE",4) || (uint64_t)read32(p+4)+8!=n)
        return audio_error(err,errlen,"invalid or truncated RIFF/WAVE length");
    const uint8_t *format=NULL,*data=NULL;
    size_t format_size=0,data_size=0;
    // Validate the entire chunk table before allocating, even when data appears
    // before fmt. Declared lengths and odd-byte padding must fit the RIFF body.
    for (size_t at=12;at<n;) {
        if (n-at<8) return audio_error(err,errlen,"truncated WAV chunk header");
        const uint8_t *chunk=p+at;
        size_t size=read32(chunk+4);at+=8;
        if (size>n-at || (size&1)>n-at-size) return audio_error(err,errlen,"truncated WAV chunk or padding");
        if (!memcmp(chunk,"fmt ",4)) {
            if (format) return audio_error(err,errlen,"duplicate WAV format chunk");
            format=p+at;format_size=size;
        } else if (!memcmp(chunk,"data",4)) {
            if (data) return audio_error(err,errlen,"duplicate WAV data chunk");
            data=p+at;data_size=size;
        }
        at+=size+(size&1);
    }
    if (!format || !data || (format_size!=16 && (format_size!=18 || read16(format+16)!=0)))
        return audio_error(err,errlen,"WAV requires one plain PCM16/float32 fmt and data chunk");
    const unsigned type=read16(format),width=type==1?2:type==3?4:0;
    if (!width || read16(format+2)!=1 || read32(format+4)!=16000 || read32(format+8)!=16000*width ||
        read16(format+12)!=width || read16(format+14)!=8*width || data_size%width)
        return audio_error(err,errlen,"WAV must be PCM16 or float32, mono, 16000 Hz with consistent sample alignment");
    const size_t count=data_size/width;
    if (!sample_count(count,max_samples)) return audio_error(err,errlen,"empty clip or sample limit exceeded");
    if (type==3) for (size_t i=0;i<count;i++)
        if ((read32(data+4*i)&0x7f800000u)==0x7f800000u) return audio_error(err,errlen,"nonfinite WAV sample");
    float *samples=malloc(count*sizeof(*samples));
    if (!samples) return audio_error(err,errlen,"out of memory");
    for (size_t i=0;i<count;i++) {
        if (type==1) {
            const unsigned u=read16(data+2*i);
            const int s=u>=32768?(int)u-65536:(int)u;
            samples[i]=(float)s/32768.f;
        } else {
            const uint32_t bits=read32(data+4*i);
            memcpy(samples+i,&bits,4); // preserve subnormals and signed zero exactly
        }
    }
    *out=(clef_audio_samples){samples,count};
    return true;
}

static bool json_sample(const jval *v,float *out) {
    double value;
    if (v->type==J_FLOAT) value=v->f;
    else if (v->type==J_INT) {
        // Every finite float32 integer fits in 39 decimal digits plus its sign.
        char text[48];
        if (!v->len || v->len>=sizeof(text)) return false;
        memcpy(text,v->s,v->len);text[v->len]=0;
        char *end;value=strtod(text,&end);
        if (end!=text+v->len) return false;
    } else return false;
    if (!isfinite(value) || fabs(value)>FLT_MAX) return false;
    *out=(float)value;
    return true;
}

bool clef_audio_from_json(const jval *value,size_t max_samples,
                          clef_audio_samples *out,char *err,size_t errlen) {
    if (!out || out->data || out->count) return audio_error(err,errlen,"output must be empty");
    if (!value || value->type!=J_ARRAY || !sample_count(value->n,max_samples))
        return audio_error(err,errlen,"expected a nonempty numeric sample array within the sample limit");
    float sample;
    for (size_t i=0;i<value->n;i++) if (!value->items[i] || !json_sample(value->items[i],&sample))
        return audio_error(err,errlen,"samples must be finite numbers representable as float32");
    float *samples=malloc(value->n*sizeof(*samples));
    if (!samples) return audio_error(err,errlen,"out of memory");
    for (size_t i=0;i<value->n;i++) {
        if (!json_sample(value->items[i],samples+i)) {
            free(samples);return audio_error(err,errlen,"sample changed during decoding");
        }
    }
    *out=(clef_audio_samples){samples,value->n};
    return true;
}

#include "clef_audio_tables.inc"

typedef struct { double re, im; } audio_complex;

/* 400 = 2^4 * 5^2. A small mixed-radix FFT avoids a codec/DSP dependency and
 * keeps scratch fixed. FP64 accumulation limits FFT error; storage and all
 * subsequent frontend stages use FP32, as in the pinned processor. */
static void audio_fft(audio_complex *out, const audio_complex *input, int stride, int n,
                       const audio_complex roots[400]) {
    if (n == 1) { *out = *input; return; }
    const int radix = n%2 == 0 ? 2 : 5, width = n/radix, root_stride = 400/n;
    for (int r = 0; r < radix; r++) audio_fft(out+r*width, input+r*stride, stride*radix, width, roots);
    audio_complex merged[400];
    for (int k = 0; k < n; k++) {
        audio_complex sum = {0};
        for (int r = 0; r < radix; r++) {
            const audio_complex v = out[r*width+k%width], w = roots[(r*k*root_stride)%400];
            sum.re += v.re*w.re-v.im*w.im;
            sum.im += v.re*w.im+v.im*w.re;
        }
        merged[k] = sum;
    }
    memcpy(out, merged, (size_t)n*sizeof(*out));
}

/* Test-only stage taps compile out of the host library. They expose the same
 * execution used by inference without adding runtime debug state or file I/O. */
#ifndef CLEF_AUDIO_STAGE
#define CLEF_AUDIO_STAGE(clip, name, frame, data, count) ((void)0)
#endif

void clef_audio_features_free(clef_audio_features *features) {
    if (features) { free(features->mel); *features = (clef_audio_features){0}; }
}

/* Lowest waveform index frame t reads: a centered 400-sample window, reflected at both ends of
 * the record's padded waveform (longest > 200 samples, so one reflection suffices). A clip reads
 * zeros at and beyond its own count, so a frame whose lowest index reaches the count reads only
 * zeros. The index first rises with t, then falls once the window reflects off the end, so those
 * frames form one run. */
static size_t frame_low(size_t t, size_t longest) {
    const int64_t a = (int64_t)t*160-200, b = a+399;
    if (a <= 0) return 0;
    if ((uint64_t)b < longest) return (size_t)a;
    const int64_t reflected = 2*(int64_t)longest-2-b;
    return (size_t)(reflected < a ? reflected : a);
}

bool clef_audio_mel_batch(const clef_audio_samples *clips, int n, clef_audio_features *out,
                          char *err, size_t errlen) {
    if (!clips || !out || n <= 0 || (size_t)n > SIZE_MAX/sizeof(*out))
        return audio_error(err, errlen, "invalid feature batch");
    size_t longest = 0;
    for (int i = 0; i < n; i++) {
        if (out[i].mel || out[i].frames || out[i].padded_frames || out[i].n_tokens)
            return audio_error(err, errlen, "feature outputs must be empty");
        if (!clips[i].data || !clips[i].count || clips[i].count > INT_MAX)
            return audio_error(err, errlen, "invalid clip length");
        if (clips[i].count > longest) longest = clips[i].count;
        for (size_t j = 0; j < clips[i].count; j++) if (!isfinite(clips[i].data[j]))
            return audio_error(err, errlen, "nonfinite input sample");
    }
    // torch reflection padding by 200 requires the padded waveform to be longer
    // than 200. A short clip is valid beside a longer clip in the same record.
    if (longest <= 200) return audio_error(err, errlen, "padded clip must contain more than 200 samples");
    const size_t frames = longest/160;
    if (frames > SIZE_MAX/(128*sizeof(float))) return audio_error(err, errlen, "feature size overflow");
    clef_audio_features *next = calloc((size_t)n, sizeof(*next));
    if (!next) return audio_error(err, errlen, "out of memory");
    const char *failure = "out of memory";
    audio_complex roots[400];
    const double pi = 3.14159265358979323846264338327950288;
    for (int k = 0; k < 400; k++) {
        double angle = -2*pi*k/400;
        roots[k] = (audio_complex){cos(angle), sin(angle)};
    }
    for (int i = 0; i < n; i++) {
        clef_audio_features *f = next+i;
        f->padded_frames = frames;
        f->frames = (clips[i].count+159)/160;
        if (f->frames > frames) f->frames = frames;
        f->n_tokens = (int)(13*(f->frames/100)+(f->frames%100+7)/8);
        /* The reference pads every clip to the longest and normalizes against the peak over
         * all its padded frames, but the tower reads only the clip's own frames. Store those,
         * and evaluate only frames whose window reaches the clip's samples: frames [first,
         * last) read zeros, giving log10(1e-10) in every bin, which enters the peak as a
         * constant. fmaxf is exact, so own frames are bitwise what full evaluation gives, and
         * cost follows the clip's length rather than the record's longest clip. */
        size_t first = 0, last = frames;
        while (first < frames && frame_low(first, longest) < clips[i].count) first++;
        while (last > first && frame_low(last-1, longest) < clips[i].count) last--;
        if (first < f->frames) { failure = "audio frame range"; goto fail; }  // own frames read samples
        f->mel = malloc(128*f->frames*sizeof(float));
        if (!f->mel) goto fail;
        float peak = first < last ? log10f(1e-10f) : -INFINITY;
        for (size_t t = 0; t < frames; t++) {
            if (t == first) t = last;
            if (t >= frames) break;
            audio_complex block[400], spectrum[400];
            for (int j = 0; j < 400; j++) {
                int64_t at = (int64_t)t*160+j-200;
                if (at < 0) at = -at;
                if ((uint64_t)at >= longest) at = 2*(int64_t)longest-2-at;
                const float sample = (uint64_t)at < clips[i].count ? clips[i].data[at] : 0;
                const float windowed = sample*audio_hann[j];
                block[j] = (audio_complex){windowed, 0};
            }
            audio_fft(spectrum, block, 1, 400, roots);
            float real[201], imag[201], power[201], mel[128];
            for (int k = 0; k < 201; k++) {
                real[k] = (float)spectrum[k].re; imag[k] = (float)spectrum[k].im;
                const float magnitude = hypotf(real[k], imag[k]);
                power[k] = magnitude*magnitude;
                if (!isfinite(power[k])) { failure = "audio power overflow"; goto fail; }
            }
            for (int m = 0; m < 128; m++) {
                float sum = 0;
                for (int j = audio_mel_offsets[m]; j < audio_mel_offsets[m+1]; j++)
                    sum += power[audio_mel_filters[j].bin]*audio_mel_filters[j].weight;
                mel[m] = sum;
                const float value = log10f(fmaxf(sum, 1e-10f));
                if (!isfinite(value)) { failure = "audio mel overflow"; goto fail; }
                if (t < f->frames) f->mel[(size_t)m*f->frames+t] = value;
                peak = fmaxf(peak, value);
            }
            CLEF_AUDIO_STAGE(i, "real", t, real, 201);
            CLEF_AUDIO_STAGE(i, "imag", t, imag, 201);
            CLEF_AUDIO_STAGE(i, "power", t, power, 201);
            CLEF_AUDIO_STAGE(i, "mel", t, mel, 128);
        }
        for (size_t j = 0; j < 128*f->frames; j++) f->mel[j] = (fmaxf(f->mel[j], peak-8.f)+4.f)/4.f;
    }
    memcpy(out, next, (size_t)n*sizeof(*out));
    free(next);
    return true;
fail:
    for (int i = 0; i < n; i++) clef_audio_features_free(next+i);
    free(next);
    return audio_error(err, errlen, failure);
}

void clef_audio_layout_free(clef_audio_layout *layout) {
    if (layout) {
        free(layout->chunks);free(layout->token_position);
        *layout = (clef_audio_layout){0};
    }
}

bool clef_audio_layout_make(const clef_audio_features *clips, int n, clef_audio_layout *out,
                            char *err, size_t errlen) {
    if (!clips || n <= 0 || !out || out->chunks || out->token_position || out->window_start || out->window_end ||
        out->n_chunks || out->tokens || out->max_chunk_frames)
        return audio_error(err,errlen,"invalid or nonempty audio layout");
    uint64_t chunks = 0,tokens = 0;
    int max_chunk = 0;
    for (int i = 0; i < n; i++) {
        const size_t frames = clips[i].frames;
        if (!clips[i].mel || !frames || frames > INT_MAX || clips[i].padded_frames > INT_MAX || frames > clips[i].padded_frames ||
            clips[i].padded_frames != clips[0].padded_frames ||
            (uint64_t)clips[i].n_tokens != 13*(frames/100)+(frames%100+7)/8)
            return audio_error(err,errlen,"inconsistent audio feature layout");
        chunks += (frames+99)/100;tokens += (uint64_t)clips[i].n_tokens;
        int width = frames < 100 ? (int)frames : 100;
        if (width > max_chunk) max_chunk = width;
        if (chunks > INT_MAX || tokens > INT_MAX) return audio_error(err,errlen,"audio layout count overflow");
    }
    if (chunks > SIZE_MAX/sizeof(clef_audio_chunk) || tokens > SIZE_MAX/(3*sizeof(int32_t)))
        return audio_error(err,errlen,"audio layout size overflow");
    clef_audio_layout next = {.n_chunks=(int)chunks,.max_chunk_frames=max_chunk,.tokens=(int)tokens};
    next.chunks = malloc((size_t)chunks*sizeof(*next.chunks));
    next.token_position = malloc((size_t)tokens*3*sizeof(int32_t));
    if (!next.chunks || !next.token_position) {
        clef_audio_layout_free(&next);return audio_error(err,errlen,"out of memory");
    }
    next.window_start = next.token_position+tokens;next.window_end = next.window_start+tokens;
    // CNN padding is shared across a record. Its longest chunk determines the
    // post-CNN attention window size even when every clip is shorter than 100.
    const int window = ((max_chunk+7)/8)*8;
    int chunk = 0,base = 0;
    for (int i = 0; i < n; i++) {
        int row = base;
        for (int start = 0; start < (int)clips[i].frames;) {
            int count = (int)clips[i].frames-start;
            if (count > 100) count = 100;
            next.chunks[chunk++] = (clef_audio_chunk){i,start,count,row};
            for (int p = 0; p < (count+7)/8; p++) next.token_position[row++] = p;
            start += count;
        }
        const int end = base+clips[i].n_tokens;
        for (int start = base; start < end;) {
            const int stop = end-start < window ? end : start+window;
            for (int t = start; t < stop; t++) { next.window_start[t]=start;next.window_end[t]=stop; }
            start = stop;
        }
        base = end;
    }
    *out = next;
    return true;
}
