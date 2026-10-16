/* Independent frontend arithmetic probe. This is not a runtime implementation:
 * FP64 direct DFT isolates rounding from FFT algorithm error before accepting
 * any host feature contract. Window/filter files come from the pinned oracle. */
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool read_exact(const char *path, float *data, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    bool ok = fread(data, 4, n, f) == n && fgetc(f) == EOF && !ferror(f);
    if (fclose(f)) ok = false;
    return ok;
}

static bool save(const char *folder, const char *name, const float *data, size_t n) {
    char path[4096];
    int length = snprintf(path, sizeof(path), "%s/%s.bin", folder, name);
    if (length < 0 || (size_t)length >= sizeof(path)) return false;
    FILE *f = fopen(path, "wb");
    if (!f) return false;
    bool ok = fwrite(data, 4, n, f) == n;
    if (fclose(f)) ok = false;
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 6) { fprintf(stderr, "usage: audio-features SAMPLES WINDOW FILTERS OUTPUT PADDED_SAMPLES\n"); return 2; }
    char *end;
    long count = strtol(argv[5], &end, 10);
    if (!*argv[5] || *end || count <= 200 || count > 4800000) return 2;
    const int frames = (int)count/160;
    float *samples = malloc((size_t)count*4), *real = calloc((size_t)201*frames, 4), *imag = calloc((size_t)201*frames, 4);
    float *power = calloc((size_t)201*frames, 4), *mel = calloc((size_t)128*frames, 4), *features = calloc((size_t)128*frames, 4);
    float window[400], native_window[400], filters[201*128];
    double *co = malloc(201*400*sizeof(double)), *si = malloc(201*400*sizeof(double));
    bool ok = false;
    if (!samples || !real || !imag || !power || !mel || !features || !co || !si) goto done;
    if (!read_exact(argv[1], samples, count) || !read_exact(argv[2], window, 400) || !read_exact(argv[3], filters, 201*128)) goto done;
    for (int j = 0; j < 400; j++) {
        float angle = (float)j*(float)(2*M_PI/400);
        native_window[j] = .5f*(1.f-cosf(angle));
    }
    for (int k = 0; k < 201; k++) for (int j = 0; j < 400; j++) {
        double angle = -2*M_PI*k*j/400;
        co[k*400+j] = cos(angle); si[k*400+j] = sin(angle);
    }
    for (int t = 0; t < frames; t++) {
        float block[400];
        for (int j = 0; j < 400; j++) {
            int at = 160*t+j-200;
            if (at < 0) at = -at;
            if (at >= count) at = 2*(int)count-2-at;
            block[j] = samples[at]*window[j];
        }
        for (int k = 0; k < 201; k++) {
            double re = 0, im = 0;
            for (int j = 0; j < 400; j++) { re += block[j]*co[k*400+j]; im += block[j]*si[k*400+j]; }
            real[k*frames+t] = (float)re; imag[k*frames+t] = (float)im;
            float magnitude = hypotf((float)re, (float)im);
            power[k*frames+t] = magnitude*magnitude;
        }
    }
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 128, frames, 201,
                1, filters, 128, power, frames, 0, mel, frames);
    float peak = -INFINITY;
    for (int i = 0; i < 128*frames; i++) {
        features[i] = log10f(fmaxf(mel[i], 1e-10f));
        peak = fmaxf(peak, features[i]);
    }
    for (int i = 0; i < 128*frames; i++) features[i] = (fmaxf(features[i], peak-8.f)+4.f)/4.f;
    ok = save(argv[4], "window", native_window, 400) && save(argv[4], "real", real, (size_t)201*frames) &&
         save(argv[4], "imag", imag, (size_t)201*frames) && save(argv[4], "power", power, (size_t)201*frames) &&
         save(argv[4], "mel", mel, (size_t)128*frames) && save(argv[4], "features", features, (size_t)128*frames);
done:
    free(samples); free(real); free(imag); free(power); free(mel); free(features); free(co); free(si);
    if (!ok) fprintf(stderr, "audio feature probe failed\n");
    return ok ? 0 : 1;
}
