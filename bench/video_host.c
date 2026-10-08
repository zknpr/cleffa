/* Time the production request encoder without model inference or patch-file I/O.
 * The wrappers belong only to this benchmark, keeping timers out of production loops.
 * make video-host-bench; ./video-host-bench MODEL.gguf REQUESTS.jsonl */
#include "../clef_record.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double stages[3];
static double clock_ms(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) { perror("clock_gettime"); exit(1); }
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}
static bool measured_base64(const char *s, size_t n, uint8_t **out, size_t *len, char *err, size_t cap) {
    double start = clock_ms();
    bool ok = clef_base64_decode(s, n, out, len, err, cap);
    stages[0] += clock_ms() - start;
    return ok;
}
static bool measured_decode(const uint8_t *s, size_t n, long limit, clef_rgb *rgb, char *err, size_t cap) {
    double start = clock_ms();
    bool ok = clef_image_decode_limited(s, n, limit, rgb, err, cap);
    stages[1] += clock_ms() - start;
    return ok;
}
static bool measured_pair(const clef_rgb *a, const clef_rgb *b, int h, int w,
                          clef_image_patches *out, char *err, size_t cap) {
    double start = clock_ms();
    bool ok = clef_video_pair_preprocess(a, b, h, w, out, err, cap);
    stages[2] += clock_ms() - start;
    return ok;
}
#define clef_base64_decode measured_base64
#define clef_image_decode_limited measured_decode
#define clef_video_pair_preprocess measured_pair
#include "../clef_record.c"

static void check(bool ok, const char *err) {
    if (!ok) { fprintf(stderr, "video host benchmark: %s\n", err); exit(1); }
}
static int order(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
int main(int argc, char **argv) {
    check(argc == 3, "expected MODEL.gguf REQUESTS.jsonl");
    char err[256], *line = NULL;
    gguf_file f;
    check(gguf_open(&f, argv[1], err, sizeof(err)), err);
    clef_tokenizer *tok = clef_tok_load(&f, err, sizeof(err));
    check(tok != NULL, err);
    clef_encode_opts opts = CLEF_ENCODE_DEFAULTS;
    opts.reject_truncation = true;
    check(clef_vision_opts_load(&f, &opts.vision, err, sizeof(err)), err);
    FILE *input = fopen(argv[2], "r");
    check(input != NULL, "cannot open requests");
    size_t capacity = 0;
    ssize_t length;
    while ((length = getline(&line, &capacity, input)) > 0) {
        double samples[5][9];
        int tokens = 0, groups = 0;
        for (int rep = 0; rep < 11; rep++) {
            memset(stages, 0, sizeof(stages));
            double start = clock_ms();
            jarena *arena = jarena_new();
            check(arena != NULL, "arena allocation failed");
            jval *req = json_parse(arena, line, (size_t)length, err, sizeof(err));
            check(req != NULL, err);
            double parsed = clock_ms();
            clef_record record;
            check(clef_encode_request(tok, req, opts, &record, err, sizeof(err)), err);
            double end = clock_ms();
            tokens = (int)record.ids.len; groups = record.n_images;
            if (rep >= 2) {
                samples[0][rep - 2] = parsed - start;
                for (int i = 0; i < 3; i++) samples[i + 1][rep - 2] = stages[i];
                samples[4][rep - 2] = end - start;
            }
            if (rep == 10) {
                const jval *id = json_get(req, "id");
                check(id && id->type == J_STRING && id->len < 128, "short request id required");
                for (int i = 0; i < 5; i++) qsort(samples[i], 9, sizeof(double), order);
                printf("%.*s tokens=%d groups=%d parse=%.3f base64=%.3f decode=%.3f preprocess=%.3f total=%.3f ms\n",
                       (int)id->len, id->s, tokens, groups, samples[0][4], samples[1][4], samples[2][4], samples[3][4], samples[4][4]);
                fflush(stdout);
            }
            clef_record_free(&record);
            jarena_free(arena);
        }
    }
    check(!ferror(input), "request read failed");
    free(line); check(fclose(input) == 0, "request close failed");
    clef_tok_free(tok); gguf_close(&f);
    return 0;
}
