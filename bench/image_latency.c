/* CPU-only image stages, before record encoding or GPU inference.
 * make image-bench; ./image-bench golden/clef-flash-vision-f32/requests.jsonl
 * First image per request; four warmups then thirty measured samples. */
#include "../clef_image.h"
#include "../clef_json.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void require(bool ok, const char *why) {
    if (!ok) { fprintf(stderr, "image benchmark: %s\n", why); exit(1); }
}

static double now(void) {
    struct timespec t;
    require(clock_gettime(CLOCK_MONOTONIC, &t) == 0, "clock_gettime failed");
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static long pixels(const jval *v, long fallback) {
    if (!v) return fallback;
    char s[32], *end;
    require(v->type == J_INT && v->len < sizeof(s), "invalid pixel bound");
    memcpy(s, v->s, v->len); s[v->len] = 0;
    errno = 0;
    long n = strtol(s, &end, 10);
    require(!errno && !*end && n > 0, "invalid pixel bound");
    return n;
}

static int compare(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    require(argc == 2, "requests.jsonl required");
    FILE *f = fopen(argv[1], "r");
    require(f != NULL, "cannot open requests");
    char *line = NULL, err[256];
    size_t cap = 0;
    ssize_t length;
    while ((length = getline(&line, &cap, f)) > 0) {
        jarena *a = jarena_new();
        require(a != NULL, "arena allocation failed");
        jval *req = json_parse(a, line, (size_t)length, err, sizeof(err));
        require(req != NULL, err);
        const jval *images = json_get(req, "images"), *id = json_get(req, "id");
        if (!images || images->type != J_ARRAY || !images->n) { jarena_free(a); continue; }
        const jval *image = images->items[0];
        if (image->type == J_OBJECT) image = json_get(image, "base64");
        require(image && image->type == J_STRING, "invalid image");
        const jval *kw = json_get(req, "media_kwargs");
        clef_image_params prm = {pixels(json_get(kw, "min_pixels"), 65536),
                                pixels(json_get(kw, "max_pixels"), 16777216), 16, 2, 2};
        double samples[4][30];
        for (int rep = 0; rep < 34; rep++) {
            uint8_t *bytes = NULL;
            size_t n = 0;
            clef_rgb rgb = {0};
            clef_image_patches patches = {0};
            double start = now();
            require(clef_base64_decode(image->s, image->len, &bytes, &n, err, sizeof(err)), err);
            double decoded64 = now();
            require(clef_image_decode(bytes, n, &rgb, err, sizeof(err)), err);
            double decoded = now();
            require(clef_image_preprocess(&rgb, &prm, &patches, err, sizeof(err)), err);
            double end = now();
            if (rep >= 4) {
                samples[0][rep - 4] = decoded64 - start;
                samples[1][rep - 4] = decoded - decoded64;
                samples[2][rep - 4] = end - decoded;
                samples[3][rep - 4] = end - start;
            }
            clef_image_patches_free(&patches); clef_rgb_free(&rgb); free(bytes);
        }
        double medians[4];
        for (int i = 0; i < 4; i++) {
            qsort(samples[i], 30, sizeof(double), compare);
            medians[i] = (samples[i][14] + samples[i][15]) * 0.5;
        }
        printf("%.*s base64 %.4f decode %.4f preprocess %.4f total %.4f ms (medians)\n",
               id && id->type == J_STRING && id->len <= INT_MAX ? (int)id->len : 1,
               id && id->type == J_STRING && id->len <= INT_MAX ? id->s : "?",
               medians[0], medians[1], medians[2], medians[3]);
        fflush(stdout);
        jarena_free(a);
    }
    require(!ferror(f), "request read failed");
    free(line);
    require(fclose(f) == 0, "request close failed");
    return 0;
}
