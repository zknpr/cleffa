/* Parallel head work must retain the unsplit BLAS results, padded strides and
 * allocation failure behavior. First-use configuration is shared across engines. */
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <pthread.h>
static bool fail_score_allocation;
static void *test_malloc(size_t n) { return fail_score_allocation ? NULL : malloc(n); }
#define malloc test_malloc
#include "../clef_head.c"
#undef malloc

static pthread_mutex_t start_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t start_cv = PTHREAD_COND_INITIALIZER;
static int ready;
static bool released;
static void *first_use(void *result) {
    assert(pthread_mutex_lock(&start_mu) == 0);
    ready++;
    assert(pthread_cond_broadcast(&start_cv) == 0);
    while (!released) assert(pthread_cond_wait(&start_cv, &start_mu) == 0);
    assert(pthread_mutex_unlock(&start_mu) == 0);
    size_t value = lin_split_min();
    for (int i = 0; i < 100; i++) assert(lin_split_min() == value);
    *(size_t *)result = value;
    return NULL;
}
static void check_first_use(void) {
    enum { N = 8 };
    pthread_t threads[N];
    size_t result[N] = {0};
    assert(setenv("CLEF_DEBUG_HEAD_SPLIT_MIN", "1048576", 1) == 0);
    for (int i = 0; i < N; i++) assert(pthread_create(&threads[i], NULL, first_use, &result[i]) == 0);
    assert(pthread_mutex_lock(&start_mu) == 0);
    while (ready != N) assert(pthread_cond_wait(&start_cv, &start_mu) == 0);
    released = true;
    assert(pthread_cond_broadcast(&start_cv) == 0);
    assert(pthread_mutex_unlock(&start_mu) == 0);
    for (int i = 0; i < N; i++) {
        assert(pthread_join(threads[i], NULL) == 0);
        assert(result[i] == LIN_SPLIT_MIN);
    }
    puts("head parallel: concurrent first-use configuration PASS");
}

/* Previous production implementation, including its single scratch matrix. */
static void unsplit_attend(const float *Qp, int ldq, int nq, const float *K, const float *V,
                           int ld, int L, const float *bv, int E, int heads, float *out) {
    const int hd = E / heads;
    float *S = malloc((size_t)nq * L * sizeof(float)); assert(S);
    const float scale = 1.0f / sqrtf((float)hd);
    for (int h = 0; h < heads; h++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nq, L, hd, scale,
                    Qp + h * hd, ldq, K + h * hd, ld, 0.0f, S, L);
        for (int r = 0; r < nq; r++) softmax_inplace(S + (size_t)r * L, L);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nq, hd, L, 1.0f,
                    S, L, V + h * hd, ld, 0.0f, out + h * hd, E);
    }
    if (bv) for (int r = 0; r < nq; r++) for (int j = 0; j < E; j++) out[(size_t)r * E + j] += bv[j];
    free(S);
}
static float value(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16;
    return (float)(x & 65535) / 32768 - 1;
}
static void check_attend(int nq, int L, int E, int heads, bool bias) {
    const int ldq = 3 * E + 8, ld = 2 * E + 16, hd = E / heads;
    const size_t ny = (size_t)nq * E;
    float *q = malloc((size_t)nq * ldq * sizeof(float));
    float *kv = malloc((size_t)L * ld * sizeof(float));
    float *bv = malloc((size_t)E * sizeof(float));
    float *want = malloc(ny * sizeof(float)), *got = malloc((ny + 32) * sizeof(float));
    double *scores = malloc((size_t)L * sizeof(double));
    assert(q && kv && bv && want && got && scores);
    for (size_t i = 0; i < (size_t)nq * ldq; i++) q[i] = NAN;
    for (size_t i = 0; i < (size_t)L * ld; i++) kv[i] = NAN;
    for (size_t i = 0; i < ny + 32; i++) got[i] = NAN;
    for (int r = 0; r < nq; r++) for (int j = 0; j < E; j++) q[(size_t)r * ldq + j] = value(r * E + j + 41);
    for (int r = 0; r < L; r++) for (int j = 0; j < 2 * E; j++) kv[(size_t)r * ld + j] = value(r * 2 * E + j + 73);
    for (int j = 0; j < E; j++) bv[j] = value(j + 91) * 0.125f;
    unsplit_attend(q, ldq, nq, kv, kv + E, ld, L, bias ? bv : NULL, E, heads, want);
    assert(attend(q, ldq, nq, kv, kv + E, ld, L, bias ? bv : NULL, E, heads, got));
    assert(memcmp(want, got, ny * sizeof(float)) == 0);
    for (size_t i = 0; i < ny; i++) assert(isfinite(got[i]));
    for (size_t i = ny; i < ny + 32; i++) assert(isnan(got[i]));
    for (int sample = 0; sample < 6; sample++) {
        const int h = sample % heads, r = sample % nq, j = sample * 7 % hd;
        double peak = -INFINITY, sum = 0, out = 0;
        for (int t = 0; t < L; t++) {
            double dot = 0;
            for (int k = 0; k < hd; k++) dot += (double)q[(size_t)r * ldq + h * hd + k] * kv[(size_t)t * ld + h * hd + k];
            scores[t] = dot / sqrt((double)hd);
            peak = fmax(peak, scores[t]);
        }
        for (int t = 0; t < L; t++) { scores[t] = exp(scores[t] - peak); sum += scores[t]; }
        for (int t = 0; t < L; t++) out += scores[t] / sum * kv[(size_t)t * ld + E + h * hd + j];
        if (bias) out += bv[h * hd + j];
        assert(fabs(got[(size_t)r * E + h * hd + j] - out) < 2e-6);
    }
    for (size_t i = 0; i < ny + 32; i++) got[i] = NAN;
    fail_score_allocation = true;
    assert(!attend(q, ldq, nq, kv, kv + E, ld, L, bias ? bv : NULL, E, heads, got));
    fail_score_allocation = false;
    for (size_t i = 0; i < ny + 32; i++) assert(isnan(got[i]));
    assert(attend(q, ldq, nq, kv, kv + E, ld, L, bias ? bv : NULL, E, heads, got));
    assert(memcmp(want, got, ny * sizeof(float)) == 0);
    free(q); free(kv); free(bv); free(want); free(got); free(scores);
    printf("head parallel: nq=%d L=%d E=%d heads=%d bias=%d exact/f64/guards/allocation recovery PASS\n", nq, L, E, heads, bias);
}
int main(int argc, char **argv) {
    check_first_use();
    if (argc == 2 && !strcmp(argv[1], "--init-only")) return 0;
    const int shapes[][4] = {{3,2730,64,8}, {3,2731,64,8}, {9,3000,96,3},
                            {3,4096,1024,8}, {9,8192,1024,8}};
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++)
        for (int bias = 0; bias < 2; bias++) check_attend(shapes[i][0], shapes[i][1], shapes[i][2], shapes[i][3], bias);
    return 0;
}
