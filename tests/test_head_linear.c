/* Packed CPU head weights must preserve the original BLAS results, including the
 * single-row path whose reduction changes when its weight matrix is transposed. */
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
static bool fail_pack_allocation;
static void *test_calloc(size_t n, size_t size) {
    return fail_pack_allocation ? NULL : calloc(n, size);
}
#define calloc test_calloc
#include "../clef_head.c"
#undef calloc

static float value(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16;
    return (float)(x & 0xffff) / 32768.0f - 1.0f;
}

/* Original full-width BLAS call, independent of the candidate's split helper. */
static void unsplit_lin(const linear *l, const float *X, int n, float *Y) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, l->out, l->in,
                1.0f, X, l->in, l->w, l->in, 0.0f, Y, l->out);
    if (l->b) for (int r = 0; r < n; r++) for (int j = 0; j < l->out; j++) Y[(size_t)r * l->out + j] += l->b[j];
}

static void check(int in, int out, int rows) {
    linear l = { .out = out, .in = in };
    l.w = malloc((size_t)out * in * sizeof(float));
    l.b = malloc((size_t)out * sizeof(float));
    float *x = malloc((size_t)rows * in * sizeof(float));
    float *want = malloc((size_t)rows * out * sizeof(float));
    float *got = malloc((size_t)rows * out * sizeof(float));
    assert(l.w && l.b && x && want && got);
    for (size_t i = 0; i < (size_t)out * in; i++) {
        float f = value((uint32_t)i + 73);
        uint32_t bits; memcpy(&bits, &f, sizeof(bits));
        l.w[i] = bf16_to_f32((uint16_t)(bits >> 16));
    }
    for (int i = 0; i < out; i++) l.b[i] = value(i + 91);
    for (size_t i = 0; i < (size_t)rows * in; i++) x[i] = value((uint32_t)i) * (i % 37 ? 1 : 8);
    unsplit_lin(&l, x, rows, want);
    lin(&l, x, rows, got);
    assert(memcmp(want, got, (size_t)rows * out * sizeof(float)) == 0);
    char err[128];
    assert(pack_linear(&l, err, sizeof(err)));
    // Padding is outside the logical tensor and must never affect any output.
    if (l.packed) for (int k = 0; k < in; k++) for (int j = out; j < l.packed_ld; j++)
        l.packed[(size_t)k * l.packed_ld + j] = NAN;
    lin(&l, x, rows, got);
    if (memcmp(want, got, (size_t)rows * out * sizeof(float))) {
        double max_error = 0;
        int nonfinite = 0;
        for (int i = 0; i < rows * out; i++) {
            max_error = fmax(max_error, fabs(want[i] - got[i]));
            nonfinite += !isfinite(got[i]);
        }
        fprintf(stderr, "head layout mismatch: in=%d out=%d rows=%d max_error=%g nonfinite=%d\n",
                in, out, rows, max_error, nonfinite);
    }
    assert(memcmp(want, got, (size_t)rows * out * sizeof(float)) == 0);
    if (rows == 1 && l.packed) {
        for (int k = 0; k < in; k++) for (int j = 0; j < out; j++)
            l.packed[(size_t)k * l.packed_ld + j] = NAN;
        lin(&l, x, rows, got);
        assert(memcmp(want, got, (size_t)out * sizeof(float)) == 0);
    }
    free_linear(&l); free(x); free(want); free(got);
}

/* Float64 reference for the batched scorer's two affine projections and GELU. */
static void check_scorer(int rows) {
    enum { W = 64, IN = 4 * W };
    linear first = { .out = W, .in = IN }, last = { .out = 1, .in = W };
    first.w = malloc(W * IN * sizeof(float)); first.b = malloc(W * sizeof(float));
    last.w = malloc(W * sizeof(float)); last.b = malloc(sizeof(float));
    float *input = malloc((size_t)rows * IN * sizeof(float));
    float *hidden = malloc(((size_t)rows * W + 1) * sizeof(float));
    float *output = malloc(((size_t)rows + 1) * sizeof(float));
    assert(first.w && first.b && last.w && last.b && input && hidden && output);
    for (int j = 0; j < W; j++) {
        first.b[j] = value(j + 81) * 0.125f;
        last.w[j] = value(j + 912) * 0.125f;
        for (int k = 0; k < IN; k++) {
            float v = value(j * IN + k + 93) * 0.125f;
            uint32_t bits; memcpy(&bits, &v, sizeof(bits));
            first.w[j * IN + k] = bf16_to_f32((uint16_t)(bits >> 16));
        }
    }
    last.b[0] = -0.125f;
    for (int i = 0; i < rows * IN; i++) input[i] = value(i + 71);
    char err[128]; assert(pack_linear(&first, err, sizeof(err)));
    for (int k = 0; k < IN; k++) for (int j = W; j < first.packed_ld; j++)
        first.packed[(size_t)k * first.packed_ld + j] = NAN;
    for (int i = 0; i <= rows * W; i++) hidden[i] = NAN;
    for (int i = 0; i <= rows; i++) output[i] = NAN;
    lin(&first, input, rows, hidden);
    for (int i = 0; i < rows * W; i++) hidden[i] = gelu(hidden[i]);
    lin(&last, hidden, rows, output);
    assert(isnan(hidden[rows * W]) && isnan(output[rows]));
    for (int r = 0; r < rows; r++) {
        double reference = last.b[0];
        for (int j = 0; j < W; j++) {
            double h = first.b[j];
            for (int k = 0; k < IN; k++) h += (double)input[r * IN + k] * first.w[j * IN + k];
            reference += last.w[j] * (0.5 * h * (1.0 + erf(h / sqrt(2.0))));
        }
        assert(isfinite(output[r]) && fabs(output[r] - reference) < 2e-6);
    }
    free_linear(&first); free_linear(&last); free(input); free(hidden); free(output);
}

int main(void) {
    const int scorer_rows[] = {1, 2, 3, 4, 9, 65};
    for (size_t i = 0; i < sizeof(scorer_rows) / sizeof(scorer_rows[0]); i++) check_scorer(scorer_rows[i]);
    const int shapes[][2] = {{63, 65}, {1024, 1024}, {4096, 1024}, {5120, 1024}, {1024, 4096}, {1024, 3072}};
    const int rows[] = {1, 2, 3, 4, 9, 32, 65, 127, 128, 129, 512, 513, 1024};
    int checks = 0;
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++)
        for (size_t r = 0; r < sizeof(rows) / sizeof(rows[0]); r++) {
            check(shapes[s][0], shapes[s][1], rows[r]); checks++;
        }
    char err[128];
    linear invalid = { .out = INT_MAX, .in = 1024 };
    assert(!pack_linear(&invalid, err, sizeof(err)) && strstr(err, "invalid"));
    assert(invalid.packed == NULL);
    linear failed = { .out = 64, .in = 64 };
    fail_pack_allocation = true;
    assert(!pack_linear(&failed, err, sizeof(err)) && strstr(err, "out of memory"));
    fail_pack_allocation = false;
    assert(failed.packed == NULL && failed.packed_ld == 0);

    /* Decoder self_in shares original QKV weights/bias, owns its packed copy. */
    clef_head *h = calloc(1, sizeof(*h)); assert(h);
    h->n_dec = 1;
    linear *q = &h->dec[0].self_attn.in_q;
    q->w = calloc(3 * 64 * 64, sizeof(float)); q->b = calloc(3 * 64, sizeof(float));
    assert(q->w && q->b);
    h->dec[0].self_in = (linear){ q->w, q->b, 3 * 64, 64, NULL, 0 };
    assert(pack_linear(&h->dec[0].self_in, err, sizeof(err)));
    clef_head_free(h);
    printf("head linear: %d exact layout checks; single-row fallback, padding, scorer FP64 reference and ownership OK\n", checks);
    return 0;
}
