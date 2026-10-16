/* Regression: the head's decoder self-attention reads Q/K/V interleaved in one [nq][3W]
 * buffer; attend() once read Q with stride W, corrupting every request with >= 2
 * questions (nq=1 was unaffected). Compares attend() with a naive reference. */
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
static bool fail_score_allocation;
static size_t score_allocation_calls, score_allocation_bytes;
static void *score_malloc(size_t n) {
    if (fail_score_allocation) {
        score_allocation_calls++; score_allocation_bytes = n;
        return NULL;
    }
    return malloc(n);
}
#define malloc score_malloc
#include "../clef_head.c"
#undef malloc

static void check_plan(int nq, int L, int heads, int requested, int expected_parts) {
    int parts; size_t bytes;
    assert(attend_plan(nq, L, heads, requested, &parts, &bytes));
    assert(parts == expected_parts && bytes == (size_t)parts * nq * L * sizeof(float));
    if (parts > 2) assert(bytes <= 8u * 1024u * 1024u);
    float out = NAN;
    fail_score_allocation = true;
    score_allocation_calls = score_allocation_bytes = 0;
    assert(!attend(NULL, 1024, nq, NULL, NULL, 2048, L, NULL, 1024, heads, requested, &out));
    fail_score_allocation = false;
    assert(score_allocation_calls == 1 && score_allocation_bytes == bytes && isnan(out));
}
static void check_planner(void) {
    check_plan(1, 4095, 16, 8, 1); check_plan(1, 4096, 16, 8, 8);
    check_plan(16, 16384, 16, 8, 8); check_plan(16, 16385, 16, 8, 4);
    check_plan(32, 16384, 16, 8, 4); check_plan(32, 16385, 16, 8, 2);
    check_plan(1024, 16347, 16, 8, 2); check_plan(16384, 16384, 16, 8, 2);
    check_plan(9, 8192, 8, 8, 8); check_plan(9, 8192, 6, 8, 2); check_plan(9, 8192, 3, 8, 1);
    check_plan(1, 4096, 16, 2, 2); check_plan(9, 8192, 16, 2, 2);
    check_plan(9, 8192, 16, 4, 4);
    int parts; size_t bytes; float out = NAN;
    fail_score_allocation = true; score_allocation_calls = 0;
    assert(!attend_plan(INT_MAX, INT_MAX, 16, 8, &parts, &bytes));
    assert(!attend(NULL, 1024, INT_MAX, NULL, NULL, 2048, INT_MAX, NULL, 1024, 16, 8, &out));
    assert(!attend(NULL, 1024, 1, NULL, NULL, 2048, 1, NULL, 1024, 0, 8, &out));
    fail_score_allocation = false;
    assert(!score_allocation_calls && isnan(out));
    puts("head attention: worker/scratch boundaries and allocation failures PASS");
}
static void ref_attend(const float *fq, int nq, int W, int heads, float *out) {
    int hd = W / heads;
    for (int h = 0; h < heads; h++)
        for (int r = 0; r < nq; r++) {
            double s[64], m = -1e300, z = 0;
            for (int c = 0; c < nq; c++) {
                double d = 0;
                for (int i = 0; i < hd; i++) d += (double)fq[r*3*W + h*hd + i] * fq[c*3*W + W + h*hd + i];
                s[c] = d / sqrt((double)hd); if (s[c] > m) m = s[c];
            }
            for (int c = 0; c < nq; c++) { s[c] = exp(s[c] - m); z += s[c]; }
            for (int i = 0; i < hd; i++) {
                double a = 0;
                for (int c = 0; c < nq; c++) a += s[c] / z * fq[c*3*W + 2*W + h*hd + i];
                out[r*W + h*hd + i] = (float)a;
            }
        }
}
int main(void) {
    check_planner();
    int fails = 0;
    const int W = 64, heads = 4;
    for (int nq = 1; nq <= 3; nq++) {
        float *fq = malloc(sizeof(float) * nq * 3 * W), *o1 = calloc(nq * W, 4), *o2 = calloc(nq * W, 4);
        for (int i = 0; i < nq * 3 * W; i++) fq[i] = sinf(i * 0.7f) * 2.0f;
        attend(fq, 3 * W, nq, fq + W, fq + 2 * W, 3 * W, nq, NULL, W, heads, 8, o1);   /* decoder self-attention call */
        ref_attend(fq, nq, W, heads, o2);
        double md = 0; for (int i = 0; i < nq * W; i++) md = fmax(md, fabs(o1[i] - o2[i]));
        printf("nq=%d  max|clef - ref| = %.3e  %s\n", nq, md, md < 1e-4 ? "OK" : "MISMATCH");
        fails += md >= 1e-4;
        free(fq); free(o1); free(o2);
    }
    return fails != 0;
}
