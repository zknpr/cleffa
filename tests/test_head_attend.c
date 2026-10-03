/* Regression: the head's decoder self-attention reads Q/K/V interleaved in one [nq][3W]
 * buffer; attend() once read Q with stride W, corrupting every request with >= 2
 * questions (nq=1 was unaffected). Compares attend() with a naive reference. */
#include "../clef_head.c"
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
    int fails = 0;
    const int W = 64, heads = 4;
    for (int nq = 1; nq <= 3; nq++) {
        float *fq = malloc(sizeof(float) * nq * 3 * W), *o1 = calloc(nq * W, 4), *o2 = calloc(nq * W, 4);
        for (int i = 0; i < nq * 3 * W; i++) fq[i] = sinf(i * 0.7f) * 2.0f;
        attend(fq, 3 * W, nq, fq + W, fq + 2 * W, 3 * W, nq, NULL, W, heads, o1);   /* decoder self-attention call */
        ref_attend(fq, nq, W, heads, o2);
        double md = 0; for (int i = 0; i < nq * W; i++) md = fmax(md, fabs(o1[i] - o2[i]));
        printf("nq=%d  max|clef - ref| = %.3e  %s\n", nq, md, md < 1e-4 ? "OK" : "MISMATCH");
        fails += md >= 1e-4;
        free(fq); free(o1); free(o2);
    }
    return fails != 0;
}
